use crate::{
    CODEBOOK_SIZE, GROUP_SIZE, ModelImportPlan, SafeTensorFile, TensorDisposition, VqConfig,
    VqError, VqMatrix,
};
use hpsi_format::{DType, ModelWriter, StreamingTensorInput, WriterStats};
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static IMPORT_SPOOL_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const COPY_CHUNK_BYTES: usize = 64 * 1024;

struct ImportSpool {
    path: PathBuf,
    file: Option<std::fs::File>,
}
impl ImportSpool {
    fn create() -> Result<Self, VqError> {
        for _ in 0..128 {
            let sequence = IMPORT_SPOOL_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                ".hpsi-import-{}-{sequence}.tmp",
                std::process::id()
            ));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => {
                    return Ok(Self {
                        path,
                        file: Some(file),
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(_) => return Err(VqError("could not create model import spool")),
            }
        }
        Err(VqError("could not allocate a unique model import spool"))
    }
}
impl Read for ImportSpool {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        self.file.as_mut().expect("spool is open").read(bytes)
    }
}
impl Write for ImportSpool {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.file.as_mut().expect("spool is open").write(bytes)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.as_mut().expect("spool is open").flush()
    }
}
impl Seek for ImportSpool {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.file.as_mut().expect("spool is open").seek(position)
    }
}
impl Drop for ImportSpool {
    fn drop(&mut self) {
        drop(self.file.take());
        let _ = fs::remove_file(&self.path);
    }
}

fn source_dtype(dtype: &str) -> Result<DType, VqError> {
    match dtype {
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::Bf16),
        "F32" => Ok(DType::F32),
        "F64" => Ok(DType::F64),
        "I64" => Ok(DType::I64),
        "I32" => Ok(DType::I32),
        "I16" => Ok(DType::I16),
        "I8" => Ok(DType::I8),
        "U8" => Ok(DType::U8),
        // HPSI has no BOOL scalar dtype; preserve its exact bytes as opaque.
        "BOOL" => Ok(DType::Opaque),
        _ => Err(VqError("source tensor dtype cannot be represented in HPSI")),
    }
}

/// Write a complete deterministic Safetensors import as one HPSI artifact.
/// Intermediate payloads are disk-backed; memory is bounded by VQ samples,
/// one source row, codebooks, directory metadata, and a 64 KiB transfer buffer.
pub fn write_model_import_to_hpsi<W: Write + Seek>(
    source: &SafeTensorFile,
    plan: &ModelImportPlan,
    output: &mut W,
    config: VqConfig,
    block_rows: usize,
) -> Result<WriterStats, VqError> {
    if block_rows == 0 || plan.tensors.is_empty() {
        return Err(VqError(
            "model import requires tensors and non-zero block rows",
        ));
    }
    let mut spool = ImportSpool::create()?;
    let mut tensors = Vec::<StreamingTensorInput>::new();
    let mut block_offsets = Vec::<Vec<u64>>::new();
    let mut provenance = Vec::with_capacity(plan.tensors.len());
    let mut next_id = 0_u32;

    for planned in &plan.tensors {
        let info = source
            .tensors()
            .iter()
            .find(|item| item.name == planned.source_name)
            .ok_or(VqError("import plan references a missing source tensor"))?;
        if info.dtype != planned.source_dtype || info.shape != planned.source_shape {
            return Err(VqError(
                "import plan no longer matches source tensor metadata",
            ));
        }
        provenance.push(json!({
            "name": planned.source_name,
            "source_dtype": planned.source_dtype,
            "source_shape": planned.source_shape,
            "disposition": match planned.disposition {
                TensorDisposition::QuantizeVq => "quantize_vq",
                TensorDisposition::PreserveRaw => "preserve_raw"
            }
        }));

        match planned.disposition {
            TensorDisposition::PreserveRaw => {
                let length = info
                    .data_end
                    .checked_sub(info.data_start)
                    .ok_or(VqError("invalid raw source payload range"))?;
                let start = spool
                    .seek(SeekFrom::End(0))
                    .map_err(|_| VqError("could not position model import spool"))?;
                let copied = source
                    .copy_raw_payload(&planned.source_name, &mut spool, COPY_CHUNK_BYTES)
                    .map_err(|_| VqError("could not spool raw source tensor"))?;
                if copied != length {
                    return Err(VqError("raw source tensor copy length mismatch"));
                }
                let mut lengths = Vec::new();
                let mut offsets = Vec::new();
                let mut remaining = length;
                let mut offset = start;
                while remaining > 0 {
                    let count = remaining.min(COPY_CHUNK_BYTES as u64);
                    lengths.push(count);
                    offsets.push(offset);
                    offset = offset
                        .checked_add(count)
                        .ok_or(VqError("raw spool offset overflow"))?;
                    remaining -= count;
                }
                let opaque = planned.source_dtype == "BOOL";
                let dimensions = if opaque {
                    vec![length]
                } else {
                    planned.source_shape.clone()
                };
                tensors.push(StreamingTensorInput {
                    tensor_id: next_id,
                    name: planned.source_name.clone(),
                    dtype: source_dtype(&planned.source_dtype)?,
                    dimensions,
                    block_lengths: lengths,
                });
                block_offsets.push(offsets);
                next_id = next_id
                    .checked_add(1)
                    .ok_or(VqError("HPSI tensor ID overflow"))?;
            }
            TensorDisposition::QuantizeVq => {
                if info.shape.len() != 2 || !matches!(info.dtype.as_str(), "F32" | "F16" | "BF16") {
                    return Err(VqError("planned VQ tensor is no longer quantizable"));
                }
                let rows = usize::try_from(info.shape[0])
                    .map_err(|_| VqError("row count exceeds address space"))?;
                let cols = usize::try_from(info.shape[1])
                    .map_err(|_| VqError("column count exceeds address space"))?;
                let selectors_per_row = (cols / GROUP_SIZE)
                    .checked_mul(config.codebook_count)
                    .ok_or(VqError("selector count overflow"))?;
                if selectors_per_row == 0 || selectors_per_row % 2 != 0 {
                    return Err(VqError("VQ selector rows are not byte aligned"));
                }
                let bytes_per_row = selectors_per_row / 2;
                let codebook = if let Some(importance) = planned.activation_second_moment.as_deref()
                {
                    VqMatrix::encode_from_reader_activation_aware_to(
                        rows,
                        cols,
                        config,
                        importance,
                        |first, count| {
                            source
                                .read_f32_elements(&planned.source_name, first, count)
                                .map_err(|_| VqError("Safetensors VQ source read failed"))
                        },
                        |_, row| {
                            spool
                                .write_all(row)
                                .map_err(|_| VqError("could not spool VQ selector row"))?;
                            Ok(())
                        },
                    )?
                } else {
                    VqMatrix::encode_from_reader_to(
                        rows,
                        cols,
                        config,
                        |first, count| {
                            source
                                .read_f32_elements(&planned.source_name, first, count)
                                .map_err(|_| VqError("Safetensors VQ source read failed"))
                        },
                        |_, row| {
                            spool
                                .write_all(row)
                                .map_err(|_| VqError("could not spool VQ selector row"))?;
                            Ok(())
                        },
                    )?
                };
                // Selector rows were emitted first; append the small codebook and
                // reference each payload by its independent spool offset.
                let indices_start = spool
                    .seek(SeekFrom::End(0))
                    .map_err(|_| VqError("could not inspect VQ spool"))?
                    .checked_sub(
                        (rows as u64)
                            .checked_mul(bytes_per_row as u64)
                            .ok_or(VqError("selector payload size overflow"))?,
                    )
                    .ok_or(VqError("selector spool length underflow"))?;
                let codebook_start = spool
                    .seek(SeekFrom::End(0))
                    .map_err(|_| VqError("could not position codebook spool"))?;
                let mut codebook_bytes = Vec::with_capacity(codebook.len() * 4);
                for value in &codebook {
                    codebook_bytes.extend_from_slice(&value.to_le_bytes());
                }
                spool
                    .write_all(&codebook_bytes)
                    .map_err(|_| VqError("could not spool VQ codebook"))?;
                let codebook_len = codebook_bytes.len() as u64;
                if codebook_len == 0 {
                    return Err(VqError("VQ codebook is empty"));
                }

                let codebook_name = planned
                    .codebook_name
                    .as_deref()
                    .ok_or(VqError("VQ plan has no codebook name"))?;
                let indices_name = planned
                    .indices_name
                    .as_deref()
                    .ok_or(VqError("VQ plan has no indices name"))?;
                tensors.push(StreamingTensorInput {
                    tensor_id: next_id,
                    name: codebook_name.to_owned(),
                    dtype: DType::F32,
                    dimensions: vec![
                        config.codebook_count as u64,
                        CODEBOOK_SIZE as u64,
                        GROUP_SIZE as u64,
                    ],
                    block_lengths: vec![codebook_len],
                });
                block_offsets.push(vec![codebook_start]);
                next_id = next_id
                    .checked_add(1)
                    .ok_or(VqError("HPSI tensor ID overflow"))?;

                let total_indices = (rows as u64)
                    .checked_mul(bytes_per_row as u64)
                    .ok_or(VqError("selector payload size overflow"))?;
                let mut lengths = Vec::new();
                let mut offsets = Vec::new();
                let mut first_row = 0usize;
                let mut byte_offset = indices_start;
                while first_row < rows {
                    let count = block_rows.min(rows - first_row);
                    let length = (count as u64)
                        .checked_mul(bytes_per_row as u64)
                        .ok_or(VqError("selector block size overflow"))?;
                    lengths.push(length);
                    offsets.push(byte_offset);
                    byte_offset = byte_offset
                        .checked_add(length)
                        .ok_or(VqError("selector spool offset overflow"))?;
                    first_row += count;
                }
                tensors.push(StreamingTensorInput {
                    tensor_id: next_id,
                    name: indices_name.to_owned(),
                    dtype: DType::Opaque,
                    dimensions: vec![total_indices],
                    block_lengths: lengths,
                });
                block_offsets.push(offsets);
                next_id = next_id
                    .checked_add(1)
                    .ok_or(VqError("HPSI tensor ID overflow"))?;
            }
        }
    }
    spool
        .flush()
        .map_err(|_| VqError("could not flush model import spool"))?;
    let metadata = json!({
        "producer": {"name": "hpsi-safetensors-import", "version": env!("CARGO_PKG_VERSION")},
        "model": {"source_format": "safetensors"},
        "user": {"source_tensors": provenance}
    });
    ModelWriter::write_streaming_with_metadata(
        output,
        &tensors,
        Some(metadata),
        |tensor_index, block_index, sink| {
            let length = *tensors
                .get(tensor_index)
                .and_then(|tensor| tensor.block_lengths.get(block_index))
                .ok_or(hpsi_format::FormatError::InvalidBlock(
                    "unexpected model import block",
                ))?;
            let mut remaining = length;
            let mut offset = *block_offsets
                .get(tensor_index)
                .and_then(|blocks| blocks.get(block_index))
                .ok_or(hpsi_format::FormatError::InvalidBlock(
                    "missing model import spool offset",
                ))?;
            let mut buffer = [0_u8; COPY_CHUNK_BYTES];
            while remaining > 0 {
                let count = remaining.min(buffer.len() as u64) as usize;
                spool
                    .seek(SeekFrom::Start(offset))
                    .map_err(|error| hpsi_format::FormatError::Io(error.to_string()))?;
                spool
                    .read_exact(&mut buffer[..count])
                    .map_err(|error| hpsi_format::FormatError::Io(error.to_string()))?;
                sink.write_all(&buffer[..count])
                    .map_err(|error| hpsi_format::FormatError::Io(error.to_string()))?;
                remaining -= count as u64;
                offset = offset.checked_add(count as u64).ok_or(
                    hpsi_format::FormatError::ArithmeticOverflow("model import spool offset"),
                )?;
            }
            Ok(length)
        },
    )
    .map_err(|_| VqError("HPSI model import writer failed"))
}
