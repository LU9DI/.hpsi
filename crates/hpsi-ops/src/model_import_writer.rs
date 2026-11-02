use crate::{
    CODEBOOK_SIZE, GROUP_SIZE, ModelImportPlan, SafeTensorFile, TensorDisposition, VqConfig,
    VqError, VqMatrix,
};
use hpsi_format::{
    DType, FileSource, ModelReader, ModelWriter, ReaderLimits, StreamingTensorInput, WriterStats,
};
use serde_json::json;
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static IMPORT_SPOOL_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const COPY_CHUNK_BYTES: usize = 64 * 1024;
const MAX_IMPORT_TRAINING_SAMPLES: usize = 262_144;
const MAX_IMPORT_VQ_ITERATIONS: usize = 256;
const MAX_IMPORT_OUTPUT_TENSORS: usize = 250_000;
const MAX_IMPORT_OUTPUT_BLOCKS: usize = 1_000_000;
const MAX_IMPORT_BLOCK_BYTES: u64 = 64 * 1024 * 1024;

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

static IMPORT_OUTPUT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Build, reopen, verify, and publish a model import without exposing a partial
/// destination. The destination is never overwritten; temporary output is
/// created in the same directory so publication can use a hard link.
pub fn write_model_import_to_hpsi_path(
    source: &SafeTensorFile,
    plan: &ModelImportPlan,
    destination: impl AsRef<Path>,
    config: VqConfig,
    block_rows: usize,
) -> Result<WriterStats, VqError> {
    let destination = destination.as_ref();
    if destination.file_name().is_none() {
        return Err(VqError("destination must name a file"));
    }
    let parent = destination
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let (temporary_path, mut temporary_file) = {
        let mut created = None;
        for _ in 0..128 {
            let sequence = IMPORT_OUTPUT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let candidate = parent.join(format!(
                ".hpsi-import-{}-{sequence}.tmp",
                std::process::id()
            ));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(file) => {
                    created = Some((candidate, file));
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(_) => return Err(VqError("could not create temporary HPSI output")),
            }
        }
        created.ok_or(VqError("could not allocate temporary HPSI output"))?
    };
    let result = (|| {
        let stats =
            write_model_import_to_hpsi(source, plan, &mut temporary_file, config, block_rows)?;
        temporary_file
            .sync_all()
            .map_err(|_| VqError("could not sync temporary HPSI output"))?;
        drop(temporary_file);
        let file_source = Arc::new(
            FileSource::open(&temporary_path)
                .map_err(|_| VqError("could not reopen temporary HPSI output"))?,
        );
        let reader = ModelReader::open(file_source, &ReaderLimits::default())
            .map_err(|_| VqError("temporary HPSI output failed to reopen"))?;
        reader
            .verify_all()
            .map_err(|_| VqError("temporary HPSI output failed integrity verification"))?;
        fs::hard_link(&temporary_path, destination)
            .map_err(|_| VqError("could not publish HPSI output without overwriting"))?;
        #[cfg(unix)]
        if let Ok(directory) = fs::File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok(stats)
    })();
    let _ = fs::remove_file(&temporary_path);
    result
}

/// Write a complete deterministic Safetensors import as one HPSI artifact.
///
/// On error, `output` may contain a partial artifact. Callers requiring
/// all-or-nothing publication should use `write_model_import_to_hpsi_path`.
/// Intermediate payloads are disk-backed; memory is bounded by VQ samples,
/// one source row, codebooks, directory metadata, and a 64 KiB transfer buffer.
pub fn write_model_import_to_hpsi<W: Write + Seek>(
    source: &SafeTensorFile,
    plan: &ModelImportPlan,
    output: &mut W,
    config: VqConfig,
    block_rows: usize,
) -> Result<WriterStats, VqError> {
    if block_rows == 0 || plan.tensors.is_empty() || plan.tensors.len() != source.tensors().len() {
        return Err(VqError(
            "model import plan must cover every source tensor and use non-zero block rows",
        ));
    }

    if config.sample_limit == 0
        || config.sample_limit > MAX_IMPORT_TRAINING_SAMPLES
        || config.iterations == 0
        || config.iterations > MAX_IMPORT_VQ_ITERATIONS
        || !(1..=4).contains(&config.codebook_count)
        || !(1..=256).contains(&config.beam_width)
    {
        return Err(VqError(
            "VQ import configuration exceeds supported resource limits",
        ));
    }

    // The plan is a public data structure and may have been constructed or
    // modified by a caller. Revalidate its full source coverage, ordering,
    // disposition fields and generated names before creating temporary files.
    let mut source_names = HashSet::new();
    source_names
        .try_reserve(source.tensors().len())
        .map_err(|_| VqError("could not reserve source tensor name index"))?;
    for info in source.tensors() {
        if !source_names.insert(info.name.as_str()) {
            return Err(VqError("source tensor names are not unique"));
        }
    }
    // The plan is sorted by source name. Build one sorted reference index so
    // validation and emission resolve each source tensor in O(log n), rather
    // than rescanning the complete source directory for every planned tensor.
    let mut source_by_name = Vec::new();
    source_by_name
        .try_reserve_exact(source.tensors().len())
        .map_err(|_| VqError("could not reserve sorted source tensor index"))?;
    source_by_name.extend(source.tensors().iter());
    source_by_name.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    let mut planned_names = HashSet::new();
    planned_names
        .try_reserve(plan.tensors.len())
        .map_err(|_| VqError("could not reserve planned tensor name index"))?;
    let generated_capacity = plan
        .tensors
        .len()
        .checked_mul(2)
        .ok_or(VqError("generated tensor name count overflow"))?;
    let mut generated_names = HashSet::new();
    generated_names
        .try_reserve(generated_capacity)
        .map_err(|_| VqError("could not reserve generated tensor name index"))?;
    let mut previous_name: Option<&str> = None;
    for planned in &plan.tensors {
        if previous_name.is_some_and(|previous| previous >= planned.source_name.as_str()) {
            return Err(VqError(
                "import plan must be uniquely sorted by source name",
            ));
        }
        previous_name = Some(&planned.source_name);
        if !planned_names.insert(planned.source_name.as_str())
            || !source_names.contains(planned.source_name.as_str())
        {
            return Err(VqError(
                "import plan contains duplicate or unknown source tensor names",
            ));
        }
        let info_index = source_by_name
            .binary_search_by(|item| item.name.as_str().cmp(planned.source_name.as_str()))
            .map_err(|_| VqError("import plan references a missing source tensor"))?;
        let info = source_by_name[info_index];
        if info.dtype != planned.source_dtype || info.shape != planned.source_shape {
            return Err(VqError(
                "import plan no longer matches source tensor metadata",
            ));
        }
        match planned.disposition {
            TensorDisposition::QuantizeVq => {
                if info.shape.len() != 2
                    || info.shape[1] % GROUP_SIZE as u64 != 0
                    || !matches!(info.dtype.as_str(), "F32" | "F16" | "BF16")
                {
                    return Err(VqError(
                        "planned VQ tensor is not eligible for quantization",
                    ));
                }
                let codebook = planned
                    .codebook_name
                    .as_deref()
                    .ok_or(VqError("VQ plan has no codebook name"))?;
                let indices = planned
                    .indices_name
                    .as_deref()
                    .ok_or(VqError("VQ plan has no indices name"))?;
                if codebook.is_empty()
                    || indices.is_empty()
                    || codebook == indices
                    || source_names.contains(codebook)
                    || source_names.contains(indices)
                    || !generated_names.insert(codebook)
                    || !generated_names.insert(indices)
                {
                    return Err(VqError(
                        "VQ plan contains invalid or colliding generated names",
                    ));
                }
                if let Some(moment) = planned.activation_second_moment.as_deref() {
                    if u64::try_from(moment.len()).ok() != Some(info.shape[1])
                        || moment
                            .iter()
                            .any(|value| !value.is_finite() || *value < 0.0)
                        || !planned
                            .calibration_samples
                            .is_some_and(|samples| samples > 0)
                    {
                        return Err(VqError("VQ plan contains invalid activation calibration"));
                    }
                } else if planned.calibration_samples.is_some() {
                    return Err(VqError(
                        "VQ calibration sample count has no activation moments",
                    ));
                }
            }
            TensorDisposition::PreserveRaw => {
                if planned.codebook_name.is_some()
                    || planned.indices_name.is_some()
                    || planned.calibration_samples.is_some()
                    || planned.activation_second_moment.is_some()
                {
                    return Err(VqError("raw-preserved tensor must not carry VQ metadata"));
                }
            }
        }
    }
    if planned_names.len() != source_names.len() {
        return Err(VqError(
            "import plan does not cover the complete source tensor set",
        ));
    }

    // Reject output layouts that cannot fit the HPSI writer limits before
    // creating spools or training any VQ codebooks.
    let mut output_tensor_count = 0usize;
    let mut output_block_count = 0usize;
    for planned in &plan.tensors {
        let info_index = source_by_name
            .binary_search_by(|item| item.name.as_str().cmp(planned.source_name.as_str()))
            .map_err(|_| VqError("import plan references a missing source tensor"))?;
        let info = source_by_name[info_index];
        let (tensor_count, block_count) = match planned.disposition {
            TensorDisposition::PreserveRaw => {
                let length = info
                    .data_end
                    .checked_sub(info.data_start)
                    .ok_or(VqError("invalid raw source payload range"))?;
                if length == 0 {
                    return Err(VqError(
                        "empty raw tensors cannot be emitted as HPSI blocks",
                    ));
                }
                let blocks = length
                    .checked_add(COPY_CHUNK_BYTES as u64 - 1)
                    .ok_or(VqError("raw block count overflow"))?
                    / COPY_CHUNK_BYTES as u64;
                (
                    1usize,
                    usize::try_from(blocks)
                        .map_err(|_| VqError("raw block count exceeds address space"))?,
                )
            }
            TensorDisposition::QuantizeVq => {
                let rows = usize::try_from(info.shape[0])
                    .map_err(|_| VqError("row count exceeds address space"))?;
                let cols = usize::try_from(info.shape[1])
                    .map_err(|_| VqError("column count exceeds address space"))?;
                let bytes_per_row = (cols / GROUP_SIZE)
                    .checked_mul(config.codebook_count)
                    .ok_or(VqError("selector count overflow"))?
                    / 2;
                let selector_block_bytes = (block_rows as u64)
                    .checked_mul(bytes_per_row as u64)
                    .ok_or(VqError("selector block size overflow"))?;
                if selector_block_bytes > MAX_IMPORT_BLOCK_BYTES {
                    return Err(VqError("selector block exceeds HPSI payload limit"));
                }
                let selector_blocks = rows
                    .checked_add(block_rows - 1)
                    .ok_or(VqError("selector block count overflow"))?
                    / block_rows;
                (
                    2usize,
                    selector_blocks
                        .checked_add(1)
                        .ok_or(VqError("VQ block count overflow"))?,
                )
            }
        };
        output_tensor_count = output_tensor_count
            .checked_add(tensor_count)
            .ok_or(VqError("output tensor count overflow"))?;
        output_block_count = output_block_count
            .checked_add(block_count)
            .ok_or(VqError("output block count overflow"))?;
        if output_tensor_count > MAX_IMPORT_OUTPUT_TENSORS {
            return Err(VqError("model import exceeds HPSI tensor count limit"));
        }
        if output_block_count > MAX_IMPORT_OUTPUT_BLOCKS {
            return Err(VqError("model import exceeds HPSI block count limit"));
        }
    }

    let mut tensors = Vec::<StreamingTensorInput>::new();
    tensors
        .try_reserve_exact(output_tensor_count)
        .map_err(|_| VqError("could not reserve HPSI tensor descriptors"))?;
    let mut block_offsets = Vec::<Vec<u64>>::new();
    block_offsets
        .try_reserve_exact(output_tensor_count)
        .map_err(|_| VqError("could not reserve HPSI block offset tables"))?;
    let mut provenance = Vec::new();
    provenance
        .try_reserve_exact(plan.tensors.len())
        .map_err(|_| VqError("could not reserve import provenance"))?;
    let mut spool = ImportSpool::create()?;
    let mut next_id = 0_u32;

    for planned in &plan.tensors {
        let info_index = source_by_name
            .binary_search_by(|item| item.name.as_str().cmp(planned.source_name.as_str()))
            .map_err(|_| VqError("import plan references a missing source tensor"))?;
        let info = source_by_name[info_index];
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
                let block_count = usize::try_from(
                    length / COPY_CHUNK_BYTES as u64
                        + u64::from(length % COPY_CHUNK_BYTES as u64 != 0),
                )
                .map_err(|_| VqError("raw block count exceeds address space"))?;
                let mut lengths = Vec::new();
                lengths
                    .try_reserve_exact(block_count)
                    .map_err(|_| VqError("could not reserve raw block lengths"))?;
                let mut offsets = Vec::new();
                offsets
                    .try_reserve_exact(block_count)
                    .map_err(|_| VqError("could not reserve raw block offsets"))?;
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
                    VqMatrix::encode_from_reader_activation_aware_into(
                        rows,
                        cols,
                        config,
                        importance,
                        |first, target| {
                            source
                                .read_f32_elements_into(&planned.source_name, first, target)
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
                    VqMatrix::encode_from_reader_into(
                        rows,
                        cols,
                        config,
                        |first, target| {
                            source
                                .read_f32_elements_into(&planned.source_name, first, target)
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
                let codebook_len = u64::try_from(codebook.len())
                    .ok()
                    .and_then(|count| count.checked_mul(4))
                    .ok_or(VqError("VQ codebook byte length overflow"))?;
                for value in &codebook {
                    spool
                        .write_all(&value.to_le_bytes())
                        .map_err(|_| VqError("could not spool VQ codebook"))?;
                }
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
                let selector_block_count = rows / block_rows
                    + usize::from(rows % block_rows != 0);
                let mut lengths = Vec::new();
                lengths
                    .try_reserve_exact(selector_block_count)
                    .map_err(|_| VqError("could not reserve selector block lengths"))?;
                let mut offsets = Vec::new();
                offsets
                    .try_reserve_exact(selector_block_count)
                    .map_err(|_| VqError("could not reserve selector block offsets"))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ActivationCalibration, ModelImportPlan, UnsupportedTensorPolicy};
    use hpsi_format::{MemorySource, ModelReader, ReaderLimits};
    use std::io::Cursor;
    use std::sync::Arc;

    fn fixture() -> (SafeTensorFile, Vec<u8>, Vec<f32>) {
        let header = br#"{"bias":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"weight":{"dtype":"F32","shape":[2,8],"data_offsets":[8,72]}}"#;
        let bias = [3.25_f32, -1.5_f32];
        let weights: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.19).sin()).collect();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        for value in bias {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in &weights {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let raw_bias = bias.iter().flat_map(|v| v.to_le_bytes()).collect();
        let source =
            SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096)
                .unwrap();
        (source, raw_bias, weights)
    }

    fn config() -> VqConfig {
        VqConfig {
            codebook_count: 2,
            sample_limit: 32,
            ..VqConfig::default()
        }
    }

    #[test]
    fn writes_raw_and_vq_tensors_into_one_verified_hpsi_artifact() {
        let (source, raw_bias, _) = fixture();
        let plan =
            ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation)
                .unwrap();
        let mut output = Cursor::new(Vec::new());
        let stats = write_model_import_to_hpsi(&source, &plan, &mut output, config(), 1).unwrap();
        assert_eq!(stats.tensor_count, 3);

        let reader = ModelReader::open(
            Arc::new(MemorySource::new(Arc::<[u8]>::from(output.into_inner()))),
            &ReaderLimits::default(),
        )
        .unwrap();
        reader.verify_all().unwrap();
        assert_eq!(reader.tensor_count(), 3);
        assert_eq!(
            reader.metadata.tensor_names[reader.tensor(0).unwrap().name_ref as usize],
            "bias"
        );
        assert_eq!(reader.read_block(0, 0).unwrap().payload, raw_bias);
        assert_eq!(
            reader.metadata.tensor_names[reader.tensor(1).unwrap().name_ref as usize],
            "weight.hpsi_vq.codebook"
        );
        assert_eq!(
            reader.metadata.tensor_names[reader.tensor(2).unwrap().name_ref as usize],
            "weight.hpsi_vq.indices"
        );
        assert_eq!(
            reader.metadata.value["user"]["source_tensors"][0]["source_dtype"],
            "F32"
        );
    }

    #[test]
    fn imports_multiple_vq_matrices_with_unique_deterministic_ids() {
        let header = br#"{"a":{"dtype":"F32","shape":[2,8],"data_offsets":[0,64]},"b":{"dtype":"F32","shape":[2,8],"data_offsets":[64,128]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        for index in 0..32 {
            bytes.extend_from_slice(&((index as f32) * 0.11).cos().to_le_bytes());
        }
        let source =
            SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096)
                .unwrap();
        let plan =
            ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation)
                .unwrap();
        let mut output = Cursor::new(Vec::new());
        let stats = write_model_import_to_hpsi(&source, &plan, &mut output, config(), 1).unwrap();
        assert_eq!(stats.tensor_count, 4);
        let reader = ModelReader::open(
            Arc::new(MemorySource::new(Arc::<[u8]>::from(output.into_inner()))),
            &ReaderLimits::default(),
        )
        .unwrap();
        reader.verify_all().unwrap();
        let names: Vec<_> = reader
            .tensors()
            .iter()
            .map(|tensor| tensor.tensor_id)
            .collect();
        assert_eq!(names, vec![0, 1, 2, 3]);
        assert_eq!(
            reader.metadata.tensor_names[reader.tensor(0).unwrap().name_ref as usize],
            "a.hpsi_vq.codebook"
        );
        assert_eq!(
            reader.metadata.tensor_names[reader.tensor(2).unwrap().name_ref as usize],
            "b.hpsi_vq.codebook"
        );
    }

    #[test]
    fn rejects_import_block_layout_overflow_before_spooling() {
        let (source, _, _) = fixture();
        let plan =
            ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation)
                .unwrap();
        let mut output = Cursor::new(Vec::new());
        assert!(
            write_model_import_to_hpsi(&source, &plan, &mut output, config(), usize::MAX,).is_err()
        );
        assert!(output.get_ref().is_empty());
    }

    #[test]
    fn rejects_plan_that_omits_or_reorders_source_tensors() {
        let (source, _, _) = fixture();
        let mut plan =
            ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation)
                .unwrap();
        plan.tensors.pop();
        let mut output = Cursor::new(Vec::new());
        assert!(write_model_import_to_hpsi(&source, &plan, &mut output, config(), 1).is_err());
    }

    #[test]
    fn imports_vq_tensor_with_activation_calibration() {
        let (source, _, _) = fixture();
        let calibration = ActivationCalibration {
            layer_name: "weight".into(),
            sample_count: 2,
            features: 8,
            activation_second_moment: vec![1.0; 8],
            model_type: "fixture".into(),
            model_revision: None,
            dataset_sha256: "b".repeat(64),
        };
        let plan = ModelImportPlan::build(
            &source,
            &[calibration],
            UnsupportedTensorPolicy::AllowRawPreservation,
        )
        .unwrap();
        let mut output = Cursor::new(Vec::new());
        let stats = write_model_import_to_hpsi(&source, &plan, &mut output, config(), 1).unwrap();
        assert_eq!(stats.tensor_count, 3);
    }
    #[test]
    fn rejects_forged_plan_with_duplicate_or_unsorted_source_names() {
        let (source, _, _) = fixture();
        let mut plan =
            ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation)
                .unwrap();
        plan.tensors.swap(0, 1);
        let mut output = Cursor::new(Vec::new());
        assert!(write_model_import_to_hpsi(&source, &plan, &mut output, config(), 1).is_err());

        let mut plan =
            ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation)
                .unwrap();
        plan.tensors[1].source_name = plan.tensors[0].source_name.clone();
        let mut output = Cursor::new(Vec::new());
        assert!(write_model_import_to_hpsi(&source, &plan, &mut output, config(), 1).is_err());
    }

    #[test]
    fn rejects_forged_vq_names_and_unbounded_training_configuration() {
        let (source, _, _) = fixture();
        let mut plan =
            ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation)
                .unwrap();
        plan.tensors[1].codebook_name = Some("bias".into());
        let mut output = Cursor::new(Vec::new());
        assert!(write_model_import_to_hpsi(&source, &plan, &mut output, config(), 1).is_err());

        let plan =
            ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation)
                .unwrap();
        let mut excessive = config();
        excessive.sample_limit = usize::MAX;
        let mut output = Cursor::new(Vec::new());
        assert!(write_model_import_to_hpsi(&source, &plan, &mut output, excessive, 1).is_err());
    }

    #[test]
    fn rejects_vq_metadata_on_raw_preserved_tensors() {
        let (source, _, _) = fixture();
        let mut plan =
            ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation)
                .unwrap();
        plan.tensors[0].activation_second_moment = Some(vec![1.0; 2]);
        plan.tensors[0].calibration_samples = Some(1);
        let mut output = Cursor::new(Vec::new());
        assert!(write_model_import_to_hpsi(&source, &plan, &mut output, config(), 1).is_err());
    }
}
#[test]
fn import_spool_is_removed_when_dropped() {
    let spool = ImportSpool::create().unwrap();
    let path = spool.path.clone();
    assert!(path.exists());
    drop(spool);
    assert!(!path.exists());
}
