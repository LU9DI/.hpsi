use crate::{
    CODEBOOK_SIZE, CalibratedModelImportPlan, CalibrationProvenance, GROUP_SIZE, ModelImportPlan,
    SafeTensorFile, TensorDisposition, VqConfig, VqError, VqMatrix,
};
use hpsi_format::{
    DType, FileSource, ModelReader, ModelWriter, ReaderLimits, StreamingTensorInput, WriterStats,
};
use serde_json::json;
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
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
const MAX_IMPORT_TENSOR_LOGICAL_BYTES: u64 = 1_u64 << 40;
const MAX_IMPORT_DIRECTORY_BYTES: u64 = 256 * 1024 * 1024;
const MAX_IMPORT_FILE_BYTES: u64 = 1_u64 << 40;

fn open_import_temporary_file(path: &Path) -> std::io::Result<std::fs::File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    options.open(path)
}

struct ImportOutputTemporaryFile {
    path: PathBuf,
    file: Option<std::fs::File>,
}

impl Drop for ImportOutputTemporaryFile {
    fn drop(&mut self) {
        drop(self.file.take());
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(unix)]
fn sync_import_parent_directory(parent: &Path) -> std::io::Result<()> {
    fs::File::open(parent)?.sync_all()
}

#[cfg(not(unix))]
fn sync_import_parent_directory(_parent: &Path) -> std::io::Result<()> {
    Ok(())
}

struct ImportSpool {
    path: PathBuf,
    file: Option<std::fs::File>,
    max_bytes: u64,
}
impl ImportSpool {
    fn create(max_bytes: u64) -> Result<Self, VqError> {
        for _ in 0..128 {
            let sequence = IMPORT_SPOOL_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                ".hpsi-import-{}-{sequence}.tmp",
                std::process::id()
            ));
            match open_import_temporary_file(&path) {
                Ok(file) => {
                    return Ok(Self {
                        path,
                        file: Some(file),
                        max_bytes,
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
        let file = self.file.as_mut().expect("spool is open");
        let position = file.stream_position()?;
        let bytes_len = u64::try_from(bytes.len()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                "import spool write exceeds format range",
            )
        })?;
        let end = position.checked_add(bytes_len).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "import spool size overflow")
        })?;
        if end > self.max_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "import spool exceeded its preflight payload limit",
            ));
        }
        file.write(bytes)
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

fn clone_import_name(name: &str) -> Result<String, VqError> {
    let mut cloned = String::new();
    cloned
        .try_reserve_exact(name.len())
        .map_err(|_| VqError("could not reserve imported tensor name"))?;
    cloned.push_str(name);
    Ok(cloned)
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
        // Safetensors has additional unsigned, sub-byte, FP8 and complex
        // types without a lossless HPSI scalar dtype. Preserve those payloads
        // as opaque bytes rather than rejecting an otherwise valid import.
        "BOOL" | "F4" | "F6_E2M3" | "F6_E3M2" | "U16" | "U32" | "U64" | "F8_E5M2" | "F8_E4M3"
        | "F8_E8M0" | "F8_E4M3FNUZ" | "F8_E5M2FNUZ" | "C64" => Ok(DType::Opaque),
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
    write_model_import_to_hpsi_path_inner(source, plan, None, destination, config, block_rows)
}

/// Publish an import and its calibration provenance atomically.
pub fn write_model_import_to_hpsi_path_calibrated(
    source: &SafeTensorFile,
    calibrated_plan: &CalibratedModelImportPlan,
    destination: impl AsRef<Path>,
    config: VqConfig,
    block_rows: usize,
) -> Result<WriterStats, VqError> {
    write_model_import_to_hpsi_path_inner(
        source,
        &calibrated_plan.plan,
        Some(&calibrated_plan.calibrations),
        destination,
        config,
        block_rows,
    )
}

fn write_model_import_to_hpsi_path_inner(
    source: &SafeTensorFile,
    plan: &ModelImportPlan,
    calibrations: Option<&[CalibrationProvenance]>,
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
    let (temporary_path, temporary_file) = {
        let mut created = None;
        for _ in 0..128 {
            let sequence = IMPORT_OUTPUT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let candidate = parent.join(format!(
                ".hpsi-import-{}-{sequence}.tmp",
                std::process::id()
            ));
            match open_import_temporary_file(&candidate) {
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
    let mut temporary = ImportOutputTemporaryFile {
        path: temporary_path.clone(),
        file: Some(temporary_file),
    };
    (|| {
        let stats = write_model_import_to_hpsi_inner(
            source,
            plan,
            calibrations,
            temporary.file.as_mut().expect("temporary output is open"),
            config,
            block_rows,
        )?;
        temporary
            .file
            .as_mut()
            .expect("temporary output is open")
            .sync_all()
            .map_err(|_| VqError("could not sync temporary HPSI output"))?;
        drop(temporary.file.take());
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
        sync_import_parent_directory(parent).map_err(|_| {
            VqError(
                "HPSI output was published but parent directory sync failed; crash durability is uncertain",
            )
        })?;
        Ok(stats)
    })()
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
    write_model_import_to_hpsi_inner(source, plan, None, output, config, block_rows)
}

/// Write an import artifact with validated calibration provenance.
pub fn write_model_import_to_hpsi_calibrated<W: Write + Seek>(
    source: &SafeTensorFile,
    calibrated_plan: &CalibratedModelImportPlan,
    output: &mut W,
    config: VqConfig,
    block_rows: usize,
) -> Result<WriterStats, VqError> {
    write_model_import_to_hpsi_inner(
        source,
        &calibrated_plan.plan,
        Some(&calibrated_plan.calibrations),
        output,
        config,
        block_rows,
    )
}

fn write_model_import_to_hpsi_inner<W: Write + Seek>(
    source: &SafeTensorFile,
    plan: &ModelImportPlan,
    calibrations: Option<&[CalibrationProvenance]>,
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
    // Validate the optional compact provenance independently of the public
    // wrapper: callers can construct or mutate it directly in Rust.
    let mut calibration_by_name = std::collections::HashMap::new();
    if let Some(calibrations) = calibrations {
        calibration_by_name
            .try_reserve(calibrations.len())
            .map_err(|_| VqError("could not reserve calibration provenance index"))?;
        let mut previous: Option<&str> = None;
        for calibration in calibrations {
            if calibration.source_name.trim().is_empty()
                || previous.is_some_and(|name| name >= calibration.source_name.as_str())
                || calibration.model_type.trim().is_empty()
                || calibration
                    .model_revision
                    .as_deref()
                    .is_some_and(|value| value.trim().is_empty())
                || calibration.sample_count == 0
                || calibration.features == 0
                || calibration.statistic != "diagonal_second_moment"
                || calibration.dataset_sha256.len() != 64
                || !calibration
                    .dataset_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(VqError(
                    "calibration provenance is invalid or not uniquely sorted",
                ));
            }
            previous = Some(&calibration.source_name);
            if calibration_by_name
                .insert(calibration.source_name.as_str(), calibration)
                .is_some()
            {
                return Err(VqError("duplicate calibration provenance tensor name"));
            }
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
        // HPSI tensor descriptors require ranks 1 through 8 and non-zero
        // dimensions. Safetensors also permits scalar and zero-sized shapes,
        // so reject those here rather than after spooling the complete payload.
        if info.shape.is_empty() || info.shape.len() > 8 || info.shape.contains(&0) {
            return Err(VqError("source tensor shape cannot be represented in HPSI"));
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
                        || planned
                            .calibration_samples
                            .is_none_or(|samples| samples == 0)
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
                    || calibration_by_name.contains_key(planned.source_name.as_str())
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
    if let Some(calibrations) = calibrations {
        for planned in &plan.tensors {
            let provenance = calibration_by_name.get(planned.source_name.as_str());
            match (planned.disposition, provenance) {
                (TensorDisposition::QuantizeVq, Some(item)) => {
                    if planned.calibration_samples != Some(item.sample_count)
                        || u64::try_from(item.features).ok() != planned.source_shape.get(1).copied()
                        || planned.activation_second_moment.as_ref().map(Vec::len)
                            != Some(item.features)
                    {
                        return Err(VqError(
                            "calibration provenance does not match planned VQ tensor",
                        ));
                    }
                }
                (TensorDisposition::QuantizeVq, None) => {
                    if planned.calibration_samples.is_some()
                        || planned.activation_second_moment.is_some()
                    {
                        return Err(VqError(
                            "calibrated VQ tensor is missing persisted provenance",
                        ));
                    }
                }
                (TensorDisposition::PreserveRaw, Some(_)) => {
                    return Err(VqError(
                        "raw-preserved tensor cannot have calibration provenance",
                    ));
                }
                (TensorDisposition::PreserveRaw, None) => {}
            }
        }
        if calibration_by_name.len() > plan.tensors.len() {
            return Err(VqError(
                "calibration provenance references an unknown source tensor",
            ));
        }
    }

    // Reject output layouts that cannot fit the HPSI writer limits before
    // creating spools or training any VQ codebooks.
    let mut output_tensor_count = 0usize;
    let mut output_block_count = 0usize;
    let mut output_payload_bytes = 0u64;
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
                if length > MAX_IMPORT_TENSOR_LOGICAL_BYTES {
                    return Err(VqError("raw tensor exceeds HPSI logical byte limit"));
                }
                output_payload_bytes = output_payload_bytes
                    .checked_add(length)
                    .ok_or(VqError("total output payload size overflow"))?;
                let blocks = length / COPY_CHUNK_BYTES as u64
                    + u64::from(length % COPY_CHUNK_BYTES as u64 != 0);
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
                if rows == 0 {
                    return Err(VqError("VQ tensors must contain at least one row"));
                }
                let selectors_per_row = (cols / GROUP_SIZE)
                    .checked_mul(config.codebook_count)
                    .ok_or(VqError("selector count overflow"))?;
                if selectors_per_row == 0 || selectors_per_row % 2 != 0 {
                    return Err(VqError("VQ selector rows are not byte aligned"));
                }
                let bytes_per_row = selectors_per_row / 2;
                let effective_block_rows = block_rows.min(rows);
                let effective_block_rows_u64 = u64::try_from(effective_block_rows)
                    .map_err(|_| VqError("selector block row count exceeds format range"))?;
                let bytes_per_row_u64 = u64::try_from(bytes_per_row)
                    .map_err(|_| VqError("VQ selector row size exceeds format range"))?;
                let selector_block_bytes = effective_block_rows_u64
                    .checked_mul(bytes_per_row_u64)
                    .ok_or(VqError("selector block size overflow"))?;
                if selector_block_bytes == 0 || selector_block_bytes > MAX_IMPORT_BLOCK_BYTES {
                    return Err(VqError("selector block exceeds HPSI payload limit"));
                }
                let rows_u64 = u64::try_from(rows)
                    .map_err(|_| VqError("VQ row count exceeds format range"))?;
                let bytes_per_row_u64 = u64::try_from(bytes_per_row)
                    .map_err(|_| VqError("VQ selector row size exceeds format range"))?;
                let indices_bytes = rows_u64
                    .checked_mul(bytes_per_row_u64)
                    .ok_or(VqError("VQ selector payload size overflow"))?;
                let codebook_bytes = u64::try_from(config.codebook_count)
                    .ok()
                    .and_then(|count| count.checked_mul(CODEBOOK_SIZE as u64))
                    .and_then(|count| count.checked_mul(GROUP_SIZE as u64))
                    .and_then(|count| count.checked_mul(4))
                    .ok_or(VqError("VQ codebook size overflow"))?;
                if indices_bytes == 0
                    || indices_bytes > MAX_IMPORT_TENSOR_LOGICAL_BYTES
                    || codebook_bytes == 0
                    || codebook_bytes > MAX_IMPORT_TENSOR_LOGICAL_BYTES
                {
                    return Err(VqError("VQ output tensor exceeds HPSI logical byte limit"));
                }
                output_payload_bytes = output_payload_bytes
                    .checked_add(indices_bytes)
                    .and_then(|total| total.checked_add(codebook_bytes))
                    .ok_or(VqError("total output payload size overflow"))?;
                let selector_blocks = rows / block_rows + usize::from(rows % block_rows != 0);
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

    let tensor_directory_bytes = u64::try_from(output_tensor_count)
        .ok()
        .and_then(|count| count.checked_mul(128))
        .ok_or(VqError("tensor directory size overflow"))?;
    let block_directory_bytes = u64::try_from(output_block_count)
        .ok()
        .and_then(|count| count.checked_mul(64))
        .ok_or(VqError("block directory size overflow"))?;
    let integrity_directory_bytes = u64::try_from(output_block_count)
        .ok()
        .and_then(|count| count.checked_add(1))
        .and_then(|count| count.checked_mul(32))
        .ok_or(VqError("integrity directory size overflow"))?;
    let directory_bytes = tensor_directory_bytes
        .checked_add(block_directory_bytes)
        .and_then(|total| total.checked_add(integrity_directory_bytes))
        .ok_or(VqError("total directory size overflow"))?;
    if directory_bytes > MAX_IMPORT_DIRECTORY_BYTES {
        return Err(VqError("model import exceeds HPSI directory byte limit"));
    }
    let minimum_file_bytes = 128u64
        .checked_add(directory_bytes)
        .and_then(|total| total.checked_add(output_payload_bytes))
        .ok_or(VqError("minimum HPSI file size overflow"))?;
    if minimum_file_bytes > MAX_IMPORT_FILE_BYTES {
        return Err(VqError("model import exceeds HPSI file size limit"));
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
    let mut metadata_tensor_names = Vec::new();
    metadata_tensor_names
        .try_reserve_exact(output_tensor_count)
        .map_err(|_| VqError("could not reserve metadata tensor names"))?;
    for planned in &plan.tensors {
        let mut source_entry = json!({
            "name": planned.source_name,
            "source_dtype": planned.source_dtype,
            "source_shape": planned.source_shape,
            "disposition": match planned.disposition {
                TensorDisposition::QuantizeVq => "quantize_vq",
                TensorDisposition::PreserveRaw => "preserve_raw"
            }
        });
        if let Some(calibration) = calibration_by_name.get(planned.source_name.as_str()) {
            let mut persisted = json!({
                "model_type": calibration.model_type,
                "dataset_sha256": calibration.dataset_sha256,
                "sample_count": calibration.sample_count,
                "features": calibration.features,
                "statistic": calibration.statistic
            });
            if let Some(revision) = calibration.model_revision.as_deref() {
                persisted["model_revision"] = json!(revision);
            }
            source_entry["calibration"] = persisted;
        }
        provenance.push(source_entry);
        match planned.disposition {
            TensorDisposition::PreserveRaw => {
                metadata_tensor_names.push(clone_import_name(&planned.source_name)?)
            }
            TensorDisposition::QuantizeVq => {
                metadata_tensor_names.push(clone_import_name(
                    planned
                        .codebook_name
                        .as_deref()
                        .ok_or(VqError("VQ plan has no codebook name"))?,
                )?);
                metadata_tensor_names.push(clone_import_name(
                    planned
                        .indices_name
                        .as_deref()
                        .ok_or(VqError("VQ plan has no indices name"))?,
                )?);
            }
        }
    }
    // Build the exact metadata document the writer will emit, validate it,
    // then strip writer-owned fields in place. This avoids cloning the full
    // provenance tree merely to perform the preflight parse.
    let mut metadata = json!({
        "producer": {"name": "hpsi-safetensors-import", "version": env!("CARGO_PKG_VERSION")},
        "model": {"source_format": "safetensors"},
        "user": {"source_tensors": provenance},
        "format": "hpsi",
        "metadata_version": 1,
        "tensor_names": metadata_tensor_names
    });
    let metadata_bytes = hpsi_format::serialize_metadata_limited(&metadata, 16 * 1024 * 1024)
        .map_err(|_| VqError("model import metadata exceeds HPSI byte or allocation limit"))?;
    hpsi_format::Metadata::parse(&metadata_bytes, &ReaderLimits::default())
        .map_err(|_| VqError("model import metadata exceeds HPSI format limits"))?;
    let metadata_object = metadata
        .as_object_mut()
        .ok_or(VqError("import metadata root is not an object"))?;
    metadata_object.remove("format");
    metadata_object.remove("metadata_version");
    metadata_object.remove("tensor_names");
    let mut spool = ImportSpool::create(output_payload_bytes)?;
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
                let opaque = source_dtype(&planned.source_dtype)? == DType::Opaque;
                let dimensions = if opaque {
                    vec![length]
                } else {
                    planned.source_shape.clone()
                };
                tensors.push(StreamingTensorInput {
                    tensor_id: next_id,
                    name: clone_import_name(&planned.source_name)?,
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
                let rows_u64 = u64::try_from(rows)
                    .map_err(|_| VqError("VQ row count exceeds format range"))?;
                let bytes_per_row_u64 = u64::try_from(bytes_per_row)
                    .map_err(|_| VqError("VQ selector row size exceeds format range"))?;
                let total_indices = rows_u64
                    .checked_mul(bytes_per_row_u64)
                    .ok_or(VqError("selector payload size overflow"))?;
                let indices_start = spool
                    .seek(SeekFrom::End(0))
                    .map_err(|_| VqError("could not inspect VQ spool"))?
                    .checked_sub(total_indices)
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
                    name: clone_import_name(codebook_name)?,
                    dtype: DType::F32,
                    dimensions: vec![
                        u64::try_from(config.codebook_count)
                            .map_err(|_| VqError("VQ codebook count exceeds format range"))?,
                        CODEBOOK_SIZE as u64,
                        GROUP_SIZE as u64,
                    ],
                    block_lengths: vec![codebook_len],
                });
                block_offsets.push(vec![codebook_start]);
                next_id = next_id
                    .checked_add(1)
                    .ok_or(VqError("HPSI tensor ID overflow"))?;

                let selector_block_count = (rows / block_rows)
                    .checked_add(usize::from(rows % block_rows != 0))
                    .ok_or(VqError("selector block count overflow"))?;
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
                    let count_u64 = u64::try_from(count)
                        .map_err(|_| VqError("selector block row count exceeds format range"))?;
                    let length = count_u64
                        .checked_mul(bytes_per_row_u64)
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
                    name: clone_import_name(indices_name)?,
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
    fn persists_calibration_provenance_and_reopens_verified_artifact() {
        let (source, _, _) = fixture();
        let calibration = ActivationCalibration {
            layer_name: "weight".into(),
            sample_count: 12,
            features: 8,
            activation_second_moment: vec![0.5; 8],
            model_type: "test-transformer".into(),
            model_revision: Some("revision-42".into()),
            dataset_sha256: "a".repeat(64),
        };
        let calibrated = ModelImportPlan::build_with_provenance(
            &source,
            &[calibration],
            UnsupportedTensorPolicy::AllowRawPreservation,
        )
        .unwrap();
        let mut output = Cursor::new(Vec::new());
        let stats =
            write_model_import_to_hpsi_calibrated(&source, &calibrated, &mut output, config(), 1)
                .unwrap();
        assert_eq!(stats.tensor_count, 3);

        let reader = ModelReader::open(
            Arc::new(MemorySource::new(Arc::<[u8]>::from(output.into_inner()))),
            &ReaderLimits::default(),
        )
        .unwrap();
        reader.verify_all().unwrap();
        let source_tensors = reader.metadata.value["user"]["source_tensors"]
            .as_array()
            .unwrap();
        let bias = source_tensors
            .iter()
            .find(|item| item["name"] == "bias")
            .unwrap();
        assert!(bias.get("calibration").is_none());
        let weight = source_tensors
            .iter()
            .find(|item| item["name"] == "weight")
            .unwrap();
        assert_eq!(weight["calibration"]["model_type"], "test-transformer");
        assert_eq!(weight["calibration"]["model_revision"], "revision-42");
        assert_eq!(weight["calibration"]["dataset_sha256"], "a".repeat(64));
        assert_eq!(weight["calibration"]["sample_count"], 12);
        assert_eq!(weight["calibration"]["features"], 8);
        assert_eq!(weight["calibration"]["statistic"], "diagonal_second_moment");
        assert!(
            weight["calibration"]
                .get("activation_second_moment")
                .is_none()
        );
    }

    #[test]
    fn rejects_tampered_calibration_provenance_before_writing() {
        let (source, _, _) = fixture();
        let calibration = ActivationCalibration {
            layer_name: "weight".into(),
            sample_count: 12,
            features: 8,
            activation_second_moment: vec![1.0; 8],
            model_type: "test-transformer".into(),
            model_revision: None,
            dataset_sha256: "b".repeat(64),
        };
        let mut calibrated = ModelImportPlan::build_with_provenance(
            &source,
            &[calibration],
            UnsupportedTensorPolicy::AllowRawPreservation,
        )
        .unwrap();
        calibrated.calibrations[0].dataset_sha256 = "z".repeat(64);
        let mut output = Cursor::new(Vec::new());
        assert!(
            write_model_import_to_hpsi_calibrated(&source, &calibrated, &mut output, config(), 1,)
                .is_err()
        );
        assert!(output.get_ref().is_empty());
    }

    #[test]
    fn imports_all_safetensors_dtypes_to_hpsi_without_payload_loss() {
        let dtypes = [
            ("BOOL", 8_u64, DType::Opaque),
            ("F4", 4, DType::Opaque),
            ("F6_E2M3", 6, DType::Opaque),
            ("F6_E3M2", 6, DType::Opaque),
            ("U8", 8, DType::U8),
            ("I8", 8, DType::I8),
            ("F8_E5M2", 8, DType::Opaque),
            ("F8_E4M3", 8, DType::Opaque),
            ("F8_E8M0", 8, DType::Opaque),
            ("F8_E4M3FNUZ", 8, DType::Opaque),
            ("F8_E5M2FNUZ", 8, DType::Opaque),
            ("I16", 16, DType::I16),
            ("U16", 16, DType::Opaque),
            ("F16", 16, DType::F16),
            ("BF16", 16, DType::Bf16),
            ("I32", 32, DType::I32),
            ("U32", 32, DType::Opaque),
            ("F32", 32, DType::F32),
            ("C64", 64, DType::Opaque),
            ("F64", 64, DType::F64),
            ("I64", 64, DType::I64),
            ("U64", 64, DType::Opaque),
        ];

        for (dtype, bits_per_element, expected_hpsi_dtype) in dtypes {
            let payload_len = usize::try_from(bits_per_element).unwrap();
            let header = format!(
                r#"{{"tensor":{{"dtype":"{dtype}","shape":[8],"data_offsets":[0,{payload_len}]}}}}"#
            );
            let payload: Vec<u8> = (0..payload_len)
                .map(|index| (index as u8).wrapping_mul(17).wrapping_add(3))
                .collect();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&payload);

            let source =
                SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096)
                    .unwrap();
            let plan =
                ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation)
                    .unwrap();
            let mut output = Cursor::new(Vec::new());
            let stats =
                write_model_import_to_hpsi(&source, &plan, &mut output, config(), 1).unwrap();
            assert_eq!(stats.tensor_count, 1, "tensor count for {dtype}");

            let reader = ModelReader::open(
                Arc::new(MemorySource::new(Arc::<[u8]>::from(output.into_inner()))),
                &ReaderLimits::default(),
            )
            .unwrap();
            reader.verify_all().unwrap();
            let descriptor = reader.tensor(0).unwrap();
            assert_eq!(
                descriptor.dtype, expected_hpsi_dtype,
                "HPSI dtype for {dtype}"
            );
            let expected_dimensions = if expected_hpsi_dtype == DType::Opaque {
                vec![payload_len as u64]
            } else {
                vec![8]
            };
            assert_eq!(
                descriptor.dimensions[..usize::from(descriptor.rank)],
                expected_dimensions,
                "HPSI dimensions for {dtype}"
            );
            assert_eq!(
                reader.read_block(0, 0).unwrap().payload,
                payload,
                "payload changed for Safetensors dtype {dtype}"
            );
            assert_eq!(
                reader.metadata.value["user"]["source_tensors"][0]["source_dtype"],
                dtype
            );
            assert_eq!(
                reader.metadata.value["user"]["source_tensors"][0]["source_shape"],
                serde_json::json!([8])
            );
        }
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
    fn accepts_block_rows_larger_than_the_vq_tensor() {
        let (source, _, _) = fixture();
        let plan =
            ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation)
                .unwrap();
        let mut output = Cursor::new(Vec::new());
        let stats =
            write_model_import_to_hpsi(&source, &plan, &mut output, config(), usize::MAX).unwrap();
        assert_eq!(stats.tensor_count, 3);
        let reader = ModelReader::open(
            Arc::new(MemorySource::new(Arc::<[u8]>::from(output.into_inner()))),
            &ReaderLimits::default(),
        )
        .unwrap();
        reader.verify_all().unwrap();
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
    fn rejects_scalar_safetensors_during_import_planning() {
        let header = br#"{"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        let source =
            SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096)
                .unwrap();

        assert!(
            ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation,)
                .is_err()
        );
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
fn import_spool_rejects_writes_beyond_preflight_limit() {
    let mut spool = ImportSpool::create(4).unwrap();
    spool.write_all(&[1, 2, 3, 4]).unwrap();
    assert!(spool.write_all(&[5]).is_err());
    assert_eq!(spool.seek(SeekFrom::End(0)).unwrap(), 4);
}

#[cfg(unix)]
#[test]
fn import_spool_is_created_with_private_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let spool = ImportSpool::create(1024).unwrap();
    let mode = fs::metadata(&spool.path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
}

#[test]
fn import_spool_is_removed_when_dropped() {
    let spool = ImportSpool::create(1024).unwrap();
    let path = spool.path.clone();
    assert!(path.exists());
    drop(spool);
    assert!(!path.exists());
}

#[test]
fn import_output_temporary_file_is_removed_during_unwind() {
    let sequence = IMPORT_OUTPUT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        ".hpsi-import-unwind-{}-{sequence}.tmp",
        std::process::id()
    ));
    let file = open_import_temporary_file(&path).unwrap();
    let temporary = ImportOutputTemporaryFile {
        path: path.clone(),
        file: Some(file),
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _temporary = temporary;
        panic!("injected import output unwind");
    }));

    assert!(result.is_err());
    assert!(!path.exists());
}
