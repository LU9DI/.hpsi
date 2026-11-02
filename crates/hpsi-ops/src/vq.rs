use hpsi_format::{
    DType, ModelReader, ModelWriter, PayloadBlock, StreamingTensorInput, TensorInput, WriterStats,
};
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static VQ_SPOOL_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn open_vq_temporary_file(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    options.open(path)
}

struct TemporarySpool {
    path: PathBuf,
    file: Option<std::fs::File>,
}

impl TemporarySpool {
    fn create() -> Result<Self, VqError> {
        let directory = std::env::temp_dir();
        for _ in 0..128 {
            let sequence = VQ_SPOOL_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = directory.join(format!(
                ".hpsi-vq-spool-{}-{sequence}.tmp",
                std::process::id()
            ));
            match open_vq_temporary_file(&path) {
                Ok(file) => {
                    return Ok(Self {
                        path,
                        file: Some(file),
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(_) => return Err(VqError("could not create temporary selector spool")),
            }
        }
        Err(VqError("could not allocate a unique selector spool"))
    }
}

impl Read for TemporarySpool {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.file
            .as_mut()
            .expect("temporary spool file is open")
            .read(buffer)
    }
}
impl Write for TemporarySpool {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.file
            .as_mut()
            .expect("temporary spool file is open")
            .write(buffer)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file
            .as_mut()
            .expect("temporary spool file is open")
            .flush()
    }
}
impl Seek for TemporarySpool {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.file
            .as_mut()
            .expect("temporary spool file is open")
            .seek(position)
    }
}
impl Drop for TemporarySpool {
    fn drop(&mut self) {
        drop(self.file.take());
        let _ = fs::remove_file(&self.path);
    }
}

pub const GROUP_SIZE: usize = 8;
const MAX_VQ_TRAINING_SAMPLES: usize = 262_144;
const MAX_VQ_ITERATIONS: usize = 256;
pub const CODEBOOK_SIZE: usize = 16;
const CODEBOOK_VALUES: usize = GROUP_SIZE * CODEBOOK_SIZE;

#[derive(Debug, Clone, Copy)]
pub struct VqConfig {
    pub iterations: usize,
    pub sample_limit: usize,
    /// Each codebook adds one 4-bit residual selector per 8-weight group.
    /// One through four additive codebooks are supported (0.5–2 bits/weight).
    pub codebook_count: usize,
    /// Number of partial additive-code combinations retained during encoding.
    pub beam_width: usize,
}

impl Default for VqConfig {
    fn default() -> Self {
        Self {
            iterations: 12,
            sample_limit: 65_536,
            codebook_count: 1,
            beam_width: 4,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VqTensorNames<'a> {
    pub codebook_id: u32,
    pub codebook_name: &'a str,
    pub indices_id: u32,
    pub indices_name: &'a str,
}

#[derive(Debug, Clone, Copy)]
pub struct VqHpsiExportOptions<'a> {
    pub block_rows: usize,
    pub tensor_names: VqTensorNames<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VqError(pub &'static str);

impl std::fmt::Display for VqError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}
impl std::error::Error for VqError {}

fn validate_vq_config(config: VqConfig) -> Result<(), VqError> {
    if config.iterations == 0 || config.iterations > MAX_VQ_ITERATIONS {
        return Err(VqError(
            "k-means iterations exceed supported resource limits",
        ));
    }
    if config.sample_limit == 0 || config.sample_limit > MAX_VQ_TRAINING_SAMPLES {
        return Err(VqError(
            "VQ training sample limit exceeds supported resource limits",
        ));
    }
    if !(1..=4).contains(&config.codebook_count) {
        return Err(VqError(
            "only one through four additive codebooks are supported",
        ));
    }
    if !(1..=256).contains(&config.beam_width) {
        return Err(VqError("beam width must be between 1 and 256"));
    }
    Ok(())
}

/// Return the number of packed 4-bit selectors in one matrix row.
/// Rows are independently addressable in HPSI blocks, so each row must end
/// on a byte boundary (an even number of selectors).
fn selectors_per_row(cols: usize, codebook_count: usize) -> Result<usize, VqError> {
    if cols == 0 || cols % GROUP_SIZE != 0 || !(1..=4).contains(&codebook_count) {
        return Err(VqError("matrix columns or codebook count are invalid"));
    }
    let selectors = (cols / GROUP_SIZE)
        .checked_mul(codebook_count)
        .ok_or(VqError("selector count per row overflow"))?;
    if selectors % 2 != 0 {
        return Err(VqError(
            "each matrix row must contain an even selector count",
        ));
    }
    Ok(selectors)
}

/// Matrix represented by packed 4-bit selectors into one through four additive codebooks.
/// Runtime multiplication consumes the representation directly.
#[derive(Debug, Clone)]
pub struct VqMatrix {
    rows: usize,
    cols: usize,
    codebook_count: usize,
    codebook: Vec<f32>,
    packed_indices: Vec<u8>,
}

impl VqMatrix {
    pub fn encode(
        weights: &[f32],
        rows: usize,
        cols: usize,
        config: VqConfig,
    ) -> Result<Self, VqError> {
        Self::encode_internal(weights, rows, cols, config, None)
    }

    /// Encode a matrix supplied by a bounded random-access reader.
    ///
    /// The source is read twice: deterministic samples train the codebooks, then
    /// one row at a time is encoded into the packed selector stream. Dense weights
    /// are never retained; peak source memory is bounded by one matrix row.
    pub fn encode_from_reader<F>(
        rows: usize,
        cols: usize,
        config: VqConfig,
        read_elements: F,
    ) -> Result<Self, VqError>
    where
        F: FnMut(u64, u64) -> Result<Vec<f32>, VqError>,
    {
        let bytes_per_row = selectors_per_row(cols, config.codebook_count)? / 2;
        let packed_len = rows
            .checked_mul(bytes_per_row)
            .ok_or(VqError("packed index count overflow"))?;
        let mut packed_indices = Vec::new();
        packed_indices
            .try_reserve_exact(packed_len)
            .map_err(|_| VqError("could not allocate packed selector matrix"))?;
        let codebook = Self::encode_from_reader_to(rows, cols, config, read_elements, |_, row| {
            packed_indices.extend_from_slice(row);
            Ok(())
        })?;
        Ok(Self {
            rows,
            cols,
            codebook_count: config.codebook_count,
            codebook,
            packed_indices,
        })
    }

    /// Train from bounded source windows, then emit one independently aligned
    /// packed selector row at a time. The complete selector matrix is never
    /// retained by this API; callers can write rows to a spool or downstream
    /// block encoder as soon as they are produced.
    pub fn encode_from_reader_to<F, E>(
        rows: usize,
        cols: usize,
        config: VqConfig,
        mut read_elements: F,
        emit_row: E,
    ) -> Result<Vec<f32>, VqError>
    where
        F: FnMut(u64, u64) -> Result<Vec<f32>, VqError>,
        E: FnMut(u64, &[u8]) -> Result<(), VqError>,
    {
        Self::encode_from_reader_into(
            rows,
            cols,
            config,
            |offset, target| {
                let values = read_elements(offset, target.len() as u64)?;
                if values.len() != target.len() {
                    return Err(VqError("reader returned an invalid window length"));
                }
                target.copy_from_slice(&values);
                Ok(())
            },
            emit_row,
        )
    }

    /// Buffer-reusing variant of the streaming VQ encoder. The callback fills
    /// caller-provided storage, allowing Safetensors decoding and VQ row
    /// processing to reuse one allocation across all rows.
    pub fn encode_from_reader_into<F, E>(
        rows: usize,
        cols: usize,
        config: VqConfig,
        mut read_elements: F,
        mut emit_row: E,
    ) -> Result<Vec<f32>, VqError>
    where
        F: FnMut(u64, &mut [f32]) -> Result<(), VqError>,
        E: FnMut(u64, &[u8]) -> Result<(), VqError>,
    {
        let expected = rows
            .checked_mul(cols)
            .ok_or(VqError("matrix element count overflow"))?;
        if rows == 0 || cols == 0 || cols % GROUP_SIZE != 0 {
            return Err(VqError("matrix shape must be non-zero and group-aligned"));
        }
        validate_vq_config(config)?;
        selectors_per_row(cols, config.codebook_count)?;

        let group_count = expected / GROUP_SIZE;
        let sample_count = group_count.min(config.sample_limit);
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(sample_count)
            .map_err(|_| VqError("could not allocate VQ training samples"))?;
        let mut sample_buffer = [0.0_f32; GROUP_SIZE];
        for sample_index in 0..sample_count {
            let group_index =
                ((sample_index as u128 * group_count as u128) / sample_count as u128) as usize;
            let offset = group_index
                .checked_mul(GROUP_SIZE)
                .ok_or(VqError("sample element offset overflow"))?;
            read_elements(offset as u64, &mut sample_buffer)?;
            if sample_buffer.iter().any(|value| !value.is_finite()) {
                return Err(VqError(
                    "reader returned an invalid or non-finite sample window",
                ));
            }
            samples.push(sample_buffer);
        }

        let mut codebooks: Vec<Vec<[f32; GROUP_SIZE]>> = Vec::new();
        codebooks
            .try_reserve_exact(config.codebook_count)
            .map_err(|_| VqError("could not allocate VQ codebook directory"))?;
        for stage in 0..config.codebook_count {
            let mut residual_samples = Vec::new();
            residual_samples
                .try_reserve_exact(samples.len())
                .map_err(|_| VqError("could not allocate VQ residual samples"))?;
            for sample in &samples {
                residual_samples
                    .push(beam_search_additive(sample, &codebooks, config.beam_width)?.0);
            }
            codebooks.push(fit_codebook(
                &residual_samples,
                config.iterations,
                stage != 0,
            )?);
        }

        let selectors_per_row = selectors_per_row(cols, config.codebook_count)?;
        let bytes_per_row = selectors_per_row / 2;
        let mut packed_row = Vec::new();
        packed_row
            .try_reserve_exact(bytes_per_row)
            .map_err(|_| VqError("could not allocate packed selector row"))?;
        let mut row_buffer = Vec::new();
        row_buffer
            .try_reserve_exact(cols)
            .map_err(|_| VqError("could not allocate VQ row buffer"))?;
        row_buffer.resize(cols, 0.0_f32);
        for row in 0..rows {
            let first_element = row
                .checked_mul(cols)
                .ok_or(VqError("row element offset overflow"))?;
            read_elements(first_element as u64, &mut row_buffer)?;
            if row_buffer.iter().any(|value| !value.is_finite()) {
                return Err(VqError(
                    "reader returned an invalid or non-finite row window",
                ));
            }
            packed_row.clear();
            packed_row.resize(bytes_per_row, 0);
            for (group_in_row, chunk) in row_buffer.chunks_exact(GROUP_SIZE).enumerate() {
                let mut residual = [0.0_f32; GROUP_SIZE];
                residual.copy_from_slice(chunk);
                let (_, codes) = beam_search_additive(&residual, &codebooks, config.beam_width)?;
                for (stage, code) in codes.iter().take(config.codebook_count).enumerate() {
                    let selector = group_in_row * config.codebook_count + stage;
                    packed_row[selector / 2] |= *code << ((selector % 2) * 4);
                }
            }
            emit_row(row as u64, &packed_row)?;
        }

        let codebook_len = config
            .codebook_count
            .checked_mul(CODEBOOK_VALUES)
            .ok_or(VqError("VQ codebook length overflow"))?;
        let mut flattened_codebook = Vec::new();
        flattened_codebook
            .try_reserve_exact(codebook_len)
            .map_err(|_| VqError("could not allocate flattened VQ codebook"))?;
        for book in codebooks {
            for centroid in book {
                flattened_codebook.extend_from_slice(&centroid);
            }
        }
        Ok(flattened_codebook)
    }

    /// Streaming equivalent of activation-aware encoding. Importance is
    /// averaged by group position, matching the in-memory encoder.
    pub fn encode_from_reader_activation_aware<F>(
        rows: usize,
        cols: usize,
        config: VqConfig,
        activation_second_moment: &[f32],
        read_elements: F,
    ) -> Result<Self, VqError>
    where
        F: FnMut(u64, u64) -> Result<Vec<f32>, VqError>,
    {
        let bytes_per_row = selectors_per_row(cols, config.codebook_count)? / 2;
        let packed_len = rows
            .checked_mul(bytes_per_row)
            .ok_or(VqError("packed index count overflow"))?;
        let mut packed_indices = Vec::new();
        packed_indices
            .try_reserve_exact(packed_len)
            .map_err(|_| VqError("could not allocate packed selector matrix"))?;
        let codebook = Self::encode_from_reader_activation_aware_to(
            rows,
            cols,
            config,
            activation_second_moment,
            read_elements,
            |_, row| {
                packed_indices.extend_from_slice(row);
                Ok(())
            },
        )?;
        Ok(Self {
            rows,
            cols,
            codebook_count: config.codebook_count,
            codebook,
            packed_indices,
        })
    }

    /// Encode a matrix from bounded source windows and write the resulting
    /// VQ tensors through the incremental HPSI writer. Selector rows are
    /// spooled to caller-provided seekable storage so the codebook tensor can
    /// precede the selector tensor without retaining the full index stream in RAM.
    pub fn encode_from_reader_to_hpsi<W, S, F>(
        out: &mut W,
        spool: &mut S,
        rows: usize,
        cols: usize,
        config: VqConfig,
        block_rows: usize,
        read_elements: F,
    ) -> Result<WriterStats, VqError>
    where
        W: Write + Seek,
        S: Read + Write + Seek,
        F: FnMut(u64, u64) -> Result<Vec<f32>, VqError>,
    {
        Self::encode_from_reader_to_hpsi_named(
            out,
            spool,
            rows,
            cols,
            config,
            VqHpsiExportOptions {
                block_rows,
                tensor_names: VqTensorNames {
                    codebook_id: 0,
                    codebook_name: "hpsi.vq.codebook",
                    indices_id: 1,
                    indices_name: "hpsi.vq.indices",
                },
            },
            read_elements,
        )
    }

    /// Same bounded-memory encoder with caller-selected tensor IDs and names.
    /// This lets a higher-level model importer assign globally unique IDs and
    /// source-derived names when several VQ matrices share one HPSI artifact.
    pub fn encode_from_reader_to_hpsi_named<W, S, F>(
        out: &mut W,
        spool: &mut S,
        rows: usize,
        cols: usize,
        config: VqConfig,
        options: VqHpsiExportOptions<'_>,
        read_elements: F,
    ) -> Result<WriterStats, VqError>
    where
        W: Write + Seek,
        S: Read + Write + Seek,
        F: FnMut(u64, u64) -> Result<Vec<f32>, VqError>,
    {
        let VqHpsiExportOptions {
            block_rows,
            tensor_names,
        } = options;
        if rows == 0 || cols == 0 || block_rows == 0 {
            return Err(VqError("matrix dimensions and block rows must be non-zero"));
        }
        if spool
            .seek(SeekFrom::End(0))
            .map_err(|_| VqError("could not inspect selector spool"))?
            != 0
        {
            return Err(VqError("selector spool must be empty"));
        }
        spool
            .seek(SeekFrom::Start(0))
            .map_err(|_| VqError("could not rewind selector spool"))?;
        let selectors = selectors_per_row(cols, config.codebook_count)?;
        let bytes_per_row = selectors / 2;
        let codebook = Self::encode_from_reader_to(rows, cols, config, read_elements, |_, row| {
            spool
                .write_all(row)
                .map_err(|_| VqError("could not spool packed selector row"))
        })?;
        spool
            .flush()
            .map_err(|_| VqError("could not flush selector spool"))?;

        let codebook_byte_len = codebook
            .len()
            .checked_mul(4)
            .ok_or(VqError("VQ codebook byte length overflow"))?;
        let mut codebook_bytes = Vec::new();
        codebook_bytes
            .try_reserve_exact(codebook_byte_len)
            .map_err(|_| VqError("could not allocate VQ codebook bytes"))?;
        for value in &codebook {
            codebook_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let codebook_len = u64::try_from(codebook_bytes.len())
            .map_err(|_| VqError("VQ codebook byte length exceeds format limits"))?;
        let block_count = (rows / block_rows)
            .checked_add(usize::from(rows % block_rows != 0))
            .ok_or(VqError("selector block count overflow"))?;
        let mut index_block_lengths = Vec::new();
        index_block_lengths
            .try_reserve_exact(block_count)
            .map_err(|_| VqError("could not allocate selector block directory"))?;
        let mut first_row = 0usize;
        while first_row < rows {
            let count = block_rows.min(rows - first_row);
            let length = count
                .checked_mul(bytes_per_row)
                .ok_or(VqError("selector block length overflow"))?;
            index_block_lengths.push(
                u64::try_from(length)
                    .map_err(|_| VqError("selector block length exceeds format limits"))?,
            );
            first_row += count;
        }
        let index_bytes = rows
            .checked_mul(bytes_per_row)
            .ok_or(VqError("selector stream size overflow"))?;
        let expected_index_bytes = u64::try_from(index_bytes)
            .map_err(|_| VqError("selector stream exceeds format limits"))?;
        let declared_index_bytes = index_block_lengths.iter().try_fold(0u64, |total, length| {
            total
                .checked_add(*length)
                .ok_or(VqError("selector block directory size overflow"))
        })?;
        if declared_index_bytes != expected_index_bytes {
            return Err(VqError(
                "selector block directory does not match the selector stream size",
            ));
        }
        let spool_len = spool
            .seek(SeekFrom::End(0))
            .map_err(|_| VqError("could not inspect selector spool length"))?;
        if spool_len != expected_index_bytes {
            return Err(VqError(
                "selector spool length does not match the selector stream size",
            ));
        }
        spool
            .seek(SeekFrom::Start(0))
            .map_err(|_| VqError("could not rewind selector spool"))?;
        let tensors = vec![
            StreamingTensorInput {
                tensor_id: tensor_names.codebook_id,
                name: tensor_names.codebook_name.into(),
                dtype: DType::F32,
                dimensions: vec![
                    config.codebook_count as u64,
                    CODEBOOK_SIZE as u64,
                    GROUP_SIZE as u64,
                ],
                block_lengths: vec![codebook_len],
            },
            StreamingTensorInput {
                tensor_id: tensor_names.indices_id,
                name: tensor_names.indices_name.into(),
                dtype: DType::Opaque,
                dimensions: vec![expected_index_bytes],
                block_lengths: index_block_lengths,
            },
        ];
        // Keep the metadata construction explicit and reject impossible empty payloads.
        if tensors[0].block_lengths[0] == 0 || tensors[1].block_lengths.is_empty() {
            return Err(VqError("generated VQ tensors have empty payload blocks"));
        }
        let mut spool_offset = 0u64;
        let mut current_block = 0usize;
        let stats =
            ModelWriter::write_streaming(out, &tensors, |tensor_index, block_index, sink| {
                if tensor_index == 0 {
                    sink.write_all(&codebook_bytes)
                        .map_err(|error| hpsi_format::FormatError::Io(error.to_string()))?;
                    return Ok(codebook_len);
                }
                if tensor_index != 1 || block_index != current_block {
                    return Err(hpsi_format::FormatError::InvalidBlock(
                        "unexpected VQ selector block request",
                    ));
                }
                let length = tensors[1].block_lengths[block_index];
                spool
                    .seek(SeekFrom::Start(spool_offset))
                    .map_err(|error| hpsi_format::FormatError::Io(error.to_string()))?;
                let mut remaining = length;
                let mut buffer = [0u8; 64 * 1024];
                while remaining > 0 {
                    let count = remaining.min(buffer.len() as u64) as usize;
                    spool
                        .read_exact(&mut buffer[..count])
                        .map_err(|error| hpsi_format::FormatError::Io(error.to_string()))?;
                    sink.write_all(&buffer[..count])
                        .map_err(|error| hpsi_format::FormatError::Io(error.to_string()))?;
                    remaining -= count as u64;
                }
                spool_offset = spool_offset.checked_add(length).ok_or(
                    hpsi_format::FormatError::ArithmeticOverflow("selector spool offset"),
                )?;
                current_block += 1;
                Ok(length)
            })
            .map_err(|_| VqError("HPSI streaming writer failed"))?;
        Ok(stats)
    }

    /// Convenience API that owns and cleans up a disk-backed selector spool.
    /// This avoids requiring callers to manage an intermediate file for large
    /// models while keeping selector memory bounded to one row and one 64 KiB
    /// transfer buffer.
    pub fn encode_from_reader_to_hpsi_with_temp_spool<W, F>(
        out: &mut W,
        rows: usize,
        cols: usize,
        config: VqConfig,
        block_rows: usize,
        read_elements: F,
    ) -> Result<WriterStats, VqError>
    where
        W: Write + Seek,
        F: FnMut(u64, u64) -> Result<Vec<f32>, VqError>,
    {
        let mut spool = TemporarySpool::create()?;
        Self::encode_from_reader_to_hpsi(
            out,
            &mut spool,
            rows,
            cols,
            config,
            block_rows,
            read_elements,
        )
    }

    /// Activation-aware variant of `encode_from_reader_to`. It preserves
    /// the in-memory encoder's group-position importance scaling while
    /// emitting each packed row to the consumer immediately.
    pub fn encode_from_reader_activation_aware_to<F, E>(
        rows: usize,
        cols: usize,
        config: VqConfig,
        activation_second_moment: &[f32],
        mut read_elements: F,
        emit_row: E,
    ) -> Result<Vec<f32>, VqError>
    where
        F: FnMut(u64, u64) -> Result<Vec<f32>, VqError>,
        E: FnMut(u64, &[u8]) -> Result<(), VqError>,
    {
        Self::encode_from_reader_activation_aware_into(
            rows,
            cols,
            config,
            activation_second_moment,
            |first, target| {
                let values = read_elements(first, target.len() as u64)?;
                if values.len() != target.len() {
                    return Err(VqError("reader returned an invalid window length"));
                }
                target.copy_from_slice(&values);
                Ok(())
            },
            emit_row,
        )
    }

    /// Activation-aware streaming encoder using a reusable caller-owned row
    /// buffer, avoiding per-row Vec allocation in the import path.
    pub fn encode_from_reader_activation_aware_into<F, E>(
        rows: usize,
        cols: usize,
        config: VqConfig,
        activation_second_moment: &[f32],
        mut read_elements: F,
        emit_row: E,
    ) -> Result<Vec<f32>, VqError>
    where
        F: FnMut(u64, &mut [f32]) -> Result<(), VqError>,
        E: FnMut(u64, &[u8]) -> Result<(), VqError>,
    {
        if rows == 0 || cols == 0 || cols % GROUP_SIZE != 0 {
            return Err(VqError("matrix shape must be non-zero and group-aligned"));
        }
        if activation_second_moment.len() != cols
            || activation_second_moment
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(VqError(
                "activation importance must contain one finite non-negative value per column",
            ));
        }
        let mut dimension_scale = [1.0_f32; GROUP_SIZE];
        for (dimension, scale) in dimension_scale.iter_mut().enumerate() {
            let mean = activation_second_moment
                .iter()
                .skip(dimension)
                .step_by(GROUP_SIZE)
                .map(|value| f64::from(*value))
                .sum::<f64>()
                / (cols / GROUP_SIZE) as f64;
            *scale = mean.max(1.0e-30).sqrt() as f32;
            if !scale.is_finite() || *scale <= 0.0 {
                return Err(VqError("activation importance scale is invalid"));
            }
        }
        let mut codebook = Self::encode_from_reader_into(
            rows,
            cols,
            config,
            |first, values| {
                read_elements(first, values)?;
                let start = usize::try_from(first)
                    .map_err(|_| VqError("reader offset exceeds address space"))?;
                for (index, value) in values.iter_mut().enumerate() {
                    let column = start
                        .checked_add(index)
                        .ok_or(VqError("reader column offset overflow"))?
                        % cols;
                    *value *= dimension_scale[column % GROUP_SIZE];
                    if !value.is_finite() {
                        return Err(VqError("activation-scaled source window is non-finite"));
                    }
                }
                Ok(())
            },
            emit_row,
        )?;
        for centroid in codebook.chunks_exact_mut(GROUP_SIZE) {
            for (value, scale) in centroid.iter_mut().zip(dimension_scale) {
                *value /= scale;
            }
        }
        Ok(codebook)
    }

    /// Encode using activation second moments as per-input-column importance.
    /// Importance is averaged by group position because codebooks are shared.
    pub fn encode_activation_aware(
        weights: &[f32],
        rows: usize,
        cols: usize,
        config: VqConfig,
        activation_second_moment: &[f32],
    ) -> Result<Self, VqError> {
        Self::encode_internal(weights, rows, cols, config, Some(activation_second_moment))
    }

    fn encode_internal(
        weights: &[f32],
        rows: usize,
        cols: usize,
        config: VqConfig,
        activation_second_moment: Option<&[f32]>,
    ) -> Result<Self, VqError> {
        let expected = rows
            .checked_mul(cols)
            .ok_or(VqError("matrix element count overflow"))?;
        if rows == 0 || cols == 0 || cols % GROUP_SIZE != 0 || weights.len() != expected {
            return Err(VqError(
                "matrix shape must be non-zero, group-aligned, and match weights",
            ));
        }
        validate_vq_config(config)?;
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(VqError("matrix weights must be finite"));
        }

        let mut dimension_scale = [1.0_f32; GROUP_SIZE];
        if let Some(importance) = activation_second_moment {
            if importance.len() != cols || importance.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(VqError(
                    "activation importance must contain one finite non-negative value per column",
                ));
            }
            for (dimension, scale) in dimension_scale.iter_mut().enumerate() {
                let mean = importance
                    .iter()
                    .skip(dimension)
                    .step_by(GROUP_SIZE)
                    .map(|v| f64::from(*v))
                    .sum::<f64>()
                    / (cols / GROUP_SIZE) as f64;
                // A fully inactive feature is valid calibration data; keep inverse scaling finite.
                *scale = mean.max(1.0e-30).sqrt() as f32;
                if !scale.is_finite() || *scale <= 0.0 {
                    return Err(VqError("activation importance scale is invalid"));
                }
            }
        }

        selectors_per_row(cols, config.codebook_count)?;
        let group_count = expected / GROUP_SIZE;
        let sample_count = group_count.min(config.sample_limit);
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(sample_count)
            .map_err(|_| VqError("could not allocate VQ training samples"))?;
        for sample_index in 0..sample_count {
            let group_index =
                ((sample_index as u128 * group_count as u128) / sample_count as u128) as usize;
            let mut vector = [0.0_f32; GROUP_SIZE];
            let start = group_index * GROUP_SIZE;
            vector.copy_from_slice(&weights[start..start + GROUP_SIZE]);
            for (value, scale) in vector.iter_mut().zip(dimension_scale) {
                *value *= scale;
            }
            samples.push(vector);
        }

        let mut codebooks: Vec<Vec<[f32; GROUP_SIZE]>> = Vec::new();
        codebooks
            .try_reserve_exact(config.codebook_count)
            .map_err(|_| VqError("could not allocate VQ codebook directory"))?;
        for stage in 0..config.codebook_count {
            let mut residual_samples = Vec::new();
            residual_samples
                .try_reserve_exact(samples.len())
                .map_err(|_| VqError("could not allocate VQ residual samples"))?;
            for sample in &samples {
                residual_samples
                    .push(beam_search_additive(sample, &codebooks, config.beam_width)?.0);
            }
            // Residual stages reserve a zero centroid. Adding a stage therefore
            // cannot increase per-group reconstruction error in weight space.
            let codebook = fit_codebook(&residual_samples, config.iterations, stage != 0)?;
            codebooks.push(codebook);
        }

        let selector_count = group_count
            .checked_mul(config.codebook_count)
            .ok_or(VqError("packed selector count overflow"))?;
        let packed_len = selector_count
            .checked_add(1)
            .ok_or(VqError("packed index count overflow"))?
            / 2;
        let mut packed_indices = Vec::new();
        packed_indices
            .try_reserve_exact(packed_len)
            .map_err(|_| VqError("could not allocate packed VQ indices"))?;
        packed_indices.resize(packed_len, 0_u8);

        for group_index in 0..group_count {
            let start = group_index * GROUP_SIZE;
            let mut residual = [0.0_f32; GROUP_SIZE];
            residual.copy_from_slice(&weights[start..start + GROUP_SIZE]);
            for (value, scale) in residual.iter_mut().zip(dimension_scale) {
                *value *= scale;
            }
            let (_, codes) = beam_search_additive(&residual, &codebooks, config.beam_width)?;
            for (stage, code) in codes.iter().take(config.codebook_count).enumerate() {
                let selector = group_index * config.codebook_count + stage;
                let byte_index = selector / 2;
                let shift = (selector % 2) * 4;
                packed_indices[byte_index] |= *code << shift;
            }
        }

        let codebook_len = config
            .codebook_count
            .checked_mul(CODEBOOK_SIZE)
            .and_then(|count| count.checked_mul(GROUP_SIZE))
            .ok_or(VqError("VQ codebook length overflow"))?;
        let mut codebook = Vec::new();
        codebook
            .try_reserve_exact(codebook_len)
            .map_err(|_| VqError("could not allocate flattened VQ codebook"))?;
        for book in codebooks {
            for mut centroid in book {
                for (value, scale) in centroid.iter_mut().zip(dimension_scale) {
                    *value /= scale;
                }
                codebook.extend_from_slice(&centroid);
            }
        }

        Ok(Self {
            rows,
            cols,
            codebook_count: config.codebook_count,
            codebook,
            packed_indices,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn codebook_count(&self) -> usize {
        self.codebook_count
    }
    pub fn codebook(&self) -> &[f32] {
        &self.codebook
    }
    pub fn packed_indices(&self) -> &[u8] {
        &self.packed_indices
    }

    pub fn stored_bytes(&self) -> usize {
        self.codebook.len() * std::mem::size_of::<f32>() + self.packed_indices.len()
    }
    pub fn dense_f32_bytes(&self) -> usize {
        self.rows * self.cols * std::mem::size_of::<f32>()
    }
    pub fn view(&self) -> VqMatrixView<'_> {
        VqMatrixView {
            rows: self.rows,
            cols: self.cols,
            codebook_count: self.codebook_count,
            codebook: &self.codebook,
            packed_indices: &self.packed_indices,
        }
    }

    /// Export codebook and packed selectors as two HPSI tensors.
    pub fn to_hpsi_tensors(
        &self,
        codebook_tensor_id: u32,
        indices_tensor_id: u32,
        block_rows: usize,
    ) -> Result<Vec<TensorInput>, VqError> {
        if codebook_tensor_id >= indices_tensor_id {
            return Err(VqError("codebook tensor ID must precede index tensor ID"));
        }
        if block_rows == 0 {
            return Err(VqError("row-streamed export requires positive block rows"));
        }
        let selectors_per_row = selectors_per_row(self.cols, self.codebook_count)?;
        let bytes_per_row = selectors_per_row / 2;
        let index_block_bytes = bytes_per_row
            .checked_mul(block_rows)
            .ok_or(VqError("index block byte count overflow"))?;

        let codebook_byte_len = self
            .codebook
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(VqError("VQ codebook byte length overflow"))?;
        let mut codebook_bytes = Vec::new();
        codebook_bytes
            .try_reserve_exact(codebook_byte_len)
            .map_err(|_| VqError("could not allocate VQ codebook bytes"))?;
        for value in &self.codebook {
            codebook_bytes.extend_from_slice(&value.to_le_bytes());
        }

        let codebook_count = u64::try_from(self.codebook_count)
            .map_err(|_| VqError("VQ codebook count exceeds format limits"))?;
        let index_byte_len = u64::try_from(self.packed_indices.len())
            .map_err(|_| VqError("VQ selector stream exceeds format limits"))?;
        let mut codebook_dimensions = Vec::new();
        codebook_dimensions
            .try_reserve_exact(3)
            .map_err(|_| VqError("could not allocate VQ codebook dimensions"))?;
        codebook_dimensions.extend_from_slice(&[
            codebook_count,
            CODEBOOK_SIZE as u64,
            GROUP_SIZE as u64,
        ]);
        let mut index_dimensions = Vec::new();
        index_dimensions
            .try_reserve_exact(1)
            .map_err(|_| VqError("could not allocate VQ selector dimensions"))?;
        index_dimensions.push(index_byte_len);

        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(2)
            .map_err(|_| VqError("could not allocate VQ tensor directory"))?;
        tensors.push(TensorInput {
            tensor_id: codebook_tensor_id,
            name: "hpsi.vq.codebook".into(),
            dtype: DType::F32,
            dimensions: codebook_dimensions,
            blocks: payload_blocks(&codebook_bytes, 64 * 1024)?,
        });
        tensors.push(TensorInput {
            tensor_id: indices_tensor_id,
            name: "hpsi.vq.indices".into(),
            dtype: DType::Opaque,
            dimensions: index_dimensions,
            blocks: payload_blocks(&self.packed_indices, index_block_bytes)?,
        });
        Ok(tensors)
    }
}

/// Borrowed view over packed selectors and shared codebooks.
#[derive(Debug, Clone, Copy)]
pub struct VqMatrixView<'a> {
    rows: usize,
    cols: usize,
    codebook_count: usize,
    codebook: &'a [f32],
    packed_indices: &'a [u8],
}

impl<'a> VqMatrixView<'a> {
    pub fn new(
        rows: usize,
        cols: usize,
        codebook_count: usize,
        codebook: &'a [f32],
        packed_indices: &'a [u8],
    ) -> Result<Self, VqError> {
        if rows == 0 {
            return Err(VqError("matrix row count must be non-zero"));
        }
        if cols == 0 || cols % GROUP_SIZE != 0 || !(1..=4).contains(&codebook_count) {
            return Err(VqError("matrix columns or codebook count are invalid"));
        }
        if codebook.len() != CODEBOOK_VALUES * codebook_count
            || codebook.iter().any(|value| !value.is_finite())
        {
            return Err(VqError("codebook shape or values are invalid"));
        }
        let groups = rows
            .checked_mul(cols / GROUP_SIZE)
            .ok_or(VqError("matrix group count overflow"))?;
        let selector_count = groups
            .checked_mul(codebook_count)
            .ok_or(VqError("packed selector count overflow"))?;
        let expected_indices = selector_count
            .checked_add(1)
            .ok_or(VqError("packed index count overflow"))?
            / 2;
        if packed_indices.len() != expected_indices {
            return Err(VqError(
                "packed index stream length does not match matrix shape",
            ));
        }
        if selector_count % 2 != 0 && packed_indices.last().is_some_and(|last| *last & 0xf0 != 0) {
            return Err(VqError("unused packed selector nibble must be zero"));
        }
        Ok(Self {
            rows,
            cols,
            codebook_count,
            codebook,
            packed_indices,
        })
    }

    pub fn matvec(&self, input: &[f32], output: &mut [f32]) -> Result<(), VqError> {
        if input.len() != self.cols || output.len() != self.rows {
            return Err(VqError("matvec input or output length mismatch"));
        }
        let groups_per_row = self.cols / GROUP_SIZE;
        // A centroid dot product depends on the input group, not on the output
        // row. Compute each of the 16 candidates once, then gather by selector
        // for every row. This avoids repeating 8 multiply-adds per row/group.
        let dot_count = groups_per_row
            .checked_mul(self.codebook_count)
            .and_then(|count| count.checked_mul(CODEBOOK_SIZE))
            .ok_or(VqError("VQ dot-product table size overflow"))?;
        let mut dot_products = Vec::new();
        dot_products
            .try_reserve_exact(dot_count)
            .map_err(|_| VqError("could not allocate VQ dot-product table"))?;
        dot_products.resize(dot_count, 0.0_f32);

        for group in 0..groups_per_row {
            let input_offset = group * GROUP_SIZE;
            for stage in 0..self.codebook_count {
                for centroid in 0..CODEBOOK_SIZE {
                    let codebook_offset = stage * CODEBOOK_VALUES + centroid * GROUP_SIZE;
                    let mut dot = 0.0_f32;
                    for dimension in 0..GROUP_SIZE {
                        dot += input[input_offset + dimension]
                            * self.codebook[codebook_offset + dimension];
                    }
                    let table_index =
                        (group * self.codebook_count + stage) * CODEBOOK_SIZE + centroid;
                    dot_products[table_index] = dot;
                }
            }
        }

        for (row, output_value) in output.iter_mut().enumerate() {
            let mut accumulator = 0.0_f32;
            for group in 0..groups_per_row {
                let global_group = row * groups_per_row + group;
                for stage in 0..self.codebook_count {
                    let selector = global_group * self.codebook_count + stage;
                    let packed = self.packed_indices[selector / 2];
                    let code = usize::from(if selector % 2 == 0 {
                        packed & 0x0f
                    } else {
                        packed >> 4
                    });
                    let table_index = (group * self.codebook_count + stage) * CODEBOOK_SIZE + code;
                    accumulator += dot_products[table_index];
                }
            }
            *output_value = accumulator;
        }
        Ok(())
    }
}

/// Compute a selected output-row range from an HPSI VQ tensor pair.
///
/// The codebook is loaded once. Only index blocks intersecting the requested
/// rows are read; each block is evaluated independently without reconstructing
/// a dense weight matrix or loading the complete selector tensor.
#[derive(Debug, Clone, Copy)]
pub struct HpsiVqLayout {
    pub codebook_tensor_id: u32,
    pub indices_tensor_id: u32,
    pub rows: usize,
    pub cols: usize,
    pub codebook_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct HpsiIndexBlock {
    block_id: u32,
    first_row: usize,
    end_row: usize,
}

/// Prepared executor for repeated row-window operations over an HPSI VQ tensor pair.
///
/// The shared codebook and index-block row map are loaded/validated once. Each
/// call reads only selector payload blocks intersecting the requested rows,
/// then computes only the requested rows within those blocks.
pub struct HpsiVqExecutor<'a> {
    reader: &'a ModelReader,
    layout: HpsiVqLayout,
    codebook: Vec<f32>,
    bytes_per_row: usize,
    index_blocks: Vec<HpsiIndexBlock>,
}

impl<'a> HpsiVqExecutor<'a> {
    pub fn new(reader: &'a ModelReader, layout: HpsiVqLayout) -> Result<Self, VqError> {
        let HpsiVqLayout {
            codebook_tensor_id,
            indices_tensor_id,
            rows,
            cols,
            codebook_count,
        } = layout;
        if rows == 0 {
            return Err(VqError("HPSI VQ layout has invalid row count"));
        }
        selectors_per_row(cols, codebook_count)?;

        let codebook_tensor = reader
            .tensor(codebook_tensor_id)
            .map_err(|_| VqError("could not locate HPSI codebook tensor"))?;
        let expected_codebook_bytes = CODEBOOK_VALUES
            .checked_mul(codebook_count)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(VqError("HPSI codebook byte length overflow"))?;
        let expected_codebook_bytes_u64 = u64::try_from(expected_codebook_bytes)
            .map_err(|_| VqError("HPSI codebook payload exceeds format limits"))?;
        if codebook_tensor.dtype != DType::F32
            || codebook_tensor.rank != 3
            || codebook_tensor.dimensions[0] != codebook_count as u64
            || codebook_tensor.dimensions[1] != CODEBOOK_SIZE as u64
            || codebook_tensor.dimensions[2] != GROUP_SIZE as u64
            || codebook_tensor.logical_nbytes != expected_codebook_bytes_u64
        {
            return Err(VqError(
                "HPSI codebook tensor descriptor does not match VQ layout",
            ));
        }

        let mut codebook_bytes = Vec::new();
        codebook_bytes
            .try_reserve_exact(expected_codebook_bytes)
            .map_err(|_| VqError("could not allocate HPSI codebook bytes"))?;
        for block_id in 0..codebook_tensor.block_count {
            let block = reader
                .read_block(codebook_tensor_id, block_id)
                .map_err(|_| VqError("failed to read HPSI codebook block"))?;
            let remaining = expected_codebook_bytes
                .checked_sub(codebook_bytes.len())
                .ok_or(VqError("HPSI codebook payload exceeds declared size"))?;
            if block.payload.len() > remaining {
                return Err(VqError("HPSI codebook block exceeds declared payload size"));
            }
            codebook_bytes.extend_from_slice(&block.payload);
        }
        if codebook_bytes.len() != expected_codebook_bytes {
            return Err(VqError("HPSI codebook payload length is invalid"));
        }
        let mut codebook = Vec::new();
        codebook
            .try_reserve_exact(expected_codebook_bytes / std::mem::size_of::<f32>())
            .map_err(|_| VqError("could not allocate decoded HPSI codebook"))?;
        for chunk in codebook_bytes.chunks_exact(std::mem::size_of::<f32>()) {
            let bytes: [u8; 4] = chunk
                .try_into()
                .map_err(|_| VqError("HPSI codebook contains an incomplete value"))?;
            codebook.push(f32::from_le_bytes(bytes));
        }
        if codebook.iter().any(|value| !value.is_finite()) {
            return Err(VqError("HPSI codebook contains non-finite values"));
        }

        let bytes_per_row = selectors_per_row(cols, codebook_count)? / 2;
        let expected_index_bytes = rows
            .checked_mul(bytes_per_row)
            .ok_or(VqError("HPSI index tensor size overflow"))?;
        let expected_index_bytes_u64 = u64::try_from(expected_index_bytes)
            .map_err(|_| VqError("HPSI index payload exceeds format limits"))?;
        let index_tensor = reader
            .tensor(indices_tensor_id)
            .map_err(|_| VqError("could not locate HPSI index tensor"))?;
        if index_tensor.dtype != DType::Opaque
            || index_tensor.rank != 1
            || index_tensor.dimensions[0] != expected_index_bytes_u64
            || index_tensor.logical_nbytes != expected_index_bytes_u64
        {
            return Err(VqError(
                "HPSI index tensor descriptor does not match VQ layout",
            ));
        }

        let index_block_count = usize::try_from(index_tensor.block_count)
            .map_err(|_| VqError("HPSI index block count exceeds address space"))?;
        let mut index_blocks = Vec::new();
        index_blocks
            .try_reserve_exact(index_block_count)
            .map_err(|_| VqError("could not allocate HPSI index block directory"))?;
        let mut next_row = 0usize;
        for block_id in 0..index_tensor.block_count {
            let descriptor = reader
                .block(indices_tensor_id, block_id)
                .map_err(|_| VqError("invalid HPSI index block descriptor"))?;
            let logical_offset = usize::try_from(descriptor.logical_offset)
                .map_err(|_| VqError("HPSI index block offset exceeds address space"))?;
            let logical_length = usize::try_from(descriptor.logical_length)
                .map_err(|_| VqError("HPSI index block length exceeds address space"))?;
            if logical_offset % bytes_per_row != 0
                || logical_length == 0
                || logical_length % bytes_per_row != 0
            {
                return Err(VqError("HPSI index block is not row-aligned"));
            }
            let first_row = logical_offset / bytes_per_row;
            let block_rows = logical_length / bytes_per_row;
            let end_row = first_row
                .checked_add(block_rows)
                .ok_or(VqError("HPSI index block row range overflow"))?;
            if first_row != next_row || end_row > rows {
                return Err(VqError(
                    "HPSI index blocks are not a contiguous row partition",
                ));
            }
            index_blocks.push(HpsiIndexBlock {
                block_id,
                first_row,
                end_row,
            });
            next_row = end_row;
        }
        if next_row != rows {
            return Err(VqError("HPSI index blocks do not cover all matrix rows"));
        }

        Ok(Self {
            reader,
            layout,
            codebook,
            bytes_per_row,
            index_blocks,
        })
    }

    /// Compute a contiguous output-row window without materializing dense weights.
    pub fn matvec_rows(
        &self,
        input: &[f32],
        first_row: usize,
        output: &mut [f32],
    ) -> Result<(), VqError> {
        if output.is_empty()
            || input.len() != self.layout.cols
            || first_row
                .checked_add(output.len())
                .is_none_or(|end| end > self.layout.rows)
        {
            return Err(VqError(
                "requested HPSI matvec row range or input shape is invalid",
            ));
        }

        let requested_end = first_row + output.len();
        let mut written_rows = 0usize;
        for descriptor in &self.index_blocks {
            if descriptor.end_row <= first_row || descriptor.first_row >= requested_end {
                continue;
            }

            let payload = self
                .reader
                .read_block(self.layout.indices_tensor_id, descriptor.block_id)
                .map_err(|_| VqError("failed to read selected HPSI index block"))?;
            let copy_start = descriptor.first_row.max(first_row);
            let copy_end = descriptor.end_row.min(requested_end);
            let source_row_start = copy_start - descriptor.first_row;
            let selected_rows = copy_end - copy_start;
            let byte_start = source_row_start
                .checked_mul(self.bytes_per_row)
                .ok_or(VqError("selected HPSI index offset overflow"))?;
            let byte_len = selected_rows
                .checked_mul(self.bytes_per_row)
                .ok_or(VqError("selected HPSI index length overflow"))?;
            let byte_end = byte_start
                .checked_add(byte_len)
                .ok_or(VqError("selected HPSI index range overflow"))?;
            let selected_indices = payload
                .payload
                .get(byte_start..byte_end)
                .ok_or(VqError("selected HPSI index range exceeds block payload"))?;

            let view = VqMatrixView::new(
                selected_rows,
                self.layout.cols,
                self.layout.codebook_count,
                &self.codebook,
                selected_indices,
            )?;
            let destination_start = copy_start - first_row;
            let destination_end = destination_start + selected_rows;
            view.matvec(input, &mut output[destination_start..destination_end])?;
            written_rows += selected_rows;
        }

        if written_rows != output.len() {
            return Err(VqError(
                "HPSI index blocks do not cover the requested output rows",
            ));
        }
        Ok(())
    }
}

/// Convenience function for one-off row windows. For repeated calls, prepare
/// an `HpsiVqExecutor` once so its codebook is not re-read for every window.
pub fn matvec_hpsi_rows(
    reader: &ModelReader,
    layout: HpsiVqLayout,
    input: &[f32],
    first_row: usize,
    output: &mut [f32],
) -> Result<(), VqError> {
    HpsiVqExecutor::new(reader, layout)?.matvec_rows(input, first_row, output)
}

fn beam_search_additive(
    vector: &[f32; GROUP_SIZE],
    codebooks: &[Vec<[f32; GROUP_SIZE]>],
    beam_width: usize,
) -> Result<([f32; GROUP_SIZE], [u8; 4]), VqError> {
    if codebooks.len() > 4 {
        return Err(VqError(
            "additive beam search supports at most four codebooks",
        ));
    }
    if !(1..=256).contains(&beam_width) {
        return Err(VqError("beam width must be between 1 and 256"));
    }

    // Keep the best partial additive reconstructions after every stage.
    let mut beam = Vec::new();
    beam.try_reserve_exact(1)
        .map_err(|_| VqError("could not allocate VQ search beam"))?;
    beam.push((0.0_f32, *vector, [0_u8; 4]));

    for (stage, codebook) in codebooks.iter().enumerate() {
        if codebook.is_empty() || codebook.len() > CODEBOOK_SIZE {
            return Err(VqError("additive codebook has an invalid centroid count"));
        }
        let expanded_capacity = beam
            .len()
            .checked_mul(codebook.len())
            .ok_or(VqError("VQ beam expansion size overflow"))?;
        let mut expanded = Vec::new();
        expanded
            .try_reserve_exact(expanded_capacity)
            .map_err(|_| VqError("could not allocate expanded VQ search beam"))?;

        for (_, residual, codes) in &beam {
            for (code, centroid) in codebook.iter().enumerate() {
                let next_residual = subtract(residual, centroid);
                let error = squared_distance(&next_residual, &[0.0; GROUP_SIZE]);
                let mut next_codes = *codes;
                next_codes[stage] = u8::try_from(code)
                    .map_err(|_| VqError("VQ centroid index exceeds selector range"))?;
                expanded.push((error, next_residual, next_codes));
            }
        }
        expanded.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));
        expanded.truncate(beam_width);
        beam = expanded;
    }

    let (_, residual, codes) = beam
        .into_iter()
        .next()
        .ok_or(VqError("additive beam search produced no candidates"))?;
    Ok((residual, codes))
}

#[cfg(test)]
mod additive_beam_search_tests {
    use super::*;

    #[test]
    fn empty_codebook_sequence_preserves_the_input_vector() {
        let vector = [0.25_f32; GROUP_SIZE];
        let (residual, codes) = beam_search_additive(&vector, &[], 4).unwrap();
        assert_eq!(residual, vector);
        assert_eq!(codes, [0; 4]);
    }

    #[test]
    fn rejects_invalid_beam_width_and_codebooks() {
        let vector = [0.0_f32; GROUP_SIZE];
        assert!(beam_search_additive(&vector, &[], 0).is_err());
        assert!(beam_search_additive(&vector, &[Vec::new()], 1).is_err());
        assert!(
            beam_search_additive(&vector, &[vec![[0.0; GROUP_SIZE]; CODEBOOK_SIZE + 1]], 1,)
                .is_err()
        );
        let too_many_codebooks = vec![vec![[0.0; GROUP_SIZE]; 1]; 5];
        assert!(beam_search_additive(&vector, &too_many_codebooks, 1).is_err());
    }
}

fn fit_codebook(
    samples: &[[f32; GROUP_SIZE]],
    iterations: usize,
    reserve_zero_centroid: bool,
) -> Result<Vec<[f32; GROUP_SIZE]>, VqError> {
    if samples.is_empty() {
        return Err(VqError("cannot fit a codebook without training samples"));
    }
    let mut centroids = Vec::new();
    centroids
        .try_reserve_exact(CODEBOOK_SIZE)
        .map_err(|_| VqError("could not allocate VQ centroids"))?;
    if reserve_zero_centroid {
        centroids.push([0.0; GROUP_SIZE]);
    } else {
        centroids.push(samples[0]);
    }

    // Deterministic farthest-first initialization spreads the initial centers
    // across the residual distribution instead of depending on input order.
    while centroids.len() < CODEBOOK_SIZE {
        let mut best_index = 0;
        let mut best_distance = f32::NEG_INFINITY;
        for (index, sample) in samples.iter().enumerate() {
            let distance = centroids
                .iter()
                .map(|centroid| squared_distance(sample, centroid))
                .fold(f32::INFINITY, f32::min);
            if distance > best_distance {
                best_distance = distance;
                best_index = index;
            }
        }
        centroids.push(samples[best_index]);
    }

    let mut nearest_distances = Vec::new();
    nearest_distances
        .try_reserve_exact(samples.len())
        .map_err(|_| VqError("could not allocate VQ nearest distances"))?;
    nearest_distances.resize(samples.len(), 0.0_f32);
    for _ in 0..iterations {
        let mut sums = [[0.0_f32; GROUP_SIZE]; CODEBOOK_SIZE];
        let mut counts = [0_usize; CODEBOOK_SIZE];
        for (sample_index, sample) in samples.iter().enumerate() {
            let cluster = nearest_centroid(sample, &centroids);
            let distance = squared_distance(sample, &centroids[cluster]);
            nearest_distances[sample_index] = distance;
            counts[cluster] += 1;
            for (sum, value) in sums[cluster].iter_mut().zip(sample) {
                *sum += *value;
            }
        }

        for cluster in 0..CODEBOOK_SIZE {
            if counts[cluster] == 0 || (reserve_zero_centroid && cluster == 0) {
                continue;
            }
            let divisor = counts[cluster] as f32;
            for dimension in 0..GROUP_SIZE {
                centroids[cluster][dimension] = sums[cluster][dimension] / divisor;
            }
            if centroids[cluster].iter().any(|value| !value.is_finite()) {
                return Err(VqError("codebook fitting overflowed"));
            }
        }

        // Empty clusters waste selector values. Re-seed them from distinct
        // worst-represented samples so subsequent Lloyd iterations can use
        // those codewords instead of retaining duplicate/obsolete centers.
        let mut reseeded = Vec::new();
        reseeded
            .try_reserve_exact(samples.len())
            .map_err(|_| VqError("could not allocate VQ reseed markers"))?;
        reseeded.resize(samples.len(), false);
        for cluster in 0..CODEBOOK_SIZE {
            if counts[cluster] != 0 || (reserve_zero_centroid && cluster == 0) {
                continue;
            }
            let candidate = nearest_distances
                .iter()
                .enumerate()
                .filter(|(index, _)| !reseeded[*index])
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index);
            if let Some(sample_index) = candidate {
                centroids[cluster] = samples[sample_index];
                reseeded[sample_index] = true;
            }
        }
    }
    Ok(centroids)
}

fn subtract(left: &[f32; GROUP_SIZE], right: &[f32; GROUP_SIZE]) -> [f32; GROUP_SIZE] {
    let mut result = [0.0; GROUP_SIZE];
    for dimension in 0..GROUP_SIZE {
        result[dimension] = left[dimension] - right[dimension];
    }
    result
}

fn squared_distance(left: &[f32; GROUP_SIZE], right: &[f32; GROUP_SIZE]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum()
}

fn nearest_centroid(vector: &[f32; GROUP_SIZE], centroids: &[[f32; GROUP_SIZE]]) -> usize {
    let mut best_index = 0;
    let mut best_distance = f32::INFINITY;
    for (index, centroid) in centroids.iter().enumerate() {
        let distance = squared_distance(vector, centroid);
        if distance < best_distance {
            best_distance = distance;
            best_index = index;
        }
    }
    best_index
}

#[cfg(test)]
mod payload_block_tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn vq_temporary_spool_is_created_with_private_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let spool = TemporarySpool::create().unwrap();
        let mode = fs::metadata(&spool.path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }

    #[test]
    fn splits_payload_without_losing_bytes() {
        let payload = (0_u8..10).collect::<Vec<_>>();
        let blocks = payload_blocks(&payload, 4).unwrap();
        assert_eq!(
            blocks
                .iter()
                .map(|block| block.bytes.as_slice())
                .collect::<Vec<_>>(),
            vec![&[0, 1, 2, 3][..], &[4, 5, 6, 7][..], &[8, 9][..]],
        );
    }

    #[test]
    fn rejects_zero_block_size() {
        assert!(payload_blocks(&[1, 2, 3], 0).is_err());
    }

    #[test]
    fn empty_payload_has_no_blocks() {
        assert!(payload_blocks(&[], 4).unwrap().is_empty());
    }
}

fn payload_blocks(bytes: &[u8], block_bytes: usize) -> Result<Vec<PayloadBlock>, VqError> {
    if block_bytes == 0 {
        return Err(VqError("payload block size must be non-zero"));
    }
    let block_count = (bytes.len() / block_bytes)
        .checked_add(usize::from(bytes.len() % block_bytes != 0))
        .ok_or(VqError("payload block count overflow"))?;
    let mut blocks = Vec::new();
    blocks
        .try_reserve_exact(block_count)
        .map_err(|_| VqError("could not allocate payload block directory"))?;
    for chunk in bytes.chunks(block_bytes) {
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(chunk.len())
            .map_err(|_| VqError("could not allocate payload block"))?;
        payload.extend_from_slice(chunk);
        blocks.push(PayloadBlock { bytes: payload });
    }
    Ok(blocks)
}

#[cfg(test)]
mod streaming_encoder_tests {
    use super::*;

    #[test]
    fn streaming_encoder_matches_in_memory_encoder() {
        let rows = 16;
        let cols = 32;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|index| ((index as f32) * 0.017).sin() * 0.6)
            .collect();
        for codebook_count in 1..=4 {
            let config = VqConfig {
                codebook_count,
                sample_limit: 32,
                ..VqConfig::default()
            };
            let expected = VqMatrix::encode(&weights, rows, cols, config).unwrap();
            let actual = VqMatrix::encode_from_reader(rows, cols, config, |first, count| {
                let start = usize::try_from(first).map_err(|_| VqError("test offset"))?;
                let end = start
                    .checked_add(count as usize)
                    .ok_or(VqError("test range"))?;
                Ok(weights
                    .get(start..end)
                    .ok_or(VqError("test range"))?
                    .to_vec())
            })
            .unwrap();
            assert_eq!(actual.codebook(), expected.codebook());
            assert_eq!(actual.packed_indices(), expected.packed_indices());
        }
    }

    #[test]
    fn streaming_activation_aware_encoder_matches_in_memory() {
        let rows = 16;
        let cols = 32;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|index| ((index as f32) * 0.031).cos() * 0.4)
            .collect();
        let mut importance = vec![1.0_f32; cols];
        for column in (0..cols).step_by(8) {
            importance[column] = 9.0;
        }
        let config = VqConfig {
            codebook_count: 2,
            sample_limit: 32,
            ..VqConfig::default()
        };
        let expected =
            VqMatrix::encode_activation_aware(&weights, rows, cols, config, &importance).unwrap();
        let actual = VqMatrix::encode_from_reader_activation_aware(
            rows,
            cols,
            config,
            &importance,
            |first, count| {
                let start = first as usize;
                Ok(weights[start..start + count as usize].to_vec())
            },
        )
        .unwrap();
        assert_eq!(actual.codebook(), expected.codebook());
        assert_eq!(actual.packed_indices(), expected.packed_indices());
    }

    #[test]
    fn streaming_encoder_rejects_bad_source_windows() {
        let config = VqConfig::default();
        assert!(VqMatrix::encode_from_reader(2, 8, config, |_, _| Ok(vec![f32::NAN; 8])).is_err());
        assert!(VqMatrix::encode_from_reader(2, 8, config, |_, _| Ok(vec![0.0; 7])).is_err());
    }
}

#[cfg(test)]
mod row_alignment_tests {
    use super::*;

    #[test]
    fn selector_rows_must_be_byte_aligned() {
        assert!(selectors_per_row(24, 1).is_err());
        assert_eq!(selectors_per_row(24, 2).unwrap(), 6);
        assert!(selectors_per_row(24, 3).is_err());
        assert_eq!(selectors_per_row(24, 4).unwrap(), 12);
        for codebooks in 1..=4 {
            assert_eq!(selectors_per_row(128, codebooks).unwrap(), 16 * codebooks);
        }
    }

    #[test]
    fn selector_row_validation_rejects_invalid_shapes_and_counts() {
        assert!(selectors_per_row(0, 1).is_err());
        assert!(selectors_per_row(7, 1).is_err());
        assert!(selectors_per_row(24, 0).is_err());
        assert!(selectors_per_row(24, 5).is_err());
    }
}

#[cfg(test)]
mod streaming_row_emission_tests {
    use super::*;

    #[test]
    fn row_emission_matches_buffered_streaming_encoder() {
        let rows = 6;
        let cols = 16;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|index| ((index * 17 % 31) as f32 - 15.0) / 7.0)
            .collect();
        for codebook_count in 1..=4 {
            let config = VqConfig {
                iterations: 4,
                sample_limit: 32,
                codebook_count,
                beam_width: 8,
            };
            if selectors_per_row(cols, codebook_count).is_err() {
                continue;
            }
            let buffered = VqMatrix::encode_from_reader(rows, cols, config, |offset, count| {
                let start = offset as usize;
                let end = start + count as usize;
                Ok(weights[start..end].to_vec())
            })
            .unwrap();
            let mut emitted = Vec::new();
            let bytes_per_row = selectors_per_row(cols, codebook_count).unwrap() / 2;
            let codebook = VqMatrix::encode_from_reader_to(
                rows,
                cols,
                config,
                |offset, count| {
                    let start = offset as usize;
                    let end = start + count as usize;
                    Ok(weights[start..end].to_vec())
                },
                |row, bytes| {
                    assert_eq!(row as usize * bytes_per_row, emitted.len());
                    emitted.extend_from_slice(bytes);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(codebook, buffered.codebook());
            assert_eq!(emitted, buffered.packed_indices());
        }
    }

    #[test]
    fn row_emitter_propagates_consumer_failure() {
        let result = VqMatrix::encode_from_reader_to(
            2,
            16,
            VqConfig::default(),
            |_, count| Ok(vec![0.25; count as usize]),
            |_, _| Err(VqError("downstream writer failed")),
        );
        assert!(matches!(result, Err(VqError("downstream writer failed"))));
    }
}

#[cfg(test)]
mod streaming_activation_row_tests {
    use super::*;

    #[test]
    fn activation_aware_row_emission_matches_buffered_encoder() {
        let rows = 5;
        let cols = 16;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|index| ((index * 13 % 29) as f32 - 14.0) / 9.0)
            .collect();
        let importance: Vec<f32> = (0..cols).map(|index| (index + 1) as f32 / 3.0).collect();
        let config = VqConfig {
            iterations: 5,
            sample_limit: 64,
            codebook_count: 2,
            beam_width: 8,
        };
        let buffered = VqMatrix::encode_from_reader_activation_aware(
            rows,
            cols,
            config,
            &importance,
            |offset, count| {
                let start = offset as usize;
                Ok(weights[start..start + count as usize].to_vec())
            },
        )
        .unwrap();
        let mut emitted = Vec::new();
        let codebook = VqMatrix::encode_from_reader_activation_aware_to(
            rows,
            cols,
            config,
            &importance,
            |offset, count| {
                let start = offset as usize;
                Ok(weights[start..start + count as usize].to_vec())
            },
            |_, row| {
                emitted.extend_from_slice(row);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(codebook, buffered.codebook());
        assert_eq!(emitted, buffered.packed_indices());

        let mut reused_emitted = Vec::new();
        let reused_codebook = VqMatrix::encode_from_reader_activation_aware_into(
            rows,
            cols,
            config,
            &importance,
            |offset, target| {
                let start = offset as usize;
                target.copy_from_slice(&weights[start..start + target.len()]);
                Ok(())
            },
            |_, row| {
                reused_emitted.extend_from_slice(row);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(reused_codebook, codebook);
        assert_eq!(reused_emitted, emitted);
    }
}

#[cfg(test)]
mod hpsi_streaming_export_tests {
    use super::*;
    use hpsi_format::{MemorySource, ReaderLimits};
    use std::io::Cursor;
    use std::sync::Arc;

    #[test]
    fn reader_encoder_accepts_caller_assigned_tensor_ids_and_names() {
        let rows = 2;
        let cols = 16;
        let weights = vec![0.25_f32; rows * cols];
        let mut output = Cursor::new(Vec::new());
        let mut spool = Cursor::new(Vec::new());
        let stats = VqMatrix::encode_from_reader_to_hpsi_named(
            &mut output,
            &mut spool,
            rows,
            cols,
            VqConfig {
                iterations: 2,
                sample_limit: 8,
                codebook_count: 1,
                beam_width: 2,
            },
            VqHpsiExportOptions {
                block_rows: 1,
                tensor_names: VqTensorNames {
                    codebook_id: 40,
                    codebook_name: "layer.weight.codebook",
                    indices_id: 41,
                    indices_name: "layer.weight.indices",
                },
            },
            |offset, count| {
                let start = offset as usize;
                Ok(weights[start..start + count as usize].to_vec())
            },
        )
        .unwrap();
        assert_eq!(stats.tensor_count, 2);
        let reader = ModelReader::open(
            Arc::new(MemorySource::new(Arc::<[u8]>::from(output.into_inner()))),
            &ReaderLimits::default(),
        )
        .unwrap();
        reader.verify_all().unwrap();
        assert!(!reader.read_block(40, 0).unwrap().payload.is_empty());
        assert!(!reader.read_block(41, 0).unwrap().payload.is_empty());
    }

    #[test]
    fn reader_encoder_round_trips_selector_blocks_for_varied_block_rows() {
        let rows = 7usize;
        let cols = 16usize;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|index| ((index * 13 % 29) as f32 - 14.0) / 9.0)
            .collect();
        let config = VqConfig {
            iterations: 3,
            sample_limit: 32,
            codebook_count: 2,
            beam_width: 4,
        };
        let expected_selector_bytes = rows * (cols / GROUP_SIZE * config.codebook_count) / 2;

        for (block_rows, expected_blocks) in [(1usize, 7usize), (3, 3), (16, 1)] {
            let mut output = Cursor::new(Vec::new());
            let mut spool = Cursor::new(Vec::new());
            let stats = VqMatrix::encode_from_reader_to_hpsi(
                &mut output,
                &mut spool,
                rows,
                cols,
                config,
                block_rows,
                |offset, count| {
                    let start = offset as usize;
                    Ok(weights[start..start + count as usize].to_vec())
                },
            )
            .unwrap();
            assert_eq!(
                stats.block_count,
                u32::try_from(expected_blocks)
                    .unwrap()
                    .checked_add(1)
                    .unwrap()
            );

            let reader = ModelReader::open(
                Arc::new(MemorySource::new(Arc::<[u8]>::from(output.into_inner()))),
                &ReaderLimits::default(),
            )
            .unwrap();
            reader.verify_all().unwrap();
            let mut selectors = Vec::new();
            for block_id in 0..expected_blocks {
                selectors
                    .extend_from_slice(&reader.read_block(1, block_id as u32).unwrap().payload);
            }
            assert_eq!(selectors.len(), expected_selector_bytes);
        }
    }

    #[test]
    fn reader_encoder_rejects_selector_spool_with_trailing_bytes() {
        let rows = 2usize;
        let cols = 8usize;
        let weights = vec![0.25_f32; rows * cols];
        let mut output = Cursor::new(Vec::new());
        let mut spool = Cursor::new(vec![0xAA]);
        let result = VqMatrix::encode_from_reader_to_hpsi(
            &mut output,
            &mut spool,
            rows,
            cols,
            VqConfig {
                iterations: 2,
                sample_limit: 8,
                codebook_count: 1,
                beam_width: 2,
            },
            1,
            |offset, count| {
                let start = offset as usize;
                Ok(weights[start..start + count as usize].to_vec())
            },
        );
        assert!(matches!(
            result,
            Err(VqError("selector spool must be empty"))
        ));
    }

    #[test]
    fn reader_encoder_streams_vq_tensors_into_hpsi_writer() {
        let rows = 8;
        let cols = 16;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|index| ((index * 19 % 37) as f32 - 18.0) / 11.0)
            .collect();
        let config = VqConfig {
            iterations: 4,
            sample_limit: 64,
            codebook_count: 2,
            beam_width: 8,
        };
        let mut output = Cursor::new(Vec::new());
        let mut spool = Cursor::new(Vec::new());
        let stats = VqMatrix::encode_from_reader_to_hpsi(
            &mut output,
            &mut spool,
            rows,
            cols,
            config,
            3,
            |offset, count| {
                let start = offset as usize;
                Ok(weights[start..start + count as usize].to_vec())
            },
        )
        .unwrap();
        assert_eq!(stats.tensor_count, 2);
        assert_eq!(stats.block_count, 4);
        let reader = ModelReader::open(
            Arc::new(MemorySource::new(Arc::<[u8]>::from(output.into_inner()))),
            &ReaderLimits::default(),
        )
        .unwrap();
        reader.verify_all().unwrap();
        let mut selectors = Vec::new();
        for block_id in 0..3 {
            selectors.extend_from_slice(&reader.read_block(1, block_id).unwrap().payload);
        }
        assert_eq!(
            selectors.len(),
            rows * (cols / GROUP_SIZE * config.codebook_count) / 2
        );
    }
}

#[cfg(test)]
mod reusable_reader_buffer_tests {
    use super::*;

    #[test]
    fn reusable_reader_buffer_matches_vec_reader_encoding() {
        let rows = 8usize;
        let cols = 16usize;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|index| ((index as f32) * 0.071).sin() * 0.5)
            .collect();
        let config = VqConfig {
            iterations: 4,
            sample_limit: 32,
            codebook_count: 2,
            beam_width: 4,
        };
        let mut legacy_rows = Vec::new();
        let legacy_codebook = VqMatrix::encode_from_reader_to(
            rows,
            cols,
            config,
            |offset, count| {
                let start = offset as usize;
                Ok(weights[start..start + count as usize].to_vec())
            },
            |_, row| {
                legacy_rows.extend_from_slice(row);
                Ok(())
            },
        )
        .unwrap();

        let mut reused_rows = Vec::new();
        let reused_codebook = VqMatrix::encode_from_reader_into(
            rows,
            cols,
            config,
            |offset, target| {
                let start = offset as usize;
                target.copy_from_slice(&weights[start..start + target.len()]);
                Ok(())
            },
            |_, row| {
                reused_rows.extend_from_slice(row);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(reused_codebook, legacy_codebook);
        assert_eq!(reused_rows, legacy_rows);
    }
}

#[cfg(test)]
mod resource_limit_tests {
    use super::*;

    #[test]
    fn rejects_vq_configurations_outside_resource_limits() {
        let weights = [0.25_f32; GROUP_SIZE];
        let invalid_configs = [
            VqConfig {
                iterations: 0,
                ..VqConfig::default()
            },
            VqConfig {
                iterations: MAX_VQ_ITERATIONS + 1,
                ..VqConfig::default()
            },
            VqConfig {
                sample_limit: 0,
                ..VqConfig::default()
            },
            VqConfig {
                sample_limit: MAX_VQ_TRAINING_SAMPLES + 1,
                ..VqConfig::default()
            },
            VqConfig {
                codebook_count: 0,
                ..VqConfig::default()
            },
            VqConfig {
                codebook_count: 5,
                ..VqConfig::default()
            },
            VqConfig {
                beam_width: 0,
                ..VqConfig::default()
            },
            VqConfig {
                beam_width: 257,
                ..VqConfig::default()
            },
        ];

        for config in invalid_configs {
            assert!(VqMatrix::encode(&weights, 1, GROUP_SIZE, config).is_err());
        }
    }

    #[test]
    fn codebook_fitting_rejects_empty_training_samples() {
        assert!(fit_codebook(&[], 1, false).is_err());
        assert!(fit_codebook(&[], 1, true).is_err());
    }
}
