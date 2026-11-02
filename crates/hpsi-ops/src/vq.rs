use hpsi_format::{DType, ModelReader, PayloadBlock, TensorInput};

pub const GROUP_SIZE: usize = 8;
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
            beam_width: 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VqError(pub &'static str);

impl std::fmt::Display for VqError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}
impl std::error::Error for VqError {}

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
        let expected = rows
            .checked_mul(cols)
            .ok_or(VqError("matrix element count overflow"))?;
        if rows == 0 || cols == 0 || cols % GROUP_SIZE != 0 || weights.len() != expected {
            return Err(VqError(
                "matrix shape must be non-zero, group-aligned, and match weights",
            ));
        }
        if config.iterations == 0 || config.sample_limit == 0 {
            return Err(VqError(
                "k-means iterations and sample limit must be non-zero",
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
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(VqError("matrix weights must be finite"));
        }

        let group_count = expected / GROUP_SIZE;
        let sample_count = group_count.min(config.sample_limit);
        let mut samples = Vec::with_capacity(sample_count);
        for sample_index in 0..sample_count {
            let group_index =
                ((sample_index as u128 * group_count as u128) / sample_count as u128) as usize;
            let mut vector = [0.0_f32; GROUP_SIZE];
            let start = group_index * GROUP_SIZE;
            vector.copy_from_slice(&weights[start..start + GROUP_SIZE]);
            samples.push(vector);
        }

        let mut codebooks: Vec<Vec<[f32; GROUP_SIZE]>> = Vec::with_capacity(config.codebook_count);
        for stage in 0..config.codebook_count {
            let residual_samples: Vec<[f32; GROUP_SIZE]> = samples
                .iter()
                .map(|sample| beam_search_additive(sample, &codebooks, config.beam_width).0)
                .collect();
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
        let mut packed_indices = vec![0_u8; packed_len];

        for group_index in 0..group_count {
            let start = group_index * GROUP_SIZE;
            let mut residual = [0.0_f32; GROUP_SIZE];
            residual.copy_from_slice(&weights[start..start + GROUP_SIZE]);
            let (_, codes) = beam_search_additive(&residual, &codebooks, config.beam_width);
            for (stage, code) in codes.iter().take(config.codebook_count).enumerate() {
                let selector = group_index * config.codebook_count + stage;
                let byte_index = selector / 2;
                let shift = (selector % 2) * 4;
                packed_indices[byte_index] |= *code << shift;
            }
        }

        let codebook = codebooks
            .into_iter()
            .flat_map(|book| book.into_iter().flat_map(|centroid| centroid.into_iter()))
            .collect();

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
        if block_rows == 0 || self.cols % 16 != 0 {
            return Err(VqError(
                "row-streamed export requires positive block rows and columns divisible by 16",
            ));
        }
        let groups_per_row = self.cols / GROUP_SIZE;
        let selectors_per_row = groups_per_row
            .checked_mul(self.codebook_count)
            .ok_or(VqError("selector count per row overflow"))?;
        let bytes_per_row = selectors_per_row / 2;
        let index_block_bytes = bytes_per_row
            .checked_mul(block_rows)
            .ok_or(VqError("index block byte count overflow"))?;

        let mut codebook_bytes = Vec::with_capacity(self.codebook.len() * 4);
        for value in &self.codebook {
            codebook_bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(vec![
            TensorInput {
                tensor_id: codebook_tensor_id,
                name: "hpsi.vq.codebook".into(),
                dtype: DType::F32,
                dimensions: vec![
                    self.codebook_count as u64,
                    CODEBOOK_SIZE as u64,
                    GROUP_SIZE as u64,
                ],
                blocks: payload_blocks(&codebook_bytes, 64 * 1024),
            },
            TensorInput {
                tensor_id: indices_tensor_id,
                name: "hpsi.vq.indices".into(),
                dtype: DType::Opaque,
                dimensions: vec![self.packed_indices.len() as u64],
                blocks: payload_blocks(&self.packed_indices, index_block_bytes),
            },
        ])
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
        if rows == 0 || cols == 0 || cols % GROUP_SIZE != 0 {
            return Err(VqError("matrix shape must be non-zero and group-aligned"));
        }
        if !(1..=4).contains(&codebook_count) {
            return Err(VqError("only one through four codebooks are supported"));
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
        if rows == 0 || cols == 0 || cols % 16 != 0 || !(1..=4).contains(&codebook_count) {
            return Err(VqError(
                "HPSI VQ layout has invalid shape or codebook count",
            ));
        }

        let codebook_tensor = reader
            .tensor(codebook_tensor_id)
            .map_err(|_| VqError("could not locate HPSI codebook tensor"))?;
        let expected_codebook_bytes = CODEBOOK_VALUES * codebook_count * 4;
        if codebook_tensor.dtype != DType::F32
            || codebook_tensor.rank != 3
            || codebook_tensor.dimensions[0] != codebook_count as u64
            || codebook_tensor.dimensions[1] != CODEBOOK_SIZE as u64
            || codebook_tensor.dimensions[2] != GROUP_SIZE as u64
            || codebook_tensor.logical_nbytes != expected_codebook_bytes as u64
        {
            return Err(VqError(
                "HPSI codebook tensor descriptor does not match VQ layout",
            ));
        }

        let mut codebook_bytes = Vec::with_capacity(expected_codebook_bytes);
        for block_id in 0..codebook_tensor.block_count {
            let block = reader
                .read_block(codebook_tensor_id, block_id)
                .map_err(|_| VqError("failed to read HPSI codebook block"))?;
            codebook_bytes.extend_from_slice(&block.payload);
        }
        if codebook_bytes.len() != expected_codebook_bytes {
            return Err(VqError("HPSI codebook payload length is invalid"));
        }
        let codebook: Vec<f32> = codebook_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect();
        if codebook.iter().any(|value| !value.is_finite()) {
            return Err(VqError("HPSI codebook contains non-finite values"));
        }

        let groups_per_row = cols / GROUP_SIZE;
        let selectors_per_row = groups_per_row
            .checked_mul(codebook_count)
            .ok_or(VqError("selector count per row overflow"))?;
        let bytes_per_row = selectors_per_row / 2;
        let expected_index_bytes = rows
            .checked_mul(bytes_per_row)
            .ok_or(VqError("HPSI index tensor size overflow"))?;
        let index_tensor = reader
            .tensor(indices_tensor_id)
            .map_err(|_| VqError("could not locate HPSI index tensor"))?;
        if index_tensor.dtype != DType::Opaque
            || index_tensor.rank != 1
            || index_tensor.dimensions[0] != expected_index_bytes as u64
            || index_tensor.logical_nbytes != expected_index_bytes as u64
        {
            return Err(VqError(
                "HPSI index tensor descriptor does not match VQ layout",
            ));
        }

        let mut index_blocks = Vec::with_capacity(index_tensor.block_count as usize);
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
) -> ([f32; GROUP_SIZE], [u8; 4]) {
    // Keep the best partial additive reconstructions after every stage.
    let mut beam = vec![(0.0_f32, *vector, [0_u8; 4])];
    for (stage, codebook) in codebooks.iter().enumerate() {
        let mut expanded = Vec::with_capacity(beam.len() * codebook.len());
        for (_, residual, codes) in &beam {
            for (code, centroid) in codebook.iter().enumerate() {
                let next_residual = subtract(residual, centroid);
                let error = squared_distance(&next_residual, &[0.0; GROUP_SIZE]);
                let mut next_codes = *codes;
                next_codes[stage] = code as u8;
                expanded.push((error, next_residual, next_codes));
            }
        }
        expanded.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));
        expanded.truncate(beam_width);
        beam = expanded;
    }
    let (_, residual, codes) = beam.into_iter().next().unwrap_or((0.0, *vector, [0; 4]));
    (residual, codes)
}

fn fit_codebook(
    samples: &[[f32; GROUP_SIZE]],
    iterations: usize,
    reserve_zero_centroid: bool,
) -> Result<Vec<[f32; GROUP_SIZE]>, VqError> {
    let mut centroids = Vec::with_capacity(CODEBOOK_SIZE);
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

    let mut nearest_distances = vec![0.0_f32; samples.len()];
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
        let mut reseeded = vec![false; samples.len()];
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

fn payload_blocks(bytes: &[u8], block_bytes: usize) -> Vec<PayloadBlock> {
    bytes
        .chunks(block_bytes)
        .map(|chunk| PayloadBlock {
            bytes: chunk.to_vec(),
        })
        .collect()
}
