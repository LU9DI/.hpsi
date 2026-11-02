use hpsi_format::{DType, PayloadBlock, TensorInput};

pub const GROUP_SIZE: usize = 8;
pub const CODEBOOK_SIZE: usize = 16;
const CODEBOOK_VALUES: usize = GROUP_SIZE * CODEBOOK_SIZE;

#[derive(Debug, Clone, Copy)]
pub struct VqConfig {
    pub iterations: usize,
    pub sample_limit: usize,
    /// One codebook uses four selector bits per group. Two codebooks use
    /// additive residual quantization and eight selector bits per group.
    pub codebook_count: usize,
}

impl Default for VqConfig {
    fn default() -> Self {
        Self {
            iterations: 12,
            sample_limit: 65_536,
            codebook_count: 1,
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

/// Matrix represented by packed selectors into one or two shared codebooks.
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
        if !(1..=2).contains(&config.codebook_count) {
            return Err(VqError("only one or two additive codebooks are supported"));
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

        let primary = fit_codebook(&samples, config.iterations, false)?;
        let mut codebooks = Vec::with_capacity(config.codebook_count);
        codebooks.push(primary);

        if config.codebook_count == 2 {
            let residual_samples: Vec<[f32; GROUP_SIZE]> = samples
                .iter()
                .map(|sample| {
                    let primary_code = nearest_centroid(sample, &codebooks[0]);
                    subtract(sample, &codebooks[0][primary_code])
                })
                .collect();
            // The zero centroid guarantees that the residual stage cannot
            // increase per-group reconstruction error in weight space.
            let residual = fit_codebook(&residual_samples, config.iterations, true)?;
            codebooks.push(residual);
        }

        let packed_len = if config.codebook_count == 1 {
            group_count
                .checked_add(1)
                .ok_or(VqError("packed index count overflow"))?
                / 2
        } else {
            group_count
        };
        let mut packed_indices = vec![0_u8; packed_len];

        for group_index in 0..group_count {
            let start = group_index * GROUP_SIZE;
            let mut vector = [0.0_f32; GROUP_SIZE];
            vector.copy_from_slice(&weights[start..start + GROUP_SIZE]);
            let primary_code = nearest_centroid(&vector, &codebooks[0]) as u8;

            if config.codebook_count == 1 {
                if group_index % 2 == 0 {
                    packed_indices[group_index / 2] = primary_code;
                } else {
                    packed_indices[group_index / 2] |= primary_code << 4;
                }
            } else {
                let residual_vector = subtract(&vector, &codebooks[0][primary_code as usize]);
                let residual_code = nearest_centroid(&residual_vector, &codebooks[1]) as u8;
                packed_indices[group_index] = primary_code | (residual_code << 4);
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
        block_bytes: usize,
    ) -> Result<Vec<TensorInput>, VqError> {
        if codebook_tensor_id >= indices_tensor_id {
            return Err(VqError("codebook tensor ID must precede index tensor ID"));
        }
        if block_bytes < 4 || block_bytes % 4 != 0 {
            return Err(VqError(
                "block size must be a positive multiple of four bytes",
            ));
        }
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
                blocks: payload_blocks(&codebook_bytes, block_bytes),
            },
            TensorInput {
                tensor_id: indices_tensor_id,
                name: "hpsi.vq.indices".into(),
                dtype: DType::Opaque,
                dimensions: vec![self.packed_indices.len() as u64],
                blocks: payload_blocks(&self.packed_indices, block_bytes),
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
        if !(1..=2).contains(&codebook_count) {
            return Err(VqError("only one or two codebooks are supported"));
        }
        if codebook.len() != CODEBOOK_VALUES * codebook_count
            || codebook.iter().any(|value| !value.is_finite())
        {
            return Err(VqError("codebook shape or values are invalid"));
        }
        let groups = rows
            .checked_mul(cols / GROUP_SIZE)
            .ok_or(VqError("matrix group count overflow"))?;
        let expected_indices = if codebook_count == 1 {
            groups
                .checked_add(1)
                .ok_or(VqError("packed index count overflow"))?
                / 2
        } else {
            groups
        };
        if packed_indices.len() != expected_indices {
            return Err(VqError(
                "packed index stream length does not match matrix shape",
            ));
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
        for (row, output_value) in output.iter_mut().enumerate() {
            let mut accumulator = 0.0_f32;
            for group in 0..groups_per_row {
                let global_group = row * groups_per_row + group;
                let (primary_code, residual_code) = if self.codebook_count == 1 {
                    let packed = self.packed_indices[global_group / 2];
                    (
                        (if global_group % 2 == 0 {
                            packed & 0x0f
                        } else {
                            packed >> 4
                        }) as usize,
                        None,
                    )
                } else {
                    let packed = self.packed_indices[global_group];
                    (usize::from(packed & 0x0f), Some(usize::from(packed >> 4)))
                };

                let input_offset = group * GROUP_SIZE;
                let primary_offset = primary_code * GROUP_SIZE;
                let residual_offset = residual_code.map(|code| CODEBOOK_VALUES + code * GROUP_SIZE);

                for dimension in 0..GROUP_SIZE {
                    let input_value = input[input_offset + dimension];
                    accumulator += input_value * self.codebook[primary_offset + dimension];
                    if let Some(offset) = residual_offset {
                        accumulator += input_value * self.codebook[offset + dimension];
                    }
                }
            }
            *output_value = accumulator;
        }
        Ok(())
    }
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

    for _ in 0..iterations {
        let mut sums = [[0.0_f32; GROUP_SIZE]; CODEBOOK_SIZE];
        let mut counts = [0_usize; CODEBOOK_SIZE];
        for sample in samples {
            let cluster = nearest_centroid(sample, &centroids);
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
