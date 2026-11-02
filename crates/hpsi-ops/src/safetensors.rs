use crate::{VqConfig, VqError, VqMatrix};
use hpsi_format::{RandomAccessSource, WriterStats};
use serde_json::Value;
use std::io::{Seek, Write};
use std::sync::Arc;

/// Bounded, random-access reader for the public Safetensors container layout.
/// The JSON header is loaded; tensor payloads remain on the underlying source.
pub struct SafeTensorFile {
    source: Arc<dyn RandomAccessSource>,
    data_start: u64,
    tensors: Vec<SafeTensorInfo>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_start: u64,
    pub data_end: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeTensorError {
    Io,
    HeaderTooLarge,
    InvalidHeader,
    InvalidTensor,
    UnsupportedDtype,
    OutOfBounds,
    Allocation,
}
impl std::fmt::Display for SafeTensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SafeTensorError {}

impl SafeTensorFile {
    pub fn open(
        source: Arc<dyn RandomAccessSource>,
        max_header_bytes: usize,
    ) -> Result<Self, SafeTensorError> {
        let mut prefix = [0_u8; 8];
        source
            .read_exact_at(0, &mut prefix)
            .map_err(|_| SafeTensorError::Io)?;
        let header_len = usize::try_from(u64::from_le_bytes(prefix))
            .map_err(|_| SafeTensorError::HeaderTooLarge)?;
        if header_len == 0 || header_len > max_header_bytes {
            return Err(SafeTensorError::HeaderTooLarge);
        }
        let data_start = 8_u64
            .checked_add(header_len as u64)
            .ok_or(SafeTensorError::InvalidHeader)?;
        if data_start > source.len().map_err(|_| SafeTensorError::Io)? {
            return Err(SafeTensorError::InvalidHeader);
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(header_len)
            .map_err(|_| SafeTensorError::Allocation)?;
        bytes.resize(header_len, 0);
        source
            .read_exact_at(8, &mut bytes)
            .map_err(|_| SafeTensorError::Io)?;
        let root: Value =
            serde_json::from_slice(&bytes).map_err(|_| SafeTensorError::InvalidHeader)?;
        let object = root.as_object().ok_or(SafeTensorError::InvalidHeader)?;
        let file_len = source.len().map_err(|_| SafeTensorError::Io)?;
        let data_len = file_len - data_start;
        let mut tensors = Vec::new();
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            let entry = value.as_object().ok_or(SafeTensorError::InvalidTensor)?;
            let dtype = entry
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or(SafeTensorError::InvalidTensor)?
                .to_owned();
            let shape = entry
                .get("shape")
                .and_then(Value::as_array)
                .ok_or(SafeTensorError::InvalidTensor)?
                .iter()
                .map(|v| v.as_u64().ok_or(SafeTensorError::InvalidTensor))
                .collect::<Result<Vec<_>, _>>()?;
            let offsets = entry
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or(SafeTensorError::InvalidTensor)?;
            if offsets.len() != 2 {
                return Err(SafeTensorError::InvalidTensor);
            }
            let start = offsets[0].as_u64().ok_or(SafeTensorError::InvalidTensor)?;
            let end = offsets[1].as_u64().ok_or(SafeTensorError::InvalidTensor)?;
            if start > end || end > data_len {
                return Err(SafeTensorError::OutOfBounds);
            }
            let elements = shape.iter().try_fold(1_u64, |a, &b| {
                a.checked_mul(b).ok_or(SafeTensorError::InvalidTensor)
            })?;
            let width = match dtype.as_str() {
                "F32" => 4,
                "F16" | "BF16" => 2,
                "F64" => 8,
                "I64" => 8,
                "I32" => 4,
                "I16" => 2,
                "I8" | "U8" | "BOOL" => 1,
                _ => return Err(SafeTensorError::UnsupportedDtype),
            };
            if elements
                .checked_mul(width)
                .ok_or(SafeTensorError::InvalidTensor)?
                != end - start
            {
                return Err(SafeTensorError::InvalidTensor);
            }
            tensors.push(SafeTensorInfo {
                name: name.to_owned(),
                dtype,
                shape,
                data_start: start,
                data_end: end,
            });
        }
        tensors.sort_by_key(|t| t.data_start);
        let mut expected_offset = 0_u64;
        for tensor in &tensors {
            if tensor.data_start != expected_offset {
                return Err(SafeTensorError::InvalidTensor);
            }
            expected_offset = tensor.data_end;
        }
        if expected_offset != data_len {
            return Err(SafeTensorError::InvalidTensor);
        }
        Ok(Self {
            source,
            data_start,
            tensors,
        })
    }
    /// Quantize a rank-2 F32/F16/BF16 Safetensors tensor and write it
    /// directly into an HPSI container. Source reads are bounded to samples
    /// and one row; packed selectors are spooled to a managed temporary file.
    pub fn encode_f32_tensor_to_hpsi<W: Write + Seek>(
        &self,
        tensor_name: &str,
        output: &mut W,
        config: VqConfig,
        block_rows: usize,
    ) -> Result<WriterStats, VqError> {
        let tensor = self
            .tensors
            .iter()
            .find(|tensor| tensor.name == tensor_name)
            .ok_or(VqError("Safetensors tensor was not found"))?;
        if tensor.shape.len() != 2 || !matches!(tensor.dtype.as_str(), "F32" | "F16" | "BF16") {
            return Err(VqError(
                "HPSI VQ import requires a rank-2 F32, F16, or BF16 tensor",
            ));
        }
        let rows = usize::try_from(tensor.shape[0])
            .map_err(|_| VqError("Safetensors row count exceeds address space"))?;
        let cols = usize::try_from(tensor.shape[1])
            .map_err(|_| VqError("Safetensors column count exceeds address space"))?;
        VqMatrix::encode_from_reader_to_hpsi_with_temp_spool(
            output,
            rows,
            cols,
            config,
            block_rows,
            |first, count| {
                self.read_f32_elements(tensor_name, first, count)
                    .map_err(|_| VqError("Safetensors source window read failed"))
            },
        )
    }

    /// Copy an original tensor payload to a writer without decoding or changing
    /// its dtype. Reads are bounded by `chunk_bytes`, so large raw tensors do
    /// not need to be materialized in memory during model import.
    pub fn copy_raw_payload<W: Write>(
        &self,
        name: &str,
        output: &mut W,
        chunk_bytes: usize,
    ) -> Result<u64, SafeTensorError> {
        if chunk_bytes == 0 {
            return Err(SafeTensorError::InvalidTensor);
        }
        let tensor = self.tensors.iter().find(|tensor| tensor.name == name)
            .ok_or(SafeTensorError::InvalidTensor)?;
        let length = tensor.data_end.checked_sub(tensor.data_start)
            .ok_or(SafeTensorError::InvalidTensor)?;
        let absolute_start = self.data_start.checked_add(tensor.data_start)
            .ok_or(SafeTensorError::OutOfBounds)?;
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(chunk_bytes).map_err(|_| SafeTensorError::Allocation)?;
        buffer.resize(chunk_bytes, 0);
        let mut copied = 0_u64;
        while copied < length {
            let count = (length - copied).min(chunk_bytes as u64) as usize;
            let offset = absolute_start.checked_add(copied).ok_or(SafeTensorError::OutOfBounds)?;
            self.source.read_exact_at(offset, &mut buffer[..count])
                .map_err(|_| SafeTensorError::Io)?;
            output.write_all(&buffer[..count]).map_err(|_| SafeTensorError::Io)?;
            copied = copied.checked_add(count as u64).ok_or(SafeTensorError::OutOfBounds)?;
        }
        Ok(copied)
    }

    pub fn tensors(&self) -> &[SafeTensorInfo] {
        &self.tensors
    }
    /// Read an arbitrary contiguous element range from a rank-1 or rank-2 floating tensor.
    /// The returned allocation is bounded by the requested range, not the full tensor size.
    pub fn read_f32_elements(
        &self,
        name: &str,
        first_element: u64,
        element_count: u64,
    ) -> Result<Vec<f32>, SafeTensorError> {
        let tensor = self
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or(SafeTensorError::InvalidTensor)?;
        if !(1..=2).contains(&tensor.shape.len()) {
            return Err(SafeTensorError::UnsupportedDtype);
        }
        let width = match tensor.dtype.as_str() {
            "F32" => 4_u64,
            "F16" | "BF16" => 2_u64,
            _ => return Err(SafeTensorError::UnsupportedDtype),
        };
        let total_elements = tensor.shape.iter().try_fold(1_u64, |total, &dimension| {
            total
                .checked_mul(dimension)
                .ok_or(SafeTensorError::InvalidTensor)
        })?;
        let end_element = first_element
            .checked_add(element_count)
            .ok_or(SafeTensorError::OutOfBounds)?;
        if end_element > total_elements {
            return Err(SafeTensorError::OutOfBounds);
        }
        let byte_count = element_count
            .checked_mul(width)
            .ok_or(SafeTensorError::OutOfBounds)?;
        let allocation = usize::try_from(byte_count).map_err(|_| SafeTensorError::Allocation)?;
        let relative_offset = first_element
            .checked_mul(width)
            .ok_or(SafeTensorError::OutOfBounds)?;
        let absolute_offset = self
            .data_start
            .checked_add(tensor.data_start)
            .and_then(|offset| offset.checked_add(relative_offset))
            .ok_or(SafeTensorError::OutOfBounds)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(allocation)
            .map_err(|_| SafeTensorError::Allocation)?;
        bytes.resize(allocation, 0);
        self.source
            .read_exact_at(absolute_offset, &mut bytes)
            .map_err(|_| SafeTensorError::Io)?;

        let values: Vec<f32> = match tensor.dtype.as_str() {
            "F32" => bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
            "BF16" => bytes
                .chunks_exact(2)
                .map(|chunk| {
                    f32::from_bits(u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16)
                })
                .collect(),
            "F16" => bytes
                .chunks_exact(2)
                .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect(),
            _ => return Err(SafeTensorError::UnsupportedDtype),
        };
        if values.iter().any(|value| !value.is_finite()) {
            return Err(SafeTensorError::InvalidTensor);
        }
        Ok(values)
    }

    /// Read a row window of a rank-2 floating tensor.
    /// Other ranks and non-floating dtypes are rejected, never silently cast.
    pub fn read_f32_rows(
        &self,
        name: &str,
        first_row: u64,
        row_count: u64,
    ) -> Result<Vec<f32>, SafeTensorError> {
        let tensor = self
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or(SafeTensorError::InvalidTensor)?;
        if tensor.shape.len() != 2 {
            return Err(SafeTensorError::UnsupportedDtype);
        }
        let cols = tensor.shape[1];
        let first_element = first_row
            .checked_mul(cols)
            .ok_or(SafeTensorError::OutOfBounds)?;
        let element_count = row_count
            .checked_mul(cols)
            .ok_or(SafeTensorError::OutOfBounds)?;
        self.read_f32_elements(name, first_element, element_count)
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let result = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let value = (fraction as f32) * 2.0_f32.powi(-24);
            return if sign == 0 { value } else { -value };
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | (u32::from(exponent + (127 - 15)) << 23) | (fraction << 13),
    };
    f32::from_bits(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hpsi_format::MemorySource;
    #[test]
    fn copies_raw_payload_in_bounded_chunks_without_dtype_conversion() {
        let header = br#"{"raw":{"dtype":"I16","shape":[3],"data_offsets":[0,6]}}"#;
        let payload = [0x34_u8, 0x12, 0xcd, 0xab, 0x01, 0x02];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&payload);
        let file = SafeTensorFile::open(
            Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096
        ).unwrap();
        let mut copied = Vec::new();
        assert_eq!(file.copy_raw_payload("raw", &mut copied, 2).unwrap(), 6);
        assert_eq!(copied, payload);
        assert!(matches!(
            file.copy_raw_payload("raw", &mut Vec::new(), 0),
            Err(SafeTensorError::InvalidTensor)
        ));
    }

    #[test]
    fn converts_f16_and_bf16_row_windows_to_f32() {
        for (dtype, one, minus_two) in [
            ("F16", 0x3c00_u16, 0xc000_u16),
            ("BF16", 0x3f80_u16, 0xc000_u16),
        ] {
            let header = format!(
                r#"{{"w":{{"dtype":"{}","shape":[1,2],"data_offsets":[0,4]}}}}"#,
                dtype
            );
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&one.to_le_bytes());
            bytes.extend_from_slice(&minus_two.to_le_bytes());
            let file =
                SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096)
                    .unwrap();
            let values = file.read_f32_rows("w", 0, 1).unwrap();
            assert_eq!(values, vec![1.0, -2.0]);
        }
    }

    #[test]
    fn reads_only_requested_f32_rows() {
        let header = br#"{"w":{"dtype":"F32","shape":[3,2],"data_offsets":[0,24]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        for v in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let file =
            SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096)
                .unwrap();
        assert_eq!(file.read_f32_rows("w", 1, 1).unwrap(), vec![3.0, 4.0]);
    }

    #[test]
    fn safetensors_element_reader_feeds_streaming_vq_encoder() {
        use crate::{VqConfig, VqMatrix};

        let rows = 8_u64;
        let cols = 16_u64;
        let values: Vec<f32> = (0..(rows * cols))
            .map(|index| ((index as f32) * 0.071).sin() * 0.5)
            .collect();
        let header = format!(
            r#"{{"weight":{{"dtype":"F32","shape":[{},{}],"data_offsets":[0,{}]}}}}"#,
            rows,
            cols,
            values.len() * 4
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for value in &values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let source =
            SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096)
                .unwrap();
        let config = VqConfig {
            codebook_count: 2,
            sample_limit: 32,
            ..VqConfig::default()
        };
        let expected = VqMatrix::encode(&values, rows as usize, cols as usize, config).unwrap();
        let actual =
            VqMatrix::encode_from_reader(rows as usize, cols as usize, config, |first, count| {
                source
                    .read_f32_elements("weight", first, count)
                    .map_err(|_| crate::VqError("Safetensors window read failed"))
            })
            .unwrap();
        assert_eq!(actual.codebook(), expected.codebook());
        assert_eq!(actual.packed_indices(), expected.packed_indices());
    }

    #[test]
    fn safetensors_tensor_imports_directly_to_hpsi() {
        use hpsi_format::{ModelReader, ReaderLimits};
        use std::io::Cursor;

        let rows = 8_u64;
        let cols = 16_u64;
        let values: Vec<f32> = (0..rows * cols)
            .map(|index| ((index as f32) * 0.037).cos())
            .collect();
        let header = format!(
            r#"{{"weight":{{"dtype":"F32","shape":[{},{}],"data_offsets":[0,{}]}}}}"#,
            rows,
            cols,
            values.len() * 4
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for value in &values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let source =
            SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096)
                .unwrap();
        let config = VqConfig {
            codebook_count: 2,
            sample_limit: 32,
            ..VqConfig::default()
        };
        let mut output = Cursor::new(Vec::new());
        let stats = source
            .encode_f32_tensor_to_hpsi("weight", &mut output, config, 2)
            .unwrap();
        assert_eq!(stats.tensor_count, 2);
        let reader = ModelReader::open(
            Arc::new(MemorySource::new(Arc::<[u8]>::from(output.into_inner()))),
            &ReaderLimits::default(),
        )
        .unwrap();
        reader.verify_all().unwrap();
        assert_eq!(
            reader.read_block(0, 0).unwrap().payload.len(),
            2 * 16 * 8 * 4
        );
    }

    #[test]
    fn reads_arbitrary_element_windows_and_rejects_non_finite_values() {
        let header = br#"{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        for value in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let file =
            SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096)
                .unwrap();
        assert_eq!(
            file.read_f32_elements("w", 2, 3).unwrap(),
            vec![3.0, 4.0, 5.0]
        );
        assert!(matches!(
            file.read_f32_elements("w", 5, 2),
            Err(SafeTensorError::OutOfBounds)
        ));

        let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&f32::INFINITY.to_le_bytes());
        let file =
            SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096)
                .unwrap();
        assert!(matches!(
            file.read_f32_elements("w", 0, 1),
            Err(SafeTensorError::InvalidTensor)
        ));
    }

    #[test]
    fn rejects_gaps_and_trailing_payload_bytes() {
        for (offsets, payload) in [("[1,2]", vec![7_u8, 8]), ("[0,1]", vec![7_u8, 8])] {
            let header =
                format!(r#"{{"w":{{"dtype":"U8","shape":[1],"data_offsets":{offsets}}}}}"#);
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&payload);
            let file =
                SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096);
            assert!(matches!(file, Err(SafeTensorError::InvalidTensor)));
        }
    }

    #[test]
    fn rejects_overlapping_tensor_ranges() {
        let header=br#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[1, 2, 3]);
        let result =
            SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096);
        assert!(matches!(result, Err(SafeTensorError::InvalidTensor)));
    }
}
