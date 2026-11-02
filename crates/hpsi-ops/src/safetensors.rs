use hpsi_format::RandomAccessSource;
use serde_json::Value;
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
        for pair in tensors.windows(2) {
            if pair[0].data_end > pair[1].data_start {
                return Err(SafeTensorError::InvalidTensor);
            }
        }
        Ok(Self {
            source,
            data_start,
            tensors,
        })
    }
    pub fn tensors(&self) -> &[SafeTensorInfo] {
        &self.tensors
    }
    /// Read a row window of a rank-2 F32 tensor. Other dtypes are rejected, never silently cast.
    pub fn read_f32_rows(
        &self,
        name: &str,
        first_row: u64,
        row_count: u64,
    ) -> Result<Vec<f32>, SafeTensorError> {
        let t = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or(SafeTensorError::InvalidTensor)?;
        if t.shape.len() != 2 {
            return Err(SafeTensorError::UnsupportedDtype);
        }
        let width = match t.dtype.as_str() {
            "F32" => 4_u64,
            "F16" | "BF16" => 2_u64,
            _ => return Err(SafeTensorError::UnsupportedDtype),
        };
        let rows = t.shape[0];
        let cols = t.shape[1];
        let end_row = first_row
            .checked_add(row_count)
            .ok_or(SafeTensorError::OutOfBounds)?;
        if end_row > rows {
            return Err(SafeTensorError::OutOfBounds);
        }
        let elements = row_count
            .checked_mul(cols)
            .ok_or(SafeTensorError::OutOfBounds)?;
        let bytes_len = usize::try_from(
            elements
                .checked_mul(width)
                .ok_or(SafeTensorError::OutOfBounds)?,
        )
        .map_err(|_| SafeTensorError::Allocation)?;
        let relative = first_row
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(width))
            .ok_or(SafeTensorError::OutOfBounds)?;
        let offset = self
            .data_start
            .checked_add(t.data_start)
            .and_then(|v| v.checked_add(relative))
            .ok_or(SafeTensorError::OutOfBounds)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(bytes_len)
            .map_err(|_| SafeTensorError::Allocation)?;
        bytes.resize(bytes_len, 0);
        self.source
            .read_exact_at(offset, &mut bytes)
            .map_err(|_| SafeTensorError::Io)?;
        let values = if width == 4 {
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        } else if t.dtype == "BF16" {
            bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect()
        } else {
            bytes
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect()
        };
        Ok(values)
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
