use crate::{VqConfig, VqError, VqMatrix};
use hpsi_format::{RandomAccessSource, WriterStats};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use std::collections::HashSet;
use std::fmt;
use std::io::{Seek, Write};
use std::sync::Arc;

struct NoDuplicateJson;

impl<'de> Deserialize<'de> for NoDuplicateJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct JsonVisitor;

        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = NoDuplicateJson;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON value without duplicate object keys")
            }

            fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(NoDuplicateJson)
            }

            fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(NoDuplicateJson)
            }

            fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(NoDuplicateJson)
            }

            fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(NoDuplicateJson)
            }

            fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(NoDuplicateJson)
            }

            fn visit_string<E>(self, _value: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(NoDuplicateJson)
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(NoDuplicateJson)
            }

            fn visit_none<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(NoDuplicateJson)
            }

            fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: Deserializer<'de>,
            {
                NoDuplicateJson::deserialize(deserializer)
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                while sequence.next_element::<NoDuplicateJson>()?.is_some() {}
                Ok(NoDuplicateJson)
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut keys = HashSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    keys.try_reserve(1).map_err(de::Error::custom)?;
                    if !keys.insert(key) {
                        return Err(de::Error::custom("duplicate JSON object key"));
                    }
                    map.next_value::<NoDuplicateJson>()?;
                }
                Ok(NoDuplicateJson)
            }
        }

        deserializer.deserialize_any(JsonVisitor)
    }
}

fn reject_duplicate_json_keys(bytes: &[u8]) -> Result<(), SafeTensorError> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    NoDuplicateJson::deserialize(&mut deserializer).map_err(|_| SafeTensorError::InvalidHeader)?;
    deserializer
        .end()
        .map_err(|_| SafeTensorError::InvalidHeader)
}

/// Bounded, random-access reader for the public Safetensors container layout.
/// The JSON header is loaded; tensor payloads remain on the underlying source.
pub struct SafeTensorFile {
    source: Arc<dyn RandomAccessSource>,
    data_start: u64,
    tensors: Vec<SafeTensorInfo>,
    tensor_name_order: Vec<usize>,
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

fn clone_header_text(value: &str) -> Result<String, SafeTensorError> {
    let mut cloned = String::new();
    cloned
        .try_reserve_exact(value.len())
        .map_err(|_| SafeTensorError::Allocation)?;
    cloned.push_str(value);
    Ok(cloned)
}

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
        let header_len_u64 =
            u64::try_from(header_len).map_err(|_| SafeTensorError::HeaderTooLarge)?;
        let data_start = 8_u64
            .checked_add(header_len_u64)
            .ok_or(SafeTensorError::InvalidHeader)?;
        let file_len = source.len().map_err(|_| SafeTensorError::Io)?;
        if data_start > file_len {
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
        reject_duplicate_json_keys(&bytes)?;
        let root: Value =
            serde_json::from_slice(&bytes).map_err(|_| SafeTensorError::InvalidHeader)?;
        let object = root.as_object().ok_or(SafeTensorError::InvalidHeader)?;
        if let Some(metadata) = object.get("__metadata__") {
            let metadata = metadata.as_object().ok_or(SafeTensorError::InvalidHeader)?;
            if metadata.values().any(|value| !value.is_string()) {
                return Err(SafeTensorError::InvalidHeader);
            }
        }
        let data_len = file_len - data_start;
        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(object.len())
            .map_err(|_| SafeTensorError::Allocation)?;
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            let entry = value.as_object().ok_or(SafeTensorError::InvalidTensor)?;
            let dtype = clone_header_text(
                entry
                    .get("dtype")
                    .and_then(Value::as_str)
                    .ok_or(SafeTensorError::InvalidTensor)?,
            )?;
            let shape_values = entry
                .get("shape")
                .and_then(Value::as_array)
                .ok_or(SafeTensorError::InvalidTensor)?;
            let mut shape = Vec::new();
            shape
                .try_reserve_exact(shape_values.len())
                .map_err(|_| SafeTensorError::Allocation)?;
            for dimension in shape_values {
                shape.push(dimension.as_u64().ok_or(SafeTensorError::InvalidTensor)?);
            }
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
            // Safetensors defines element widths in bits. In particular, F4
            // and F6 are sub-byte formats and must be checked for byte alignment.
            let bits_per_element = match dtype.as_str() {
                "F4" => 4_u64,
                "F6_E2M3" | "F6_E3M2" => 6,
                "BOOL"
                | "U8"
                | "I8"
                | "F8_E5M2"
                | "F8_E4M3"
                | "F8_E8M0"
                | "F8_E4M3FNUZ"
                | "F8_E5M2FNUZ" => 8,
                "I16" | "U16" | "F16" | "BF16" => 16,
                "I32" | "U32" | "F32" => 32,
                "C64" | "F64" | "I64" | "U64" => 64,
                _ => return Err(SafeTensorError::UnsupportedDtype),
            };
            let tensor_bits = elements
                .checked_mul(bits_per_element)
                .ok_or(SafeTensorError::InvalidTensor)?;
            if tensor_bits % 8 != 0 || tensor_bits / 8 != end - start {
                return Err(SafeTensorError::InvalidTensor);
            }
            tensors.push(SafeTensorInfo {
                name: clone_header_text(name)?,
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
        let mut tensor_name_order = Vec::new();
        tensor_name_order
            .try_reserve_exact(tensors.len())
            .map_err(|_| SafeTensorError::Allocation)?;
        tensor_name_order.extend(0..tensors.len());
        tensor_name_order
            .sort_unstable_by(|&left, &right| tensors[left].name.cmp(&tensors[right].name));
        Ok(Self {
            source,
            data_start,
            tensors,
            tensor_name_order,
        })
    }
    fn tensor_by_name(&self, name: &str) -> Option<&SafeTensorInfo> {
        let position = self
            .tensor_name_order
            .binary_search_by(|&index| self.tensors[index].name.as_str().cmp(name))
            .ok()?;
        self.tensor_name_order
            .get(position)
            .and_then(|&index| self.tensors.get(index))
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
            .tensor_by_name(tensor_name)
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
        // Keep memory use bounded even when a caller accidentally requests a
        // very large chunk. Smaller caller-selected chunks remain respected.
        let chunk_bytes = chunk_bytes.min(64 * 1024);
        let tensor = self
            .tensor_by_name(name)
            .ok_or(SafeTensorError::InvalidTensor)?;
        let length = tensor
            .data_end
            .checked_sub(tensor.data_start)
            .ok_or(SafeTensorError::InvalidTensor)?;
        let absolute_start = self
            .data_start
            .checked_add(tensor.data_start)
            .ok_or(SafeTensorError::OutOfBounds)?;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(chunk_bytes)
            .map_err(|_| SafeTensorError::Allocation)?;
        buffer.resize(chunk_bytes, 0);
        let mut copied = 0_u64;
        while copied < length {
            let count = (length - copied).min(chunk_bytes as u64) as usize;
            let offset = absolute_start
                .checked_add(copied)
                .ok_or(SafeTensorError::OutOfBounds)?;
            self.source
                .read_exact_at(offset, &mut buffer[..count])
                .map_err(|_| SafeTensorError::Io)?;
            output
                .write_all(&buffer[..count])
                .map_err(|_| SafeTensorError::Io)?;
            copied = copied
                .checked_add(count as u64)
                .ok_or(SafeTensorError::OutOfBounds)?;
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
        let count = usize::try_from(element_count).map_err(|_| SafeTensorError::Allocation)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| SafeTensorError::Allocation)?;
        values.resize(count, 0.0_f32);
        self.read_f32_elements_into(name, first_element, &mut values)?;
        Ok(values)
    }

    /// Decode a Safetensors window into caller-owned storage. Reusing this
    /// slice across calls avoids allocating a new F32 vector for every row.
    pub fn read_f32_elements_into(
        &self,
        name: &str,
        first_element: u64,
        values: &mut [f32],
    ) -> Result<(), SafeTensorError> {
        let element_count =
            u64::try_from(values.len()).map_err(|_| SafeTensorError::OutOfBounds)?;
        let tensor = self
            .tensor_by_name(name)
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
        usize::try_from(byte_count).map_err(|_| SafeTensorError::Allocation)?;
        let relative_offset = first_element
            .checked_mul(width)
            .ok_or(SafeTensorError::OutOfBounds)?;
        let absolute_offset = self
            .data_start
            .checked_add(tensor.data_start)
            .and_then(|offset| offset.checked_add(relative_offset))
            .ok_or(SafeTensorError::OutOfBounds)?;
        // Decode directly into the final f32 allocation. The previous
        // implementation allocated a second full-sized byte Vec and then a
        // third iterator-collection allocation for the converted values.
        // A fixed-size transfer buffer keeps temporary byte memory bounded
        // even when a caller requests a very large row.
        let element_count_usize = values.len();
        let element_width = width as usize;
        let elements_per_chunk = (64 * 1024) / element_width;
        let mut bytes = [0_u8; 64 * 1024];
        let mut completed = 0usize;
        while completed < element_count_usize {
            let count = (element_count_usize - completed).min(elements_per_chunk);
            let byte_len = count * element_width;
            let byte_offset = (completed as u64)
                .checked_mul(width)
                .and_then(|offset| absolute_offset.checked_add(offset))
                .ok_or(SafeTensorError::OutOfBounds)?;
            self.source
                .read_exact_at(byte_offset, &mut bytes[..byte_len])
                .map_err(|_| SafeTensorError::Io)?;
            for (slot, chunk) in values[completed..completed + count]
                .iter_mut()
                .zip(bytes[..byte_len].chunks_exact(element_width))
            {
                *slot = match tensor.dtype.as_str() {
                    "F32" => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                    "BF16" => {
                        f32::from_bits(u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16)
                    }
                    "F16" => f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])),
                    _ => return Err(SafeTensorError::UnsupportedDtype),
                };
            }
            completed += count;
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(SafeTensorError::InvalidTensor);
        }
        Ok(())
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
            .tensor_by_name(name)
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
    fn rejects_duplicate_json_keys_at_every_object_depth() {
        for header in [
            r#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            r#"{"weight":{"dtype":"U8","dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#,
            r#"{"__metadata__":{"format":"one","format":"two"}}"#,
            r#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"we\u0069ght":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        ] {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            assert!(matches!(
                SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096,),
                Err(SafeTensorError::InvalidHeader)
            ));
        }
    }

    #[test]
    fn accepts_string_metadata_and_rejects_non_object_or_non_string_values() {
        for header in [
            r#"{"__metadata__":{"format":"weights","version":"1"}}"#,
            r#"{"__metadata__":null}"#,
            r#"{"__metadata__":[]}"#,
            r#"{"__metadata__":{"version":1}}"#,
            r#"{"__metadata__":{"nested":{"key":"value"}}}"#,
        ] {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            let result =
                SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096);
            if header.contains(r#""format":"weights""#) {
                assert!(result.is_ok());
            } else {
                assert!(matches!(result, Err(SafeTensorError::InvalidHeader)));
            }
        }
    }

    #[test]
    fn accepts_all_reference_safetensors_dtypes_for_raw_preservation() {
        let dtypes = [
            ("BOOL", 8_u64),
            ("F4", 4),
            ("F6_E2M3", 6),
            ("F6_E3M2", 6),
            ("U8", 8),
            ("I8", 8),
            ("F8_E5M2", 8),
            ("F8_E4M3", 8),
            ("F8_E8M0", 8),
            ("F8_E4M3FNUZ", 8),
            ("F8_E5M2FNUZ", 8),
            ("I16", 16),
            ("U16", 16),
            ("F16", 16),
            ("BF16", 16),
            ("I32", 32),
            ("U32", 32),
            ("F32", 32),
            ("C64", 64),
            ("F64", 64),
            ("I64", 64),
            ("U64", 64),
        ];

        for (dtype, bits_per_element) in dtypes {
            let bit_length = 8_u64 * bits_per_element;
            let payload_len = usize::try_from(bit_length / 8).unwrap();
            let header = format!(
                r#"{{"tensor":{{"dtype":"{dtype}","shape":[8],"data_offsets":[0,{payload_len}]}}}}"#
            );
            let payload: Vec<u8> = (0..payload_len).map(|index| index as u8).collect();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&payload);

            let file = SafeTensorFile::open(
                Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))),
                4096,
            )
            .unwrap_or_else(|error| panic!("reference dtype {dtype} was rejected: {error}"));
            let mut copied = Vec::new();
            assert_eq!(file.copy_raw_payload("tensor", &mut copied, 3).unwrap(), payload_len as u64);
            assert_eq!(copied, payload, "raw payload changed for dtype {dtype}");
        }
    }

    #[test]
    fn rejects_misaligned_sub_byte_safetensors_payloads() {
        for dtype in ["F4", "F6_E2M3", "F6_E3M2"] {
            let header = format!(
                r#"{{"tensor":{{"dtype":"{dtype}","shape":[1],"data_offsets":[0,1]}}}}"#
            );
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.push(0);

            assert!(matches!(
                SafeTensorFile::open(
                    Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))),
                    4096,
                ),
                Err(SafeTensorError::InvalidTensor)
            ), "misaligned dtype {dtype} should be rejected");
        }
    }

    #[test]
    fn name_index_finds_tensors_without_changing_payload_order() {
        let header = br#"{"z":{"dtype":"U8","shape":[1],"data_offsets":[1,2]},"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[10, 20]);
        let file =
            SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096)
                .unwrap();

        assert_eq!(file.tensor_by_name("a").unwrap().data_start, 0);
        assert_eq!(file.tensor_by_name("z").unwrap().data_start, 1);
        assert!(file.tensor_by_name("missing").is_none());
        assert_eq!(file.tensors()[0].name, "a");
        assert_eq!(file.tensors()[1].name, "z");
    }

    #[test]
    fn copies_raw_payload_in_bounded_chunks_without_dtype_conversion() {
        let header = br#"{"raw":{"dtype":"I16","shape":[3],"data_offsets":[0,6]}}"#;
        let payload = [0x34_u8, 0x12, 0xcd, 0xab, 0x01, 0x02];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&payload);
        let file =
            SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096)
                .unwrap();
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
            let mut reused = [0.0_f32; 2];
            file.read_f32_elements_into("w", 0, &mut reused).unwrap();
            assert_eq!(reused.as_slice(), values.as_slice());
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