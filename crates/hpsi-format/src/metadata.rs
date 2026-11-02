use std::collections::HashSet;

use serde_json::Value;

use crate::{FormatError, ReaderLimits, Result};

#[derive(Debug, Clone)]
pub struct Metadata {
    pub tensor_names: Vec<String>,
    pub value: Value,
}

impl Metadata {
    pub fn parse(bytes: &[u8], limits: &ReaderLimits) -> Result<Self> {
        if bytes.len() as u64 > limits.max_metadata_bytes {
            return Err(FormatError::LimitExceeded("metadata bytes"));
        }
        let value: Value = serde_json::from_slice(bytes)
            .map_err(|_| FormatError::InvalidMetadata("metadata is not valid UTF-8 JSON"))?;
        let object = value.as_object()
            .ok_or(FormatError::InvalidMetadata("metadata root must be an object"))?;

        for key in object.keys() {
            if !matches!(key.as_str(), "format" | "metadata_version" | "tensor_names" | "model" | "producer" | "user") {
                return Err(FormatError::InvalidMetadata("unknown top-level metadata property"));
            }
        }

        if object.get("format").and_then(Value::as_str) != Some("hpsi") {
            return Err(FormatError::InvalidMetadata("format must be hpsi"));
        }
        if object.get("metadata_version").and_then(Value::as_u64) != Some(1) {
            return Err(FormatError::InvalidMetadata("metadata_version must be 1"));
        }

        let names = object.get("tensor_names").and_then(Value::as_array)
            .ok_or(FormatError::InvalidMetadata("tensor_names must be an array"))?;
        if names.len() > limits.max_tensors as usize {
            return Err(FormatError::LimitExceeded("metadata tensor name count"));
        }

        let mut seen = HashSet::new();
        seen.try_reserve(names.len())
            .map_err(|_| FormatError::LimitExceeded("tensor name index allocation failed"))?;
        let mut tensor_names = Vec::new();
        tensor_names.try_reserve_exact(names.len())
            .map_err(|_| FormatError::LimitExceeded("tensor name allocation failed"))?;
        for item in names {
            let name = item.as_str()
                .ok_or(FormatError::InvalidMetadata("tensor name must be a string"))?;
            if name.is_empty() || name.contains('\0') || name.len() > limits.max_tensor_name_bytes {
                return Err(FormatError::InvalidMetadata("tensor name is empty, contains NUL, or exceeds limit"));
            }
            if !seen.insert(name.to_owned()) {
                return Err(FormatError::InvalidMetadata("duplicate tensor name"));
            }
            tensor_names.push(name.to_owned());
        }

        let mut pending = vec![(&value, 1_usize)];
        while let Some((node, depth)) = pending.pop() {
            if depth > limits.max_json_depth {
                return Err(FormatError::LimitExceeded("JSON nesting depth"));
            }
            match node {
                Value::Array(items) => {
                    if items.len() > limits.max_json_container_items {
                        return Err(FormatError::LimitExceeded("JSON array item count"));
                    }
                    pending.extend(items.iter().map(|child| (child, depth + 1)));
                }
                Value::Object(items) => {
                    if items.len() > limits.max_json_container_items {
                        return Err(FormatError::LimitExceeded("JSON object member count"));
                    }
                    pending.extend(items.values().map(|child| (child, depth + 1)));
                }
                Value::String(text) if text.len() > limits.max_json_string_bytes => {
                    return Err(FormatError::LimitExceeded("JSON string length"));
                }
                _ => {}
            }
        }

        Ok(Self { tensor_names, value })
    }
}
