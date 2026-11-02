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
        let object = value.as_object().ok_or(FormatError::InvalidMetadata(
            "metadata root must be an object",
        ))?;

        for key in object.keys() {
            if !matches!(
                key.as_str(),
                "format" | "metadata_version" | "tensor_names" | "model" | "producer" | "user"
            ) {
                return Err(FormatError::InvalidMetadata(
                    "unknown top-level metadata property",
                ));
            }
        }

        for key in ["model", "producer", "user"] {
            if object.get(key).is_some_and(|value| !value.is_object()) {
                return Err(FormatError::InvalidMetadata(
                    "model, producer, and user metadata must be objects",
                ));
            }
        }

        if object.get("format").and_then(Value::as_str) != Some("hpsi") {
            return Err(FormatError::InvalidMetadata("format must be hpsi"));
        }
        if object.get("metadata_version").and_then(Value::as_u64) != Some(1) {
            return Err(FormatError::InvalidMetadata("metadata_version must be 1"));
        }

        let names = object.get("tensor_names").and_then(Value::as_array).ok_or(
            FormatError::InvalidMetadata("tensor_names must be an array"),
        )?;
        if names.len() > limits.max_tensors as usize {
            return Err(FormatError::LimitExceeded("metadata tensor name count"));
        }

        // Borrow names from the parsed JSON tree for duplicate detection.
        // This avoids allocating a second owned String for every tensor name.
        let mut seen: HashSet<&str> = HashSet::new();
        seen.try_reserve(names.len())
            .map_err(|_| FormatError::LimitExceeded("tensor name index allocation failed"))?;
        let mut tensor_names = Vec::new();
        tensor_names
            .try_reserve_exact(names.len())
            .map_err(|_| FormatError::LimitExceeded("tensor name allocation failed"))?;
        for item in names {
            let name = item
                .as_str()
                .ok_or(FormatError::InvalidMetadata("tensor name must be a string"))?;
            if name.is_empty() || name.contains('\0') || name.len() > limits.max_tensor_name_bytes {
                return Err(FormatError::InvalidMetadata(
                    "tensor name is empty, contains NUL, or exceeds limit",
                ));
            }
            if !seen.insert(name) {
                return Err(FormatError::InvalidMetadata("duplicate tensor name"));
            }
            tensor_names.push(name.to_owned());
        }

        let mut pending = Vec::new();
        pending
            .try_reserve(1)
            .map_err(|_| FormatError::LimitExceeded("JSON traversal allocation"))?;
        pending.push((&value, 1_usize));
        let max_nodes = limits.max_json_container_items.saturating_mul(4).max(1);
        let mut visited_nodes = 0usize;
        while let Some((node, depth)) = pending.pop() {
            visited_nodes = visited_nodes
                .checked_add(1)
                .ok_or(FormatError::ArithmeticOverflow("JSON node count"))?;
            if visited_nodes > max_nodes {
                return Err(FormatError::LimitExceeded("JSON total node count"));
            }
            if depth > limits.max_json_depth {
                return Err(FormatError::LimitExceeded("JSON nesting depth"));
            }
            match node {
                Value::Array(items) => {
                    if items.len() > limits.max_json_container_items {
                        return Err(FormatError::LimitExceeded("JSON array item count"));
                    }
                    pending
                        .try_reserve(items.len())
                        .map_err(|_| FormatError::LimitExceeded("JSON traversal allocation"))?;
                    pending.extend(items.iter().map(|child| (child, depth + 1)));
                }
                Value::Object(items) => {
                    if items.len() > limits.max_json_container_items {
                        return Err(FormatError::LimitExceeded("JSON object member count"));
                    }
                    pending
                        .try_reserve(items.len())
                        .map_err(|_| FormatError::LimitExceeded("JSON traversal allocation"))?;
                    pending.extend(items.values().map(|child| (child, depth + 1)));
                }
                Value::String(text) if text.len() > limits.max_json_string_bytes => {
                    return Err(FormatError::LimitExceeded("JSON string length"));
                }
                _ => {}
            }
        }

        Ok(Self {
            tensor_names,
            value,
        })
    }
}

#[cfg(test)]
mod resource_limit_tests {
    use super::*;

    #[test]
    fn descriptive_metadata_fields_must_be_objects() {
        for (key, value) in [
            ("model", serde_json::json!("model-name")),
            ("producer", serde_json::json!(["producer"])),
            ("user", serde_json::json!(42)),
        ] {
            let bytes = serde_json::to_vec(&serde_json::json!({
                "format": "hpsi",
                "metadata_version": 1,
                "tensor_names": [],
                key: value
            }))
            .unwrap();
            assert!(matches!(
                Metadata::parse(&bytes, &ReaderLimits::default()),
                Err(FormatError::InvalidMetadata(
                    "model, producer, and user metadata must be objects"
                ))
            ));
        }
    }

    #[test]
    fn rejects_metadata_with_excessive_total_json_nodes() {
        let values: Vec<u32> = (0..900).collect();
        let groups = vec![
            values.clone(),
            values.clone(),
            values.clone(),
            values.clone(),
            values,
        ];
        let bytes = serde_json::to_vec(&serde_json::json!({
            "format": "hpsi",
            "metadata_version": 1,
            "tensor_names": [],
            "producer": { "groups": groups }
        }))
        .unwrap();
        let limits = ReaderLimits {
            max_json_container_items: 1000,
            ..ReaderLimits::default()
        };
        assert!(matches!(
            Metadata::parse(&bytes, &limits),
            Err(FormatError::LimitExceeded("JSON total node count"))
        ));
    }
}
