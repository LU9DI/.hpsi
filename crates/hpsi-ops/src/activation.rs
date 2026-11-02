use serde_json::Value;

/// Validated per-layer activation calibration emitted by the Python capture tool.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationCalibration {
    pub layer_name: String,
    pub sample_count: usize,
    pub features: usize,
    pub activation_second_moment: Vec<f32>,
    pub model_type: String,
    pub model_revision: Option<String>,
    pub dataset_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalibrationError(pub &'static str);

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for CalibrationError {}

impl ActivationCalibration {
    /// Validate provenance and numeric invariants, including values created
    /// directly through the public Rust struct rather than parsed from JSON.
    pub fn validate(&self) -> Result<(), CalibrationError> {
        if self.layer_name.trim().is_empty() {
            return Err(CalibrationError("layer_name must not be empty"));
        }
        if self.sample_count == 0 {
            return Err(CalibrationError("sample_count must be positive"));
        }
        if self.features == 0 || self.activation_second_moment.len() != self.features {
            return Err(CalibrationError("activation moment count must match positive feature count"));
        }
        if self.activation_second_moment.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(CalibrationError("activation moments must be finite and non-negative"));
        }
        if self.model_type.trim().is_empty() {
            return Err(CalibrationError("model_type must not be empty"));
        }
        if self.model_revision.as_deref().is_some_and(|revision| revision.trim().is_empty()) {
            return Err(CalibrationError("model revision must not be empty when present"));
        }
        if self.dataset_sha256.len() != 64
            || !self.dataset_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(CalibrationError("dataset sha256 must be 64 hexadecimal characters"));
        }
        Ok(())
    }

    /// Parse one versioned HPSI activation-calibration JSON document.
    pub fn from_json(input: &str) -> Result<Self, CalibrationError> {
        let root: Value = serde_json::from_str(input)
            .map_err(|_| CalibrationError("calibration JSON is malformed"))?;
        let object = root
            .as_object()
            .ok_or(CalibrationError("calibration document must be an object"))?;
        if required_string(object, "format")? != "hpsi-activation-calibration" {
            return Err(CalibrationError("unsupported calibration format"));
        }
        if object.get("version").and_then(Value::as_u64) != Some(1) {
            return Err(CalibrationError("unsupported calibration version"));
        }
        if required_string(object, "statistic")? != "diagonal_second_moment" {
            return Err(CalibrationError("unsupported activation statistic"));
        }
        let layer_name = required_string(object, "layer_name")?.to_owned();
        let sample_count = object
            .get("sample_count")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .filter(|v| *v > 0)
            .ok_or(CalibrationError("sample_count must be positive"))?;
        let shape = object
            .get("shape")
            .and_then(Value::as_array)
            .ok_or(CalibrationError("shape must be [sample_count, features]"))?;
        if shape.len() != 2
            || shape[0].as_u64().and_then(|v| usize::try_from(v).ok()) != Some(sample_count)
        {
            return Err(CalibrationError("shape does not match sample_count"));
        }
        let features = shape[1]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .filter(|v| *v > 0)
            .ok_or(CalibrationError("feature dimension must be positive"))?;
        let values = object
            .get("activation_second_moment")
            .and_then(Value::as_array)
            .ok_or(CalibrationError(
                "activation_second_moment must be an array",
            ))?;
        if values.len() != features {
            return Err(CalibrationError(
                "moment count does not match feature dimension",
            ));
        }
        let mut activation_second_moment = Vec::new();
        activation_second_moment
            .try_reserve_exact(features)
            .map_err(|_| CalibrationError("could not allocate activation calibration values"))?;
        for value in values {
            let number = value
                .as_f64()
                .ok_or(CalibrationError("activation moments must be numeric"))?;
            if !number.is_finite() || number < 0.0 || number > f64::from(f32::MAX) {
                return Err(CalibrationError(
                    "activation moments must be finite, non-negative f32 values",
                ));
            }
            activation_second_moment.push(number as f32);
        }
        let model = object
            .get("model")
            .and_then(Value::as_object)
            .ok_or(CalibrationError("model provenance is missing"))?;
        let model_type = required_string(model, "model_type")?.to_owned();
        let model_revision = model
            .get("revision")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let dataset = object
            .get("dataset")
            .and_then(Value::as_object)
            .ok_or(CalibrationError("dataset provenance is missing"))?;
        let dataset_sha256 = required_string(dataset, "sha256")?.to_owned();
        if dataset_sha256.len() != 64 || !dataset_sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(CalibrationError(
                "dataset sha256 must be 64 hexadecimal characters",
            ));
        }
        Ok(Self {
            layer_name,
            sample_count,
            features,
            activation_second_moment,
            model_type,
            model_revision,
            dataset_sha256,
        })
    }
}

fn required_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &'static str,
) -> Result<&'a str, CalibrationError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|v| !v.trim().is_empty())
        .ok_or(CalibrationError(
            "required string field is missing or empty",
        ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document() -> String {
        format!(
            r#"{{"format":"hpsi-activation-calibration","version":1,"model":{{"model_type":"llama","revision":"abc"}},"dataset":{{"sha256":"{}"}},"layer_name":"model.layers.0.mlp.up_proj","shape":[32,8],"sample_count":32,"statistic":"diagonal_second_moment","activation_second_moment":[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]}}"#,
            "a".repeat(64)
        )
    }

    #[test]
    fn parses_valid_calibration_and_preserves_zero_moments() {
        let calibration = ActivationCalibration::from_json(&document()).unwrap();
        assert_eq!(calibration.layer_name, "model.layers.0.mlp.up_proj");
        assert_eq!(calibration.sample_count, 32);
        assert_eq!(calibration.features, 8);
        assert_eq!(calibration.activation_second_moment[0], 0.0);
        assert_eq!(calibration.model_type, "llama");
    }

    #[test]
    fn rejects_inconsistent_shapes_and_negative_moments() {
        assert!(ActivationCalibration::from_json(&document().replace("[32,8]", "[31,8]")).is_err());
        assert!(
            ActivationCalibration::from_json(&document().replace("[0.0,0.1", "[-0.1,0.1")).is_err()
        );
    }

    #[test]
    fn rejects_unsupported_version_and_invalid_dataset_hash() {
        assert!(
            ActivationCalibration::from_json(&document().replace("\"version\":1", "\"version\":2"))
                .is_err()
        );
        assert!(
            ActivationCalibration::from_json(&document().replace(&"a".repeat(64), "bad")).is_err()
        );
    }
}
