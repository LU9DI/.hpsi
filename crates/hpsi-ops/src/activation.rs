use serde_json::Value;

/// Orientation of a rank-2 weight matrix in its source Safetensors container,
/// declared explicitly by the capture bundle (`module_type`), never inferred
/// from the tensor name or from its dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightOrientation {
    /// Source layout `[out_features, in_features]` (`torch.nn.Linear`).
    LinearOutIn,
    /// Source layout `[in_features, out_features]` (`transformers.pytorch_utils.Conv1D`).
    Conv1DInOut,
}

impl WeightOrientation {
    /// Stable persisted identifier used in calibration provenance and import metadata.
    pub fn as_str(self) -> &'static str {
        match self {
            WeightOrientation::LinearOutIn => "linear_out_in",
            WeightOrientation::Conv1DInOut => "conv1d_in_out",
        }
    }

    /// Parse the persisted identifier. Rejects anything else.
    pub fn parse(value: &str) -> Result<Self, CalibrationError> {
        match value {
            "linear_out_in" => Ok(WeightOrientation::LinearOutIn),
            "conv1d_in_out" => Ok(WeightOrientation::Conv1DInOut),
            _ => Err(CalibrationError(
                "weight orientation must be 'linear_out_in' or 'conv1d_in_out'",
            )),
        }
    }

    /// Logical input feature dimension for a source rank-2 shape, i.e. the
    /// dimension the VQ encoder groups along (GROUP_SIZE contiguous features).
    pub fn in_features(self, source_shape: &[u64]) -> Option<u64> {
        match (self, source_shape) {
            (WeightOrientation::LinearOutIn, [_, input]) => Some(*input),
            (WeightOrientation::Conv1DInOut, [input, _]) => Some(*input),
            _ => None,
        }
    }

    /// Logical output feature dimension for a source rank-2 shape.
    pub fn out_features(self, source_shape: &[u64]) -> Option<u64> {
        match (self, source_shape) {
            (WeightOrientation::LinearOutIn, [output, _]) => Some(*output),
            (WeightOrientation::Conv1DInOut, [_, output]) => Some(*output),
            _ => None,
        }
    }

    /// The `[out_features, in_features]` shape the executor consumes.
    pub fn logical_shape(self, source_shape: &[u64]) -> Option<[u64; 2]> {
        Some([
            self.out_features(source_shape)?,
            self.in_features(source_shape)?,
        ])
    }
}

/// How the orientation of a planned tensor was decided. Persisted so a
/// consumer can tell an explicit capture declaration from the legacy default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightOrientationSource {
    /// Declared by the capture bundle (`module_type`).
    Calibration,
    /// Legacy bundle without orientation metadata; defaults to `LinearOutIn`.
    LegacyDefault,
}

impl WeightOrientationSource {
    pub fn as_str(self) -> &'static str {
        match self {
            WeightOrientationSource::Calibration => "calibration",
            WeightOrientationSource::LegacyDefault => "legacy_default",
        }
    }

    pub fn parse(value: &str) -> Result<Self, CalibrationError> {
        match value {
            "calibration" => Ok(WeightOrientationSource::Calibration),
            "legacy_default" => Ok(WeightOrientationSource::LegacyDefault),
            _ => Err(CalibrationError(
                "weight orientation source must be 'calibration' or 'legacy_default'",
            )),
        }
    }
}

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
    /// Explicit projection kind captured from the model ("linear" or "conv1d").
    /// Absent in legacy bundles without orientation metadata.
    pub module_type: Option<String>,
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
            return Err(CalibrationError(
                "activation moment count must match positive feature count",
            ));
        }
        if self
            .activation_second_moment
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(CalibrationError(
                "activation moments must be finite and non-negative",
            ));
        }
        if self.model_type.trim().is_empty() {
            return Err(CalibrationError("model_type must not be empty"));
        }
        if self
            .model_revision
            .as_deref()
            .is_some_and(|revision| revision.trim().is_empty())
        {
            return Err(CalibrationError(
                "model revision must not be empty when present",
            ));
        }
        if let Some(module_type) = self.module_type.as_deref() {
            if !matches!(module_type, "linear" | "conv1d") {
                return Err(CalibrationError(
                    "module_type must be 'linear' or 'conv1d' when present",
                ));
            }
        }
        if self.dataset_sha256.len() != 64
            || !self
                .dataset_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(CalibrationError(
                "dataset sha256 must be 64 hexadecimal characters",
            ));
        }
        // Orientation metadata, when present, must be unambiguous. The legacy
        // absence is handled by `weight_orientation` (LinearOutIn default).
        self.weight_orientation()?;
        Ok(())
    }

    /// Expected logical orientation of the layer's weight matrix.
    /// Bundles without orientation metadata predate the capture fix and are
    /// assumed Linear (`[out, in]`); that is the only safe legacy default.
    pub fn weight_orientation(&self) -> Result<WeightOrientation, CalibrationError> {
        match self.module_type.as_deref() {
            None => Ok(WeightOrientation::LinearOutIn),
            Some("linear") => Ok(WeightOrientation::LinearOutIn),
            Some("conv1d") => Ok(WeightOrientation::Conv1DInOut),
            Some(_) => Err(CalibrationError(
                "module_type must be 'linear' or 'conv1d' when present",
            )),
        }
    }

    /// How the orientation was declared (explicit capture vs legacy default).
    pub fn orientation_source(&self) -> WeightOrientationSource {
        match self.module_type {
            Some(_) => WeightOrientationSource::Calibration,
            None => WeightOrientationSource::LegacyDefault,
        }
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
        let model_revision = match model.get("revision") {
            Some(Value::String(revision)) => Some(revision.clone()),
            Some(Value::Null) | None => None,
            Some(_) => {
                return Err(CalibrationError(
                    "model revision must be a string or null when present",
                ));
            }
        };
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
        let module_type = match object.get("module_type") {
            None => None,
            Some(Value::String(value)) => {
                if !matches!(value.as_str(), "linear" | "conv1d") {
                    return Err(CalibrationError("module_type must be 'linear' or 'conv1d'"));
                }
                Some(value.clone())
            }
            Some(_) => {
                return Err(CalibrationError(
                    "module_type must be a string when present",
                ));
            }
        };
        let calibration = Self {
            layer_name,
            sample_count,
            features,
            activation_second_moment,
            model_type,
            model_revision,
            dataset_sha256,
            module_type,
        };
        calibration.validate()?;
        Ok(calibration)
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
    fn rejects_empty_or_non_string_model_revision() {
        let valid = document();
        assert!(
            ActivationCalibration::from_json(
                &valid.replace(r#""revision":"abc""#, r#""revision":"  ""#)
            )
            .is_err()
        );
        assert!(
            ActivationCalibration::from_json(
                &valid.replace(r#""revision":"abc""#, r#""revision":42"#)
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_inconsistent_shapes_and_negative_moments() {
        assert!(ActivationCalibration::from_json(&document().replace("[32,8]", "[31,8]")).is_err());
        assert!(
            ActivationCalibration::from_json(&document().replace("[0.0,0.1", "[-0.1,0.1")).is_err()
        );
    }

    #[test]
    fn validates_public_struct_provenance_without_json_parsing() {
        let mut calibration = ActivationCalibration::from_json(&document()).unwrap();
        assert!(calibration.validate().is_ok());

        calibration.model_type.clear();
        assert!(calibration.validate().is_err());

        let mut calibration = ActivationCalibration::from_json(&document()).unwrap();
        calibration.model_revision = Some("  ".into());
        assert!(calibration.validate().is_err());

        let mut calibration = ActivationCalibration::from_json(&document()).unwrap();
        calibration.dataset_sha256 = "z".repeat(64);
        assert!(calibration.validate().is_err());
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

    #[test]
    fn parses_module_type_and_rejects_invalid_values() {
        let linear = document().replace(
            r#""layer_name":"model.layers.0.mlp.up_proj""#,
            r#""layer_name":"model.layers.0.mlp.up_proj","module_type":"linear""#,
        );
        let calibration = ActivationCalibration::from_json(&linear).unwrap();
        assert_eq!(calibration.module_type.as_deref(), Some("linear"));
        assert_eq!(
            calibration.weight_orientation(),
            Ok(WeightOrientation::LinearOutIn)
        );
        assert_eq!(
            calibration.orientation_source(),
            WeightOrientationSource::Calibration
        );

        let conv1d = linear.replace(r#""module_type":"linear""#, r#""module_type":"conv1d""#);
        let calibration = ActivationCalibration::from_json(&conv1d).unwrap();
        assert_eq!(
            calibration.weight_orientation(),
            Ok(WeightOrientation::Conv1DInOut)
        );

        assert!(
            ActivationCalibration::from_json(
                &linear.replace(r#""module_type":"linear""#, r#""module_type":"Conv1D""#)
            )
            .is_err()
        );
        assert!(
            ActivationCalibration::from_json(
                &linear.replace(r#""module_type":"linear""#, r#""module_type":42"#)
            )
            .is_err()
        );
    }

    #[test]
    fn absence_of_module_type_defaults_to_linear_legacy() {
        let calibration = ActivationCalibration::from_json(&document()).unwrap();
        assert_eq!(calibration.module_type, None);
        assert_eq!(
            calibration.weight_orientation(),
            Ok(WeightOrientation::LinearOutIn)
        );
        assert_eq!(
            calibration.orientation_source(),
            WeightOrientationSource::LegacyDefault
        );
    }
}
