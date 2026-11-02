use crate::{ActivationCalibration, SafeTensorFile, VqError};
use std::collections::{HashMap, HashSet};

/// A deterministic, validated plan for a multi-tensor Safetensors import.
/// Planning does not write output; it prevents silent tensor loss and name collisions.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelImportPlan {
    pub tensors: Vec<PlannedTensor>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlannedTensor {
    pub source_name: String,
    pub source_dtype: String,
    pub source_shape: Vec<u64>,
    pub disposition: TensorDisposition,
    pub codebook_name: Option<String>,
    pub indices_name: Option<String>,
    pub calibration_samples: Option<usize>,
    /// Exact diagonal second moments used by activation-aware VQ encoding.
    pub activation_second_moment: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDisposition {
    QuantizeVq,
    PreserveRaw,
}

fn clone_plan_text(value: &str) -> Result<String, VqError> {
    let mut cloned = String::new();
    cloned
        .try_reserve_exact(value.len())
        .map_err(|_| VqError("could not allocate model import name"))?;
    cloned.push_str(value);
    Ok(cloned)
}

fn generated_tensor_name(source_name: &str, suffix: &str) -> Result<String, VqError> {
    let capacity = source_name
        .len()
        .checked_add(suffix.len())
        .ok_or(VqError("generated tensor name length overflow"))?;
    let mut name = String::new();
    name.try_reserve_exact(capacity)
        .map_err(|_| VqError("could not allocate generated tensor name"))?;
    name.push_str(source_name);
    name.push_str(suffix);
    Ok(name)
}

fn clone_plan_shape(shape: &[u64]) -> Result<Vec<u64>, VqError> {
    let mut cloned = Vec::new();
    cloned
        .try_reserve_exact(shape.len())
        .map_err(|_| VqError("could not allocate model import tensor shape"))?;
    cloned.extend_from_slice(shape);
    Ok(cloned)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedTensorPolicy {
    /// Preserve non-VQ tensors byte-for-byte instead of silently omitting them.
    AllowRawPreservation,
    /// Reject the plan if any tensor cannot be VQ-quantized.
    Reject,
}

impl ModelImportPlan {
    /// Build a stable plan sorted by source tensor name. Only rank-2 F32,
    /// F16, and BF16 tensors with a feature dimension divisible by the VQ group
    /// size are selected for VQ; all other supported tensors are preserved raw
    /// or rejected according to policy. Calibration uses exact source names
    /// and must match the input feature count.
    pub fn build(
        source: &SafeTensorFile,
        calibrations: &[ActivationCalibration],
        unsupported_policy: UnsupportedTensorPolicy,
    ) -> Result<Self, VqError> {
        let mut calibration_by_name = HashMap::new();
        calibration_by_name
            .try_reserve(calibrations.len())
            .map_err(|_| VqError("could not allocate activation calibration index"))?;
        for calibration in calibrations {
            calibration.validate().map_err(|_| {
                VqError("activation calibration contains invalid values or provenance")
            })?;
            if calibration_by_name
                .insert(calibration.layer_name.as_str(), calibration)
                .is_some()
            {
                return Err(VqError("duplicate activation calibration layer name"));
            }
        }
        let mut names = Vec::new();
        names
            .try_reserve_exact(source.tensors().len())
            .map_err(|_| VqError("could not allocate source tensor ordering"))?;
        names.extend(source.tensors().iter());
        names.sort_by(|a, b| a.name.cmp(&b.name));
        let mut source_names: HashSet<&str> = HashSet::new();
        source_names
            .try_reserve(source.tensors().len())
            .map_err(|_| VqError("could not allocate source tensor name index"))?;
        source_names.extend(source.tensors().iter().map(|tensor| tensor.name.as_str()));
        if calibration_by_name
            .keys()
            .any(|layer_name| !source_names.contains(layer_name))
        {
            return Err(VqError(
                "activation calibration references a tensor absent from the source model",
            ));
        }
        let generated_name_capacity = names
            .len()
            .checked_mul(2)
            .ok_or(VqError("generated tensor name count overflow"))?;
        let mut generated_names = HashSet::new();
        generated_names
            .try_reserve(generated_name_capacity)
            .map_err(|_| VqError("could not allocate generated tensor name index"))?;
        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(names.len())
            .map_err(|_| VqError("could not allocate model import tensor plan"))?;

        for tensor in names {
            if tensor.shape.is_empty() || tensor.shape.len() > 8 || tensor.shape.contains(&0) {
                return Err(VqError(
                    "source tensor rank or dimensions are incompatible with HPSI",
                ));
            }
            let quantizable = tensor.shape.len() == 2
                && tensor.shape.iter().all(|dimension| *dimension > 0)
                && tensor.shape[1] % crate::vq::GROUP_SIZE as u64 == 0
                && matches!(tensor.dtype.as_str(), "F32" | "F16" | "BF16");
            if calibration_by_name.contains_key(tensor.name.as_str()) && !quantizable {
                return Err(VqError(
                    "activation calibration targets a tensor that is not eligible for VQ",
                ));
            }
            if !quantizable && unsupported_policy == UnsupportedTensorPolicy::Reject {
                return Err(VqError(
                    "model contains a tensor unsupported by VQ import policy",
                ));
            }
            let (
                disposition,
                codebook_name,
                indices_name,
                calibration_samples,
                activation_second_moment,
            ) = if quantizable {
                let codebook = generated_tensor_name(&tensor.name, ".hpsi_vq.codebook")?;
                let indices = generated_tensor_name(&tensor.name, ".hpsi_vq.indices")?;
                if !generated_names.insert(clone_plan_text(&codebook)?)
                    || !generated_names.insert(clone_plan_text(&indices)?)
                    || codebook == indices
                {
                    return Err(VqError("generated VQ tensor name collision"));
                }
                if source_names.contains(codebook.as_str())
                    || source_names.contains(indices.as_str())
                {
                    return Err(VqError(
                        "generated VQ tensor name collides with source tensor",
                    ));
                }
                let calibration_samples = match calibration_by_name.get(tensor.name.as_str()) {
                    Some(calibration) => {
                        if u64::try_from(calibration.features).ok() != Some(tensor.shape[1]) {
                            return Err(VqError("activation calibration feature count mismatch"));
                        }
                        Some(calibration.sample_count)
                    }
                    None => None,
                };
                let activation_second_moment = calibration_by_name
                    .get(tensor.name.as_str())
                    .map(|calibration| {
                        let mut values = Vec::new();
                        values
                            .try_reserve_exact(calibration.activation_second_moment.len())
                            .map_err(|_| {
                                VqError("could not allocate activation calibration copy")
                            })?;
                        values.extend_from_slice(&calibration.activation_second_moment);
                        Ok(values)
                    })
                    .transpose()?;
                (
                    TensorDisposition::QuantizeVq,
                    Some(codebook),
                    Some(indices),
                    calibration_samples,
                    activation_second_moment,
                )
            } else {
                (TensorDisposition::PreserveRaw, None, None, None, None)
            };
            tensors.push(PlannedTensor {
                source_name: clone_plan_text(&tensor.name)?,
                source_dtype: clone_plan_text(&tensor.dtype)?,
                source_shape: clone_plan_shape(&tensor.shape)?,
                disposition,
                codebook_name,
                indices_name,
                calibration_samples,
                activation_second_moment,
            });
        }
        Ok(Self { tensors })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hpsi_format::MemorySource;
    use std::sync::Arc;

    fn source() -> SafeTensorFile {
        let header = br#"{"bias":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"weight":{"dtype":"F32","shape":[2,8],"data_offsets":[8,72]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0_u8; 72]);
        SafeTensorFile::open(Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))), 4096).unwrap()
    }

    fn calibration() -> ActivationCalibration {
        ActivationCalibration {
            layer_name: "weight".into(),
            sample_count: 12,
            features: 8,
            activation_second_moment: vec![1.0; 8],
            model_type: "test".into(),
            model_revision: None,
            dataset_sha256: "a".repeat(64),
        }
    }

    #[test]
    fn plans_all_tensors_deterministically_and_carries_calibration() {
        let source = source();
        let first = ModelImportPlan::build(
            &source,
            &[calibration()],
            UnsupportedTensorPolicy::AllowRawPreservation,
        )
        .unwrap();
        let second = ModelImportPlan::build(
            &source,
            &[calibration()],
            UnsupportedTensorPolicy::AllowRawPreservation,
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.tensors.len(), 2);
        assert_eq!(first.tensors[0].source_name, "bias");
        assert_eq!(first.tensors[0].disposition, TensorDisposition::PreserveRaw);
        assert_eq!(first.tensors[1].source_name, "weight");
        assert_eq!(first.tensors[1].disposition, TensorDisposition::QuantizeVq);
        assert_eq!(first.tensors[1].calibration_samples, Some(12));
        assert_eq!(
            first.tensors[1].activation_second_moment.as_ref().unwrap(),
            &vec![1.0; 8]
        );
    }

    #[test]
    fn rejects_raw_preservation_when_policy_forbids_it() {
        assert!(ModelImportPlan::build(&source(), &[], UnsupportedTensorPolicy::Reject).is_err());
    }

    #[test]
    fn rejects_invalid_activation_calibration_values() {
        let source = source();
        let mut invalid = calibration();
        invalid.sample_count = 0;
        assert!(
            ModelImportPlan::build(
                &source,
                &[invalid],
                UnsupportedTensorPolicy::AllowRawPreservation
            )
            .is_err()
        );

        let mut invalid = calibration();
        invalid.activation_second_moment[0] = f32::NAN;
        assert!(
            ModelImportPlan::build(
                &source,
                &[invalid],
                UnsupportedTensorPolicy::AllowRawPreservation
            )
            .is_err()
        );

        let mut invalid = calibration();
        invalid.activation_second_moment[0] = -1.0;
        assert!(
            ModelImportPlan::build(
                &source,
                &[invalid],
                UnsupportedTensorPolicy::AllowRawPreservation
            )
            .is_err()
        );

        let mut invalid = calibration();
        invalid.activation_second_moment.pop();
        assert!(
            ModelImportPlan::build(
                &source,
                &[invalid],
                UnsupportedTensorPolicy::AllowRawPreservation
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_calibration_for_non_quantizable_tensor() {
        let mut invalid = calibration();
        invalid.layer_name = "bias".into();
        invalid.features = 2;
        invalid.activation_second_moment = vec![1.0; 2];
        assert!(
            ModelImportPlan::build(
                &source(),
                &[invalid],
                UnsupportedTensorPolicy::AllowRawPreservation
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_calibration_for_unknown_source_tensor() {
        let mut invalid = calibration();
        invalid.layer_name = "missing.weight".into();
        assert!(
            ModelImportPlan::build(
                &source(),
                &[invalid],
                UnsupportedTensorPolicy::AllowRawPreservation
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_calibration_feature_mismatch() {
        let mut invalid = calibration();
        invalid.features = 7;
        assert!(
            ModelImportPlan::build(
                &source(),
                &[invalid],
                UnsupportedTensorPolicy::AllowRawPreservation
            )
            .is_err()
        );
    }
}
