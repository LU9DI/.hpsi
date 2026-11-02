use crate::{ActivationCalibration, SafeTensorFile, VqError};
use std::collections::{BTreeMap, HashSet};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedTensorPolicy {
    /// Fail rather than silently omit a tensor this importer cannot quantize.
    AllowRawPreservation,
    /// Reject the plan if any tensor cannot be VQ-quantized.
    Reject,
}

impl ModelImportPlan {
    /// Build a stable plan sorted by source tensor name. Rank-2 floating tensors
    /// are selected for VQ; all other currently supported Safetensors dtypes
    /// are either marked for raw preservation or rejected by policy.
    /// Calibration is matched by exact source tensor name and input feature count.
    pub fn build(
        source: &SafeTensorFile,
        calibrations: &[ActivationCalibration],
        unsupported_policy: UnsupportedTensorPolicy,
    ) -> Result<Self, VqError> {
        let mut calibration_by_name = BTreeMap::new();
        for calibration in calibrations {
            if calibration_by_name
                .insert(calibration.layer_name.as_str(), calibration)
                .is_some()
            {
                return Err(VqError("duplicate activation calibration layer name"));
            }
        }
        let mut names: Vec<_> = source.tensors().iter().collect();
        names.sort_by(|a, b| a.name.cmp(&b.name));
        let source_names: HashSet<&str> = source
            .tensors()
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect();
        let mut generated_names = HashSet::new();
        let mut tensors = Vec::with_capacity(names.len());

        for tensor in names {
            let quantizable = tensor.shape.len() == 2
                && tensor.shape.iter().all(|dimension| *dimension > 0)
                && tensor.shape[1] % crate::vq::GROUP_SIZE as u64 == 0
                && matches!(tensor.dtype.as_str(), "F32" | "F16" | "BF16");
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
                let codebook = format!("{}.hpsi_vq.codebook", tensor.name);
                let indices = format!("{}.hpsi_vq.indices", tensor.name);
                if !generated_names.insert(codebook.clone())
                    || !generated_names.insert(indices.clone())
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
                        if calibration.features as u64 != tensor.shape[1] {
                            return Err(VqError("activation calibration feature count mismatch"));
                        }
                        Some(calibration.sample_count)
                    }
                    None => None,
                };
                let activation_second_moment = calibration_by_name
                    .get(tensor.name.as_str())
                    .map(|calibration| calibration.activation_second_moment.clone());
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
                source_name: tensor.name.clone(),
                source_dtype: tensor.dtype.clone(),
                source_shape: tensor.shape.clone(),
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
