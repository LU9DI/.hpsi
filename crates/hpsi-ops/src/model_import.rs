use crate::{ActivationCalibration, SafeTensorFile, VqError};
use std::collections::{BTreeMap, HashSet};

/// A deterministic, validated plan for a multi-tensor Safetensors import.
/// Planning does not write output; it prevents silent tensor loss and name collisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelImportPlan {
    pub tensors: Vec<PlannedTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedTensor {
    pub source_name: String,
    pub source_dtype: String,
    pub source_shape: Vec<u64>,
    pub disposition: TensorDisposition,
    pub codebook_name: Option<String>,
    pub indices_name: Option<String>,
    pub calibration_samples: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDisposition {
    QuantizeVq,
    PreserveRaw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedTensorPolicy {
    /// Fail rather than silently omit a tensor this importer cannot quantize.
    PreserveRaw,
    /// Reject the plan if any tensor cannot be VQ-quantized.
    Reject,
}

impl ModelImportPlan {
    /// Build a stable plan sorted by source tensor name. Rank-2 floating tensors
    /// are selected for VQ; all other currently supported Safetensors dtypes
    /// are either marked for lossless raw preservation or rejected by policy.
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
        let mut generated_names = HashSet::new();
        let mut tensors = Vec::with_capacity(names.len());

        for tensor in names {
            let quantizable = tensor.shape.len() == 2
                && matches!(tensor.dtype.as_str(), "F32" | "F16" | "BF16");
            if !quantizable && unsupported_policy == UnsupportedTensorPolicy::Reject {
                return Err(VqError("model contains a tensor unsupported by VQ import policy"));
            }
            let (disposition, codebook_name, indices_name, calibration_samples) =
                if quantizable {
                    let codebook = format!("{}.hpsi_vq.codebook", tensor.name);
                    let indices = format!("{}.hpsi_vq.indices", tensor.name);
                    if !generated_names.insert(codebook.clone())
                        || !generated_names.insert(indices.clone())
                        || codebook == indices
                    {
                        return Err(VqError("generated VQ tensor name collision"));
                    }
                    if source.tensors().iter().any(|source_tensor| {
                        source_tensor.name == codebook || source_tensor.name == indices
                    }) {
                        return Err(VqError("generated VQ tensor name collides with source tensor"));
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
                    (
                        TensorDisposition::QuantizeVq,
                        Some(codebook),
                        Some(indices),
                        calibration_samples,
                    )
                } else {
                    (TensorDisposition::PreserveRaw, None, None, None)
                };
            tensors.push(PlannedTensor {
                source_name: tensor.name.clone(),
                source_dtype: tensor.dtype.clone(),
                source_shape: tensor.shape.clone(),
                disposition,
                codebook_name,
                indices_name,
                calibration_samples,
            });
        }
        Ok(Self { tensors })
    }
}
