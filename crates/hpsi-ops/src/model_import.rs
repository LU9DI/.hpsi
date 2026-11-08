use crate::orientation::{ArchitectureFacts, TensorRole};
use crate::{
    ActivationCalibration, SafeTensorFile, VqError, WeightOrientation, WeightOrientationSource,
};
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
    /// Explicit orientation of the source weight matrix (never inferred).
    pub weight_orientation: WeightOrientation,
    /// Whether the orientation was declared by the capture bundle or assumed.
    pub orientation_source: WeightOrientationSource,
    /// Architectural role, taken from the capture's module class. A plan built
    /// without any class evidence records `unknown` rather than assuming a
    /// projection: an orientation token describes an layout, and assuming
    /// one has never been evidence of what the tensor is for.
    pub role: TensorRole,
    /// Which axis a lookup table indexes. Only `apply_architecture_roles` can
    /// set it, because only a model definition states it.
    pub index_axis: Option<usize>,
}

impl PlannedTensor {
    /// The role a capture bundle's `module_type` evidences.
    ///
    /// `linear` and `conv1d` name classes, not name patterns; a bundle that
    /// predates the module-type field states no class, so the role stays
    /// explicitly unknown while the documented legacy layout still applies.
    fn role_from_calibration(calibration: &ActivationCalibration) -> TensorRole {
        match calibration.module_type.as_deref() {
            Some("linear") => TensorRole::LinearProjection,
            Some("conv1d") => TensorRole::Conv1dProjection,
            _ => TensorRole::Unknown,
        }
    }
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
            let (weight_orientation, orientation_source, role) =
                match calibration_by_name.get(tensor.name.as_str()) {
                    Some(calibration) => (
                        calibration.weight_orientation().map_err(|_| {
                            VqError("activation calibration contains invalid orientation metadata")
                        })?,
                        calibration.orientation_source(),
                        PlannedTensor::role_from_calibration(calibration),
                    ),
                    None => (
                        WeightOrientation::LinearOutIn,
                        WeightOrientationSource::LegacyDefault,
                        TensorRole::Unknown,
                    ),
                };
            let features_in = if tensor.shape.len() == 2 {
                Some(
                    weight_orientation
                        .in_features(&tensor.shape)
                        .ok_or(VqError(
                            "activation calibration references a malformed weight shape",
                        ))?,
                )
            } else {
                // Raw ranks other than 2 are preserved verbatim and carry no
                // input feature dimension.
                None
            };
            let quantizable = features_in.is_some_and(|features| {
                tensor.shape.iter().all(|dimension| *dimension > 0)
                    && features % crate::vq::GROUP_SIZE as u64 == 0
                    && matches!(tensor.dtype.as_str(), "F32" | "F16" | "BF16")
            });
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
                        if u64::try_from(calibration.features).ok() != features_in {
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
                weight_orientation,
                orientation_source,
                role,
                index_axis: None,
            });
        }
        Ok(Self { tensors })
    }
}

impl ModelImportPlan {
    /// Refine the plan with the roles a model definition evidences.
    ///
    /// A capture bundle can only speak about the modules it hooked, so it never
    /// states what a lookup table is: an embedding has no input activations to
    /// take a second moment of. A model definition does know the class, and this
    /// is the only legitimate route by which a table gains a role. Nothing is
    /// inferred from names, suffixes or shapes: a tensor the manifest does not
    /// describe keeps exactly the role the calibration gave it.
    ///
    /// A table with no projection evidence is preserved RAW. That is the existing
    /// documented path for a tensor HPSI must not reinterpret; the alternative
    /// would be serving a lookup through the matvec kernel, and no lookup kernel
    /// ships in this contract. A table that the definition also reaches as a
    /// projection (a tied embedding/output head) keeps its projection encoding
    /// and records the projection role, because that is the function it was
    /// quantized under.
    pub fn apply_architecture_roles(&mut self, facts: &ArchitectureFacts) -> Result<(), String> {
        for planned in &mut self.tensors {
            let Some(expectation) = facts.expectation(&planned.source_name) else {
                continue;
            };
            if expectation.roles.is_empty() {
                continue;
            }
            let projection = expectation.roles.iter().find(|role| role.is_projection());
            if let Some(role) = projection.copied() {
                let required = match role {
                    TensorRole::Conv1dProjection => WeightOrientation::Conv1DInOut,
                    _ => WeightOrientation::LinearOutIn,
                };
                if planned.weight_orientation != required {
                    return Err(format!(
                        "{}: the model definition reaches this tensor as {}, which is laid out {:?}, but the plan would encode it {:?}",
                        planned.source_name,
                        role.as_str(),
                        required.as_str(),
                        planned.weight_orientation.as_str()
                    ));
                }
                planned.role = role;
                // v0.1 records one role per tensor: the function it was actually
                // encoded under. A tied weight reached as both a table and a
                // projection is recorded as the projection, because that is what
                // its selector bytes mean; the additional table role stays in the
                // model definition, where the audit surfaces it as
                // `architecture_roles`. `index_axis` belongs to the recorded role,
                // so it is not emitted for a projection.
                planned.index_axis = None;
                continue;
            }
            if !expectation
                .roles
                .iter()
                .any(|role| matches!(role, TensorRole::EmbeddingTable))
            {
                // A role that states neither a projection nor a table: recorded
                // as the definition gives it, with no layout claim implied.
                planned.role = expectation.roles[0];
                planned.index_axis = None;
                continue;
            }
            let axis = expectation.index_axis.ok_or_else(|| {
                format!(
                    "{}: embedding_table requires index_axis 0 or 1",
                    planned.source_name
                )
            })?;
            if planned.source_shape.len() != 2 {
                return Err(format!(
                    "{}: an indexed table must be rank 2, found shape {:?}",
                    planned.source_name, planned.source_shape
                ));
            }
            if axis == 1 {
                // Axis 1 needs the `[feature, index]` layout, i.e. the
                // `conv1d_in_out` encoding, on a tensor that is not a Conv1D
                // layer. v0.1 permits that pairing only through
                // `orientation_source: calibration`, which requires an activation
                // object an indexed table cannot honestly carry. Saying so here,
                // instead of quietly relabelling the table as a Conv1D, is the
                // contract decision this phase deliberately leaves pending.
                return Err(format!(
                    "{}: an embedding table indexed on axis 1 cannot state its layout under v0.1 without inventing a calibration claim; this needs the role schema decision in docs/orientation-policy.md",
                    planned.source_name
                ));
            }
            // A tensor the capture measured input activations for is used as a
            // multiplication somewhere in the model, so the definition calling it
            // purely an indexed table is a disagreement to resolve in the model
            // definition, not something to encode. v0.1 could not express the
            // combination anyway: a RAW tensor must not carry calibration, and an
            // indexing role with calibration provenance would need a calibration
            // object.
            if planned.orientation_source == WeightOrientationSource::Calibration {
                return Err(format!(
                    "{}: the capture bundle measured input activations for this tensor while the model definition states it is only an indexed table; a tensor reached as a projection must keep the projection role it was encoded under",
                    planned.source_name
                ));
            }
            planned.role = TensorRole::EmbeddingTable;
            planned.index_axis = Some(axis);
            // Features stay on the grouped axis, so the recorded layout is
            // the one an indexed-first table implies.
            planned.weight_orientation = WeightOrientation::LinearOutIn;
            planned.disposition = TensorDisposition::PreserveRaw;
            planned.codebook_name = None;
            planned.indices_name = None;
            planned.calibration_samples = None;
            planned.activation_second_moment = None;
        }
        Ok(())
    }
}

/// Compact calibration provenance persisted in HPSI metadata. Numeric moments
/// remain in the import plan for encoding and are deliberately not serialized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationProvenance {
    pub source_name: String,
    pub model_type: String,
    pub model_revision: Option<String>,
    pub dataset_sha256: String,
    pub sample_count: usize,
    pub features: usize,
    pub statistic: String,
    /// Explicit orientation of the calibrated layer's weight matrix.
    pub weight_orientation: String,
    /// Whether the orientation was declared by the capture bundle or assumed.
    pub orientation_source: String,
}

/// An import plan paired with the calibration provenance required to explain
/// every activation-aware VQ encoding. The legacy plan type remains unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedModelImportPlan {
    pub plan: ModelImportPlan,
    pub calibrations: Vec<CalibrationProvenance>,
}

impl ModelImportPlan {
    /// Build a regular import plan and retain compact provenance for every
    /// supplied calibration without copying its potentially large moment vector.
    pub fn build_with_provenance(
        source: &SafeTensorFile,
        calibrations: &[ActivationCalibration],
        unsupported_policy: UnsupportedTensorPolicy,
    ) -> Result<CalibratedModelImportPlan, VqError> {
        let plan = Self::build(source, calibrations, unsupported_policy)?;
        let mut provenance = Vec::new();
        provenance
            .try_reserve_exact(calibrations.len())
            .map_err(|_| VqError("could not allocate calibration provenance"))?;
        for calibration in calibrations {
            calibration.validate().map_err(|_| {
                VqError("activation calibration contains invalid values or provenance")
            })?;
            provenance.push(CalibrationProvenance {
                source_name: clone_plan_text(&calibration.layer_name)?,
                model_type: clone_plan_text(&calibration.model_type)?,
                model_revision: calibration
                    .model_revision
                    .as_deref()
                    .map(clone_plan_text)
                    .transpose()?,
                dataset_sha256: clone_plan_text(&calibration.dataset_sha256)?,
                sample_count: calibration.sample_count,
                features: calibration.features,
                statistic: "diagonal_second_moment".to_owned(),
                weight_orientation: clone_plan_text(
                    calibration
                        .weight_orientation()
                        .map_err(|_| {
                            VqError("activation calibration contains invalid orientation metadata")
                        })?
                        .as_str(),
                )?,
                orientation_source: clone_plan_text(calibration.orientation_source().as_str())?,
            });
        }
        provenance.sort_unstable_by(|left, right| left.source_name.cmp(&right.source_name));
        Ok(CalibratedModelImportPlan {
            plan,
            calibrations: provenance,
        })
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
            module_type: Some("linear".into()),
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
        assert_eq!(
            first.tensors[1].weight_orientation,
            crate::WeightOrientation::LinearOutIn
        );
        assert_eq!(
            first.tensors[1].orientation_source,
            crate::WeightOrientationSource::Calibration
        );
    }

    #[test]
    fn rejects_invalid_orientation_module_type() {
        let mut invalid = calibration();
        invalid.module_type = Some("Conv1D".into());
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
