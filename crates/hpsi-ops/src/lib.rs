#[cfg(test)]
mod orient_tests;

mod activation;
mod model_import;
mod model_import_writer;
mod orientation;
mod projection;
mod safetensors;
mod vq;

pub use activation::{
    ActivationCalibration, CalibrationError, WeightOrientation, WeightOrientationSource,
};
pub use model_import::{
    CalibratedModelImportPlan, CalibrationProvenance, ModelImportPlan, PlannedTensor,
    TensorDisposition, UnsupportedTensorPolicy,
};
pub use model_import_writer::{
    write_model_import_to_hpsi, write_model_import_to_hpsi_calibrated,
    write_model_import_to_hpsi_path, write_model_import_to_hpsi_path_calibrated,
};
pub use orientation::{
    ARCHITECTURE_MANIFEST_FORMAT, ARCHITECTURE_MANIFEST_VERSION, ArchitectureExpectation,
    ArchitectureFacts, EntryLayout, LayoutContext, LayoutViolation, OrientationAssurance,
    OrientationAudit, OrientationError, ResolvedVqTensor, RoleAssurance, RoleEvidence,
    TensorOrientationFacts, TensorRole, audit_container, resolve_vq_layout, validate_entry_layout,
};
pub use projection::{
    ProjectionCandidate, ProjectionError, discover_projection_candidates,
    try_discover_projection_candidates,
};
pub use safetensors::{SafeTensorError, SafeTensorFile, SafeTensorInfo, TransposedFloatReader};

pub use vq::{
    CODEBOOK_SIZE, GROUP_SIZE, HpsiVqExecutor, HpsiVqLayout, VqConfig, VqError,
    VqHpsiExportOptions, VqMatrix, VqMatrixView, VqTensorNames, matvec_hpsi_rows,
};
