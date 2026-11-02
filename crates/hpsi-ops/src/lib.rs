mod activation;
mod model_import;
mod model_import_writer;
mod projection;
mod safetensors;
mod vq;

pub use activation::{ActivationCalibration, CalibrationError};
pub use model_import::{
    ModelImportPlan, PlannedTensor, TensorDisposition, UnsupportedTensorPolicy,
};
pub use model_import_writer::write_model_import_to_hpsi;
pub use projection::{ProjectionCandidate, discover_projection_candidates};
pub use safetensors::{SafeTensorError, SafeTensorFile, SafeTensorInfo};

pub use vq::{
    CODEBOOK_SIZE, GROUP_SIZE, HpsiVqExecutor, HpsiVqLayout, VqConfig, VqError, VqMatrix,
    VqMatrixView, matvec_hpsi_rows,
};
