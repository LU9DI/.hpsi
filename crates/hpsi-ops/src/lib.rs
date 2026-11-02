mod vq;

pub use vq::{
    CODEBOOK_SIZE, GROUP_SIZE, VqConfig, VqError, VqMatrix, VqMatrixView, HpsiVqLayout, matvec_hpsi_rows,
};
