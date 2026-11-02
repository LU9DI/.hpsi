use hpsi_format::{MemorySource, ModelReader, ModelWriter, ReaderLimits};
use hpsi_ops::{VqConfig, VqMatrix, VqMatrixView};
use std::io::Cursor;
use std::sync::Arc;

fn synthetic_weights(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| {
            let value = index as f32;
            (value * 0.017).sin() * 0.6 + (value * 0.0031).cos() * 0.2
        })
        .collect()
}

#[test]
fn direct_matvec_from_hpsi_payloads_matches_in_memory_vq_view() {
    let rows = 64;
    let cols = 64;
    let weights = synthetic_weights(rows, cols);
    let matrix = VqMatrix::encode(&weights, rows, cols, VqConfig::default()).unwrap();
    let input: Vec<f32> = (0..cols).map(|index| (index as f32 * 0.11).sin()).collect();
    let mut expected = vec![0.0; rows];
    matrix.view().matvec(&input, &mut expected).unwrap();

    let tensors = matrix.to_hpsi_tensors(10, 11, 4096).unwrap();
    let mut artifact = Cursor::new(Vec::new());
    ModelWriter::write(&mut artifact, &tensors).unwrap();
    let reader = ModelReader::open(
        Arc::new(MemorySource::new(artifact.into_inner())),
        &ReaderLimits::default(),
    )
    .unwrap();

    let codebook_payload = reader.read_block(10, 0).unwrap().payload;
    let indices_payload = reader.read_block(11, 0).unwrap().payload;
    let codebook: Vec<f32> = codebook_payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let stored_view = VqMatrixView::new(rows, cols, &codebook, &indices_payload).unwrap();
    let mut actual = vec![0.0; rows];
    stored_view.matvec(&input, &mut actual).unwrap();

    assert_eq!(actual, expected);
    assert_eq!(reader.verify_all().unwrap(), 3);
}

#[test]
fn packed_representation_is_smaller_than_dense_f32_weights() {
    let rows = 256;
    let cols = 256;
    let weights = synthetic_weights(rows, cols);
    let matrix = VqMatrix::encode(&weights, rows, cols, VqConfig::default()).unwrap();
    assert!(matrix.stored_bytes() * 16 < matrix.dense_f32_bytes());
}

#[test]
fn view_rejects_mismatched_shapes_and_index_lengths() {
    let codebook = vec![0.0_f32; 16 * 8];
    let indices = vec![0_u8; 1];
    assert!(VqMatrixView::new(2, 16, &codebook, &indices).is_err());
    assert!(VqMatrixView::new(1, 16, &codebook, &indices).is_ok());
}

#[test]
fn encoder_rejects_non_finite_weights() {
    let mut weights = vec![0.0_f32; 16];
    weights[3] = f32::NAN;
    assert!(VqMatrix::encode(&weights, 1, 16, VqConfig::default()).is_err());
}
