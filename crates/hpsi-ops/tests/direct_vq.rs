use hpsi_format::{MemorySource, ModelReader, ModelWriter, ReaderLimits};
use hpsi_ops::{VqConfig, VqMatrix, VqMatrixView, matvec_hpsi_rows};
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

fn heldout_relative_l2(weights: &[f32], rows: usize, cols: usize, matrix: &VqMatrix) -> f64 {
    let view = matrix.view();
    let mut error_squared = 0.0_f64;
    let mut reference_squared = 0.0_f64;
    for sample in 0..8 {
        let input: Vec<f32> = (0..cols)
            .map(|index| ((index as f32 + sample as f32) * 0.11).sin())
            .collect();
        let mut reference = vec![0.0; rows];
        let mut approximate = vec![0.0; rows];
        for row in 0..rows {
            for col in 0..cols {
                reference[row] += weights[row * cols + col] * input[col];
            }
        }
        view.matvec(&input, &mut approximate).unwrap();
        for (expected, actual) in reference.iter().zip(&approximate) {
            let difference = expected - actual;
            error_squared += f64::from(difference) * f64::from(difference);
            reference_squared += f64::from(*expected) * f64::from(*expected);
        }
    }
    (error_squared / reference_squared.max(f64::MIN_POSITIVE)).sqrt()
}

#[test]
fn direct_matvec_from_hpsi_payloads_matches_in_memory_additive_view() {
    let rows = 64;
    let cols = 64;
    let weights = synthetic_weights(rows, cols);
    let matrix = VqMatrix::encode(
        &weights,
        rows,
        cols,
        VqConfig {
            codebook_count: 2,
            ..VqConfig::default()
        },
    )
    .unwrap();
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
    let stored_view = VqMatrixView::new(rows, cols, 2, &codebook, &indices_payload).unwrap();
    let mut actual = vec![0.0; rows];
    stored_view.matvec(&input, &mut actual).unwrap();

    assert_eq!(actual, expected);
    assert_eq!(reader.verify_all().unwrap(), 3);
}

#[test]
fn additive_codebook_reduces_weight_domain_error() {
    let rows = 64;
    let cols = 64;
    let weights = synthetic_weights(rows, cols);
    let single = VqMatrix::encode(&weights, rows, cols, VqConfig::default()).unwrap();
    let additive = VqMatrix::encode(
        &weights,
        rows,
        cols,
        VqConfig {
            codebook_count: 2,
            ..VqConfig::default()
        },
    )
    .unwrap();

    assert!(additive.codebook_count() == 2);
    assert!(additive.stored_bytes() > single.stored_bytes());
    let single_error = heldout_relative_l2(&weights, rows, cols, &single);
    let additive_error = heldout_relative_l2(&weights, rows, cols, &additive);
    assert!(additive_error <= single_error);
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
    assert!(VqMatrixView::new(2, 16, 1, &codebook, &indices).is_err());
    assert!(VqMatrixView::new(1, 16, 1, &codebook, &indices).is_ok());
}

#[test]
fn encoder_rejects_non_finite_weights() {
    let mut weights = vec![0.0_f32; 16];
    weights[3] = f32::NAN;
    assert!(VqMatrix::encode(&weights, 1, 16, VqConfig::default()).is_err());
}

#[test]
fn row_streamed_hpsi_matvec_reads_only_selected_index_blocks() {
    let rows = 64;
    let cols = 64;
    let weights = synthetic_weights(rows, cols);
    let matrix = VqMatrix::encode(
        &weights,
        rows,
        cols,
        VqConfig {
            codebook_count: 2,
            ..VqConfig::default()
        },
    )
    .unwrap();
    let input: Vec<f32> = (0..cols).map(|index| (index as f32 * 0.09).cos()).collect();
    let mut expected = vec![0.0; rows];
    matrix.view().matvec(&input, &mut expected).unwrap();

    let tensors = matrix.to_hpsi_tensors(20, 21, 8).unwrap();
    let mut artifact = Cursor::new(Vec::new());
    ModelWriter::write(&mut artifact, &tensors).unwrap();
    let reader = ModelReader::open(
        Arc::new(MemorySource::new(artifact.into_inner())),
        &ReaderLimits::default(),
    )
    .unwrap();

    let first_row = 16;
    let mut selected = vec![0.0; 8];
    matvec_hpsi_rows(
        &reader,
        20,
        21,
        rows,
        cols,
        2,
        &input,
        first_row,
        &mut selected,
    )
    .unwrap();

    assert_eq!(selected, expected[first_row..first_row + selected.len()]);
    let codebook_bytes = 2 * 16 * 8 * 4;
    let one_index_block_bytes = (cols / 8) * 8;
    assert_eq!(
        reader.stats().payload_bytes_read,
        (codebook_bytes + one_index_block_bytes) as u64
    );
}
