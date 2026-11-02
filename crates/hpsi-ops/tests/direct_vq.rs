use hpsi_format::{MemorySource, ModelReader, ModelWriter, ReaderLimits};
use hpsi_ops::{HpsiVqExecutor, HpsiVqLayout, VqConfig, VqMatrix, VqMatrixView, matvec_hpsi_rows};
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

fn weight_relative_l2(weights: &[f32], matrix: &VqMatrix) -> f64 {
    let codebook = matrix.codebook();
    let selectors = matrix.packed_indices();
    let codebook_count = matrix.codebook_count();
    let mut error_squared = 0.0_f64;
    let mut reference_squared = 0.0_f64;

    for (group, source) in weights.chunks_exact(8).enumerate() {
        for dimension in 0..8 {
            let mut reconstructed = 0.0_f32;
            for stage in 0..codebook_count {
                let selector = group * codebook_count + stage;
                let packed = selectors[selector / 2];
                let code = usize::from(if selector % 2 == 0 {
                    packed & 0x0f
                } else {
                    packed >> 4
                });
                reconstructed += codebook[stage * 16 * 8 + code * 8 + dimension];
            }
            let difference = source[dimension] - reconstructed;
            error_squared += f64::from(difference) * f64::from(difference);
            reference_squared += f64::from(source[dimension]) * f64::from(source[dimension]);
        }
    }

    (error_squared / reference_squared.max(f64::MIN_POSITIVE)).sqrt()
}

fn scalar_reference_matvec(matrix: &VqMatrix, input: &[f32], output: &mut [f32]) {
    let groups_per_row = matrix.cols() / 8;
    let codebook = matrix.codebook();
    let selectors = matrix.packed_indices();
    for (row, output_value) in output.iter_mut().enumerate() {
        let mut accumulator = 0.0_f32;
        for group in 0..groups_per_row {
            let global_group = row * groups_per_row + group;
            for stage in 0..matrix.codebook_count() {
                let selector = global_group * matrix.codebook_count() + stage;
                let packed = selectors[selector / 2];
                let code = usize::from(if selector % 2 == 0 {
                    packed & 0x0f
                } else {
                    packed >> 4
                });
                let codebook_offset = stage * 16 * 8 + code * 8;
                for dimension in 0..8 {
                    accumulator +=
                        input[group * 8 + dimension] * codebook[codebook_offset + dimension];
                }
            }
        }
        *output_value = accumulator;
    }
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
    let single_error = weight_relative_l2(&weights, &single);
    let mut previous_error = single_error;
    let mut previous_size = single.stored_bytes();

    for codebook_count in 2..=4 {
        let additive = VqMatrix::encode(
            &weights,
            rows,
            cols,
            VqConfig {
                codebook_count,
                ..VqConfig::default()
            },
        )
        .unwrap();

        assert_eq!(additive.codebook_count(), codebook_count);
        assert!(additive.stored_bytes() > previous_size);
        let additive_error = weight_relative_l2(&weights, &additive);
        assert!(
            additive_error <= previous_error,
            "{codebook_count} codebooks increased weight-domain error: {additive_error} > {previous_error}"
        );
        assert!(heldout_relative_l2(&weights, rows, cols, &additive).is_finite());
        previous_error = additive_error;
        previous_size = additive.stored_bytes();
    }
}

#[test]
fn centroid_dot_table_matches_scalar_reference_for_all_rates() {
    let rows = 96;
    let cols = 128;
    let weights = synthetic_weights(rows, cols);
    let input: Vec<f32> = (0..cols)
        .map(|index| ((index as f32) * 0.037).cos() * 0.8)
        .collect();

    for codebook_count in 1..=4 {
        let matrix = VqMatrix::encode(
            &weights,
            rows,
            cols,
            VqConfig {
                codebook_count,
                ..VqConfig::default()
            },
        )
        .unwrap();
        let mut optimized = vec![0.0_f32; rows];
        let mut scalar = vec![0.0_f32; rows];
        matrix.view().matvec(&input, &mut optimized).unwrap();
        scalar_reference_matvec(&matrix, &input, &mut scalar);

        for (row, (actual, expected)) in optimized.iter().zip(&scalar).enumerate() {
            let difference = (actual - expected).abs();
            assert!(
                difference <= 2.0e-5,
                "rate {codebook_count}, row {row}: optimized={actual}, scalar={expected}, diff={difference}"
            );
        }
    }
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

    let odd_group_indices = [0xf0_u8];
    assert!(VqMatrixView::new(1, 8, 1, &codebook, &odd_group_indices).is_err());
    let valid_odd_group_indices = [0x0f_u8];
    assert!(VqMatrixView::new(1, 8, 1, &codebook, &valid_odd_group_indices).is_ok());
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
    let input: Vec<f32> = (0..cols).map(|index| (index as f32 * 0.09).cos()).collect();
    let first_row = 16;

    for codebook_count in 1..=4 {
        let matrix = VqMatrix::encode(
            &weights,
            rows,
            cols,
            VqConfig {
                codebook_count,
                ..VqConfig::default()
            },
        )
        .unwrap();
        let mut expected = vec![0.0; rows];
        matrix.view().matvec(&input, &mut expected).unwrap();

        let tensors = matrix.to_hpsi_tensors(20, 21, 8).unwrap();
        let mut artifact = Cursor::new(Vec::new());
        ModelWriter::write(&mut artifact, &tensors).unwrap();
        let artifact_bytes = artifact.into_inner();
        let reader = ModelReader::open(
            Arc::new(MemorySource::new(artifact_bytes.clone())),
            &ReaderLimits::default(),
        )
        .unwrap();

        let layout = HpsiVqLayout {
            codebook_tensor_id: 20,
            indices_tensor_id: 21,
            rows,
            cols,
            codebook_count,
        };
        let mut selected = vec![0.0; 8];
        matvec_hpsi_rows(&reader, layout, &input, first_row, &mut selected).unwrap();

        assert_eq!(selected, expected[first_row..first_row + selected.len()]);
        let codebook_bytes = codebook_count * 16 * 8 * 4;
        let one_index_block_bytes = (cols / 8) * codebook_count / 2 * 8;
        assert_eq!(
            reader.stats().payload_bytes_read,
            (codebook_bytes + one_index_block_bytes) as u64
        );

        let prepared_reader = ModelReader::open(
            Arc::new(MemorySource::new(artifact_bytes)),
            &ReaderLimits::default(),
        )
        .unwrap();
        let executor = HpsiVqExecutor::new(&prepared_reader, layout).unwrap();
        assert_eq!(
            prepared_reader.stats().payload_bytes_read,
            codebook_bytes as u64,
            "prepared executor should load its shared codebook exactly once"
        );

        // Exercise a partial window inside a larger 8-row payload block.
        let mut partial = vec![0.0; 3];
        executor.matvec_rows(&input, 19, &mut partial).unwrap();
        assert_eq!(partial, expected[19..22]);
        let bytes_after_first_window = prepared_reader.stats().payload_bytes_read;
        assert_eq!(
            bytes_after_first_window,
            (codebook_bytes + one_index_block_bytes) as u64
        );

        executor.matvec_rows(&input, 20, &mut partial).unwrap();
        assert_eq!(partial, expected[20..23]);
        assert_eq!(
            prepared_reader.stats().payload_bytes_read - bytes_after_first_window,
            one_index_block_bytes as u64,
            "repeated windows must not reload the codebook"
        );

        // A request crossing the boundary between two payload blocks reads
        // both blocks but computes/copies only the four requested rows.
        let bytes_before_crossing = prepared_reader.stats().payload_bytes_read;
        let mut crossing = vec![0.0; 4];
        executor.matvec_rows(&input, 7, &mut crossing).unwrap();
        assert_eq!(crossing, expected[7..11]);
        assert_eq!(
            prepared_reader.stats().payload_bytes_read - bytes_before_crossing,
            (2 * one_index_block_bytes) as u64
        );
    }
}
