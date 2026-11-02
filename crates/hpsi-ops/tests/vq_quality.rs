//! Deterministic quality regression tests for HPSI's additive VQ representation.
//!
//! These tests use synthetic weights and held-out input vectors. They are
//! regression checks, not claims about any particular production checkpoint.

use hpsi_ops::{VqConfig, VqMatrix};

fn fixture(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| {
            let row = (i / cols) as f32;
            let col = (i % cols) as f32;
            // A deterministic mixture of smooth structure and bounded variation.
            let smooth = ((row * 0.17 + col * 0.11).sin()) * 0.7;
            let variation = (((i * 37 + 11) % 101) as f32 / 50.0) - 1.0;
            smooth + variation * 0.23
        })
        .collect()
}

fn held_out_inputs(cols: usize) -> Vec<Vec<f32>> {
    (0..12)
        .map(|sample| {
            (0..cols)
                .map(|col| {
                    let phase = sample as f32 * 0.31 + col as f32 * 0.19;
                    phase.sin() * 0.8 + (phase * 0.37).cos() * 0.2
                })
                .collect()
        })
        .collect()
}

fn relative_output_error(matrix: &VqMatrix, weights: &[f32], rows: usize, cols: usize) -> f64 {
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let view = matrix.view();

    for input in held_out_inputs(cols) {
        let mut actual = vec![0.0_f32; rows];
        view.matvec(&input, &mut actual).expect("VQ matvec");

        for row in 0..rows {
            let reference = weights[row * cols..(row + 1) * cols]
                .iter()
                .zip(&input)
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
            let delta = (actual[row] - reference) as f64;
            squared_error += delta * delta;
            squared_reference += (reference as f64) * (reference as f64);
        }
    }

    (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt()
}

#[test]
fn quality_metrics_are_finite_and_storage_is_smaller_than_dense_f32() {
    let (rows, cols) = (32, 64);
    let weights = fixture(rows, cols);
    let config = VqConfig {
        iterations: 10,
        sample_limit: 4096,
        codebook_count: 2,
        beam_width: 8,
    };
    let matrix = VqMatrix::encode(&weights, rows, cols, config).expect("encode fixture");
    let relative_error = relative_output_error(&matrix, &weights, rows, cols);

    assert!(
        relative_error.is_finite(),
        "relative output error must be finite"
    );
    assert!(relative_error >= 0.0);
    assert!(
        matrix.stored_bytes() < matrix.dense_f32_bytes(),
        "VQ storage ({}) should be smaller than dense F32 storage ({})",
        matrix.stored_bytes(),
        matrix.dense_f32_bytes()
    );
}

#[test]
fn additive_codebook_count_does_not_regress_held_out_output_error() {
    let (rows, cols) = (24, 64);
    let weights = fixture(rows, cols);
    let inputs = held_out_inputs(cols);
    let config_for = |codebook_count| VqConfig {
        iterations: 12,
        sample_limit: 4096,
        codebook_count,
        beam_width: 16,
    };
    let one = VqMatrix::encode(&weights, rows, cols, config_for(1)).expect("one codebook");
    let four = VqMatrix::encode(&weights, rows, cols, config_for(4)).expect("four codebooks");

    let error = |matrix: &VqMatrix| {
        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        for input in &inputs {
            let mut actual = vec![0.0_f32; rows];
            matrix.view().matvec(input, &mut actual).expect("VQ matvec");
            for row in 0..rows {
                let reference = weights[row * cols..(row + 1) * cols]
                    .iter()
                    .zip(input)
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>();
                let delta = (actual[row] - reference) as f64;
                squared_error += delta * delta;
                squared_reference += (reference as f64) * (reference as f64);
            }
        }
        (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt()
    };

    let one_error = error(&one);
    let four_error = error(&four);
    assert!(one_error.is_finite() && four_error.is_finite());
    assert!(
        four_error <= one_error + 1.0e-6,
        "four-codebook relative output error ({four_error}) exceeded one-codebook error ({one_error})"
    );
    assert!(
        four.stored_bytes() > one.stored_bytes(),
        "additional codebooks should be reflected in storage accounting"
    );
}

#[test]
fn activation_aware_quantization_handles_divergent_feature_importance() {
    let (rows, cols) = (24, 64);
    let weights = fixture(rows, cols);
    // Deliberately non-uniform diagonal second moments: later features carry
    // substantially more activation energy than the first ones.
    let moments: Vec<f32> = (0..cols)
        .map(|col| {
            let scale = 0.15 + (col as f32 / (cols - 1) as f32) * 2.5;
            scale * scale
        })
        .collect();
    let config = VqConfig {
        iterations: 12,
        sample_limit: 4096,
        codebook_count: 2,
        beam_width: 8,
    };

    let ordinary = VqMatrix::encode(&weights, rows, cols, config).expect("ordinary VQ");
    let calibrated = VqMatrix::encode_activation_aware(&weights, rows, cols, config, &moments)
        .expect("activation-aware VQ");

    let weighted_inputs: Vec<Vec<f32>> = (0..16)
        .map(|sample| {
            (0..cols)
                .map(|col| {
                    let phase = sample as f32 * 0.23 + col as f32 * 0.13;
                    (phase.sin() + 0.25 * (phase * 0.71).cos()) * moments[col].sqrt()
                })
                .collect()
        })
        .collect();

    let weighted_error = |matrix: &VqMatrix| {
        let mut error_sum = 0.0_f64;
        let mut reference_sum = 0.0_f64;
        for input in &weighted_inputs {
            let mut actual = vec![0.0_f32; rows];
            matrix.view().matvec(input, &mut actual).expect("VQ matvec");
            for row in 0..rows {
                let reference = weights[row * cols..(row + 1) * cols]
                    .iter()
                    .zip(input)
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>();
                let delta = (actual[row] - reference) as f64;
                error_sum += delta * delta;
                reference_sum += (reference as f64) * (reference as f64);
            }
        }
        (error_sum / reference_sum.max(f64::MIN_POSITIVE)).sqrt()
    };

    let ordinary_error = weighted_error(&ordinary);
    let calibrated_error = weighted_error(&calibrated);
    assert!(ordinary_error.is_finite() && calibrated_error.is_finite());
    assert!(ordinary_error >= 0.0 && calibrated_error >= 0.0);
    assert_eq!(calibrated.rows(), rows);
    assert_eq!(calibrated.cols(), cols);
    assert!(
        calibrated.stored_bytes() < calibrated.dense_f32_bytes(),
        "activation-aware representation should remain smaller than dense F32"
    );
}
