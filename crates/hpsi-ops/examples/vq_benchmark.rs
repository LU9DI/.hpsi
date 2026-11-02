use hpsi_format::{ModelWriter, WriterStats};
use hpsi_ops::{VqConfig, VqMatrix};
use std::hint::black_box;
use std::io::Cursor;
use std::time::{Duration, Instant};

const ROWS: usize = 256;
const COLS: usize = 256;
const QUALITY_SAMPLES: usize = 16;
const LATENCY_REPETITIONS: usize = 100;

fn next_value(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let unit = ((*state >> 40) as u32) as f32 / ((1_u32 << 24) - 1) as f32;
    (unit * 2.0 - 1.0) * 0.1
}

fn make_weights() -> Vec<f32> {
    let mut state = 0x4850_5349_5651_3031_u64;
    (0..ROWS * COLS).map(|_| next_value(&mut state)).collect()
}

fn make_input(seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..COLS).map(|_| next_value(&mut state) * 10.0).collect()
}

fn dense_matvec(weights: &[f32], input: &[f32], output: &mut [f32]) {
    for row in 0..ROWS {
        let mut sum = 0.0_f32;
        for col in 0..COLS {
            sum += weights[row * COLS + col] * input[col];
        }
        output[row] = sum;
    }
}

fn percentile_ms(samples: &mut [Duration], percentile: usize) -> f64 {
    samples.sort_unstable();
    let index = ((samples.len() - 1) * percentile).div_ceil(100);
    samples[index].as_secs_f64() * 1_000.0
}

fn measure_dense(weights: &[f32], input: &[f32]) -> (f64, f64, f32) {
    let mut output = vec![0.0_f32; ROWS];
    let mut durations = Vec::with_capacity(LATENCY_REPETITIONS);
    let mut checksum = 0.0_f32;
    for _ in 0..LATENCY_REPETITIONS {
        let start = Instant::now();
        dense_matvec(weights, input, &mut output);
        durations.push(start.elapsed());
        checksum += black_box(output[0]);
    }
    let p50 = percentile_ms(&mut durations, 50);
    let p95 = percentile_ms(&mut durations, 95);
    (p50, p95, checksum)
}

fn measure_vq(matrix: &VqMatrix, input: &[f32]) -> (f64, f64, f32) {
    let view = matrix.view();
    let mut output = vec![0.0_f32; ROWS];
    let mut durations = Vec::with_capacity(LATENCY_REPETITIONS);
    let mut checksum = 0.0_f32;
    for _ in 0..LATENCY_REPETITIONS {
        let start = Instant::now();
        view.matvec(input, &mut output)
            .expect("valid VQ dimensions");
        durations.push(start.elapsed());
        checksum += black_box(output[0]);
    }
    let p50 = percentile_ms(&mut durations, 50);
    let p95 = percentile_ms(&mut durations, 95);
    (p50, p95, checksum)
}

fn evaluate_quality(weights: &[f32], matrix: &VqMatrix) -> (f64, f64) {
    let view = matrix.view();
    let mut error_squared = 0.0_f64;
    let mut reference_squared = 0.0_f64;
    let mut max_absolute_error = 0.0_f32;
    let mut reference = vec![0.0_f32; ROWS];
    let mut approximate = vec![0.0_f32; ROWS];

    for sample in 0..QUALITY_SAMPLES {
        let input = make_input(0x5155_414c_4954_5900_u64 + sample as u64);
        dense_matvec(weights, &input, &mut reference);
        view.matvec(&input, &mut approximate)
            .expect("valid VQ dimensions");
        for (expected, actual) in reference.iter().zip(&approximate) {
            let difference = expected - actual;
            error_squared += f64::from(difference) * f64::from(difference);
            reference_squared += f64::from(*expected) * f64::from(*expected);
            max_absolute_error = max_absolute_error.max(difference.abs());
        }
    }

    let relative_l2 = (error_squared / reference_squared.max(f64::MIN_POSITIVE)).sqrt();
    (relative_l2, f64::from(max_absolute_error))
}

fn write_hpsi_artifact(matrix: &VqMatrix) -> WriterStats {
    let tensors = matrix
        .to_hpsi_tensors(10, 11, 64 * 1024)
        .expect("valid tensor export");
    let mut artifact = Cursor::new(Vec::new());
    ModelWriter::write(&mut artifact, &tensors).expect("valid HPSI export")
}

fn main() {
    let weights = make_weights();
    let encode_start = Instant::now();
    let matrix = VqMatrix::encode(&weights, ROWS, COLS, VqConfig::default())
        .expect("valid synthetic matrix");
    let encode_ms = encode_start.elapsed().as_secs_f64() * 1_000.0;

    let artifact_stats = write_hpsi_artifact(&matrix);
    let quality = evaluate_quality(&weights, &matrix);
    let input = make_input(0x4245_4e43_484d_4152);
    let dense_latency = measure_dense(&weights, &input);
    let vq_latency = measure_vq(&matrix, &input);

    println!("HPSI direct VQ matvec — deterministic synthetic baseline");
    println!("matrix_shape: {ROWS}x{COLS}");
    println!("codebook_shape: 16x8 F32");
    println!("selector_bits_per_group: 4");
    println!("group_size: 8");
    println!("encode_ms: {encode_ms:.3}");
    println!("dense_f32_bytes: {}", matrix.dense_f32_bytes());
    println!("vq_representation_bytes: {}", matrix.stored_bytes());
    println!("hpsi_artifact_bytes: {}", artifact_stats.total_bytes);
    println!(
        "artifact_compression_vs_dense_f32: {:.2}x",
        matrix.dense_f32_bytes() as f64 / artifact_stats.total_bytes as f64
    );
    println!("heldout_relative_l2_error: {:.6}", quality.0);
    println!("heldout_max_absolute_error: {:.6}", quality.1);
    println!("dense_matvec_p50_ms: {:.6}", dense_latency.0);
    println!("dense_matvec_p95_ms: {:.6}", dense_latency.1);
    println!("vq_direct_matvec_p50_ms: {:.6}", vq_latency.0);
    println!("vq_direct_matvec_p95_ms: {:.6}", vq_latency.1);
    println!("checksum_guard: {:.6}", dense_latency.2 + vq_latency.2);
    println!("latency_samples_per_path: {LATENCY_REPETITIONS}");
    println!("heldout_input_vectors: {QUALITY_SAMPLES}");
}
