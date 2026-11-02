use hpsi_format::{FileSource, MemorySource, ModelReader, ModelWriter, ReaderLimits, WriterStats};
use hpsi_ops::{HpsiVqExecutor, HpsiVqLayout, VqConfig, VqMatrix};
use std::fs::{OpenOptions, remove_file};
use std::hint::black_box;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
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

fn benchmark_shape_weights(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..rows * cols).map(|_| next_value(&mut state)).collect()
}

fn dense_matvec_shape(weights: &[f32], cols: usize, input: &[f32], output: &mut [f32]) {
    for (row, output_value) in output.iter_mut().enumerate() {
        let mut sum = 0.0_f32;
        for col in 0..cols {
            sum += weights[row * cols + col] * input[col];
        }
        *output_value = sum;
    }
}

fn measure_dense_shape(weights: &[f32], rows: usize, cols: usize, input: &[f32]) -> (f64, f64) {
    let mut output = vec![0.0_f32; rows];
    let mut durations = Vec::with_capacity(LATENCY_REPETITIONS);
    for _ in 0..LATENCY_REPETITIONS {
        let start = Instant::now();
        dense_matvec_shape(weights, cols, input, &mut output);
        durations.push(start.elapsed());
        black_box(output[0]);
    }
    (
        percentile_ms(&mut durations, 50),
        percentile_ms(&mut durations, 95),
    )
}

fn measure_vq_shape(matrix: &VqMatrix, input: &[f32]) -> (f64, f64) {
    let mut output = vec![0.0_f32; matrix.rows()];
    let mut durations = Vec::with_capacity(LATENCY_REPETITIONS);
    for _ in 0..LATENCY_REPETITIONS {
        let start = Instant::now();
        matrix
            .view()
            .matvec(input, &mut output)
            .expect("valid VQ shape");
        durations.push(start.elapsed());
        black_box(output[0]);
    }
    (
        percentile_ms(&mut durations, 50),
        percentile_ms(&mut durations, 95),
    )
}

fn benchmark_row_scaling() {
    const COLS: usize = 256;
    println!("row_scaling_cols: {COLS}");
    for rows in [16, 64, 256, 1024] {
        let weights = benchmark_shape_weights(rows, COLS, 0x5343_414c_4500_0000_u64 + rows as u64);
        let input = make_input(0x5343_414c_494e_5055_u64 + rows as u64);
        let dense = measure_dense_shape(&weights, rows, COLS, &input);
        println!("scaling_rows_{rows}_dense_p50_ms: {:.6}", dense.0);
        println!("scaling_rows_{rows}_dense_p95_ms: {:.6}", dense.1);

        for codebook_count in [1, 4] {
            let matrix = VqMatrix::encode(
                &weights,
                rows,
                COLS,
                VqConfig {
                    codebook_count,
                    ..VqConfig::default()
                },
            )
            .expect("valid row-scaling matrix");
            let vq = measure_vq_shape(&matrix, &input);
            println!("scaling_rows_{rows}_vq{codebook_count}_p50_ms: {:.6}", vq.0);
            println!("scaling_rows_{rows}_vq{codebook_count}_p95_ms: {:.6}", vq.1);
        }
    }
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

fn make_hpsi_file_reader(
    matrix: &VqMatrix,
    block_rows: usize,
) -> (ModelReader, WriterStats, PathBuf) {
    static NEXT_ARTIFACT_ID: AtomicU64 = AtomicU64::new(0);
    let artifact_id = NEXT_ARTIFACT_ID.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "hpsi-vq-benchmark-{}-{}-{}.hpsi",
        std::process::id(),
        matrix.codebook_count(),
        artifact_id
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("create temporary HPSI artifact");
    let tensors = matrix
        .to_hpsi_tensors(10, 11, block_rows)
        .expect("valid file-backed tensor export");
    let stats = ModelWriter::write(&mut file, &tensors).expect("write file-backed HPSI artifact");
    file.sync_all().expect("flush file-backed HPSI artifact");
    drop(file);

    let reader = ModelReader::open(
        Arc::new(FileSource::open(&path).expect("open file-backed HPSI source")),
        &ReaderLimits::default(),
    )
    .expect("open file-backed HPSI model");
    (reader, stats, path)
}

fn make_hpsi_reader(matrix: &VqMatrix, block_rows: usize) -> (ModelReader, WriterStats) {
    let tensors = matrix
        .to_hpsi_tensors(10, 11, block_rows)
        .expect("valid tensor export");
    let mut artifact = Cursor::new(Vec::new());
    let stats = ModelWriter::write(&mut artifact, &tensors).expect("valid HPSI export");
    let reader = ModelReader::open(
        Arc::new(MemorySource::new(artifact.into_inner())),
        &ReaderLimits::default(),
    )
    .expect("valid HPSI artifact");
    (reader, stats)
}

fn measure_hpsi_row_window(
    reader: &ModelReader,
    matrix: &VqMatrix,
    input: &[f32],
) -> (f64, f64, u64, f64) {
    const FIRST_ROW: usize = 64;
    const WINDOW_ROWS: usize = 32;
    let layout = HpsiVqLayout {
        codebook_tensor_id: 10,
        indices_tensor_id: 11,
        rows: matrix.rows(),
        cols: matrix.cols(),
        codebook_count: matrix.codebook_count(),
    };
    let mut expected = vec![0.0_f32; matrix.rows()];
    matrix
        .view()
        .matvec(input, &mut expected)
        .expect("valid VQ view");
    let init_start = Instant::now();
    let executor = HpsiVqExecutor::new(reader, layout).expect("valid prepared HPSI executor");
    let init_ms = init_start.elapsed().as_secs_f64() * 1_000.0;
    let mut output = vec![0.0_f32; WINDOW_ROWS];
    executor
        .matvec_rows(input, FIRST_ROW, &mut output)
        .expect("valid HPSI row window");
    assert_eq!(output, expected[FIRST_ROW..FIRST_ROW + WINDOW_ROWS]);

    let bytes_before = reader.stats().payload_bytes_read;
    let mut durations = Vec::with_capacity(LATENCY_REPETITIONS);
    for _ in 0..LATENCY_REPETITIONS {
        let start = Instant::now();
        executor
            .matvec_rows(input, FIRST_ROW, &mut output)
            .expect("valid HPSI row window");
        durations.push(start.elapsed());
    }
    let bytes_read = reader.stats().payload_bytes_read - bytes_before;
    let p50 = percentile_ms(&mut durations, 50);
    let p95 = percentile_ms(&mut durations, 95);
    (p50, p95, bytes_read / LATENCY_REPETITIONS as u64, init_ms)
}

fn benchmark_beam_widths(weights: &[f32]) {
    let input = make_input(0x4245_414d_5749_4454);
    println!("beam_width_sweep_codebooks: 4");
    for beam_width in [1, 2, 4, 8, 16] {
        let start = Instant::now();
        let matrix = VqMatrix::encode(
            weights,
            ROWS,
            COLS,
            VqConfig {
                codebook_count: 4,
                beam_width,
                ..VqConfig::default()
            },
        )
        .expect("valid beam-width sweep matrix");
        let encode_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let quality = evaluate_quality(weights, &matrix);
        let latency = measure_vq(&matrix, &input);
        println!("beam_{beam_width}_encode_ms: {encode_ms:.3}");
        println!(
            "beam_{beam_width}_artifact_bytes: {}",
            matrix.stored_bytes()
        );
        println!("beam_{beam_width}_heldout_relative_l2: {:.6}", quality.0);
        println!("beam_{beam_width}_matvec_p50_ms: {:.6}", latency.0);
    }
}

fn main() {
    let weights = make_weights();
    let encode_start = Instant::now();
    let matrix4 = VqMatrix::encode(&weights, ROWS, COLS, VqConfig::default())
        .expect("valid synthetic matrix");
    let encode4_ms = encode_start.elapsed().as_secs_f64() * 1_000.0;

    let encode_additive_start = Instant::now();
    let matrix8 = VqMatrix::encode(
        &weights,
        ROWS,
        COLS,
        VqConfig {
            codebook_count: 2,
            ..VqConfig::default()
        },
    )
    .expect("valid additive synthetic matrix");
    let encode8_ms = encode_additive_start.elapsed().as_secs_f64() * 1_000.0;

    let encode_three_start = Instant::now();
    let matrix12 = VqMatrix::encode(
        &weights,
        ROWS,
        COLS,
        VqConfig {
            codebook_count: 3,
            ..VqConfig::default()
        },
    )
    .expect("valid three-stage synthetic matrix");
    let encode12_ms = encode_three_start.elapsed().as_secs_f64() * 1_000.0;

    let encode_four_start = Instant::now();
    let matrix16 = VqMatrix::encode(
        &weights,
        ROWS,
        COLS,
        VqConfig {
            codebook_count: 4,
            ..VqConfig::default()
        },
    )
    .expect("valid four-stage synthetic matrix");
    let encode16_ms = encode_four_start.elapsed().as_secs_f64() * 1_000.0;

    // Eight independently addressable 32-row index blocks make the
    // selected-row benchmark exercise the actual HPSI block reader.
    let (reader4, artifact4) = make_hpsi_reader(&matrix4, 32);
    let (reader8, artifact8) = make_hpsi_reader(&matrix8, 32);
    let (reader12, artifact12) = make_hpsi_reader(&matrix12, 32);
    let (reader16, artifact16) = make_hpsi_reader(&matrix16, 32);
    let quality4 = evaluate_quality(&weights, &matrix4);
    let quality8 = evaluate_quality(&weights, &matrix8);
    let quality12 = evaluate_quality(&weights, &matrix12);
    let quality16 = evaluate_quality(&weights, &matrix16);
    let input = make_input(0x4245_4e43_484d_4152);
    let dense_latency = measure_dense(&weights, &input);
    let vq4_latency = measure_vq(&matrix4, &input);
    let vq8_latency = measure_vq(&matrix8, &input);
    let vq12_latency = measure_vq(&matrix12, &input);
    let vq16_latency = measure_vq(&matrix16, &input);
    let hpsi4_latency = measure_hpsi_row_window(&reader4, &matrix4, &input);
    let hpsi8_latency = measure_hpsi_row_window(&reader8, &matrix8, &input);
    let hpsi12_latency = measure_hpsi_row_window(&reader12, &matrix12, &input);
    let hpsi16_latency = measure_hpsi_row_window(&reader16, &matrix16, &input);

    // FileSource uses OS random-access reads. These timings are explicitly
    // warm-cache measurements; the benchmark does not drop the OS page cache.
    let (file_reader4, _, file_path4) = make_hpsi_file_reader(&matrix4, 32);
    let (file_reader8, _, file_path8) = make_hpsi_file_reader(&matrix8, 32);
    let (file_reader12, _, file_path12) = make_hpsi_file_reader(&matrix12, 32);
    let (file_reader16, _, file_path16) = make_hpsi_file_reader(&matrix16, 32);
    let file_hpsi4_latency = measure_hpsi_row_window(&file_reader4, &matrix4, &input);
    let file_hpsi8_latency = measure_hpsi_row_window(&file_reader8, &matrix8, &input);
    let file_hpsi12_latency = measure_hpsi_row_window(&file_reader12, &matrix12, &input);
    let file_hpsi16_latency = measure_hpsi_row_window(&file_reader16, &matrix16, &input);

    println!("HPSI direct VQ matvec — deterministic synthetic baseline");
    println!("matrix_shape: {ROWS}x{COLS}");
    println!("group_size: 8");
    println!("dense_f32_bytes: {}", matrix4.dense_f32_bytes());
    println!("dense_matvec_p50_ms: {:.6}", dense_latency.0);
    println!("dense_matvec_p95_ms: {:.6}", dense_latency.1);

    println!("single_codebook_count: 1");
    println!("single_selector_bits_per_group: 4");
    println!("single_encode_ms: {encode4_ms:.3}");
    println!("single_representation_bytes: {}", matrix4.stored_bytes());
    println!("single_hpsi_artifact_bytes: {}", artifact4.total_bytes);
    println!(
        "single_artifact_compression_vs_dense_f32: {:.2}x",
        matrix4.dense_f32_bytes() as f64 / artifact4.total_bytes as f64
    );
    println!("single_heldout_relative_l2_error: {:.6}", quality4.0);
    println!("single_heldout_max_absolute_error: {:.6}", quality4.1);
    println!("single_direct_matvec_p50_ms: {:.6}", vq4_latency.0);
    println!("single_direct_matvec_p95_ms: {:.6}", vq4_latency.1);
    println!("single_hpsi_32_row_window_p50_ms: {:.6}", hpsi4_latency.0);
    println!("single_hpsi_32_row_window_p95_ms: {:.6}", hpsi4_latency.1);
    println!("single_hpsi_executor_init_ms: {:.6}", hpsi4_latency.3);
    println!(
        "single_hpsi_payload_bytes_per_32_row_window: {}",
        hpsi4_latency.2
    );

    println!("additive_codebook_count: 2");
    println!("additive_selector_bits_per_group: 8");
    println!("additive_encode_ms: {encode8_ms:.3}");
    println!("additive_representation_bytes: {}", matrix8.stored_bytes());
    println!("additive_hpsi_artifact_bytes: {}", artifact8.total_bytes);
    println!(
        "additive_artifact_compression_vs_dense_f32: {:.2}x",
        matrix8.dense_f32_bytes() as f64 / artifact8.total_bytes as f64
    );
    println!("additive_heldout_relative_l2_error: {:.6}", quality8.0);
    println!("additive_heldout_max_absolute_error: {:.6}", quality8.1);
    println!("additive_direct_matvec_p50_ms: {:.6}", vq8_latency.0);
    println!("additive_direct_matvec_p95_ms: {:.6}", vq8_latency.1);
    println!("additive_hpsi_32_row_window_p50_ms: {:.6}", hpsi8_latency.0);
    println!("additive_hpsi_32_row_window_p95_ms: {:.6}", hpsi8_latency.1);
    println!("additive_hpsi_executor_init_ms: {:.6}", hpsi8_latency.3);
    println!(
        "additive_hpsi_payload_bytes_per_32_row_window: {}",
        hpsi8_latency.2
    );

    println!("three_codebook_count: 3");
    println!("three_selector_bits_per_group: 12");
    println!("three_encode_ms: {encode12_ms:.3}");
    println!("three_representation_bytes: {}", matrix12.stored_bytes());
    println!("three_hpsi_artifact_bytes: {}", artifact12.total_bytes);
    println!(
        "three_artifact_compression_vs_dense_f32: {:.2}x",
        matrix12.dense_f32_bytes() as f64 / artifact12.total_bytes as f64
    );
    println!("three_heldout_relative_l2_error: {:.6}", quality12.0);
    println!("three_heldout_max_absolute_error: {:.6}", quality12.1);
    println!("three_direct_matvec_p50_ms: {:.6}", vq12_latency.0);
    println!("three_direct_matvec_p95_ms: {:.6}", vq12_latency.1);
    println!("three_hpsi_32_row_window_p50_ms: {:.6}", hpsi12_latency.0);
    println!("three_hpsi_32_row_window_p95_ms: {:.6}", hpsi12_latency.1);
    println!("three_hpsi_executor_init_ms: {:.6}", hpsi12_latency.3);
    println!(
        "three_hpsi_payload_bytes_per_32_row_window: {}",
        hpsi12_latency.2
    );

    println!("four_codebook_count: 4");
    println!("four_selector_bits_per_group: 16");
    println!("four_encode_ms: {encode16_ms:.3}");
    println!("four_representation_bytes: {}", matrix16.stored_bytes());
    println!("four_hpsi_artifact_bytes: {}", artifact16.total_bytes);
    println!(
        "four_artifact_compression_vs_dense_f32: {:.2}x",
        matrix16.dense_f32_bytes() as f64 / artifact16.total_bytes as f64
    );
    println!("four_heldout_relative_l2_error: {:.6}", quality16.0);
    println!("four_heldout_max_absolute_error: {:.6}", quality16.1);
    println!("four_direct_matvec_p50_ms: {:.6}", vq16_latency.0);
    println!("four_direct_matvec_p95_ms: {:.6}", vq16_latency.1);
    println!("four_hpsi_32_row_window_p50_ms: {:.6}", hpsi16_latency.0);
    println!("four_hpsi_32_row_window_p95_ms: {:.6}", hpsi16_latency.1);
    println!("four_hpsi_executor_init_ms: {:.6}", hpsi16_latency.3);
    println!(
        "four_hpsi_payload_bytes_per_32_row_window: {}",
        hpsi16_latency.2
    );

    println!(
        "single_file_hpsi_row_window_p50_ms: {:.6}",
        file_hpsi4_latency.0
    );
    println!(
        "single_file_hpsi_row_window_p95_ms: {:.6}",
        file_hpsi4_latency.1
    );
    println!(
        "single_file_hpsi_executor_init_ms: {:.6}",
        file_hpsi4_latency.3
    );
    println!(
        "single_file_hpsi_index_bytes_per_window: {}",
        file_hpsi4_latency.2
    );
    println!(
        "additive_file_hpsi_row_window_p50_ms: {:.6}",
        file_hpsi8_latency.0
    );
    println!(
        "additive_file_hpsi_row_window_p95_ms: {:.6}",
        file_hpsi8_latency.1
    );
    println!(
        "additive_file_hpsi_executor_init_ms: {:.6}",
        file_hpsi8_latency.3
    );
    println!(
        "additive_file_hpsi_index_bytes_per_window: {}",
        file_hpsi8_latency.2
    );
    println!(
        "three_file_hpsi_row_window_p50_ms: {:.6}",
        file_hpsi12_latency.0
    );
    println!(
        "three_file_hpsi_row_window_p95_ms: {:.6}",
        file_hpsi12_latency.1
    );
    println!(
        "three_file_hpsi_executor_init_ms: {:.6}",
        file_hpsi12_latency.3
    );
    println!(
        "three_file_hpsi_index_bytes_per_window: {}",
        file_hpsi12_latency.2
    );
    println!(
        "four_file_hpsi_row_window_p50_ms: {:.6}",
        file_hpsi16_latency.0
    );
    println!(
        "four_file_hpsi_row_window_p95_ms: {:.6}",
        file_hpsi16_latency.1
    );
    println!(
        "four_file_hpsi_executor_init_ms: {:.6}",
        file_hpsi16_latency.3
    );
    println!(
        "four_file_hpsi_index_bytes_per_window: {}",
        file_hpsi16_latency.2
    );

    drop((file_reader4, file_reader8, file_reader12, file_reader16));
    for path in [file_path4, file_path8, file_path12, file_path16] {
        remove_file(path).expect("remove temporary HPSI benchmark artifact");
    }

    println!(
        "checksum_guard: {:.6}",
        dense_latency.2 + vq4_latency.2 + vq8_latency.2 + vq12_latency.2 + vq16_latency.2
    );
    println!("latency_samples_per_path: {LATENCY_REPETITIONS}");
    println!("heldout_input_vectors: {QUALITY_SAMPLES}");
    benchmark_row_scaling();
    benchmark_beam_widths(&weights);
}
