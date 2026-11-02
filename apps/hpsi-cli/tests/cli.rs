use hpsi_format::{DType, ModelWriter, PayloadBlock, TensorInput};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn artifact_path() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "hpsi-cli-{}-{}.hpsi",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn inspect_reports_shape_and_proves_it_does_not_read_payload() {
    let path = artifact_path();
    let tensor = TensorInput {
        tensor_id: 3,
        name: "dense.weight".into(),
        dtype: DType::F32,
        dimensions: vec![2, 2],
        blocks: vec![PayloadBlock { bytes: vec![0; 16] }],
    };
    ModelWriter::write_path(&path, &[tensor]).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_hpsi"))
        .arg("inspect")
        .arg(&path)
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success());
    assert!(stdout.contains("shape=2x2"));
    assert!(stdout.contains("logical_bytes=16"));
    assert!(stdout.contains("stored_bytes=16"));
    assert!(stdout.contains("payload_bytes_read_during_inspect: 0"));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn verify_command_reports_crc32c_success() {
    let path = artifact_path();
    let tensor = TensorInput {
        tensor_id: 3,
        name: "dense.weight".into(),
        dtype: DType::U8,
        dimensions: vec![2],
        blocks: vec![PayloadBlock { bytes: vec![1, 2] }],
    };
    ModelWriter::write_path(&path, &[tensor]).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_hpsi"))
        .arg("verify")
        .arg(&path)
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success());
    assert!(stdout.contains("OK: verified 2 CRC32C records"));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn calibrate_streams_f32_activations_across_unaligned_blocks() {
    let path = artifact_path();
    let values = [1.0_f32, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let tensor = TensorInput {
        tensor_id: 7,
        name: "layer.0.input".into(),
        dtype: DType::F32,
        dimensions: vec![2, 2],
        blocks: vec![
            PayloadBlock {
                bytes: bytes[..5].to_vec(),
            },
            PayloadBlock {
                bytes: bytes[5..].to_vec(),
            },
        ],
    };
    ModelWriter::write_path(&path, &[tensor]).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_hpsi"))
        .arg("calibrate")
        .arg(&path)
        .arg("7")
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let calibration: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(calibration["format"], "hpsi-activation-calibration");
    assert_eq!(calibration["shape"], serde_json::json!([2, 2]));
    assert_eq!(calibration["sample_count"], 2);
    assert_eq!(
        calibration["activation_second_moment"],
        serde_json::json!([5.0, 10.0])
    );
    assert_eq!(calibration["source_payload_bytes_read"], 16);
    assert_eq!(calibration["source_read_operations"], 2);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn calibrate_rejects_non_finite_activation_values() {
    let path = artifact_path();
    let bytes = f32::NAN.to_le_bytes().to_vec();
    let tensor = TensorInput {
        tensor_id: 9,
        name: "layer.invalid.input".into(),
        dtype: DType::F32,
        dimensions: vec![1, 1],
        blocks: vec![PayloadBlock { bytes }],
    };
    ModelWriter::write_path(&path, &[tensor]).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_hpsi"))
        .arg("calibrate")
        .arg(&path)
        .arg("9")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("non-finite F32"));
    std::fs::remove_file(path).unwrap();
}
