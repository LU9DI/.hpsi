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
        blocks: vec![PayloadBlock {
            bytes: vec![0; 16],
        }],
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
