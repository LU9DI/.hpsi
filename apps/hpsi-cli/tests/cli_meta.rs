//! Integration tests for the CLI help, version and error surface.
//!
//! These tests exercise the binary through `CARGO_BIN_EXE_hpsi`, so they check
//! the same observable behaviour a user sees.

use hpsi_format::{DType, ModelWriter, PayloadBlock, TensorInput};
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn unique(stem: &str, extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "hpsi-cli-meta-{}-{}-{}.{}",
        std::process::id(),
        stem,
        NEXT.fetch_add(1, Ordering::Relaxed),
        extension
    ))
}

fn write_valid_container() -> PathBuf {
    let path = unique("valid", "hpsi");
    let tensor = TensorInput {
        tensor_id: 1,
        name: "weights".to_owned(),
        dtype: DType::U8,
        dimensions: vec![16],
        blocks: vec![PayloadBlock { bytes: vec![0; 16] }],
    };
    ModelWriter::write_path(&path, &[tensor]).expect("CLI fixture must write");
    path
}

fn run(args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_hpsi"));
    for arg in args {
        command.arg(arg);
    }
    command.output().expect("CLI must be executable")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("CLI stdout is UTF-8")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

#[test]
fn long_help_succeeds_and_lists_real_commands() {
    let output = run(&["--help"]);
    assert!(output.status.success(), "--help must exit 0");
    let text = stdout(&output);
    assert!(text.contains("Commands:"), "help must describe commands");
    assert!(text.contains("inspect"), "help must list inspect");
    assert!(text.contains("verify"), "help must list verify");
    assert!(
        text.contains("safetensors-import"),
        "help must list safetensors-import"
    );
    assert!(
        text.contains(env!("CARGO_PKG_VERSION")),
        "help must show the package version"
    );
}

#[test]
fn short_help_succeeds() {
    let output = run(&["-h"]);
    assert!(output.status.success(), "-h must exit 0");
    assert!(stdout(&output).contains("Commands:"));
}

#[test]
fn long_version_prints_package_version() {
    let output = run(&["--version"]);
    assert!(output.status.success(), "--version must exit 0");
    assert_eq!(
        stdout(&output).trim(),
        format!("hpsi {}", env!("CARGO_PKG_VERSION"))
    );
}

#[test]
fn short_version_prints_package_version() {
    let output = run(&["-V"]);
    assert!(output.status.success(), "-V must exit 0");
    assert_eq!(
        stdout(&output).trim(),
        format!("hpsi {}", env!("CARGO_PKG_VERSION"))
    );
}

#[test]
fn no_arguments_reports_usage_and_fails() {
    let output = run(&[]);
    assert!(!output.status.success(), "no arguments must fail");
    assert!(stderr(&output).contains("usage"));
}

#[test]
fn unknown_command_reports_usage_and_fails() {
    let output = run(&["bogus"]);
    assert!(!output.status.success(), "unknown command must fail");
    assert!(stderr(&output).contains("usage"));
}

#[test]
fn inspect_keeps_working() {
    let path = write_valid_container();
    let output = run(&["inspect", path.to_str().expect("utf-8 path")]);
    assert!(output.status.success(), "inspect must exit 0");
    assert!(stdout(&output).contains("HPSI 0.1"));
}

#[test]
fn verify_keeps_working() {
    let path = write_valid_container();
    let output = run(&["verify", path.to_str().expect("utf-8 path")]);
    assert!(output.status.success(), "verify must exit 0");
    assert!(stdout(&output).contains("CRC32C"));
}

#[test]
fn truncated_container_is_rejected() {
    let valid = write_valid_container();
    let bytes = std::fs::read(&valid).expect("fixture readable");
    let truncated = unique("truncated", "hpsi");
    std::fs::write(&truncated, &bytes[..bytes.len().min(64)]).expect("write truncated fixture");
    let output = run(&["inspect", truncated.to_str().expect("utf-8 path")]);
    assert!(!output.status.success(), "truncated container must fail");
    assert!(!stderr(&output).is_empty());
}
