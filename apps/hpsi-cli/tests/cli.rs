use hpsi_format::{DType, ModelWriter, PayloadBlock, StreamingTensorInput, TensorInput};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn artifact_path() -> PathBuf {
    std::env::temp_dir().join(format!(
        "hpsi-cli-{}-{}.hpsi",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

fn unique(stem: &str, extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "hpsi-cli-{}-{}-{}.{}",
        std::process::id(),
        stem,
        NEXT.fetch_add(1, Ordering::Relaxed),
        extension
    ))
}

fn run(args: &[&str]) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_hpsi"));
    for arg in args {
        command.arg(arg);
    }
    command.output().expect("CLI must be executable")
}

fn stdout_of(output: &std::process::Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("CLI stdout is UTF-8")
}

fn stderr_of(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

/// Orientation provenance shaped exactly like the importer's output.
fn provenance_entry(
    source_shape: [u64; 2],
    orientation: &str,
    source: &str,
    logical_shape: [u64; 2],
) -> Value {
    json!({
        "name": "dense.weight",
        "source_dtype": "F32",
        "source_shape": [source_shape[0], source_shape[1]],
        "disposition": "preserve_raw",
        "weight_orientation": orientation,
        "orientation_source": source,
        "logical_shape": [logical_shape[0], logical_shape[1]],
    })
}

/// One RAW `dense.weight` matrix carrying the given orientation provenance.
fn write_orientation_container(path: &Path, source_shape: [u64; 2], entries: Vec<Value>) {
    let payload_length = source_shape[0] * source_shape[1] * 4;
    let tensors = [StreamingTensorInput {
        tensor_id: 1,
        name: "dense.weight".into(),
        dtype: DType::F32,
        dimensions: source_shape.to_vec(),
        block_lengths: vec![payload_length],
    }];
    let payload = vec![7_u8; payload_length as usize];
    let mut file = std::fs::File::create(path).expect("create handcrafted container");
    ModelWriter::write_streaming_with_metadata(
        &mut file,
        &tensors,
        Some(json!({ "user": { "source_tensors": entries } })),
        |_, _, sink| {
            sink.write_all(&payload)
                .map_err(|_| hpsi_format::FormatError::InvalidBlock("handcrafted write failed"))?;
            Ok(payload_length)
        },
    )
    .expect("handcrafted container must write");
}

/// An architecture manifest derived from a model definition, not from the container.
fn write_architecture_manifest(path: &Path, entries: &[(&str, &str, Option<[u64; 2]>)]) {
    let tensors: Vec<Value> = entries
        .iter()
        .map(|(name, orientation, shape)| match shape {
            Some(shape) => json!({
                "name": name,
                "weight_orientation": orientation,
                "source_shape": [shape[0], shape[1]],
            }),
            None => json!({ "name": name, "weight_orientation": orientation }),
        })
        .collect();
    std::fs::write(
        path,
        json!({
            "format": "hpsi-architecture-manifest",
            "version": 1,
            "model_type": "fixture-model",
            "tensors": tensors,
        })
        .to_string(),
    )
    .expect("write architecture manifest");
}

/// A single F32 `weight` matrix in the Safetensors subset the importer reads.
fn write_f32_safetensors(path: &Path, rows: u64, cols: u64) {
    let data_bytes = rows * cols * 4;
    let header = format!(
        r#"{{"weight":{{"dtype":"F32","shape":[{rows},{cols}],"data_offsets":[0,{data_bytes}]}}}}"#
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for index in 0..rows * cols {
        bytes.extend_from_slice(&(index as f32 * 0.017_f32).sin().to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write safetensors fixture");
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

#[test]
fn inspect_reports_a_declared_orientation_as_unverified_without_architecture() {
    let path = artifact_path();
    write_orientation_container(
        &path,
        [16, 32],
        vec![provenance_entry(
            [16, 32],
            "linear_out_in",
            "legacy_default",
            [16, 32],
        )],
    );

    let output = run(&["inspect", &path.display().to_string()]);
    let stdout = stdout_of(&output);
    assert!(output.status.success(), "{}", stderr_of(&output));
    assert!(stdout.contains("orientation summary: rank2=1 confirmed=0 declared_not_verified=1"));
    assert!(stdout.contains("declared=linear_out_in"));
    assert!(stdout.contains("orientation_source=legacy_default"));
    assert!(stdout.contains("logical_shape=16x32"));
    assert!(stdout.contains("square=false"));
    assert!(stdout.contains("assurance=declared_not_verified"));
    assert!(stdout.contains("no architecture manifest supplied"));
    assert!(!stdout.contains("assurance=architecturally_confirmed"));
    assert!(stdout.contains("payload_bytes_read_during_inspect: 0"));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn inspect_confirms_orientation_only_when_an_external_manifest_agrees() {
    let path = artifact_path();
    let manifest = unique("confirm", "json");
    write_orientation_container(
        &path,
        [16, 32],
        vec![provenance_entry(
            [16, 32],
            "linear_out_in",
            "legacy_default",
            [16, 32],
        )],
    );
    write_architecture_manifest(
        &manifest,
        &[("dense.weight", "linear_out_in", Some([16, 32]))],
    );

    let output = run(&[
        "inspect",
        &path.display().to_string(),
        "--architecture",
        &manifest.display().to_string(),
    ]);
    let stdout = stdout_of(&output);
    assert!(output.status.success(), "{}", stderr_of(&output));
    assert!(stdout.contains("orientation summary: rank2=1 confirmed=1"));
    assert!(stdout.contains("assurance=architecturally_confirmed"));
    assert!(stdout.contains("architecture=linear_out_in"));
    assert!(stdout.contains("an architecture manifest was supplied"));
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(manifest).unwrap();
}

#[test]
fn a_manifest_naming_other_tensors_confirms_nothing() {
    let path = artifact_path();
    let manifest = unique("no-match", "json");
    write_orientation_container(
        &path,
        [16, 32],
        vec![provenance_entry(
            [16, 32],
            "linear_out_in",
            "legacy_default",
            [16, 32],
        )],
    );
    write_architecture_manifest(&manifest, &[("blocks.0.attn.qkv", "linear_out_in", None)]);

    let output = run(&[
        "inspect",
        &path.display().to_string(),
        "--architecture",
        &manifest.display().to_string(),
    ]);
    let stdout = stdout_of(&output);
    assert!(output.status.success(), "{}", stderr_of(&output));
    assert!(stdout.contains("orientation summary: rank2=1 confirmed=0"));
    assert!(stdout.contains("matched none of the tensor names"));
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(manifest).unwrap();
}

#[test]
fn verify_fails_when_the_manifest_disagrees_with_the_declared_orientation() {
    let path = artifact_path();
    let manifest = unique("disagree", "json");
    write_orientation_container(
        &path,
        [16, 32],
        vec![provenance_entry(
            [16, 32],
            "linear_out_in",
            "legacy_default",
            [16, 32],
        )],
    );
    write_architecture_manifest(&manifest, &[("dense.weight", "conv1d_in_out", None)]);

    let output = run(&[
        "verify",
        &path.display().to_string(),
        "--architecture",
        &manifest.display().to_string(),
    ]);
    let stdout = stdout_of(&output);
    let stderr = stderr_of(&output);
    // CRC32C still passes: integrity and architectural compatibility are separate claims.
    assert!(stdout.contains("OK: verified 2 CRC32C records"));
    assert!(stdout.contains("orientation finding: dense.weight: declared linear_out_in"));
    assert!(!output.status.success());
    assert!(stderr.contains("orientation contract violated"));
    assert!(stderr.contains("the model definition requires conv1d_in_out"));
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(manifest).unwrap();
}

#[test]
fn inspect_labels_a_self_inconsistent_logical_shape_as_a_finding() {
    let path = artifact_path();
    write_orientation_container(
        &path,
        [16, 32],
        vec![provenance_entry(
            [16, 32],
            "linear_out_in",
            "legacy_default",
            [32, 16],
        )],
    );

    let output = run(&["inspect", &path.display().to_string()]);
    let stdout = stdout_of(&output);
    assert!(output.status.success(), "{}", stderr_of(&output));
    assert!(stdout.contains("orientation finding: dense.weight: logical_shape"));
    assert!(stdout.contains("contradicts linear_out_in"));
    assert!(stdout.contains(
        "container integrity, metadata self-consistency, architectural compatibility and reconstruction correctness are separate claims"
    ));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn inspect_reports_a_square_declaration_as_a_claim_it_cannot_verify() {
    let path = artifact_path();
    write_orientation_container(
        &path,
        [16, 16],
        vec![provenance_entry(
            [16, 16],
            "linear_out_in",
            "legacy_default",
            [16, 16],
        )],
    );

    let output = run(&["inspect", &path.display().to_string()]);
    let stdout = stdout_of(&output);
    assert!(output.status.success(), "{}", stderr_of(&output));
    assert!(stdout.contains("orientation summary: rank2=1 confirmed=0"));
    assert!(stdout.contains("square_not_verifiable=1"));
    assert!(stdout.contains("square=true"));
    assert!(stdout.contains("assurance=square_not_verifiable"));
    assert!(stdout.contains("is a claim, not a confirmation"));
    // A square matrix is unverifiable, not invalid: integrity alone must still pass.
    let verified = run(&["verify", &path.display().to_string()]);
    let verified_stdout = stdout_of(&verified);
    assert!(verified.status.success(), "{}", stderr_of(&verified));
    assert!(verified_stdout.contains("OK: verified 2 CRC32C records"));
    assert!(verified_stdout.contains("assurance=square_not_verifiable"));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn an_external_manifest_is_the_only_way_to_confirm_a_square_matrix() {
    let path = artifact_path();
    let manifest = unique("square-confirm", "json");
    write_orientation_container(
        &path,
        [16, 16],
        vec![provenance_entry(
            [16, 16],
            "linear_out_in",
            "legacy_default",
            [16, 16],
        )],
    );
    write_architecture_manifest(&manifest, &[("dense.weight", "linear_out_in", None)]);

    let output = run(&[
        "inspect",
        &path.display().to_string(),
        "--architecture",
        &manifest.display().to_string(),
    ]);
    let stdout = stdout_of(&output);
    assert!(output.status.success(), "{}", stderr_of(&output));
    assert!(stdout.contains("orientation summary: rank2=1 confirmed=1"));
    assert!(stdout.contains("square_not_verifiable=0"));
    assert!(stdout.contains("assurance=architecturally_confirmed"));
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(manifest).unwrap();
}

#[test]
fn verify_still_succeeds_on_a_legacy_container_that_declares_nothing() {
    let path = artifact_path();
    let tensor = TensorInput {
        tensor_id: 3,
        name: "dense.weight".into(),
        dtype: DType::F32,
        dimensions: vec![2, 2],
        blocks: vec![PayloadBlock { bytes: vec![0; 16] }],
    };
    ModelWriter::write_path(&path, &[tensor]).unwrap();

    let output = run(&["verify", &path.display().to_string()]);
    let stdout = stdout_of(&output);
    assert!(output.status.success(), "{}", stderr_of(&output));
    assert!(stdout.contains("OK: verified 2 CRC32C records"));
    assert!(stdout.contains("legacy_undeclared=1 undeclared_tensors=1"));
    assert!(stdout.contains("declared=<missing>"));
    assert!(stdout.contains("assurance=legacy_undeclared"));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn inspect_rejects_a_manifest_that_omits_orientation_instead_of_guessing_it() {
    let path = artifact_path();
    let manifest = unique("guessed", "json");
    write_orientation_container(
        &path,
        [16, 32],
        vec![provenance_entry(
            [16, 32],
            "linear_out_in",
            "legacy_default",
            [16, 32],
        )],
    );
    std::fs::write(
        &manifest,
        json!({
            "format": "hpsi-architecture-manifest",
            "version": 1,
            "tensors": [{ "name": "c_attn" }],
        })
        .to_string(),
    )
    .unwrap();

    let output = run(&[
        "inspect",
        &path.display().to_string(),
        "--architecture",
        &manifest.display().to_string(),
    ]);
    assert!(!output.status.success());
    assert!(stderr_of(&output).contains("orientation is never inferred from a name"));
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(manifest).unwrap();
}

#[test]
fn safetensors_import_reports_the_orientation_audit_of_the_container_it_wrote() {
    let source = unique("import-audit", "safetensors");
    let destination = unique("import-audit", "hpsi");
    write_f32_safetensors(&source, 8, 16);

    let output = run(&[
        "safetensors-import",
        &source.display().to_string(),
        &destination.display().to_string(),
    ]);
    let stdout = stdout_of(&output);
    assert!(output.status.success(), "{}", stderr_of(&output));
    assert!(stdout.contains("OK: imported"));
    assert!(stdout.contains("orientation summary:"));
    assert!(stdout.contains("orientation tensor=weight"));
    assert!(stdout.contains("declared=linear_out_in"));
    assert!(stdout.contains("assurance=declared_not_verified"));
    assert!(stdout.contains("no architecture manifest supplied"));
    std::fs::remove_file(source).unwrap();
    std::fs::remove_file(destination).unwrap();
}

#[test]
fn safetensors_import_refuses_a_manifest_contradiction_before_writing() {
    let source = unique("import-refuse", "safetensors");
    let destination = unique("import-refuse", "hpsi");
    let manifest = unique("import-refuse", "json");
    write_f32_safetensors(&source, 8, 16);
    write_architecture_manifest(&manifest, &[("weight", "conv1d_in_out", Some([8, 16]))]);

    let output = run(&[
        "safetensors-import",
        &source.display().to_string(),
        &destination.display().to_string(),
        "--architecture",
        &manifest.display().to_string(),
    ]);
    assert!(!output.status.success());
    let stderr = stderr_of(&output);
    assert!(stderr.contains("import would encode linear_out_in"));
    assert!(stderr.contains("the model definition requires conv1d_in_out"));
    assert!(
        !destination.exists(),
        "a container with a knowingly wrong transposition must not be written"
    );
    std::fs::remove_file(source).unwrap();
    std::fs::remove_file(manifest).unwrap();
}
