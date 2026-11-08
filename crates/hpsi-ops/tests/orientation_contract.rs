//! Orientation-contract closure tests (Phase 4).
//!
//! Every test here drives the public API the way a consumer does — import, write,
//! reopen, audit, resolve a layout, decode — and then asserts on observable
//! behaviour. Two rules shape the expectations:
//!
//! * A container alone can only falsify a claim. The audit is therefore expected
//!   to *accept* what it cannot disprove and label it unverified, and to *reject*
//!   what contradicts the stored bytes, the calibration, or a supplied model
//!   definition.
//! * Architectural correctness is decided against references that are not the
//!   code under test: the original Safetensors fixture, an explicit architecture
//!   manifest, or a hand-written provenance document. Nothing here asks the
//!   audit to grade its own reasoning.
//!
//! No test relies on tensor-name heuristics, and none asserts that a square
//! matrix orientation is verified without external information.

use hpsi_format::{
    DType, FormatError, MemorySource, ModelReader, ModelWriter, ReaderLimits, StreamingTensorInput,
};
use hpsi_ops::{
    ActivationCalibration, ArchitectureFacts, CODEBOOK_SIZE, GROUP_SIZE, HpsiVqExecutor,
    HpsiVqLayout, ModelImportPlan, OrientationAssurance, RoleAssurance, SafeTensorFile, TensorRole,
    UnsupportedTensorPolicy, VqConfig, WeightOrientation, audit_container, resolve_vq_layout,
    write_model_import_to_hpsi, write_model_import_to_hpsi_calibrated,
};
use serde_json::{Value, json};
use std::io::Cursor;
use std::sync::Arc;

const TEST_CONFIG: VqConfig = VqConfig {
    iterations: 3,
    sample_limit: 64,
    codebook_count: 1,
    beam_width: 4,
};

// ---------------------------------------------------------------- fixtures

fn pattern(rows: usize, cols: usize, salt: f32) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| ((index % 11) as f32 * 0.13 + (index / 11) as f32 * 0.07 + salt) - 0.4)
        .collect()
}

fn transpose(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = values[row * cols + col];
        }
    }
    output
}

/// Relative L2 distance between a reference matrix (row-major) and a decoded one.
fn relative_l2(reference: &[f32], decoded: &[f32]) -> f64 {
    assert_eq!(reference.len(), decoded.len(), "matrix shape mismatch");
    let mut difference = 0.0_f64;
    let mut norm = 0.0_f64;
    for (expected, actual) in reference.iter().zip(decoded.iter()) {
        let delta = f64::from(*actual) - f64::from(*expected);
        difference += delta * delta;
        norm += f64::from(*expected) * f64::from(*expected);
    }
    assert!(norm > 0.0, "reference matrix must be non-zero");
    difference.sqrt() / norm.sqrt()
}

fn safetensors(entries: &[(&str, Vec<u64>, Vec<f32>)]) -> SafeTensorFile {
    let mut header_entries = Vec::new();
    let mut payload = Vec::new();
    for (name, shape, values) in entries {
        let start = payload.len();
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let end = payload.len();
        header_entries.push(format!(
            r#""{}":{{"dtype":"F32","shape":[{}],"data_offsets":[{},{}]}}"#,
            name,
            shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(","),
            start,
            end
        ));
    }
    let header = format!("{{{}}}", header_entries.join(","));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&payload);
    SafeTensorFile::open(
        Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))),
        64 * 1024 * 1024,
    )
    .expect("fixture Safetensors must parse")
}

fn calibration(
    layer_name: &str,
    module_type: Option<&str>,
    features: usize,
) -> ActivationCalibration {
    ActivationCalibration {
        layer_name: layer_name.to_owned(),
        sample_count: 8,
        features,
        activation_second_moment: vec![1.0; features],
        model_type: "fixture-transformer".to_owned(),
        model_revision: None,
        dataset_sha256: "a".repeat(64),
        module_type: module_type.map(str::to_owned),
    }
}

/// Import with capture provenance so the orientation is declared by calibration.
fn import_calibrated(
    entries: &[(&str, Vec<u64>, Vec<f32>)],
    cals: &[ActivationCalibration],
) -> Vec<u8> {
    let source = safetensors(entries);
    let plan = ModelImportPlan::build_with_provenance(
        &source,
        cals,
        UnsupportedTensorPolicy::AllowRawPreservation,
    )
    .expect("calibrated plan must build");
    let mut output = Cursor::new(Vec::new());
    write_model_import_to_hpsi_calibrated(&source, &plan, &mut output, TEST_CONFIG, 4)
        .expect("calibrated import must write");
    output.into_inner()
}

/// Import without any capture: the importer falls back to the legacy default.
fn import_uncalibrated(entries: &[(&str, Vec<u64>, Vec<f32>)]) -> Vec<u8> {
    let source = safetensors(entries);
    let plan = ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation)
        .expect("plan must build");
    let mut output = Cursor::new(Vec::new());
    write_model_import_to_hpsi(&source, &plan, &mut output, TEST_CONFIG, 4)
        .expect("import must write");
    output.into_inner()
}

fn open(bytes: Vec<u8>) -> ModelReader {
    ModelReader::open(
        Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))),
        &ReaderLimits::default(),
    )
    .expect("written container must reopen")
}

fn manifest(entries: &[(&str, &str)]) -> ArchitectureFacts {
    let tensors = entries
        .iter()
        .map(|(name, orientation)| json!({"name": name, "weight_orientation": orientation}))
        .collect::<Vec<_>>();
    ArchitectureFacts::from_json(
        &json!({
            "format": "hpsi-architecture-manifest",
            "version": 1,
            "model_type": "fixture-transformer",
            "tensors": tensors,
        })
        .to_string(),
    )
    .expect("manifest must parse")
}

fn descriptor_by_name(reader: &ModelReader, name: &str) -> hpsi_format::TensorDescriptor {
    reader
        .tensors()
        .iter()
        .find(|descriptor| {
            reader
                .metadata
                .tensor_names
                .get(descriptor.name_ref as usize)
                .is_some_and(|candidate| candidate == name)
        })
        .unwrap_or_else(|| panic!("{name} must exist in the container"))
        .clone()
}

/// Decode a stored VQ pair with the production executor and basis vectors.
///
/// The companion names are the importer's documented naming convention; the
/// audit resolves the same pair from metadata, and this helper reads it the way
/// an external consumer would.
fn decode_pair(reader: &ModelReader, source_name: &str, rows: usize, cols: usize) -> Vec<f32> {
    let codebook = descriptor_by_name(reader, &format!("{source_name}.hpsi_vq.codebook"));
    let indices = descriptor_by_name(reader, &format!("{source_name}.hpsi_vq.indices"));
    let layout = HpsiVqLayout {
        codebook_tensor_id: codebook.tensor_id,
        indices_tensor_id: indices.tensor_id,
        rows,
        cols,
        codebook_count: usize::try_from(codebook.dimensions[0]).expect("codebook count fits"),
    };
    let executor = HpsiVqExecutor::new(reader, layout).expect("executor must build");
    let mut dense = vec![0.0_f32; rows * cols];
    let mut input = vec![0.0_f32; cols];
    let mut column = vec![0.0_f32; rows];
    for col in 0..cols {
        input.iter_mut().for_each(|value| *value = 0.0);
        input[col] = 1.0;
        executor
            .matvec_rows(&input, 0, &mut column)
            .expect("matvec must succeed");
        for row in 0..rows {
            dense[row * cols + col] = column[row];
        }
    }
    dense
}

// ------------------------------------------------------- metadata crafting

/// Castagnoli CRC32C, written here so a tampering test does not depend on the
/// private helper inside `hpsi-format`. The assertions that require integrity to
/// pass would fail loudly if this disagreed with the format.
fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0x82f6_3b78
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn occurrences(haystack: &[u8], needle: &[u8]) -> usize {
    let mut count = 0;
    let mut cursor = 0;
    while cursor + needle.len() <= haystack.len() {
        if haystack[cursor..].starts_with(needle) {
            count += 1;
            cursor += needle.len();
        } else {
            cursor += 1;
        }
    }
    count
}

/// Replace equal-length bytes inside the metadata region and refresh its CRC32C
/// record, producing an artifact whose integrity check still passes.
///
/// Only same-length edits are allowed on purpose: changing the metadata length
/// would move every region and the container would stop parsing for a reason
/// unrelated to the orientation contract.
fn patch_metadata(bytes: &mut [u8], find: &str, replace: &str) -> usize {
    let (find, replace) = (find.as_bytes(), replace.as_bytes());
    assert_eq!(find.len(), replace.len(), "tampering must preserve length");
    let layout = ModelReader::open(
        Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes.to_owned()))),
        &ReaderLimits::default(),
    )
    .expect("container must parse before tampering");
    let metadata_start = usize::try_from(layout.header.metadata.offset).expect("offset fits");
    let metadata_end =
        metadata_start + usize::try_from(layout.header.metadata.length).expect("length fits");
    let integrity_start =
        usize::try_from(layout.header.integrity_directory.offset).expect("offset fits");
    let record_count = usize::try_from(layout.header.integrity_count).expect("count fits");
    drop(layout);

    let region = &mut bytes[metadata_start..metadata_end];
    assert!(
        occurrences(region, find) > 0,
        "tampering target not present in metadata: {find:?}"
    );
    let mut cursor = 0;
    while cursor + find.len() <= region.len() {
        if region[cursor..].starts_with(find) {
            region[cursor..cursor + find.len()].copy_from_slice(replace);
            cursor += find.len();
        } else {
            cursor += 1;
        }
    }
    let replaced = occurrences(region, replace);

    let record = (0..record_count)
        .map(|index| integrity_start + index * 32)
        .find(|at| u32::from_le_bytes(bytes[*at..*at + 4].try_into().expect("fixed slice")) == 1)
        .expect("a metadata integrity record must exist");
    let checksum = crc32c(&bytes[metadata_start..metadata_end]);
    bytes[record + 12..record + 16].copy_from_slice(&checksum.to_le_bytes());
    replaced
}

/// Write a container whose provenance is exactly the supplied entries.
///
/// These are hand-authored documents, not importer output, which is what makes
/// them usable as "metadata that a producer got wrong" cases.
fn handcrafted_with(tensors: &[(&str, DType, Vec<u64>)], entries: Vec<Value>) -> Vec<u8> {
    let inputs: Vec<StreamingTensorInput> = tensors
        .iter()
        .enumerate()
        .map(|(index, (name, dtype, shape))| StreamingTensorInput {
            tensor_id: index as u32 + 1,
            name: (*name).to_owned(),
            dtype: *dtype,
            dimensions: shape.clone(),
            block_lengths: vec![shape.iter().product::<u64>() * item_bytes(*dtype)],
        })
        .collect();
    let payloads: Vec<Vec<u8>> = inputs
        .iter()
        .map(|tensor| vec![0_u8; tensor.block_lengths[0] as usize])
        .collect();
    let mut output = Cursor::new(Vec::new());
    ModelWriter::write_streaming_with_metadata(
        &mut output,
        &inputs,
        Some(json!({"user": {"source_tensors": entries}})),
        |tensor_index, _block_index, sink| {
            let payload = payloads
                .get(tensor_index)
                .ok_or(FormatError::InvalidBlock("unknown handcrafted tensor"))?;
            sink.write_all(payload)
                .map_err(|_| FormatError::InvalidBlock("handcrafted write failed"))?;
            Ok(payload.len() as u64)
        },
    )
    .expect("handcrafted container must write");
    output.into_inner()
}

fn item_bytes(dtype: DType) -> u64 {
    match dtype {
        DType::F32 | DType::F64 | DType::I32 | DType::I64 => 4,
        DType::F16 | DType::Bf16 | DType::I16 => 2,
        _ => 1,
    }
}

fn handcrafted(tensors: &[(&str, Vec<u64>)], entries: Vec<Value>) -> Vec<u8> {
    let typed: Vec<(&str, DType, Vec<u64>)> = tensors
        .iter()
        .map(|(name, shape)| (*name, DType::F32, shape.clone()))
        .collect();
    handcrafted_with(&typed, entries)
}

/// The VQ companions of a `quantize_vq` entry for a `[rows, cols]` logical
/// matrix: one codebook and the packed selector bytes that layout implies
/// (`cols / GROUP_SIZE` selectors per row, two selectors per byte).
fn vq_pair(name: &str, logical: [u64; 2]) -> Vec<(String, DType, Vec<u64>)> {
    let selectors_per_row = logical[1] / GROUP_SIZE as u64;
    let bytes_per_row = selectors_per_row / 2;
    vec![
        (
            format!("{name}.hpsi_vq.codebook"),
            DType::F32,
            vec![1, CODEBOOK_SIZE as u64, GROUP_SIZE as u64],
        ),
        (
            format!("{name}.hpsi_vq.indices"),
            DType::Opaque,
            vec![logical[0] * bytes_per_row],
        ),
    ]
}

/// A well-formed F32 `[16, 32]` RAW entry that every case can mutate.
fn raw_entry(overrides: Value) -> Value {
    let mut entry = json!({
        "name": "dense.weight",
        "source_dtype": "F32",
        "source_shape": [16, 32],
        "disposition": "preserve_raw",
        "weight_orientation": "linear_out_in",
        "orientation_source": "legacy_default",
        "logical_shape": [16, 32],
    });
    apply_overrides(&mut entry, overrides);
    entry
}

/// The same entry as a quantized tensor backed by `vq_pair` companions.
fn quantized_entry(overrides: Value) -> Value {
    let mut entry = json!({
        "name": "dense.weight",
        "source_dtype": "F32",
        "source_shape": [16, 32],
        "disposition": "quantize_vq",
        "weight_orientation": "linear_out_in",
        "orientation_source": "legacy_default",
        "logical_shape": [16, 32],
    });
    apply_overrides(&mut entry, overrides);
    entry
}

fn apply_overrides(entry: &mut Value, overrides: Value) {
    for (key, value) in overrides.as_object().expect("overrides must be an object") {
        if value.is_null() {
            entry.as_object_mut().expect("entry").remove(key);
        } else {
            entry[key] = value.clone();
        }
    }
}

fn handcrafted_audit(entries: Vec<Value>) -> hpsi_ops::OrientationAudit {
    let reader = open(handcrafted(&[("dense.weight", vec![16, 32])], entries));
    audit_container(&reader, None)
}

fn quantized_audit(entries: Vec<Value>) -> hpsi_ops::OrientationAudit {
    let pair = vq_pair("dense.weight", [16, 32]);
    let tensors: Vec<(&str, DType, Vec<u64>)> = pair
        .iter()
        .map(|(name, dtype, dims)| (name.as_str(), *dtype, dims.clone()))
        .collect();
    let reader = open(handcrafted_with(&tensors, entries));
    audit_container(&reader, None)
}

// ------------------------------------------------------- Linear, rectangular

#[test]
fn linear_rectangular_orientation_is_confirmed_by_an_external_definition() {
    let source = pattern(24, 32, 0.5);
    let bytes = import_calibrated(
        &[("mlp.fc.weight", vec![24, 32], source.clone())],
        &[calibration("mlp.fc.weight", Some("linear"), 32)],
    );
    let reader = open(bytes);
    let audit = audit_container(
        &reader,
        Some(&manifest(&[("mlp.fc.weight", "linear_out_in")])),
    );

    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    assert_eq!(audit.confirmed(), 1);
    let facts = audit.facts("mlp.fc.weight").expect("entry must be audited");
    assert_eq!(
        facts.assurance,
        OrientationAssurance::ArchitecturallyConfirmed
    );
    assert_eq!(facts.logical_shape, Some([24, 32]));
    assert!(!facts.square);

    // The consumer that trusts metadata alone decodes the architectural matrix.
    let resolved = resolve_vq_layout(
        &reader,
        "mlp.fc.weight",
        Some(&manifest(&[("mlp.fc.weight", "linear_out_in")])),
        false,
    )
    .expect("confirmed orientation must resolve");
    let decoded = decode_pair(
        &reader,
        "mlp.fc.weight",
        resolved.layout.rows,
        resolved.layout.cols,
    );
    assert!(
        relative_l2(&source, &decoded) < 0.2,
        "decoded matrix must approximate the source"
    );
}

#[test]
fn linear_rectangular_without_external_information_stays_unverified() {
    let bytes = import_calibrated(
        &[("mlp.fc.weight", vec![24, 32], pattern(24, 32, 0.5))],
        &[calibration("mlp.fc.weight", Some("linear"), 32)],
    );
    let reader = open(bytes);
    let audit = audit_container(&reader, None);
    assert!(audit.invalid.is_empty());
    assert_eq!(audit.confirmed(), 0);
    assert_eq!(
        audit.facts("mlp.fc.weight").expect("entry").assurance,
        OrientationAssurance::DeclaredNotVerified
    );

    // The same declaration cannot be used for a guaranteed operation.
    let refused = resolve_vq_layout(&reader, "mlp.fc.weight", None, false)
        .expect_err("unverified orientation must not resolve silently");
    let message = refused.to_string();
    assert!(message.contains("declared_not_verified"), "{message}");
    assert!(message.contains("--architecture"), "{message}");

    // ...unless the operator explicitly accepts the claim.
    let accepted = resolve_vq_layout(&reader, "mlp.fc.weight", None, true)
        .expect("explicit confirmation must be honoured");
    assert_eq!(accepted.logical_shape(), [24, 32]);
}

#[test]
fn linear_rectangular_orientation_contradicting_the_definition_is_rejected() {
    let bytes = import_calibrated(
        &[("mlp.fc.weight", vec![24, 32], pattern(24, 32, 0.5))],
        &[calibration("mlp.fc.weight", Some("linear"), 32)],
    );
    let reader = open(bytes);
    let wrong = manifest(&[("mlp.fc.weight", "conv1d_in_out")]);
    let audit = audit_container(&reader, Some(&wrong));

    assert_eq!(audit.confirmed(), 0);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(
        audit.invalid[0].contains("requires conv1d_in_out"),
        "{}",
        audit.invalid[0]
    );
    let error = resolve_vq_layout(&reader, "mlp.fc.weight", Some(&wrong), true)
        .expect_err("a contradiction cannot be waived away");
    assert!(error.to_string().contains("requires conv1d_in_out"));
}

#[test]
fn a_wrong_rectangular_declaration_is_caught_by_the_stored_selector_bytes() {
    // Encode [24, 32] (out, in). Rewriting the declaration to the transposed
    // orientation is self-consistent on paper, but the packed index payload was
    // produced by [24, 32] and cannot have come from [32, 24].
    let mut bytes = import_calibrated(
        &[("mlp.fc.weight", vec![24, 32], pattern(24, 32, 0.5))],
        &[calibration("mlp.fc.weight", Some("linear"), 32)],
    );
    patch_metadata(&mut bytes, "linear_out_in", "conv1d_in_out");
    // The forged declaration must stay self-consistent in the new dimension
    // too: a container whose role contradicts its orientation is caught by the
    // role matrix, not by the property under test. Both tokens are 17 bytes, so
    // the swap keeps the metadata length and the CRC refresh intact.
    patch_metadata(&mut bytes, "linear_projection", "conv1d_projection");
    patch_metadata(
        &mut bytes,
        "\"logical_shape\":[24,32]",
        "\"logical_shape\":[32,24]",
    );
    patch_metadata(&mut bytes, "\"features\":32", "\"features\":24");

    let reader = open(bytes);
    reader
        .verify_all()
        .expect("a tampered container can still pass integrity");
    let audit = audit_container(&reader, None);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(
        audit.invalid[0].contains("selector bytes") || audit.invalid[0].contains("not encodable"),
        "{}",
        audit.invalid[0]
    );
}

#[test]
fn a_transposition_the_container_cannot_distinguish_is_left_unverified() {
    // [16, 32] and [32, 16] both pack to 32 selector bytes, so neither the
    // descriptor nor the payload can decide between them. This is the case the
    // policy says must be reported as unverified rather than confirmed.
    let mut bytes = import_calibrated(
        &[("mlp.fc.weight", vec![16, 32], pattern(16, 32, 0.5))],
        &[calibration("mlp.fc.weight", Some("linear"), 32)],
    );
    patch_metadata(&mut bytes, "linear_out_in", "conv1d_in_out");
    // The forged declaration must stay self-consistent in the new dimension
    // too: a container whose role contradicts its orientation is caught by the
    // role matrix, not by the property under test. Both tokens are 17 bytes, so
    // the swap keeps the metadata length and the CRC refresh intact.
    patch_metadata(&mut bytes, "linear_projection", "conv1d_projection");
    patch_metadata(
        &mut bytes,
        "\"logical_shape\":[16,32]",
        "\"logical_shape\":[32,16]",
    );
    patch_metadata(&mut bytes, "\"features\":32", "\"features\":16");

    let reader = open(bytes);
    reader.verify_all().expect("integrity must still pass");
    let audit = audit_container(&reader, None);
    assert!(
        audit.invalid.is_empty(),
        "the container cannot falsify this claim: {:?}",
        audit.invalid
    );
    assert_eq!(audit.confirmed(), 0);
    assert!(
        resolve_vq_layout(&reader, "mlp.fc.weight", None, false).is_err(),
        "and no consumer may act on it without external information"
    );

    // With the model definition the same artifact is a hard failure.
    let audit = audit_container(
        &reader,
        Some(&manifest(&[("mlp.fc.weight", "linear_out_in")])),
    );
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("requires linear_out_in"));
}

// ---------------------------------------------------------- Linear, square

#[test]
fn square_linear_declaration_is_not_verifiable_from_the_container() {
    let bytes = import_calibrated(
        &[("attn.q.weight", vec![16, 16], pattern(16, 16, 0.5))],
        &[calibration("attn.q.weight", Some("linear"), 16)],
    );
    let reader = open(bytes);
    let audit = audit_container(&reader, None);
    assert!(audit.invalid.is_empty());
    let facts = audit.facts("attn.q.weight").expect("entry");
    assert!(facts.square);
    assert_eq!(facts.assurance, OrientationAssurance::SquareNotVerifiable);
    assert_eq!(audit.square_unverified(), 1);
    assert_eq!(audit.confirmed(), 0);
    assert!(resolve_vq_layout(&reader, "attn.q.weight", None, false).is_err());
}

#[test]
fn square_linear_declaration_is_confirmed_by_the_definition_and_by_the_matrix() {
    let source = pattern(16, 16, 0.5);
    let bytes = import_calibrated(
        &[("attn.q.weight", vec![16, 16], source.clone())],
        &[calibration("attn.q.weight", Some("linear"), 16)],
    );
    let reader = open(bytes);
    let correct = manifest(&[("attn.q.weight", "linear_out_in")]);
    let audit = audit_container(&reader, Some(&correct));
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    assert_eq!(audit.square_unverified(), 0);
    assert_eq!(audit.confirmed(), 1);

    let resolved = resolve_vq_layout(&reader, "attn.q.weight", Some(&correct), false)
        .expect("confirmed square orientation must resolve");
    let decoded = decode_pair(
        &reader,
        "attn.q.weight",
        resolved.layout.rows,
        resolved.layout.cols,
    );
    assert!(relative_l2(&source, &decoded) < 0.2);
}

#[test]
fn a_wrong_square_declaration_survives_every_container_check_and_fails_the_matrix() {
    // Flipping a square entry leaves the metadata self-consistent: the logical
    // shape, the feature count and the selector bytes are all identical under
    // both orientations. Only an external reference can decide.
    let source = pattern(16, 16, 0.5);
    assert_ne!(
        source,
        transpose(&source, 16, 16),
        "the fixture must be non-symmetric for this test to mean anything"
    );
    let mut bytes = import_calibrated(
        &[("attn.q.weight", vec![16, 16], source.clone())],
        &[calibration("attn.q.weight", Some("linear"), 16)],
    );
    patch_metadata(&mut bytes, "linear_out_in", "conv1d_in_out");
    // The forged declaration must stay self-consistent in the new dimension
    // too: a container whose role contradicts its orientation is caught by the
    // role matrix, not by the property under test. Both tokens are 17 bytes, so
    // the swap keeps the metadata length and the CRC refresh intact.
    patch_metadata(&mut bytes, "linear_projection", "conv1d_projection");

    let reader = open(bytes);
    reader
        .verify_all()
        .expect("the tampered container keeps valid CRC32C records");
    let audit = audit_container(&reader, None);
    assert!(
        audit.invalid.is_empty(),
        "nothing inside the container contradicts the forged claim: {:?}",
        audit.invalid
    );
    assert_eq!(audit.square_unverified(), 1);
    assert_eq!(audit.confirmed(), 0);

    // Independent oracle: the model definition.
    let audit = audit_container(
        &reader,
        Some(&manifest(&[("attn.q.weight", "linear_out_in")])),
    );
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("requires linear_out_in"));

    // Independent reference: the source matrix itself. Under the forged
    // `conv1d_in_out` claim the stored logical matrix should be the transpose of
    // the source; it is not, while the declared-but-unverified `linear_out_in`
    // reading is what the bytes actually hold.
    let decoded = decode_pair(&reader, "attn.q.weight", 16, 16);
    let as_declared = relative_l2(&transpose(&source, 16, 16), &decoded);
    let as_imported = relative_l2(&source, &decoded);
    assert!(
        as_imported < as_declared,
        "the reference must separate the two orientations ({as_imported} vs {as_declared})"
    );
    assert!(
        as_imported < 0.2,
        "the imported reading must match the source"
    );
}

// ------------------------------------------------------------------- Conv1D

#[test]
fn conv1d_rectangular_transposition_is_confirmed_and_decoded_as_the_logical_matrix() {
    // A Conv1D weight arrives as [in, out] = [32, 24]; the logical matrix is
    // [out, in] = [24, 32], i.e. the transpose of the source.
    let source = pattern(32, 24, 0.25);
    let bytes = import_calibrated(
        &[("attn.out.weight", vec![32, 24], source.clone())],
        &[calibration("attn.out.weight", Some("conv1d"), 32)],
    );
    let reader = open(bytes);
    let correct = manifest(&[("attn.out.weight", "conv1d_in_out")]);
    let audit = audit_container(&reader, Some(&correct));
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    let facts = audit.facts("attn.out.weight").expect("entry");
    assert_eq!(
        facts.assurance,
        OrientationAssurance::ArchitecturallyConfirmed
    );
    assert_eq!(facts.logical_shape, Some([24, 32]));

    let resolved = resolve_vq_layout(&reader, "attn.out.weight", Some(&correct), false)
        .expect("confirmed conv1d layout must resolve");
    assert_eq!(resolved.logical_shape(), [24, 32]);
    let decoded = decode_pair(
        &reader,
        "attn.out.weight",
        resolved.layout.rows,
        resolved.layout.cols,
    );
    let logical = transpose(&source, 32, 24);
    assert!(
        relative_l2(&logical, &decoded) < 0.2,
        "the logical matrix, not the source layout, must be what is stored"
    );
    assert!(relative_l2(&logical, &decoded) < relative_l2(&source, &decoded));
}

#[test]
fn conv1d_rectangular_container_silently_rejects_the_wrong_declaration() {
    let bytes = import_calibrated(
        &[("attn.out.weight", vec![32, 24], pattern(32, 24, 0.25))],
        &[calibration("attn.out.weight", Some("conv1d"), 32)],
    );
    let reader = open(bytes);
    let audit = audit_container(
        &reader,
        Some(&manifest(&[("attn.out.weight", "linear_out_in")])),
    );
    assert_eq!(audit.invalid.len(), 1);
    assert!(audit.invalid[0].contains("requires linear_out_in"));
}

#[test]
fn a_wrong_conv1d_declaration_is_caught_by_the_stored_selector_bytes() {
    let mut bytes = import_calibrated(
        &[("attn.out.weight", vec![32, 24], pattern(32, 24, 0.25))],
        &[calibration("attn.out.weight", Some("conv1d"), 32)],
    );
    patch_metadata(&mut bytes, "conv1d_in_out", "linear_out_in");
    // Keep the forged role consistent with the forged layout, for the same
    // reason as in the linear case above.
    patch_metadata(&mut bytes, "conv1d_projection", "linear_projection");
    patch_metadata(
        &mut bytes,
        "\"logical_shape\":[24,32]",
        "\"logical_shape\":[32,24]",
    );
    patch_metadata(&mut bytes, "\"features\":32", "\"features\":24");

    let reader = open(bytes);
    reader.verify_all().expect("integrity is refreshed");
    let audit = audit_container(&reader, None);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(
        audit.invalid[0].contains("not encodable") || audit.invalid[0].contains("selector bytes"),
        "{}",
        audit.invalid[0]
    );
}

#[test]
fn conv1d_square_matrix_needs_the_definition_and_the_matrix_agrees() {
    let source = pattern(16, 16, 0.25);
    let bytes = import_calibrated(
        &[("attn.conv.weight", vec![16, 16], source.clone())],
        &[calibration("attn.conv.weight", Some("conv1d"), 16)],
    );
    let reader = open(bytes);
    let audit = audit_container(&reader, None);
    assert!(audit.invalid.is_empty());
    assert_eq!(audit.square_unverified(), 1);
    assert_eq!(
        audit.facts("attn.conv.weight").expect("entry").assurance,
        OrientationAssurance::SquareNotVerifiable
    );

    let correct = manifest(&[("attn.conv.weight", "conv1d_in_out")]);
    let audit = audit_container(&reader, Some(&correct));
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    assert_eq!(audit.confirmed(), 1);

    // Independent reference: the stored logical matrix is the source transpose,
    // not the source, even though both are 16x16.
    let decoded = decode_pair(&reader, "attn.conv.weight", 16, 16);
    let logical = transpose(&source, 16, 16);
    assert!(relative_l2(&logical, &decoded) < 0.2);
    assert!(relative_l2(&logical, &decoded) < relative_l2(&source, &decoded));
}

// ------------------------------------------------------------- legacy paths

#[test]
fn legacy_default_orientation_is_declared_but_never_confirmed() {
    let bytes = import_uncalibrated(&[("mlp.fc.weight", vec![24, 32], pattern(24, 32, 0.5))]);
    let reader = open(bytes);
    let audit = audit_container(&reader, None);
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    let facts = audit.facts("mlp.fc.weight").expect("entry");
    assert_eq!(
        facts.source.map(|source| source.as_str()),
        Some("legacy_default")
    );
    assert_eq!(facts.assurance, OrientationAssurance::DeclaredNotVerified);
    assert_eq!(audit.confirmed(), 0);

    // A legacy default on a square matrix is the same claim with less evidence.
    let bytes = import_uncalibrated(&[("attn.q.weight", vec![16, 16], pattern(16, 16, 0.5))]);
    let reader = open(bytes);
    let audit = audit_container(&reader, None);
    assert!(audit.invalid.is_empty());
    assert_eq!(audit.square_unverified(), 1);
}

#[test]
fn a_container_without_any_provenance_is_reported_as_undeclared_not_invalid() {
    let bytes = import_uncalibrated(&[("mlp.fc.weight", vec![24, 32], pattern(24, 32, 0.5))]);
    let persisted = json_of_source_tensors(&bytes).is_some();
    assert!(persisted, "the importer must persist provenance");

    // A pre-contract artifact: written by the plain writer, no user metadata.
    let legacy = {
        let mut output = Cursor::new(Vec::new());
        let tensors = vec![StreamingTensorInput {
            tensor_id: 1,
            name: "dense.weight".to_owned(),
            dtype: DType::F32,
            dimensions: vec![16, 32],
            block_lengths: vec![16 * 32 * 4],
        }];
        let payload = vec![0_u8; 16 * 32 * 4];
        ModelWriter::write_streaming(&mut output, &tensors, |_, _, sink| {
            sink.write_all(&payload)
                .map_err(|_| FormatError::InvalidBlock("write failed"))?;
            Ok(payload.len() as u64)
        })
        .expect("legacy container must write");
        output.into_inner()
    };
    let reader = open(legacy);
    let audit = audit_container(&reader, None);
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    assert_eq!(audit.undeclared_rank2_tensors, 1);
    assert_eq!(audit.rank2_tensors, 1);
    let facts = audit.facts("dense.weight").expect("tensor must be visible");
    assert_eq!(facts.assurance, OrientationAssurance::LegacyUndeclared);
    assert_eq!(facts.declared, None);
    reader.verify_all().expect("legacy container verifies");
    assert!(
        resolve_vq_layout(&reader, "dense.weight", None, true).is_err(),
        "an undeclared orientation cannot be resolved even with explicit waiver"
    );
}

#[test]
fn raw_payload_is_byte_identical_and_never_carries_a_logical_reinterpretation() {
    let vector = vec![3_u8, 5, 7, 11, 13, 17, 19, 23];
    let source = {
        let header = r#"{"norm.weight":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&vector);
        SafeTensorFile::open(
            Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))),
            64 * 1024 * 1024,
        )
        .expect("raw fixture must parse")
    };
    let plan = ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation)
        .expect("raw plan must build");
    let mut output = Cursor::new(Vec::new());
    write_model_import_to_hpsi(&source, &plan, &mut output, TEST_CONFIG, 4).expect("write");
    let bytes = output.into_inner();

    let reader = open(bytes);
    reader.verify_all().expect("integrity must pass");
    let tensor = reader
        .tensors()
        .iter()
        .find(|tensor| {
            reader
                .metadata
                .tensor_names
                .get(tensor.name_ref as usize)
                .is_some_and(|name| name == "norm.weight")
        })
        .expect("RAW tensor must exist");
    let block = reader.read_block(tensor.tensor_id, 0).expect("block read");
    assert_eq!(block.payload.as_slice(), vector.as_slice());

    let audit = audit_container(&reader, None);
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    assert!(audit.tensors.iter().all(|facts| !facts.square));
}

// -------------------------------------------------------- malformed metadata

#[test]
fn unknown_orientation_tokens_are_findings() {
    let audit = handcrafted_audit(vec![raw_entry(json!({
        "weight_orientation": "linear_out_out",
    }))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("unknown weight_orientation"));

    let audit = handcrafted_audit(vec![raw_entry(json!({
        "orientation_source": "guesswork",
    }))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("unknown orientation_source"));
}

#[test]
fn orientation_fields_must_appear_together() {
    let audit = handcrafted_audit(vec![raw_entry(json!({"weight_orientation": null}))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("must appear together"));

    let audit = handcrafted_audit(vec![raw_entry(json!({"orientation_source": null}))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("must appear together"));
}

#[test]
fn missing_or_inconsistent_logical_shape_is_a_finding() {
    let audit = handcrafted_audit(vec![raw_entry(json!({"logical_shape": null}))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("logical_shape"));

    let audit = handcrafted_audit(vec![raw_entry(json!({"logical_shape": [32, 16]}))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("contradicts"));

    let audit = handcrafted_audit(vec![raw_entry(json!({"logical_shape": "16x32"}))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
}

#[test]
fn legacy_default_cannot_claim_a_conv1d_layout() {
    let audit = handcrafted_audit(vec![raw_entry(json!({
        "weight_orientation": "conv1d_in_out",
        "logical_shape": [32, 16],
    }))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("legacy_default"));
}

#[test]
fn a_calibration_claim_without_a_calibration_object_is_a_finding() {
    let audit = handcrafted_audit(vec![raw_entry(
        json!({"orientation_source": "calibration"}),
    )]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("requires a calibration object"));
}

#[test]
fn calibration_moments_must_match_the_declared_input_features() {
    let good = json!({
        "model_type": "fixture-transformer",
        "dataset_sha256": "a".repeat(64),
        "sample_count": 8,
        "features": 32,
        "statistic": "diagonal_second_moment",
        "weight_orientation": "linear_out_in",
        "orientation_source": "legacy_default",
    });
    let audit = quantized_audit(vec![quantized_entry(json!({"calibration": good.clone()}))]);
    assert!(
        audit.invalid.is_empty(),
        "moments without a module type stay a legacy declaration: {:?}",
        audit.invalid
    );
    assert_eq!(
        audit.facts("dense.weight").expect("entry").assurance,
        OrientationAssurance::DeclaredNotVerified
    );

    // The moment count is the input feature count of the orientation that was
    // encoded: `linear_out_in` over [16, 32] has 32 inputs, so 16 is a lie
    // about which axis was quantized.
    let mismatched = json!({
        "model_type": "fixture-transformer",
        "dataset_sha256": "a".repeat(64),
        "sample_count": 8,
        "features": 16,
        "statistic": "diagonal_second_moment",
        "weight_orientation": "linear_out_in",
        "orientation_source": "legacy_default",
    });
    let audit = quantized_audit(vec![quantized_entry(json!({"calibration": mismatched}))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("input features"));
}

#[test]
fn calibration_object_disagreeing_with_the_entry_is_a_finding() {
    let mut crossed = json!({
        "features": 32,
        "weight_orientation": "conv1d_in_out",
        "orientation_source": "legacy_default",
    });
    let audit = quantized_audit(vec![quantized_entry(
        json!({"calibration": crossed.clone()}),
    )]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("calibration weight_orientation"));

    crossed["weight_orientation"] = json!("linear_out_in");
    crossed["orientation_source"] = json!("calibration");
    let audit = quantized_audit(vec![quantized_entry(json!({"calibration": crossed}))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("calibration orientation_source"));
}

#[test]
fn a_raw_tensor_must_not_carry_calibration_or_a_missing_source_shape() {
    let audit = handcrafted_audit(vec![raw_entry(json!({
        "calibration": {"features": 32, "weight_orientation": "linear_out_in", "orientation_source": "legacy_default"},
    }))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("RAW-preserved"));

    let audit = handcrafted_audit(vec![json!({
        "name": "absent.weight",
        "disposition": "preserve_raw",
    })]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("no tensor of that name"));
}

#[test]
fn unknown_disposition_and_non_array_provenance_are_findings() {
    let audit = handcrafted_audit(vec![raw_entry(json!({"disposition": "quantize_later"}))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("disposition"));

    let reader = open(handcrafted(
        &[("dense.weight", vec![16, 32])],
        vec![json!("not an entry object")],
    ));
    let audit = audit_container(&reader, None);
    assert_eq!(audit.invalid.len(), 1);
    assert!(audit.invalid[0].contains("lacks a string `name`"));
}

#[test]
fn duplicate_and_undeclared_entries_are_visible_in_the_summary() {
    let entry = raw_entry(
        json!({"orientation_source": null, "weight_orientation": null, "logical_shape": null}),
    );
    let reader = open(handcrafted(
        &[
            ("dense.weight", vec![16, 32]),
            ("other.weight", vec![8, 16]),
        ],
        vec![entry.clone(), entry],
    ));
    let audit = audit_container(&reader, None);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("more than once"));
    // `other.weight` exists but is never described: reported, not rejected.
    assert_eq!(audit.undeclared_rank2_tensors, 1);
    assert_eq!(audit.rank2_tensors, 2);
    assert_eq!(
        audit
            .facts("other.weight")
            .expect("undeclared tensor must be visible")
            .assurance,
        OrientationAssurance::LegacyUndeclared
    );
    assert_eq!(
        audit
            .facts("dense.weight")
            .expect("declared tensor")
            .assurance,
        OrientationAssurance::LegacyUndeclared
    );
}

#[test]
fn orientation_metadata_on_a_non_matrix_tensor_is_not_a_logical_claim() {
    // The importer repeats the orientation tokens for every source tensor,
    // including rank-1 ones. Those values are still validated, but no logical
    // shape can be derived from them and none is required.
    let reader = open(handcrafted(
        &[("dense.bias", vec![8])],
        vec![json!({
            "name": "dense.bias",
            "source_shape": [8],
            "disposition": "preserve_raw",
            "weight_orientation": "linear_out_in",
            "orientation_source": "legacy_default",
        })],
    ));
    let audit = audit_container(&reader, None);
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    assert!(audit.tensors.is_empty());

    let reader = open(handcrafted(
        &[("dense.bias", vec![8])],
        vec![json!({
            "name": "dense.bias",
            "source_shape": [8],
            "disposition": "preserve_raw",
            "weight_orientation": "linear_out_in",
            "orientation_source": "legacy_default",
            "logical_shape": [8, 1],
        })],
    ));
    let audit = audit_container(&reader, None);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(audit.invalid[0].contains("rank-2"));

    let reader = open(handcrafted(
        &[("dense.bias", vec![8])],
        vec![json!({
            "name": "dense.bias",
            "source_shape": [8],
            "disposition": "preserve_raw",
            "weight_orientation": "nope",
            "orientation_source": "legacy_default",
        })],
    ));
    let audit = audit_container(&reader, None);
    assert_eq!(audit.invalid.len(), 1);
    assert!(audit.invalid[0].contains("unknown weight_orientation"));
}

// ------------------------------------------------------- integrity vs claims

#[test]
fn crc32c_and_orientation_are_independent_claims() {
    let mut bytes = import_calibrated(
        &[("mlp.fc.weight", vec![24, 32], pattern(24, 32, 0.5))],
        &[calibration("mlp.fc.weight", Some("linear"), 32)],
    );
    let baseline = audit_container(&open(bytes.clone()), None);
    assert!(baseline.invalid.is_empty());

    // Tamper without refreshing the record: integrity fails first.
    patch_metadata_without_crc_refresh(&mut bytes, "linear_out_in", "conv1d_in_out");
    let reader = open(bytes.clone());
    assert!(
        reader.verify_all().is_err(),
        "a stale CRC32C must not be accepted"
    );

    // Same edit with a refreshed record: integrity passes, and the orientation
    // audit still catches it through the stored selector bytes.
    let mut refreshed = import_calibrated(
        &[("mlp.fc.weight", vec![24, 32], pattern(24, 32, 0.5))],
        &[calibration("mlp.fc.weight", Some("linear"), 32)],
    );
    patch_metadata(&mut refreshed, "linear_out_in", "conv1d_in_out");
    let reader = open(refreshed);
    reader.verify_all().expect("refreshed record must verify");
    assert!(
        !audit_container(&reader, None).invalid.is_empty(),
        "and the audit must still report it"
    );
}

fn patch_metadata_without_crc_refresh(bytes: &mut [u8], find: &str, replace: &str) {
    let start = {
        let layout = ModelReader::open(
            Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes.to_owned()))),
            &ReaderLimits::default(),
        )
        .expect("container must parse");
        (
            usize::try_from(layout.header.metadata.offset).expect("offset"),
            usize::try_from(layout.header.metadata.length).expect("length"),
        )
    };
    let region = &mut bytes[start.0..start.0 + start.1];
    let find = find.as_bytes();
    let replace = replace.as_bytes();
    let mut cursor = 0;
    while cursor + find.len() <= region.len() {
        if region[cursor..].starts_with(find) {
            region[cursor..cursor + find.len()].copy_from_slice(replace);
            cursor += find.len();
        } else {
            cursor += 1;
        }
    }
}

fn json_of_source_tensors(bytes: &[u8]) -> Option<Vec<Value>> {
    let reader = open(bytes.to_vec());
    reader
        .metadata
        .value
        .get("user")?
        .get("source_tensors")?
        .as_array()
        .cloned()
}

// ------------------------------------------------------------- name matching

#[test]
fn architecture_expectations_match_exact_names_only() {
    let bytes = import_calibrated(
        &[("mlp.fc.weight", vec![24, 32], pattern(24, 32, 0.5))],
        &[calibration("mlp.fc.weight", Some("linear"), 32)],
    );
    let reader = open(bytes);
    // A manifest about a different tensor must neither confirm nor contradict.
    let audit = audit_container(
        &reader,
        Some(&manifest(&[(
            "mlp.fc.weight.hpsi_codebook",
            "conv1d_in_out",
        )])),
    );
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    assert_eq!(audit.confirmed(), 0);
    assert_eq!(
        audit.facts("mlp.fc.weight").expect("entry").assurance,
        OrientationAssurance::DeclaredNotVerified
    );

    // Suffix and prefix similarity is not a match either.
    let facts = manifest(&[("fc.weight", "conv1d_in_out")]);
    assert!(facts.expectation("mlp.fc.weight").is_none());
    assert!(facts.expectation("fc.weight").is_some());
}

#[test]
fn a_manifest_without_orientation_information_is_refused_not_guessed() {
    let error = ArchitectureFacts::from_json(
        &json!({
            "format": "hpsi-architecture-manifest",
            "version": 1,
            "tensors": [{"name": "mlp.fc.weight"}],
        })
        .to_string(),
    )
    .expect_err("a manifest entry must state the orientation explicitly");
    assert!(error.to_string().contains("never inferred"), "{error}");

    let error = ArchitectureFacts::from_json(
        &json!({
            "format": "hpsi-architecture-manifest",
            "version": 1,
            "tensors": [{"name": "a.weight", "weight_orientation": "linear_out_in"},
                        {"name": "a.weight", "weight_orientation": "conv1d_in_out"}],
        })
        .to_string(),
    )
    .expect_err("duplicate names are not allowed");
    assert!(
        error.to_string().contains("declares the same tensor twice"),
        "{error}"
    );
}

// --------------------------------------------------- Architectural roles (4.2)
//
// A role is a second dimension: what a tensor is for. An orientation says how a
// matrix is laid out. These tests hold the two apart, and hold the legacy
// reading of pre-role artifacts unchanged.

/// One architecture-manifest entry: name, roles, index axis, stated layout.
type RoleManifestEntry = (
    &'static str,
    &'static [&'static str],
    Option<usize>,
    Option<&'static str>,
);

fn role_manifest(entries: &[RoleManifestEntry]) -> ArchitectureFacts {
    let tensors = entries
        .iter()
        .map(|(name, roles, index_axis, orientation)| {
            let mut entry = json!({"name": name, "tensor_role": roles});
            if let Some(axis) = index_axis {
                entry["index_axis"] = json!(axis);
            }
            if let Some(token) = orientation {
                entry["weight_orientation"] = json!(token);
            }
            entry
        })
        .collect::<Vec<_>>();
    ArchitectureFacts::from_json(
        &json!({
            "format": "hpsi-architecture-manifest",
            "version": 1,
            "model_type": "fixture-transformer",
            "tensors": tensors,
        })
        .to_string(),
    )
    .expect("role manifest must parse")
}

fn quantized_reader(entries: Vec<Value>) -> ModelReader {
    let pair = vq_pair("dense.weight", [16, 32]);
    let tensors: Vec<(&str, DType, Vec<u64>)> = pair
        .iter()
        .map(|(name, dtype, dims)| (name.as_str(), *dtype, dims.clone()))
        .collect();
    open(handcrafted_with(&tensors, entries))
}

#[test]
fn an_indexed_table_records_a_role_and_stays_out_of_the_projection_path() {
    let reader = quantized_reader(vec![quantized_entry(json!({
        "tensor_role": "embedding_table",
        "index_axis": 0,
    }))]);
    let audit = audit_container(&reader, None);
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    let facts = audit.facts("dense.weight").expect("table must be audited");
    assert_eq!(facts.role, Some(TensorRole::EmbeddingTable));
    assert_eq!(facts.index_axis, Some(0));
    assert_eq!(facts.role_assurance, RoleAssurance::DeclaredNotConfirmed);
    // The recorded orientation is still only an encoding fact: a role stated by
    // the container itself cannot confirm a layout.
    assert_eq!(facts.assurance, OrientationAssurance::DeclaredNotVerified);
    assert_eq!(audit.indexed_tables, 1);
    // And matvec is the wrong operation for a lookup, whatever the acceptance
    // flag says: no lookup kernel ships in this contract.
    let error = resolve_vq_layout(&reader, "dense.weight", None, true)
        .expect_err("an indexed table must not enter the projection path");
    assert!(error.to_string().contains("is not a projection"), "{error}");
}

#[test]
fn a_manifest_stating_only_a_role_entails_and_confirms_the_layout() {
    let reader = quantized_reader(vec![quantized_entry(json!({
        "tensor_role": "embedding_table",
        "index_axis": 0,
    }))]);
    let audit = audit_container(
        &reader,
        Some(&role_manifest(&[(
            "dense.weight",
            &["embedding_table"],
            Some(0),
            None,
        )])),
    );
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    let facts = audit.facts("dense.weight").expect("audited");
    assert_eq!(facts.role_assurance, RoleAssurance::Confirmed);
    // An indexed-first table puts its features on axis 1, which is the axis the
    // quantizer grouped: the role entails the recorded layout.
    assert_eq!(
        facts.assurance,
        OrientationAssurance::ArchitecturallyConfirmed
    );
}

#[test]
fn a_role_that_states_no_layout_confers_no_layout_assurance() {
    // The manifest knows this tensor is an "other" matrix and states nothing
    // about the layout. Confirming that claim must not hand the orientation
    // an architectural guarantee it never earned.
    let reader = quantized_reader(vec![quantized_entry(json!({"tensor_role": "other"}))]);
    let audit = audit_container(
        &reader,
        Some(&role_manifest(&[("dense.weight", &["other"], None, None)])),
    );
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    let facts = audit.facts("dense.weight").expect("audited");
    assert_eq!(facts.role_assurance, RoleAssurance::Confirmed);
    assert_eq!(facts.assurance, OrientationAssurance::DeclaredNotVerified);
    assert!(!facts.assurance.is_confirmed());
}

#[test]
fn an_indexed_table_on_axis_one_is_rejected_not_resupported() {
    // The type allows 0 or 1; the contract does not. Axis 1 groups features
    // along the source rows, which v0.1 can only record as `conv1d_in_out` with
    // calibration provenance, and an indexed table must not carry activation
    // statistics. No producer path can write it, so it is refused by name rather
    // than quietly accepted because the value parsed.
    let audit = quantized_audit(vec![quantized_entry(json!({
        "tensor_role": "embedding_table",
        "index_axis": 1,
    }))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(
        audit.invalid[0].starts_with("dense.weight: axis_one_unsupported:"),
        "{}",
        audit.invalid[0]
    );
    assert!(
        audit.invalid[0].contains("No producer path"),
        "{}",
        audit.invalid[0]
    );
}

#[test]
fn an_indexed_table_without_an_index_axis_is_a_finding() {
    let audit = quantized_audit(vec![quantized_entry(json!({
        "tensor_role": "embedding_table",
    }))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(
        audit.invalid[0].contains("requires index_axis"),
        "{}",
        audit.invalid[0]
    );
}

#[test]
fn a_projection_role_without_its_layout_is_a_finding() {
    let audit = quantized_audit(vec![quantized_entry(json!({
        "tensor_role": "linear_projection",
        "weight_orientation": Value::Null,
        "orientation_source": Value::Null,
        "logical_shape": Value::Null,
    }))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(
        audit.invalid[0]
            .contains("role linear_projection requires weight_orientation \"linear_out_in\""),
        "{}",
        audit.invalid[0]
    );
}

#[test]
fn an_index_axis_without_a_role_is_a_finding() {
    let audit = quantized_audit(vec![quantized_entry(json!({"index_axis": 0}))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(
        audit.invalid[0].contains("only meaningful alongside a tensor_role"),
        "{}",
        audit.invalid[0]
    );
}

#[test]
fn a_role_contradicting_the_definition_is_a_finding() {
    let reader = quantized_reader(vec![quantized_entry(
        json!({"tensor_role": "linear_projection"}),
    )]);
    let audit = audit_container(
        &reader,
        Some(&role_manifest(&[(
            "dense.weight",
            &["other"],
            None,
            Some("linear_out_in"),
        )])),
    );
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(
        audit.invalid[0].contains(
            "declared role linear_projection but the model definition evidences only [other]"
        ),
        "{}",
        audit.invalid[0]
    );
}

#[test]
fn a_tied_tensor_with_two_roles_is_confirmed_by_either_one() {
    // A tied embedding/output head really is reached as a table and as a
    // projection. Recording the projection role must not read as a
    // contradiction when the definition states both.
    let reader = quantized_reader(vec![quantized_entry(json!({
        "tensor_role": "linear_projection",
    }))]);
    let audit = audit_container(
        &reader,
        Some(&role_manifest(&[(
            "dense.weight",
            &["embedding_table", "linear_projection"],
            Some(0),
            None,
        )])),
    );
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    let facts = audit.facts("dense.weight").expect("audited");
    assert_eq!(facts.role_assurance, RoleAssurance::Confirmed);
    assert_eq!(
        facts.assurance,
        OrientationAssurance::ArchitecturallyConfirmed
    );
    assert_eq!(facts.architecture_roles.len(), 2);
}

#[test]
fn role_unknown_is_a_different_claim_from_role_other() {
    // unknown: the producer never established the function, so the ordinary
    // unverified gate applies and an explicit acceptance can lift it.
    let unknown = quantized_reader(vec![quantized_entry(json!({"tensor_role": "unknown"}))]);
    let audit = audit_container(&unknown, None);
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    assert_eq!(
        audit.facts("dense.weight").expect("x").role_token(),
        "unknown"
    );
    assert!(resolve_vq_layout(&unknown, "dense.weight", None, false).is_err());
    resolve_vq_layout(&unknown, "dense.weight", None, true).expect("explicit acceptance lifts it");

    // other: the function is known and is not a projection, so no flag opens it.
    let other = quantized_reader(vec![quantized_entry(json!({"tensor_role": "other"}))]);
    let audit = audit_container(&other, None);
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    assert_eq!(
        audit.facts("dense.weight").expect("x").role_token(),
        "other"
    );
    let error = resolve_vq_layout(&other, "dense.weight", None, true)
        .expect_err("a known non-projection must stay out of matvec");
    assert!(
        error.to_string().contains("role other is not a projection"),
        "{error}"
    );
}

#[test]
fn an_uncalibrated_import_records_an_unknown_role_instead_of_assuming_linear() {
    let bytes = import_uncalibrated(&[("mlp.fc.weight", vec![16, 32], pattern(16, 32, 0.5))]);
    let reader = open(bytes);
    let entries = reader.metadata.value["user"]["source_tensors"]
        .as_array()
        .expect("provenance array");
    let entry = &entries[0];
    assert_eq!(
        entry["tensor_role"].as_str(),
        Some("unknown"),
        "a plan with no class evidence must not claim a projection role"
    );
    // The layout claim stays, exactly as before: the role dimension does
    // not reinterpret what the importer already recorded.
    assert_eq!(entry["weight_orientation"].as_str(), Some("linear_out_in"));
    assert_eq!(entry["orientation_source"].as_str(), Some("legacy_default"));
}

#[test]
fn a_calibrated_import_records_the_class_the_capture_observed() {
    let bytes = import_calibrated(
        &[("attn.q.weight", vec![16, 32], pattern(16, 32, 0.5))],
        &[calibration("attn.q.weight", Some("conv1d"), 16)],
    );
    let reader = open(bytes);
    let entries = reader.metadata.value["user"]["source_tensors"]
        .as_array()
        .expect("provenance array");
    assert_eq!(
        entries[0]["tensor_role"].as_str(),
        Some("conv1d_projection"),
        "module_type is class evidence, so the role follows from it"
    );
    let audit = audit_container(&reader, None);
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    assert_eq!(audit.roles_declared, 1);
}

#[test]
fn a_bundle_without_module_type_declares_no_role() {
    let bytes = import_calibrated(
        &[("attn.q.weight", vec![16, 32], pattern(16, 32, 0.5))],
        &[calibration("attn.q.weight", None, 32)],
    );
    let reader = open(bytes);
    let entries = reader.metadata.value["user"]["source_tensors"]
        .as_array()
        .expect("provenance array");
    assert_eq!(entries[0]["tensor_role"].as_str(), Some("unknown"));
    // A bundle that states no module class also states no orientation, so the
    // documented legacy layout applies and is labelled as such.
    assert_eq!(
        entries[0]["orientation_source"].as_str(),
        Some("legacy_default")
    );
}

#[test]
fn entries_that_state_no_role_are_read_exactly_as_before() {
    let audit = quantized_audit(vec![quantized_entry(json!({}))]);
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    assert_eq!(audit.roles_declared, 0);
    assert_eq!(audit.count_role(RoleAssurance::NotDeclared), 1);
    let facts = audit.facts("dense.weight").expect("audited");
    assert_eq!(facts.assurance, OrientationAssurance::DeclaredNotVerified);
    assert!(
        resolve_vq_layout(
            &quantized_reader(vec![quantized_entry(json!({}))]),
            "dense.weight",
            None,
            true
        )
        .is_ok()
    );
}

#[test]
fn an_indexed_table_may_not_carry_activation_statistics() {
    let audit = quantized_audit(vec![quantized_entry(json!({
        "tensor_role": "embedding_table",
        "index_axis": 0,
        "orientation_source": "calibration",
        "calibration": {
            "model_type": "fixture",
            "dataset_sha256": "0".repeat(64),
            "sample_count": 8,
            "features": 32,
            "statistic": "diagonal_second_moment",
            "weight_orientation": "linear_out_in",
            "orientation_source": "calibration",
        },
    }))]);
    assert_eq!(audit.invalid.len(), 1, "{:?}", audit.invalid);
    assert!(
        audit.invalid[0].contains("indexed, so it has no input activations"),
        "{}",
        audit.invalid[0]
    );
}

#[test]
fn the_writer_refuses_a_plan_whose_role_contradicts_its_layout() {
    let source = safetensors(&[("dense.weight", vec![16, 32], pattern(16, 32, 0.5))]);
    let mut plan =
        ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation)
            .expect("plan must build");
    // A producer that measured nothing about this tensor cannot claim it is a
    // Conv1D layer while encoding it as a Linear one.
    plan.tensors[0].role = TensorRole::Conv1dProjection;
    let mut output = Cursor::new(Vec::new());
    let error = write_model_import_to_hpsi(&source, &plan, &mut output, TEST_CONFIG, 4)
        .expect_err("a contradictory plan must not be written");
    assert!(error.to_string().contains("role contradicts"), "{error}");
}

#[test]
fn an_indexed_table_from_the_definition_leaves_the_vq_path() {
    let source = safetensors(&[("embed.table.weight", vec![24, 16], pattern(24, 16, 0.25))]);
    let mut plan =
        ModelImportPlan::build(&source, &[], UnsupportedTensorPolicy::AllowRawPreservation)
            .expect("plan must build");
    let facts = role_manifest(&[("embed.table.weight", &["embedding_table"], Some(0), None)]);
    plan.apply_architecture_roles(&facts)
        .expect("roles must apply");
    assert_eq!(plan.tensors[0].role, TensorRole::EmbeddingTable);
    assert_eq!(plan.tensors[0].index_axis, Some(0));
    assert_eq!(
        plan.tensors[0].weight_orientation,
        WeightOrientation::LinearOutIn
    );
    assert!(
        matches!(
            plan.tensors[0].disposition,
            hpsi_ops::TensorDisposition::PreserveRaw
        ),
        "an indexed table with no projection evidence is preserved, not multiplied"
    );
}

#[test]
fn a_manifest_cannot_turn_a_calibrated_projection_into_a_bare_table() {
    let source = safetensors(&[("attn.q.weight", vec![16, 32], pattern(16, 32, 0.5))]);
    let mut plan = ModelImportPlan::build_with_provenance(
        &source,
        &[calibration("attn.q.weight", Some("conv1d"), 16)],
        UnsupportedTensorPolicy::AllowRawPreservation,
    )
    .expect("calibrated plan must build");
    let facts = role_manifest(&[("attn.q.weight", &["embedding_table"], Some(0), None)]);
    let error = plan
        .plan
        .apply_architecture_roles(&facts)
        .expect_err("the definition contradicts the capture");
    assert!(error.contains("must keep the projection role"), "{error}");
}
