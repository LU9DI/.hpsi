//! End-to-end contract test for embedding tables (Phase 4.4).
//!
//! One `cargo test` command exercises the whole chain an operator would walk:
//! build a source checkpoint, state the architectural roles from a model
//! definition, plan, apply the roles, write a container, reopen it, verify every
//! CRC32C record and every padding byte, audit it with and without the manifest,
//! refuse the projection path for the tables, reconstruct the weights
//! numerically against the original arrays, and compare a tied head and a legacy
//! artifact. The Phase 4.2 and 4.3 evidence for this path was a list of commands;
//! it is now a test.
//!
//! Rules exercised: `docs/format-v0.1.md` §4.1-§4.2, `docs/orientation-policy.md`,
//! and the shared reason-code table in
//! `tests/fixtures/embedding_layout_cases.json`.

use hpsi_format::{
    DType, FormatError, MemorySource, ModelReader, ModelWriter, ReaderLimits, StreamingTensorInput,
    TensorDescriptor,
};
use hpsi_ops::{
    ActivationCalibration, ArchitectureFacts, CalibratedModelImportPlan, EntryLayout,
    HpsiVqExecutor, HpsiVqLayout, LayoutContext, LayoutViolation, ModelImportPlan,
    OrientationAssurance, RoleAssurance, SafeTensorFile, TensorDisposition, TensorRole,
    UnsupportedTensorPolicy, VqConfig, WeightOrientation, audit_container, resolve_vq_layout,
    validate_entry_layout, write_model_import_to_hpsi_calibrated,
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
const GROUP: u64 = 8;

const TABLE: &str = "embed.table.weight";
const POSITION: &str = "embed.pos.weight";
const LINEAR: &str = "proj.linear.weight";
const CONV: &str = "proj.conv.weight";
const SQUARE: &str = "proj.conv.square.weight";
const MYSTERY: &str = "mystery.matrix";
const TIED: &str = "transformer.wte.weight";
const VECTOR: &str = "proj.linear.bias";

// ---------------------------------------------------------------- fixtures

fn pattern(rows: usize, cols: usize, salt: f32) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| {
            let row = index / cols;
            let col = index % cols;
            (row as f32 * 0.31 + col as f32 * 0.17 + salt).sin() + col as f32 * 0.02
                - row as f32 * 0.011
        })
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

fn fixture_entries() -> Vec<(&'static str, Vec<u64>, Vec<f32>)> {
    vec![
        (TABLE, vec![24, 16], pattern(24, 16, 0.1)),
        (POSITION, vec![32, 16], pattern(32, 16, 0.2)),
        (LINEAR, vec![32, 16], pattern(32, 16, 0.3)),
        (CONV, vec![16, 32], pattern(16, 32, 0.4)),
        (SQUARE, vec![16, 16], pattern(16, 16, 0.5)),
        (MYSTERY, vec![8, 32], pattern(8, 32, 0.6)),
        (TIED, vec![24, 16], pattern(24, 16, 0.7)),
        (VECTOR, vec![16], pattern(16, 1, 0.8)),
    ]
}

fn source() -> SafeTensorFile {
    safetensors(&fixture_entries())
}

/// A model definition as an architecture manifest: classes for the projections,
/// indexing roles for the tables, and deliberately nothing for the two tensors
/// whose function this definition does not know.
/// One manifest row: the tensor's exact name, the classes that reach it, the
/// axis an indexed table is gathered on, and the orientation a projection is
/// encoded with.
type ManifestEntry<'a> = (&'a str, &'a [&'a str], Option<usize>, Option<&'a str>);

fn manifest_document(entries: &[ManifestEntry<'_>]) -> Result<ArchitectureFacts, String> {
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
            "model_type": "e2e-transformer",
            "tensors": tensors,
        })
        .to_string(),
    )
    .map_err(|error| error.to_string())
}

fn roles() -> ArchitectureFacts {
    manifest_document(&[
        (TABLE, &["embedding_table"], Some(0), None),
        (POSITION, &["embedding_table"], Some(0), None),
        (LINEAR, &["linear_projection"], None, Some("linear_out_in")),
        (CONV, &["conv1d_projection"], None, Some("conv1d_in_out")),
        (SQUARE, &["conv1d_projection"], None, Some("conv1d_in_out")),
        (
            TIED,
            &["embedding_table", "linear_projection"],
            Some(0),
            Some("linear_out_in"),
        ),
    ])
    .expect("the e2e manifest must parse")
}

fn calibration(layer: &str, module_type: Option<&str>, features: usize) -> ActivationCalibration {
    ActivationCalibration {
        layer_name: layer.to_owned(),
        sample_count: 8,
        features,
        activation_second_moment: vec![1.0; features],
        model_type: "e2e-transformer".to_owned(),
        model_revision: None,
        dataset_sha256: "b".repeat(64),
        module_type: module_type.map(str::to_owned),
    }
}

/// The capture bundle, as a real run would produce it: it hooked the four
/// projection modules and nothing else. The tables are absent because an
/// embedding has no input activations to measure, and `mystery.matrix` is absent
/// because the capture never recognised it either.
fn capture() -> Vec<ActivationCalibration> {
    vec![
        calibration(LINEAR, Some("linear"), 16),
        calibration(CONV, Some("conv1d"), 16),
        calibration(SQUARE, Some("conv1d"), 16),
        calibration(TIED, Some("linear"), 16),
    ]
}

fn planned() -> CalibratedModelImportPlan {
    let source = source();
    let mut calibrated = ModelImportPlan::build_with_provenance(
        &source,
        &capture(),
        UnsupportedTensorPolicy::AllowRawPreservation,
    )
    .expect("calibrated plan must build");
    calibrated
        .plan
        .apply_architecture_roles(&roles())
        .expect("the manifest roles must apply");
    calibrated
}

fn written() -> Vec<u8> {
    let source = source();
    let calibrated = planned();
    let mut output = Cursor::new(Vec::new());
    write_model_import_to_hpsi_calibrated(&source, &calibrated, &mut output, TEST_CONFIG, 4)
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

fn descriptor(reader: &ModelReader, name: &str) -> TensorDescriptor {
    reader
        .tensors()
        .iter()
        .find(|tensor| {
            reader
                .metadata
                .tensor_names
                .get(tensor.name_ref as usize)
                .map(String::as_str)
                == Some(name)
        })
        .cloned()
        .unwrap_or_else(|| panic!("{name} is absent from the container"))
}

fn stored_bytes(reader: &ModelReader, name: &str) -> Vec<u8> {
    let tensor = descriptor(reader, name);
    let mut bytes = Vec::new();
    for block_id in 0..tensor.block_count {
        let block = reader
            .read_block(tensor.tensor_id, block_id)
            .unwrap_or_else(|error| panic!("{name} block {block_id}: {error:?}"));
        bytes.extend_from_slice(&block.payload);
    }
    bytes
}

fn source_f32(name: &str) -> Vec<f32> {
    let source = source();
    let info = source
        .tensors()
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("{name} is absent from the source"));
    let total = info.shape.iter().product::<u64>();
    source
        .read_f32_elements(name, 0, total)
        .unwrap_or_else(|error| panic!("{name}: {error:?}"))
}

fn source_raw(name: &str) -> Vec<u8> {
    let source = source();
    let mut bytes = Vec::new();
    source
        .copy_raw_payload(name, &mut bytes, 4096)
        .unwrap_or_else(|error| panic!("{name}: {error:?}"));
    bytes
}

/// Rebuild a dense logical matrix through the production VQ decode path, one
/// column at a time, so no block layout is assumed by the test.
fn reconstruct(reader: &ModelReader, name: &str, rows: usize, cols: usize) -> Vec<f32> {
    let codebook = descriptor(reader, &format!("{name}.hpsi_vq.codebook"));
    let indices = descriptor(reader, &format!("{name}.hpsi_vq.indices"));
    let layout = HpsiVqLayout {
        codebook_tensor_id: codebook.tensor_id,
        indices_tensor_id: indices.tensor_id,
        rows,
        cols,
        codebook_count: usize::try_from(codebook.dimensions[0]).expect("codebook count fits"),
    };
    let executor = HpsiVqExecutor::new(reader, layout).expect("executor must prepare");
    let mut dense = vec![0.0_f32; rows * cols];
    let mut input = vec![0.0_f32; cols];
    let mut column = vec![0.0_f32; rows];
    for col in 0..cols {
        input.iter_mut().for_each(|value| *value = 0.0);
        input[col] = 1.0;
        executor
            .matvec_rows(&input, 0, &mut column)
            .unwrap_or_else(|error| panic!("{name} column {col}: {error}"));
        for row in 0..rows {
            dense[row * cols + col] = column[row];
        }
    }
    dense
}

// ------------------------------------------------------------------- plan

#[test]
fn tables_are_planned_as_tables_before_anything_is_written() {
    let plan = planned().plan;
    let by_name = plan
        .tensors
        .iter()
        .map(|tensor| (tensor.source_name.as_str(), tensor))
        .collect::<std::collections::BTreeMap<_, _>>();
    for name in [TABLE, POSITION] {
        let tensor = by_name[name];
        assert_eq!(tensor.role, TensorRole::EmbeddingTable, "{name}");
        assert_eq!(tensor.index_axis, Some(0), "{name}");
        assert!(
            matches!(tensor.disposition, TensorDisposition::PreserveRaw),
            "{name} must not enter the VQ projection path"
        );
        assert_eq!(
            tensor.weight_orientation,
            WeightOrientation::LinearOutIn,
            "{name}: v0.1 still records which axis was grouped"
        );
    }
    for (name, role) in [
        (LINEAR, TensorRole::LinearProjection),
        (CONV, TensorRole::Conv1dProjection),
        (SQUARE, TensorRole::Conv1dProjection),
        (TIED, TensorRole::LinearProjection),
    ] {
        let tensor = by_name[name];
        assert_eq!(tensor.role, role, "{name}");
        assert!(
            matches!(tensor.disposition, TensorDisposition::QuantizeVq),
            "{name} is a projection and must be quantized"
        );
    }
    assert_eq!(by_name[MYSTERY].role, TensorRole::Unknown);
    assert_eq!(by_name[VECTOR].role, TensorRole::Unknown);
}

// ---------------------------------------------------------------- container

#[test]
fn the_written_container_verifies_and_reports_roles_and_axes() {
    let reader = open(written());
    let records = reader
        .verify_all()
        .expect("every CRC32C record must verify");
    assert!(records > 0, "the container must carry integrity records");
    reader
        .verify_padding(reader.header.file_size)
        .expect("every padding byte must be zero");

    let audit = audit_container(&reader, Some(&roles()));
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);

    let table = audit.facts(TABLE).expect("table is audited");
    assert_eq!(table.role, Some(TensorRole::EmbeddingTable));
    assert_eq!(table.index_axis, Some(0));
    assert_eq!(table.role_assurance, RoleAssurance::Confirmed);
    assert_eq!(table.disposition, "preserve_raw");
    assert!(!table.is_projection(), "a table is not a projection");
    assert_eq!(
        audit
            .facts(POSITION)
            .expect("position table")
            .role_assurance,
        RoleAssurance::Confirmed
    );
    assert_eq!(audit.indexed_tables, 2);

    for name in [LINEAR, CONV, SQUARE, TIED] {
        let facts = audit.facts(name).expect(name);
        assert!(facts.is_projection(), "{name}");
        assert_eq!(
            facts.role_assurance,
            RoleAssurance::Confirmed,
            "{name} role is evidenced by the definition"
        );
    }
    let tied = audit.facts(TIED).expect("tied head is audited");
    assert_eq!(tied.role, Some(TensorRole::LinearProjection));
    assert!(
        tied.architecture_roles
            .contains(&TensorRole::EmbeddingTable),
        "{:?} should also evidence the table role",
        tied.architecture_roles
    );

    let mystery = audit.facts(MYSTERY).expect("mystery is audited");
    assert_eq!(mystery.role, Some(TensorRole::Unknown));
    assert_eq!(
        mystery.role_assurance,
        RoleAssurance::DeclaredNotConfirmed,
        "an unrelated manifest confirms nothing about a tensor it does not name"
    );
    assert_eq!(
        mystery.assurance,
        OrientationAssurance::DeclaredNotVerified,
        "and a grouping token alone earns no architectural guarantee"
    );
    assert_eq!(audit.roles_confirmed, 6);
    assert_eq!(audit.confirmed(), 6);
    assert_eq!(audit.rank2_tensors, 7);
}

#[test]
fn without_a_manifest_nothing_is_architecturally_confirmed() {
    let reader = open(written());
    let audit = audit_container(&reader, None);
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    assert_eq!(audit.confirmed(), 0);
    assert_eq!(audit.count_role(RoleAssurance::Confirmed), 0);
    assert_eq!(audit.count_role(RoleAssurance::DeclaredNotConfirmed), 7);
    assert_eq!(audit.count(OrientationAssurance::SquareNotVerifiable), 1);
    assert_eq!(audit.count(OrientationAssurance::DeclaredNotVerified), 6);
    let table = audit.facts(TABLE).expect("table");
    assert_eq!(table.role, Some(TensorRole::EmbeddingTable));
    assert!(!table.assurance.is_confirmed());
}

#[test]
fn the_projection_executor_refuses_tables_and_accepts_projections() {
    let reader = open(written());
    for name in [TABLE, POSITION] {
        for allow_unverified in [false, true] {
            let error = resolve_vq_layout(&reader, name, Some(&roles()), allow_unverified)
                .err()
                .unwrap_or_else(|| panic!("{name} must never enter the projection path"));
            let message = error.to_string();
            assert!(
                message.contains("is not a projection")
                    || message.contains("no VQ layout to resolve"),
                "{name}: {message}"
            );
            assert!(
                !message.contains("allow-unverified"),
                "{name}: no acceptance flag may open a lookup table: {message}"
            );
        }
    }
    let resolved =
        resolve_vq_layout(&reader, LINEAR, Some(&roles()), false).expect("projection resolves");
    assert_eq!(resolved.logical_shape(), [32, 16]);
}

#[test]
fn a_quantized_table_is_refused_by_the_role_gate_even_when_its_payload_is_valid() {
    // The disposition gate above is a proximate reason. This case isolates the
    // role gate itself: a producer that did quantize a table, with a perfectly
    // consistent selector payload, is still refused by `tensor_role`.
    let rows = 24_u64;
    let cols = 16_u64;
    let codebooks = 1_u64;
    let index_bytes = rows * ((cols / GROUP) * codebooks) / 2;
    let inputs = vec![
        StreamingTensorInput {
            tensor_id: 1,
            name: "embed.table.weight.hpsi_vq.codebook".to_owned(),
            dtype: DType::F32,
            dimensions: vec![codebooks, 16, GROUP],
            block_lengths: vec![codebooks * 16 * GROUP * 4],
        },
        StreamingTensorInput {
            tensor_id: 2,
            name: "embed.table.weight.hpsi_vq.indices".to_owned(),
            dtype: DType::Opaque,
            dimensions: vec![index_bytes],
            block_lengths: vec![index_bytes],
        },
    ];
    let payloads = inputs
        .iter()
        .map(|tensor| vec![0_u8; tensor.block_lengths[0] as usize])
        .collect::<Vec<_>>();
    let entries = vec![json!({
        "name": "embed.table.weight",
        "source_dtype": "F32",
        "source_shape": [rows, cols],
        "disposition": "quantize_vq",
        "weight_orientation": "linear_out_in",
        "orientation_source": "legacy_default",
        "tensor_role": "embedding_table",
        "index_axis": 0,
        "logical_shape": [rows, cols],
    })];
    let mut output = Cursor::new(Vec::new());
    ModelWriter::write_streaming_with_metadata(
        &mut output,
        &inputs,
        Some(json!({"user": {"source_tensors": entries}})),
        |tensor_index, _block_index, sink| {
            let payload = payloads
                .get(tensor_index)
                .ok_or(FormatError::InvalidBlock("unknown table tensor"))?;
            sink.write_all(payload)
                .map_err(|_| FormatError::InvalidBlock("table write failed"))?;
            Ok(payload.len() as u64)
        },
    )
    .expect("a table container must write");
    let reader = open(output.into_inner());
    reader.verify_all().expect("integrity must verify");
    let audit = audit_container(&reader, None);
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    assert_eq!(audit.indexed_tables, 1);
    let error = resolve_vq_layout(&reader, "embed.table.weight", None, true)
        .expect_err("the role gate must refuse a quantized table");
    assert!(error.to_string().contains("is not a projection"), "{error}");
}

// ------------------------------------------------------------- reconstruction

#[test]
fn tables_reconstruct_byte_for_byte_and_projections_by_their_orientation() {
    let reader = open(written());
    for name in [TABLE, POSITION] {
        assert_eq!(
            stored_bytes(&reader, name),
            source_raw(name),
            "{name} must be the source table verbatim"
        );
    }

    // (name, logical rows, logical cols, source rows, source cols)
    let cases = [
        (LINEAR, 32_usize, 16_usize, 32_usize, 16_usize),
        (CONV, 32, 16, 16, 32),
        (TIED, 24, 16, 24, 16),
        (SQUARE, 16, 16, 16, 16),
    ];
    for (name, rows, cols, source_rows, source_cols) in cases {
        let decoded = reconstruct(&reader, name, rows, cols);
        let stored = source_f32(name);
        // The architectural logical matrix, as the model definition arranges it.
        let reference = if name == CONV || name == SQUARE {
            transpose(&stored, source_rows, source_cols)
        } else {
            stored.clone()
        };
        let direct = relative_l2(&reference, &decoded);
        let flipped = relative_l2(&transpose(&reference, rows, cols), &decoded);
        assert!(
            direct < flipped,
            "{name}: decoded must fit its architectural orientation ({direct} vs {flipped})"
        );
        assert!(direct.is_finite() && direct > 0.0, "{name}: {direct}");
    }
}

#[test]
fn a_square_matrix_needs_the_definition_and_the_numeric_control_proves_it() {
    // With no manifest the square matrix is unverifiable by shape; the numeric
    // control still distinguishes the two layouts, and the container's own
    // bytes do not.
    let reader = open(written());
    let audit = audit_container(&reader, None);
    let square = audit.facts(SQUARE).expect("square is audited");
    assert!(square.square);
    assert_eq!(
        square.assurance,
        OrientationAssurance::SquareNotVerifiable,
        "shape alone cannot decide a square orientation"
    );
    let decoded = reconstruct(&reader, SQUARE, 16, 16);
    let as_conv1d = transpose(&source_f32(SQUARE), 16, 16);
    let direct = relative_l2(&as_conv1d, &decoded);
    let wrong = relative_l2(&source_f32(SQUARE), &decoded);
    assert!(
        direct < wrong,
        "square: right {direct} must beat the wrong layout {wrong}"
    );
}

// ---------------------------------------------------------------- forged data

#[test]
fn the_layout_matrix_rejects_a_payload_that_disagrees_with_the_declared_axis() {
    // The Phase 4.3 hole: an entry can state a coherent axis and shape while the
    // stored selectors say otherwise. The rule is physical, so a recomputed CRC
    // cannot hide it, and both auditors report the same code.
    let (code, reason) = validate_entry_layout(&EntryLayout {
        source_shape: vec![24, 16],
        orientation: Some(WeightOrientation::LinearOutIn),
        logical_shape: Some([24, 16]),
        role: Some(TensorRole::EmbeddingTable),
        index_axis: Some(0),
        context: LayoutContext::SourceEntry,
        quantized: true,
        codebook_count: 1,
        stored_index_bytes: 10_000,
        calibrated: false,
    })
    .expect_err("a mismatched selector payload must be rejected");
    assert_eq!(code, LayoutViolation::SelectorBytesMismatch);
    assert!(reason.contains("selector bytes"), "{reason}");

    // Same entry, honest payload: accepted.
    let honest = validate_entry_layout(&EntryLayout {
        source_shape: vec![24, 16],
        orientation: Some(WeightOrientation::LinearOutIn),
        logical_shape: Some([24, 16]),
        role: Some(TensorRole::EmbeddingTable),
        index_axis: Some(0),
        context: LayoutContext::SourceEntry,
        quantized: true,
        codebook_count: 1,
        stored_index_bytes: 24 * (16 / GROUP) / 2,
        calibrated: false,
    });
    assert!(honest.is_ok(), "{honest:?}");
}

#[test]
fn axis_one_is_refused_by_the_manifest_that_claims_it() {
    let error = manifest_document(&[(TABLE, &["embedding_table"], Some(1), None)])
        .expect_err("axis 1 has no producer path and must be refused");
    assert!(error.contains("axis_one_unsupported"), "{error}");
}

// ------------------------------------------------------------------- legacy

#[test]
fn a_legacy_container_without_roles_is_read_exactly_as_before() {
    let rows = 16_u64;
    let cols = 32_u64;
    let codebook_count = 1_u64;
    let index_bytes = rows * ((cols / GROUP) * codebook_count) / 2;
    let inputs = vec![
        StreamingTensorInput {
            tensor_id: 1,
            name: "dense.weight.hpsi_vq.codebook".to_owned(),
            dtype: DType::F32,
            dimensions: vec![codebook_count, 16, GROUP],
            block_lengths: vec![codebook_count * 16 * GROUP * 4],
        },
        StreamingTensorInput {
            tensor_id: 2,
            name: "dense.weight.hpsi_vq.indices".to_owned(),
            dtype: DType::Opaque,
            dimensions: vec![index_bytes],
            block_lengths: vec![index_bytes],
        },
    ];
    let payloads = inputs
        .iter()
        .map(|tensor| vec![0_u8; tensor.block_lengths[0] as usize])
        .collect::<Vec<_>>();
    let entries = vec![json!({
        "name": "dense.weight",
        "source_dtype": "F32",
        "source_shape": [rows, cols],
        "disposition": "quantize_vq",
        "weight_orientation": "linear_out_in",
        "orientation_source": "legacy_default",
        "logical_shape": [rows, cols],
    })];
    let mut output = Cursor::new(Vec::new());
    ModelWriter::write_streaming_with_metadata(
        &mut output,
        &inputs,
        Some(json!({"user": {"source_tensors": entries}})),
        |tensor_index, _block_index, sink| {
            let payload = payloads
                .get(tensor_index)
                .ok_or(FormatError::InvalidBlock("unknown legacy tensor"))?;
            sink.write_all(payload)
                .map_err(|_| FormatError::InvalidBlock("legacy write failed"))?;
            Ok(payload.len() as u64)
        },
    )
    .expect("a legacy container must write");
    let container: Vec<u8> = output.into_inner();
    let reader = open(container);
    reader.verify_all().expect("legacy integrity must verify");

    let audit = audit_container(&reader, None);
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);
    let facts = audit.facts("dense.weight").expect("audited");
    assert_eq!(facts.role, None, "no role was ever declared");
    assert_eq!(facts.role_assurance, RoleAssurance::NotDeclared);
    assert_eq!(audit.roles_declared, 0);
    assert_eq!(
        facts.assurance,
        OrientationAssurance::DeclaredNotVerified,
        "the pre-contract reading is unchanged"
    );
    let provenance: Vec<&Value> = reader.metadata.value["user"]["source_tensors"]
        .as_array()
        .expect("provenance")
        .iter()
        .collect();
    assert!(
        provenance[0].get("tensor_role").is_none(),
        "a legacy artifact must not grow a role"
    );
}

#[test]
fn a_vector_never_becomes_a_table_holding_an_index_axis() {
    // Order is part of the contract: an index axis is meaningless on a non-matrix
    // before the role itself is even considered, and both auditors report that
    // first code. The role-only case is covered by the shared table.
    let error = validate_entry_layout(&EntryLayout {
        source_shape: vec![16],
        orientation: Some(WeightOrientation::LinearOutIn),
        logical_shape: None,
        role: Some(TensorRole::EmbeddingTable),
        index_axis: Some(0),
        context: LayoutContext::SourceEntry,
        quantized: false,
        codebook_count: 0,
        stored_index_bytes: 0,
        calibrated: false,
    })
    .expect_err("a rank-1 tensor cannot be an indexed table");
    assert_eq!(error.0, LayoutViolation::AxisForNonIndexedRole);

    let role_only = validate_entry_layout(&EntryLayout {
        source_shape: vec![16],
        orientation: Some(WeightOrientation::LinearOutIn),
        logical_shape: None,
        role: Some(TensorRole::EmbeddingTable),
        index_axis: None,
        context: LayoutContext::SourceEntry,
        quantized: false,
        codebook_count: 0,
        stored_index_bytes: 0,
        calibrated: false,
    })
    .expect_err("a rank-1 tensor cannot carry a matrix role either");
    assert_eq!(role_only.0, LayoutViolation::MatrixRoleOnNonMatrix);
}
