//! Parameter identity, aliases and multiple roles (Phase 4.5).
//!
//! The capture side now states, per *parameter*, every alias the model reaches it
//! under, every architectural role those aliases evidence, and which module
//! carried which role. This test is the consumer half of that contract: it parses
//! the manifests the real capture produces, refuses pairings that contradict the
//! roles they claim, plans and writes containers from them, and shows that a tied
//! embedding keeps both functions while two equal-but-distinct parameters stay
//! two tensors.
//!
//! Rules exercised: `docs/orientation-policy.md` §3 and §8, and
//! `docs/transformer-activation-capture.md` (capture bundle version 2).

use hpsi_format::{MemorySource, ModelReader, ReaderLimits, TensorDescriptor};
use hpsi_ops::{
    ActivationCalibration, ArchitectureFacts, CalibratedModelImportPlan, HpsiVqExecutor,
    HpsiVqLayout, ModelImportPlan, RoleAssurance, SafeTensorFile, TensorDisposition, TensorRole,
    UnsupportedTensorPolicy, VqConfig, WeightOrientation, audit_container, resolve_vq_layout,
    write_model_import_to_hpsi_calibrated,
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

const WTE: &str = "transformer.wte.weight";
const HEAD: &str = "lm_head.weight";
const POSITION: &str = "transformer.wpe.weight";

const VOCAB: usize = 10;
const HIDDEN: usize = 16;
const POSITIONS: usize = 4;

// ---------------------------------------------------------------- fixtures

/// Deterministic weights: the same `salt` produces the same array, so the untied
/// fixture can hold two parameters of *identical values* on purpose.
fn weights(rows: usize, cols: usize, salt: f32) -> Vec<f32> {
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

/// Tied model: one stored array, reached as an embedding and as the output head.
fn tied_source() -> SafeTensorFile {
    safetensors(&[
        (
            WTE,
            vec![VOCAB as u64, HIDDEN as u64],
            weights(VOCAB, HIDDEN, 0.1),
        ),
        (
            POSITION,
            vec![POSITIONS as u64, HIDDEN as u64],
            weights(POSITIONS, HIDDEN, 0.2),
        ),
    ])
}

/// Untied model: `lm_head.weight` holds the *same values* as `transformer.wte.weight`.
fn untied_source() -> SafeTensorFile {
    let table = weights(VOCAB, HIDDEN, 0.1);
    safetensors(&[
        (WTE, vec![VOCAB as u64, HIDDEN as u64], table.clone()),
        (HEAD, vec![VOCAB as u64, HIDDEN as u64], table),
        (
            POSITION,
            vec![POSITIONS as u64, HIDDEN as u64],
            weights(POSITIONS, HIDDEN, 0.2),
        ),
    ])
}

fn calibration(layer: &str, module_type: &str, features: usize) -> ActivationCalibration {
    ActivationCalibration {
        layer_name: layer.to_owned(),
        sample_count: 8,
        features,
        activation_second_moment: vec![1.0; features],
        model_type: "gpt2".to_owned(),
        model_revision: None,
        dataset_sha256: "c".repeat(64),
        module_type: Some(module_type.to_owned()),
    }
}

/// The tied capture measured the head, whose weight is the table's array, so the
/// layer is recorded under the stored key with the projection class.
fn tied_capture() -> Vec<ActivationCalibration> {
    vec![calibration(WTE, "linear", HIDDEN)]
}

fn untied_capture() -> Vec<ActivationCalibration> {
    vec![calibration(HEAD, "linear", HIDDEN)]
}

fn embedding_evidence(alias: &str, module: &str) -> Value {
    json!({
        "alias": alias,
        "module": module,
        "module_class": "embedding",
        "tensor_role": "embedding_table",
        "index_axis": 0,
    })
}

fn linear_evidence(alias: &str, module: &str) -> Value {
    json!({
        "alias": alias,
        "module": module,
        "module_class": "linear",
        "tensor_role": "linear_projection",
        "weight_orientation": "linear_out_in",
    })
}

/// The tied manifest the capture tool writes for a real `tie_word_embeddings` model.
fn tied_manifest_json() -> Value {
    json!({
        "format": "hpsi-architecture-manifest",
        "version": 1,
        "model_type": "gpt2",
        "tensors": [
            {
                "name": WTE,
                "tensor_role": ["embedding_table", "linear_projection"],
                "weight_orientation": "linear_out_in",
                "index_axis": 0,
                "aliases": [HEAD, WTE],
                "role_evidence": [
                    linear_evidence(HEAD, "lm_head"),
                    embedding_evidence(WTE, "transformer.wte"),
                ],
            },
            {
                "name": POSITION,
                "tensor_role": ["embedding_table"],
                "index_axis": 0,
                "aliases": [POSITION],
                "role_evidence": [embedding_evidence(POSITION, "transformer.wpe")],
            },
        ],
        "oracle": "module classes recorded by the activation capture",
    })
}

/// The untied manifest: two independent parameters, identical values.
fn untied_manifest_json() -> Value {
    json!({
        "format": "hpsi-architecture-manifest",
        "version": 1,
        "model_type": "gpt2",
        "tensors": [
            {
                "name": WTE,
                "tensor_role": ["embedding_table"],
                "index_axis": 0,
                "aliases": [WTE],
                "role_evidence": [embedding_evidence(WTE, "transformer.wte")],
            },
            {
                "name": HEAD,
                "tensor_role": ["linear_projection"],
                "weight_orientation": "linear_out_in",
                "aliases": [HEAD],
                "role_evidence": [linear_evidence(HEAD, "lm_head")],
            },
            {
                "name": POSITION,
                "tensor_role": ["embedding_table"],
                "index_axis": 0,
                "aliases": [POSITION],
                "role_evidence": [embedding_evidence(POSITION, "transformer.wpe")],
            },
        ],
        "oracle": "module classes recorded by the activation capture",
    })
}

fn facts(manifest: &Value) -> ArchitectureFacts {
    ArchitectureFacts::from_json(&manifest.to_string())
        .unwrap_or_else(|error| panic!("manifest must parse: {error}"))
}

fn tied_facts() -> ArchitectureFacts {
    facts(&tied_manifest_json())
}

fn untied_facts() -> ArchitectureFacts {
    facts(&untied_manifest_json())
}

fn planned(
    source: &SafeTensorFile,
    capture: &[ActivationCalibration],
    manifest: &Value,
) -> CalibratedModelImportPlan {
    let mut calibrated = ModelImportPlan::build_with_provenance(
        source,
        capture,
        UnsupportedTensorPolicy::AllowRawPreservation,
    )
    .expect("calibrated plan must build");
    calibrated
        .plan
        .apply_architecture_roles(&facts(manifest))
        .expect("the manifest roles must apply");
    calibrated
}

fn written(
    source: &SafeTensorFile,
    capture: &[ActivationCalibration],
    manifest: &Value,
) -> Vec<u8> {
    let calibrated = planned(source, capture, manifest);
    let mut output = Cursor::new(Vec::new());
    write_model_import_to_hpsi_calibrated(source, &calibrated, &mut output, TEST_CONFIG, 4)
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

fn source_f32(source: &SafeTensorFile, name: &str) -> Vec<f32> {
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

fn source_bytes(source: &SafeTensorFile, name: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    source
        .copy_raw_payload(name, &mut bytes, 64 * 1024)
        .unwrap_or_else(|error| panic!("{name}: {error:?}"));
    bytes
}

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

/// Reject a manifest and return the reason, so each rule can name its own text.
fn rejection(manifest: &Value) -> String {
    ArchitectureFacts::from_json(&manifest.to_string())
        .err()
        .unwrap_or_else(|| panic!("{manifest} must be refused"))
        .to_string()
}

// ------------------------------------------------------ the manifest contract

#[test]
fn the_manifest_keeps_both_roles_and_names_the_alias_behind_each() {
    let manifest = tied_facts();
    let tied = manifest
        .expectation(WTE)
        .expect("the tied parameter is described by its stored key");

    assert_eq!(
        tied.roles,
        vec![TensorRole::EmbeddingTable, TensorRole::LinearProjection],
        "sorted and deduplicated, both functions preserved"
    );
    assert_eq!(tied.index_axis, Some(0));
    assert_eq!(tied.orientation, Some(WeightOrientation::LinearOutIn));
    assert_eq!(tied.aliases, vec![HEAD.to_owned(), WTE.to_owned()]);

    let evidence: Vec<(String, TensorRole, String)> = tied
        .role_evidence
        .iter()
        .map(|item| (item.alias.clone(), item.role, item.module_class.clone()))
        .collect();
    assert_eq!(
        evidence,
        vec![
            (
                HEAD.to_owned(),
                TensorRole::LinearProjection,
                "linear".to_owned()
            ),
            (
                WTE.to_owned(),
                TensorRole::EmbeddingTable,
                "embedding".to_owned()
            ),
        ],
        "the pairing, not just the set of roles"
    );
    assert_eq!(
        tied.role_evidence[0].orientation,
        Some(WeightOrientation::LinearOutIn)
    );
    assert_eq!(tied.role_evidence[1].index_axis, Some(0));
}

#[test]
fn a_positional_table_is_recorded_without_a_projection_claim() {
    let manifest = tied_facts();
    let position = manifest
        .expectation(POSITION)
        .expect("the positional table is described");
    assert_eq!(position.roles, vec![TensorRole::EmbeddingTable]);
    assert_eq!(position.index_axis, Some(0));
    assert_eq!(
        position.orientation, None,
        "an indexing role states no layout of its own"
    );
    assert_eq!(
        position.effective_orientation(),
        Some(WeightOrientation::LinearOutIn),
        "and axis 0 entails how the quantizer grouped it"
    );
    assert_eq!(position.aliases, vec![POSITION.to_owned()]);
}

#[test]
fn a_pairing_that_disagrees_with_its_roles_is_refused() {
    let cases: Vec<(Value, &str)> = vec![
        (
            {
                let mut manifest = tied_manifest_json();
                manifest["tensors"][0]["tensor_role"] = json!(["linear_projection"]);
                manifest
            },
            "which `tensor_role` does not declare",
        ),
        (
            {
                let mut manifest = tied_manifest_json();
                manifest["tensors"][0]["role_evidence"] = json!([linear_evidence(HEAD, "lm_head")]);
                manifest
            },
            "with no role_evidence naming the module that evidences it",
        ),
        (
            {
                let mut manifest = tied_manifest_json();
                manifest["tensors"][0]["aliases"] = json!([WTE]);
                manifest
            },
            "is not among `aliases`",
        ),
        (
            {
                let mut manifest = tied_manifest_json();
                manifest["tensors"][0]["role_evidence"][1] =
                    json!({"alias": WTE, "module": "transformer.wte", "module_class": "embedding"});
                manifest
            },
            "manifest role_evidence lacks a non-empty `tensor_role`",
        ),
        (
            {
                let mut manifest = tied_manifest_json();
                manifest["tensors"][0]["role_evidence"][1]
                    .as_object_mut()
                    .expect("row")
                    .remove("index_axis");
                manifest
            },
            "names embedding_table without an index_axis",
        ),
        (
            {
                let mut manifest = tied_manifest_json();
                manifest["tensors"][0]["role_evidence"][1]["index_axis"] = json!(1);
                manifest
            },
            "while the entry declares",
        ),
        (
            {
                let mut manifest = tied_manifest_json();
                manifest["tensors"][0]["role_evidence"][1]["weight_orientation"] =
                    json!("conv1d_in_out");
                manifest
            },
            "which that role does not imply",
        ),
        (
            {
                let mut manifest = tied_manifest_json();
                manifest["tensors"][0]["role_evidence"][0]["weight_orientation"] =
                    json!("conv1d_in_out");
                manifest
            },
            "which that role does not imply",
        ),
        (
            {
                let mut manifest = tied_manifest_json();
                manifest["tensors"][0]["role_evidence"][0]["index_axis"] = json!(0);
                manifest
            },
            "only meaningful for embedding_table",
        ),
        (
            {
                let mut manifest = tied_manifest_json();
                manifest["tensors"][0]["role_evidence"][0]["alias"] = json!("  ");
                manifest
            },
            "lacks a non-empty `alias`",
        ),
        (
            {
                let mut manifest = tied_manifest_json();
                manifest["tensors"][0]["role_evidence"] = json!([
                    linear_evidence(HEAD, "lm_head"),
                    embedding_evidence(WTE, "transformer.wte"),
                    embedding_evidence(WTE, "transformer.wte")
                ]);
                manifest
            },
            "",
        ),
    ];

    for (manifest, fragment) in cases {
        let reason = rejection(&manifest);
        assert!(
            fragment.is_empty() || reason.contains(fragment),
            "expected {fragment:?}, got {reason}"
        );
    }
}

#[test]
fn duplicate_evidence_rows_are_refused_rather_than_quietly_merged() {
    let mut manifest = tied_manifest_json();
    let rows = manifest["tensors"][0]["role_evidence"]
        .as_array_mut()
        .expect("array");
    rows.push(rows[0].clone());
    let reason = rejection(&manifest);
    assert!(
        reason.contains("evidences this parameter more than once") || reason.contains("twice"),
        "{reason}"
    );
}

#[test]
fn a_manifest_without_the_new_keys_reads_exactly_as_before() {
    // The Phase 4.2-4.4 vocabulary: a role list, no aliases, no pairing.
    let legacy = json!({
        "format": "hpsi-architecture-manifest",
        "version": 1,
        "model_type": "gpt2",
        "tensors": [
            {"name": WTE, "tensor_role": ["embedding_table"], "index_axis": 0},
            {"name": HEAD, "weight_orientation": "linear_out_in"},
        ],
    });
    let facts = facts(&legacy);

    assert_eq!(
        facts.expectation(WTE).expect("entry").roles,
        vec![TensorRole::EmbeddingTable]
    );
    assert!(
        facts.expectation(WTE).expect("entry").aliases.is_empty(),
        "no alias evidence stated, so none is invented"
    );
    assert!(
        facts
            .expectation(WTE)
            .expect("entry")
            .role_evidence
            .is_empty()
    );
    assert_eq!(
        facts
            .expectation(HEAD)
            .expect("orientation-only entry")
            .roles,
        Vec::<TensorRole>::new(),
        "an entry that states only a layout still states no role"
    );
}

// ------------------------------------------------------------ the tied chain

#[test]
fn the_tied_weight_is_written_once_and_audited_under_both_functions() {
    let source = tied_source();
    let reader = open(written(&source, &tied_capture(), &tied_manifest_json()));
    let records = reader
        .verify_all()
        .expect("every CRC32C record must verify");
    assert!(records > 0);
    reader
        .verify_padding(reader.header.file_size)
        .expect("every padding byte must be zero");

    let audit = audit_container(&reader, Some(&tied_facts()));
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);

    let tied = audit.facts(WTE).expect("the tied parameter is audited");
    // The recorded role is the function the payload was encoded under, while the
    // model definition keeps both, with the alias that carries each.
    assert_eq!(tied.role, Some(TensorRole::LinearProjection));
    assert_eq!(tied.role_assurance, RoleAssurance::Confirmed);
    assert_eq!(
        tied.architecture_roles,
        vec![TensorRole::EmbeddingTable, TensorRole::LinearProjection,]
    );
    assert_eq!(
        tied.architecture_aliases,
        vec![HEAD.to_owned(), WTE.to_owned()]
    );
    assert_eq!(tied.role_evidence.len(), 2);
    assert_eq!(tied.disposition, "quantize_vq");

    let position = audit.facts(POSITION).expect("the positional table");
    assert_eq!(position.role, Some(TensorRole::EmbeddingTable));
    assert_eq!(position.index_axis, Some(0));
    assert_eq!(position.role_assurance, RoleAssurance::Confirmed);
    assert_eq!(position.disposition, "preserve_raw");
    assert!(!position.is_projection());
    assert_eq!(position.role_evidence[0].module_class, "embedding");

    assert_eq!(
        audit.indexed_tables, 1,
        "only the unmeasured table is indexed"
    );

    // One array, one payload: nothing was written twice for the tied parameter.
    let stored: Vec<String> = reader
        .tensors()
        .iter()
        .map(|tensor| {
            reader
                .metadata
                .tensor_names
                .get(tensor.name_ref as usize)
                .cloned()
                .unwrap_or_default()
        })
        .collect();
    assert!(
        !stored.iter().any(|name| name.starts_with("lm_head.")),
        "the tied head has no separate payload: {stored:?}"
    );
}

#[test]
fn the_tied_weight_reconstructs_by_the_projection_layout_it_was_encoded_under() {
    let source = tied_source();
    let reader = open(written(&source, &tied_capture(), &tied_manifest_json()));
    let reference = source_f32(&source, WTE);
    let decoded = reconstruct(&reader, WTE, VOCAB, HIDDEN);

    let direct = relative_l2(&reference, &decoded);
    let flipped = relative_l2(&transpose(&reference, VOCAB, HIDDEN), &decoded);
    assert!(
        direct < flipped,
        "the tied payload must follow the head's layout: {direct} vs {flipped}"
    );
    assert!(
        direct < 0.2,
        "quantization of the tied weight is coarse but ordered: {direct}"
    );
}

#[test]
fn the_projection_path_refuses_a_table_the_definition_only_indexes() {
    let source = tied_source();
    let reader = open(written(&source, &tied_capture(), &tied_manifest_json()));
    let error = resolve_vq_layout(&reader, POSITION, Some(&tied_facts()), false)
        .expect_err("a table is not a projection");
    let reason = error.to_string();
    assert!(
        reason.contains("is not a projection") || reason.contains("no VQ layout to resolve"),
        "{reason}"
    );
    assert!(
        !reason.contains("allow-unverified"),
        "the role gate must not offer to bypass a table: {reason}"
    );
}

// ---------------------------------------------------------- the untied chain

#[test]
fn two_parameters_of_equal_values_stay_two_tensors_with_two_functions() {
    let source = untied_source();
    // The fixture's premise, checked rather than assumed.
    assert_eq!(
        source_bytes(&source, WTE),
        source_bytes(&source, HEAD),
        "the untied fixture holds identical values on purpose"
    );

    let reader = open(written(&source, &untied_capture(), &untied_manifest_json()));
    reader.verify_all().expect("integrity must verify");
    let audit = audit_container(&reader, Some(&untied_facts()));
    assert!(audit.invalid.is_empty(), "{:?}", audit.invalid);

    let table = audit.facts(WTE).expect("the embedding table");
    assert_eq!(table.role, Some(TensorRole::EmbeddingTable));
    assert_eq!(table.index_axis, Some(0));
    assert_eq!(table.role_assurance, RoleAssurance::Confirmed);
    assert_eq!(table.disposition, "preserve_raw");
    assert_eq!(table.architecture_roles, vec![TensorRole::EmbeddingTable]);
    assert_eq!(table.architecture_aliases, vec![WTE.to_owned()]);

    let head = audit.facts(HEAD).expect("the output head");
    assert_eq!(head.role, Some(TensorRole::LinearProjection));
    assert_eq!(head.index_axis, None);
    assert_eq!(head.disposition, "quantize_vq");
    assert_eq!(head.architecture_aliases, vec![HEAD.to_owned()]);

    // Identity kept them apart: the table is verbatim source bytes, the head a
    // separate quantized payload, and neither claims the other's aliases.
    assert_eq!(
        stored_bytes(&reader, WTE),
        source_bytes(&source, WTE),
        "the table stays RAW and byte-identical"
    );
    assert_eq!(audit.indexed_tables, 2);
}

#[test]
fn the_untied_head_reconstructs_and_the_untied_table_is_not_a_projection() {
    let source = untied_source();
    let reader = open(written(&source, &untied_capture(), &untied_manifest_json()));

    let reference = source_f32(&source, HEAD);
    let decoded = reconstruct(&reader, HEAD, VOCAB, HIDDEN);
    let direct = relative_l2(&reference, &decoded);
    let flipped = relative_l2(&transpose(&reference, VOCAB, HIDDEN), &decoded);
    assert!(direct < flipped, "{direct} vs {flipped}");

    let error = resolve_vq_layout(&reader, WTE, Some(&untied_facts()), false)
        .expect_err("the table is indexed, not multiplied");
    let reason = error.to_string();
    assert!(
        reason.contains("is not a projection") || reason.contains("no VQ layout to resolve"),
        "{reason}"
    );
    assert!(
        !reason.contains("allow-unverified"),
        "a table is refused outright, not offered a bypass: {reason}"
    );
}

// ----------------------------------------------------------- conflict refusal

#[test]
fn a_table_role_on_a_measured_projection_is_refused_before_anything_is_written() {
    // The untied manifest calls WTE only an indexed table. Fed to the *tied*
    // capture, where the same array was measured as a projection, the two claims
    // cannot both be encoded: the plan refuses instead of choosing one silently.
    let source = tied_source();
    let mut calibrated = ModelImportPlan::build_with_provenance(
        &source,
        &tied_capture(),
        UnsupportedTensorPolicy::AllowRawPreservation,
    )
    .expect("calibrated plan must build");
    let error = calibrated
        .plan
        .apply_architecture_roles(&untied_facts())
        .expect_err("a measured projection may not be relabelled a table");
    assert!(
        error.contains("only an indexed table") || error.contains("must keep the projection role"),
        "{error}"
    );
}

#[test]
fn an_alias_that_another_identity_already_claims_is_refused() {
    let mut manifest = untied_manifest_json();
    // Both entries now name the same alias: no single parameter can own it.
    manifest["tensors"][0]["aliases"] = json!([WTE, HEAD]);
    manifest["tensors"][0]["role_evidence"] = json!([
        embedding_evidence(WTE, "transformer.wte"),
        embedding_evidence(HEAD, "transformer.wte")
    ]);
    manifest["tensors"][0]["tensor_role"] = json!(["embedding_table"]);
    let reason = rejection(&manifest);
    assert!(reason.contains("twice"), "{reason}");
}

#[test]
fn aliases_may_be_stated_where_no_pairing_is() {
    let manifest = json!({
        "format": "hpsi-architecture-manifest",
        "version": 1,
        "model_type": "gpt2",
        "tensors": [
            {"name": WTE, "tensor_role": ["embedding_table"], "index_axis": 0,
             "aliases": [WTE, HEAD]},
        ],
    });
    let facts = facts(&manifest);
    let entry = facts.expectation(WTE).expect("entry");
    assert_eq!(entry.aliases, vec![HEAD.to_owned(), WTE.to_owned()]);
    assert!(
        entry.role_evidence.is_empty(),
        "stating names without pairing them stays absence of the association"
    );
}

#[test]
fn the_pairing_fills_in_the_alias_list_when_the_producer_omits_it() {
    let mut manifest = tied_manifest_json();
    manifest["tensors"][0]
        .as_object_mut()
        .expect("entry")
        .remove("aliases");
    let entry = facts(&manifest).expectation(WTE).expect("entry").clone();
    assert_eq!(
        entry.aliases,
        vec![HEAD.to_owned(), WTE.to_owned()],
        "the evidence rows already name the parameter"
    );
}

#[test]
fn an_unknown_role_in_the_pairing_is_refused_with_the_shared_code() {
    let mut manifest = tied_manifest_json();
    manifest["tensors"][0]["role_evidence"][0]["tensor_role"] = json!("linear");
    let reason = rejection(&manifest);
    assert!(reason.contains("role_token_unknown"), "{reason}");
}

#[test]
fn a_planned_tied_tensor_keeps_the_layout_the_head_needs() {
    let source = tied_source();
    let calibrated = planned(&source, &tied_capture(), &tied_manifest_json());
    let tied = calibrated
        .plan
        .tensors
        .iter()
        .find(|tensor| tensor.source_name == WTE)
        .expect("the tied parameter is planned");
    assert_eq!(tied.role, TensorRole::LinearProjection);
    assert!(matches!(tied.disposition, TensorDisposition::QuantizeVq));
    assert_eq!(tied.index_axis, None);
    assert_eq!(tied.weight_orientation, WeightOrientation::LinearOutIn);

    let position = calibrated
        .plan
        .tensors
        .iter()
        .find(|tensor| tensor.source_name == POSITION)
        .expect("the positional table is planned");
    assert_eq!(position.role, TensorRole::EmbeddingTable);
    assert_eq!(position.index_axis, Some(0));
    assert!(matches!(
        position.disposition,
        TensorDisposition::PreserveRaw
    ));
}
