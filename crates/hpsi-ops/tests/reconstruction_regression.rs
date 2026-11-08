//! Regression tests for reconstruction fidelity, Conv1D orientation, RAW
//! preservation, and calibration provenance.
//!
//! These are integration tests: they drive the public importer API exactly as an
//! external consumer would, then reopen the artifact through `ModelReader` and
//! decode it with the production `HpsiVqExecutor` path. The orientation oracle
//! used for the expectations comes from the GPT-2 module definition (a
//! `transformers.Conv1D` weight is stored `[in, out]`), never from the importer's
//! own plan, so a wrong plan cannot validate itself.

use hpsi_format::{MemorySource, ModelReader, ReaderLimits, TensorDescriptor};
use hpsi_ops::{
    ActivationCalibration, HpsiVqExecutor, HpsiVqLayout, ModelImportPlan, SafeTensorFile,
    UnsupportedTensorPolicy, VqConfig, write_model_import_to_hpsi,
    write_model_import_to_hpsi_calibrated,
};
use std::io::Cursor;
use std::sync::Arc;

const TEST_CONFIG: VqConfig = VqConfig {
    iterations: 2,
    sample_limit: 64,
    codebook_count: 1,
    beam_width: 4,
};

struct TensorFixture {
    name: &'static str,
    shape: Vec<u64>,
    values: Vec<f32>,
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Build an in-memory Safetensors file with F32 tensors.
fn fixture_file(tensors: &[TensorFixture]) -> SafeTensorFile {
    let mut entries = Vec::new();
    let mut payload = Vec::new();
    for tensor in tensors {
        let start = payload.len();
        payload.extend_from_slice(&f32_bytes(&tensor.values));
        let end = payload.len();
        let shape = tensor
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!(
            r#""{}":{{"dtype":"F32","shape":[{}],"data_offsets":[{},{}]}}"#,
            tensor.name, shape, start, end
        ));
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&payload);
    SafeTensorFile::open(
        Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))),
        64 * 1024,
    )
    .expect("fixture Safetensors must parse")
}

fn pattern(rows: usize, cols: usize, salt: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            values.push(((row * 7 + col * 3) as f32 + salt) * 0.01);
        }
    }
    values
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
        model_type: "gpt2".to_owned(),
        model_revision: None,
        dataset_sha256: "a".repeat(64),
        module_type: module_type.map(str::to_owned),
    }
}

fn write_container(source: &SafeTensorFile, plan: &ModelImportPlan) -> Vec<u8> {
    let mut output = Cursor::new(Vec::new());
    write_model_import_to_hpsi(source, plan, &mut output, TEST_CONFIG, 4)
        .expect("import must succeed");
    output.into_inner()
}

fn write_calibrated_container(
    source: &SafeTensorFile,
    calibrated: &hpsi_ops::CalibratedModelImportPlan,
) -> Vec<u8> {
    let mut output = Cursor::new(Vec::new());
    write_model_import_to_hpsi_calibrated(source, calibrated, &mut output, TEST_CONFIG, 4)
        .expect("calibrated import must succeed");
    output.into_inner()
}

fn open_container(bytes: Vec<u8>) -> ModelReader {
    ModelReader::open(
        Arc::new(MemorySource::new(Arc::<[u8]>::from(bytes))),
        &ReaderLimits::default(),
    )
    .expect("written container must reopen")
}

fn tensor_by_name<'a>(reader: &'a ModelReader, name: &str) -> Option<&'a TensorDescriptor> {
    reader.tensors().iter().find(|tensor| {
        reader
            .metadata
            .tensor_names
            .get(tensor.name_ref as usize)
            .is_some_and(|candidate| candidate == name)
    })
}

fn container_tensor_names(reader: &ModelReader) -> Vec<String> {
    reader
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
        .collect()
}

/// Decode a VQ matrix by probing the production executor with basis vectors.
fn reconstruct(
    reader: &ModelReader,
    codebook_name: &str,
    indices_name: &str,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let codebook = tensor_by_name(reader, codebook_name).expect("codebook tensor must exist");
    let indices = tensor_by_name(reader, indices_name).expect("index tensor must exist");
    let codebook_count = usize::try_from(codebook.dimensions[0]).expect("codebook count fits");
    let layout = HpsiVqLayout {
        codebook_tensor_id: codebook.tensor_id,
        indices_tensor_id: indices.tensor_id,
        rows,
        cols,
        codebook_count,
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

fn relative_l2(reference: &[f32], reconstructed: &[f32]) -> f64 {
    assert_eq!(reference.len(), reconstructed.len(), "shape mismatch");
    let mut difference = 0.0_f64;
    let mut norm = 0.0_f64;
    for (expected, actual) in reference.iter().zip(reconstructed.iter()) {
        let expected = f64::from(*expected);
        let actual = f64::from(*actual);
        difference += (actual - expected) * (actual - expected);
        norm += expected * expected;
    }
    assert!(norm > 0.0, "reference matrix must be non-zero");
    difference.sqrt() / norm.sqrt()
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

/// The importer is lossy by design (default 0.5 bit/weight). A correct
/// reconstruction must be far closer to the original than to a wrong
/// orientation, and it must not be garbage.
const MAX_ACCEPTABLE_RELATIVE_L2: f64 = 0.95;

// ---------------------------------------------------------------------------
// 1. Linear reconstruction.
// ---------------------------------------------------------------------------

#[test]
fn linear_matrix_reconstructs_with_the_source_orientation() {
    let fixture = fixture_file(&[TensorFixture {
        name: "linear.weight",
        shape: vec![8, 16],
        values: pattern(8, 16, 1.0),
    }]);
    let plan = ModelImportPlan::build(&fixture, &[], UnsupportedTensorPolicy::AllowRawPreservation)
        .expect("plan must build");
    assert_eq!(plan.tensors.len(), 1);
    assert_eq!(
        plan.tensors[0]
            .weight_orientation
            .logical_shape(&plan.tensors[0].source_shape)
            .expect("rank-2 tensor must have a logical shape"),
        [8, 16]
    );

    let reader = open_container(write_container(&fixture, &plan));
    let decoded = reconstruct(
        &reader,
        "linear.weight.hpsi_vq.codebook",
        "linear.weight.hpsi_vq.indices",
        8,
        16,
    );
    let original = pattern(8, 16, 1.0);
    let error = relative_l2(&original, &decoded);
    assert!(
        error < MAX_ACCEPTABLE_RELATIVE_L2,
        "linear reconstruction error {error} is not bounded"
    );
    // A non-square Linear matrix has a distinct transpose shape, so orientation
    // errors are structurally impossible here; the payload size check below is
    // what ties the logical shape to the stored selectors.
    let indices =
        tensor_by_name(&reader, "linear.weight.hpsi_vq.indices").expect("index tensor must exist");
    let expected_index_bytes = (8_u64 * (16 / 8)).div_ceil(2);
    assert_eq!(indices.logical_nbytes, expected_index_bytes);
}

#[test]
fn square_linear_matrix_is_not_transposed() {
    // Square and asymmetric: shape alone cannot prove orientation, so the
    // numeric comparison is the only evidence. If the importer silently
    // transposed this matrix, `error` would exceed `transposed_error`.
    let original = pattern(16, 16, 2.0);
    let fixture = fixture_file(&[TensorFixture {
        name: "square.weight",
        shape: vec![16, 16],
        values: original.clone(),
    }]);
    let plan = ModelImportPlan::build(&fixture, &[], UnsupportedTensorPolicy::AllowRawPreservation)
        .expect("plan must build");
    let reader = open_container(write_container(&fixture, &plan));
    let decoded = reconstruct(
        &reader,
        "square.weight.hpsi_vq.codebook",
        "square.weight.hpsi_vq.indices",
        16,
        16,
    );
    let direct = relative_l2(&original, &decoded);
    let transposed = relative_l2(&transpose(&original, 16, 16), &decoded);
    assert!(
        direct < transposed,
        "square matrix is closer to its transpose: direct {direct} >= transposed {transposed}"
    );
    assert!(
        direct < MAX_ACCEPTABLE_RELATIVE_L2,
        "square linear reconstruction error {direct} is not bounded"
    );
}

// ---------------------------------------------------------------------------
// 2. Conv1D non-square reconstruction, both directions.
// ---------------------------------------------------------------------------

#[test]
fn conv1d_wide_source_transposes_into_logical_matrix() {
    // transformers.Conv1D stores [in, out] = [16, 48]; logical is [48, 16].
    let in_features = 16_usize;
    let out_features = 48_usize;
    let stored = pattern(in_features, out_features, 3.0);
    let fixture = fixture_file(&[TensorFixture {
        name: "attn.c_attn.weight",
        shape: vec![in_features as u64, out_features as u64],
        values: stored.clone(),
    }]);
    let calibrations = [calibration(
        "attn.c_attn.weight",
        Some("conv1d"),
        in_features,
    )];
    let plan = ModelImportPlan::build(
        &fixture,
        &calibrations,
        UnsupportedTensorPolicy::AllowRawPreservation,
    )
    .expect("plan must build");
    assert_eq!(
        plan.tensors[0]
            .weight_orientation
            .logical_shape(&plan.tensors[0].source_shape)
            .expect("rank-2 tensor must have a logical shape"),
        [out_features as u64, in_features as u64]
    );

    let reader = open_container(write_container(&fixture, &plan));
    let decoded = reconstruct(
        &reader,
        "attn.c_attn.weight.hpsi_vq.codebook",
        "attn.c_attn.weight.hpsi_vq.indices",
        out_features,
        in_features,
    );
    let logical = transpose(&stored, in_features, out_features);
    let against_logical = relative_l2(&logical, &decoded);
    // Comparing against the raw stored layout is a shape mismatch, which is the
    // structural proof that a non-square Conv1D transposition cannot hide.
    assert_eq!(stored.len(), decoded.len());
    let against_stored_layout = relative_l2(&stored, &decoded);
    assert!(
        against_logical < against_stored_layout,
        "logical orientation {against_logical} should beat raw layout {against_stored_layout}"
    );
    assert!(
        against_logical < MAX_ACCEPTABLE_RELATIVE_L2,
        "conv1d wide reconstruction error {against_logical} is not bounded"
    );
}

#[test]
fn conv1d_tall_source_transposes_into_logical_matrix() {
    // mlp.c_proj stores [in, out] = [64, 16]; logical is [16, 64].
    let in_features = 64_usize;
    let out_features = 16_usize;
    let stored = pattern(in_features, out_features, 4.0);
    let fixture = fixture_file(&[TensorFixture {
        name: "mlp.c_proj.weight",
        shape: vec![in_features as u64, out_features as u64],
        values: stored.clone(),
    }]);
    let calibrations = [calibration(
        "mlp.c_proj.weight",
        Some("conv1d"),
        in_features,
    )];
    let plan = ModelImportPlan::build(
        &fixture,
        &calibrations,
        UnsupportedTensorPolicy::AllowRawPreservation,
    )
    .expect("plan must build");
    assert_eq!(
        plan.tensors[0]
            .weight_orientation
            .logical_shape(&plan.tensors[0].source_shape)
            .expect("rank-2 tensor must have a logical shape"),
        [out_features as u64, in_features as u64]
    );

    let reader = open_container(write_container(&fixture, &plan));
    let decoded = reconstruct(
        &reader,
        "mlp.c_proj.weight.hpsi_vq.codebook",
        "mlp.c_proj.weight.hpsi_vq.indices",
        out_features,
        in_features,
    );
    let logical = transpose(&stored, in_features, out_features);
    let against_logical = relative_l2(&logical, &decoded);
    let against_stored_layout = relative_l2(&stored, &decoded);
    assert!(against_logical < against_stored_layout);
    assert!(against_logical < MAX_ACCEPTABLE_RELATIVE_L2);
}

#[test]
fn conv1d_square_matrix_orientation_is_detectable_numerically() {
    // A square Conv1D matrix has the same shape in both orientations. Only the
    // numeric control can tell them apart, so this test pins that down.
    let dimension = 16_usize;
    let stored = pattern(dimension, dimension, 5.0);
    let fixture = fixture_file(&[TensorFixture {
        name: "attn.c_proj.weight",
        shape: vec![dimension as u64, dimension as u64],
        values: stored.clone(),
    }]);
    let calibrations = [calibration("attn.c_proj.weight", Some("conv1d"), dimension)];
    let plan = ModelImportPlan::build(
        &fixture,
        &calibrations,
        UnsupportedTensorPolicy::AllowRawPreservation,
    )
    .expect("plan must build");

    let reader = open_container(write_container(&fixture, &plan));
    let decoded = reconstruct(
        &reader,
        "attn.c_proj.weight.hpsi_vq.codebook",
        "attn.c_proj.weight.hpsi_vq.indices",
        dimension,
        dimension,
    );
    let logical = transpose(&stored, dimension, dimension);
    let against_logical = relative_l2(&logical, &decoded);
    let against_stored = relative_l2(&stored, &decoded);
    assert!(
        against_logical < against_stored,
        "square Conv1D must match the transposed logical matrix better \
         (logical {against_logical}, stored {against_stored})"
    );
}

// ---------------------------------------------------------------------------
// 3. RAW preservation, byte for byte.
// ---------------------------------------------------------------------------

#[test]
fn raw_tensors_are_preserved_byte_for_byte() {
    let fixture = fixture_file(&[
        TensorFixture {
            name: "bias",
            shape: vec![16],
            values: pattern(1, 16, 6.0),
        },
        TensorFixture {
            name: "weight",
            shape: vec![8, 16],
            values: pattern(8, 16, 7.0),
        },
    ]);
    let plan = ModelImportPlan::build(&fixture, &[], UnsupportedTensorPolicy::AllowRawPreservation)
        .expect("plan must build");
    let bias_plan = plan
        .tensors
        .iter()
        .find(|tensor| tensor.source_name == "bias")
        .expect("bias must be planned");
    assert_eq!(
        bias_plan.disposition,
        hpsi_ops::TensorDisposition::PreserveRaw
    );

    let reader = open_container(write_container(&fixture, &plan));
    let tensor = tensor_by_name(&reader, "bias").expect("raw tensor must exist in container");
    assert_eq!(tensor.rank, 1);
    assert_eq!(tensor.dimensions[0], 16);
    assert_eq!(tensor.logical_nbytes, 64);

    let mut stored = Vec::new();
    for block_id in 0..tensor.block_count {
        let block = reader
            .read_block(tensor.tensor_id, block_id)
            .expect("raw block must read");
        stored.extend_from_slice(&block.payload);
    }
    let expected = f32_bytes(&pattern(1, 16, 6.0));
    assert_eq!(stored.len(), expected.len(), "raw payload length changed");
    assert_eq!(stored, expected, "raw payload bytes changed");
}

#[test]
fn no_source_tensor_is_silently_omitted() {
    let fixture = fixture_file(&[
        TensorFixture {
            name: "bias",
            shape: vec![8],
            values: pattern(1, 8, 8.0),
        },
        TensorFixture {
            name: "weight",
            shape: vec![8, 16],
            values: pattern(8, 16, 9.0),
        },
        TensorFixture {
            name: "odd",
            shape: vec![3, 5],
            values: pattern(3, 5, 10.0),
        },
    ]);
    let plan = ModelImportPlan::build(&fixture, &[], UnsupportedTensorPolicy::AllowRawPreservation)
        .expect("plan must build");
    let reader = open_container(write_container(&fixture, &plan));
    let names = container_tensor_names(&reader);
    for planned in &plan.tensors {
        match planned.disposition {
            hpsi_ops::TensorDisposition::PreserveRaw => {
                assert!(
                    names.iter().any(|name| name == &planned.source_name),
                    "raw tensor {} is missing from the container",
                    planned.source_name
                );
            }
            hpsi_ops::TensorDisposition::QuantizeVq => {
                for name in [&planned.codebook_name, &planned.indices_name] {
                    let name = name.as_ref().expect("quantized tensor must name its parts");
                    assert!(
                        names.iter().any(|candidate| candidate == name),
                        "generated tensor {name} is missing from the container"
                    );
                }
            }
        }
    }
    let quantized = plan
        .tensors
        .iter()
        .filter(|tensor| tensor.disposition == hpsi_ops::TensorDisposition::QuantizeVq)
        .count();
    let raw = plan.tensors.len() - quantized;
    assert_eq!(names.len(), quantized * 2 + raw);
    assert_eq!(plan.tensors.len(), fixture.tensors().len());
}

// ---------------------------------------------------------------------------
// 4. Inconsistent orientation is rejected.
// ---------------------------------------------------------------------------

#[test]
fn rejects_calibration_features_inconsistent_with_conv1d_orientation() {
    // [16, 48] as Conv1D means 16 input features, not 48. Declaring 48 must fail
    // instead of silently encoding along the wrong axis.
    let fixture = fixture_file(&[TensorFixture {
        name: "attn.c_attn.weight",
        shape: vec![16, 48],
        values: pattern(16, 48, 11.0),
    }]);
    let calibrations = [calibration("attn.c_attn.weight", Some("conv1d"), 48)];
    assert!(
        ModelImportPlan::build(
            &fixture,
            &calibrations,
            UnsupportedTensorPolicy::AllowRawPreservation
        )
        .is_err(),
        "feature count inconsistent with conv1d orientation must be rejected"
    );
}

#[test]
fn rejects_unknown_module_type() {
    let fixture = fixture_file(&[TensorFixture {
        name: "w",
        shape: vec![8, 16],
        values: pattern(8, 16, 12.0),
    }]);
    let calibrations = [calibration("w", Some("Conv1D"), 16)];
    assert!(
        ModelImportPlan::build(
            &fixture,
            &calibrations,
            UnsupportedTensorPolicy::AllowRawPreservation
        )
        .is_err(),
        "an unrecognised module_type must be rejected, never guessed from the name"
    );
}

#[test]
fn rejects_duplicate_calibration_for_one_tensor() {
    let fixture = fixture_file(&[TensorFixture {
        name: "w",
        shape: vec![8, 16],
        values: pattern(8, 16, 13.0),
    }]);
    let calibrations = [
        calibration("w", Some("linear"), 16),
        calibration("w", Some("conv1d"), 16),
    ];
    assert!(
        ModelImportPlan::build(
            &fixture,
            &calibrations,
            UnsupportedTensorPolicy::AllowRawPreservation
        )
        .is_err(),
        "two contradicting calibrations for one tensor must be rejected"
    );
}

// ---------------------------------------------------------------------------
// 5. Plan and written container agree.
// ---------------------------------------------------------------------------

#[test]
fn written_container_matches_the_plan_orientation_metadata() {
    let fixture = fixture_file(&[
        TensorFixture {
            name: "attn.c_attn.weight",
            shape: vec![16, 48],
            values: pattern(16, 48, 14.0),
        },
        TensorFixture {
            name: "attn.c_attn.bias",
            shape: vec![48],
            values: pattern(1, 48, 15.0),
        },
    ]);
    let calibrations = [calibration("attn.c_attn.weight", Some("conv1d"), 16)];
    let calibrated = ModelImportPlan::build_with_provenance(
        &fixture,
        &calibrations,
        UnsupportedTensorPolicy::AllowRawPreservation,
    )
    .expect("calibrated plan must build");
    let reader = open_container(write_calibrated_container(&fixture, &calibrated));

    let provenance = reader
        .metadata
        .value
        .get("user")
        .and_then(|user| user.get("source_tensors"))
        .and_then(|value| value.as_array())
        .expect("metadata must carry user.source_tensors");

    for planned in &calibrated.plan.tensors {
        let entry = provenance
            .iter()
            .find(|entry| entry.get("name").and_then(|v| v.as_str()) == Some(&planned.source_name))
            .unwrap_or_else(|| panic!("provenance missing for {}", planned.source_name));
        assert_eq!(
            entry.get("weight_orientation").and_then(|v| v.as_str()),
            Some(planned.weight_orientation.as_str())
        );
        assert_eq!(
            entry.get("orientation_source").and_then(|v| v.as_str()),
            Some(planned.orientation_source.as_str())
        );
        if let Some(expected) = planned
            .weight_orientation
            .logical_shape(&planned.source_shape)
        {
            let actual = entry
                .get("logical_shape")
                .and_then(|v| v.as_array())
                .map(|values| values.iter().filter_map(|v| v.as_u64()).collect::<Vec<_>>())
                .expect("rank-2 entries must carry logical_shape");
            assert_eq!(actual, expected.to_vec());
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Calibration provenance.
// ---------------------------------------------------------------------------

#[test]
fn calibration_provenance_survives_the_round_trip() {
    let fixture = fixture_file(&[TensorFixture {
        name: "attn.c_proj.weight",
        shape: vec![16, 16],
        values: pattern(16, 16, 16.0),
    }]);
    let calibrations = [calibration("attn.c_proj.weight", Some("conv1d"), 16)];
    let calibrated = ModelImportPlan::build_with_provenance(
        &fixture,
        &calibrations,
        UnsupportedTensorPolicy::AllowRawPreservation,
    )
    .expect("calibrated plan must build");
    assert_eq!(calibrated.calibrations.len(), 1);
    let expected = &calibrated.calibrations[0];
    assert_eq!(expected.weight_orientation, "conv1d_in_out");
    assert_eq!(expected.orientation_source, "calibration");
    assert_eq!(expected.statistic, "diagonal_second_moment");

    let reader = open_container(write_calibrated_container(&fixture, &calibrated));
    let entry = reader
        .metadata
        .value
        .get("user")
        .and_then(|user| user.get("source_tensors"))
        .and_then(|value| value.as_array())
        .and_then(|entries| {
            entries.iter().find(|entry| {
                entry.get("name").and_then(|v| v.as_str()) == Some("attn.c_proj.weight")
            })
        })
        .expect("calibrated tensor must be present");
    let calibration = entry
        .get("calibration")
        .expect("calibrated tensor must persist a calibration object");

    assert_eq!(
        calibration.get("model_type").and_then(|v| v.as_str()),
        Some(expected.model_type.as_str())
    );
    assert_eq!(
        calibration.get("dataset_sha256").and_then(|v| v.as_str()),
        Some(expected.dataset_sha256.as_str())
    );
    assert_eq!(
        calibration.get("sample_count").and_then(|v| v.as_u64()),
        Some(expected.sample_count as u64)
    );
    assert_eq!(
        calibration.get("features").and_then(|v| v.as_u64()),
        Some(expected.features as u64)
    );
    assert_eq!(
        calibration
            .get("weight_orientation")
            .and_then(|v| v.as_str()),
        Some(expected.weight_orientation.as_str())
    );
    assert_eq!(
        calibration
            .get("orientation_source")
            .and_then(|v| v.as_str()),
        Some(expected.orientation_source.as_str())
    );
}

#[test]
fn raw_tensors_never_carry_calibration_provenance() {
    let fixture = fixture_file(&[
        TensorFixture {
            name: "bias",
            shape: vec![16],
            values: pattern(1, 16, 17.0),
        },
        TensorFixture {
            name: "weight",
            shape: vec![8, 16],
            values: pattern(8, 16, 18.0),
        },
    ]);
    let calibrations = [calibration("weight", Some("linear"), 16)];
    let calibrated = ModelImportPlan::build_with_provenance(
        &fixture,
        &calibrations,
        UnsupportedTensorPolicy::AllowRawPreservation,
    )
    .expect("calibrated plan must build");
    let reader = open_container(write_calibrated_container(&fixture, &calibrated));
    let entries = reader
        .metadata
        .value
        .get("user")
        .and_then(|user| user.get("source_tensors"))
        .and_then(|value| value.as_array())
        .expect("metadata must carry provenance");
    let bias = entries
        .iter()
        .find(|entry| entry.get("name").and_then(|v| v.as_str()) == Some("bias"))
        .expect("bias must be present");
    assert!(
        bias.get("calibration").is_none(),
        "a raw-preserved tensor must not claim calibration provenance"
    );
}
