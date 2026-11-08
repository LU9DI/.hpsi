//! Orientação de pesos Linear vs Conv1D no importador Safetensors → HPSI.
//!
//! O encoder VQ interpreta linhas como saídas e colunas como entradas, então
//! a matriz lógica precisa ser `[out_features, in_features]`. O Safetensors do
//! GPT-2 guarda `Linear.weight` como `[out, in]` e `transformers.Conv1D.weight`
//! como `[in, out]`. A orientação vêm da calibração (`module_type`), nunca de
//! inferência por nome ou dimensão.
use crate::vq::{GROUP_SIZE, HpsiVqExecutor, HpsiVqLayout, VqConfig, VqMatrix};
use crate::{
    ActivationCalibration, ModelImportPlan, SafeTensorFile, UnsupportedTensorPolicy,
    WeightOrientation, WeightOrientationSource,
};
use hpsi_format::{MemorySource, ModelReader, ReaderLimits, TensorDescriptor};
use std::io::Cursor;
use std::sync::Arc;

const TEST_CONFIG: VqConfig = VqConfig {
    iterations: 2,
    sample_limit: 16,
    codebook_count: 1,
    beam_width: 4,
};

struct TensorFixture {
    name: &'static str,
    dtype: &'static str,
    shape: Vec<u64>,
    payload: Vec<u8>,
}

fn fixture_file(tensors: &[TensorFixture]) -> SafeTensorFile {
    let mut entries = Vec::new();
    let mut payload = Vec::new();
    for tensor in tensors {
        let start = payload.len();
        payload.extend_from_slice(&tensor.payload);
        let end = payload.len();
        let shape = tensor
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!(
            r#""{}":{{"dtype":"{}","shape":[{}],"data_offsets":[{},{}]}}"#,
            tensor.name, tensor.dtype, shape, start, end
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
    .unwrap()
}

fn f32_payload(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn calibration(layer_name: &str, module_type: &str, features: usize) -> ActivationCalibration {
    ActivationCalibration {
        layer_name: layer_name.into(),
        sample_count: 16,
        features,
        activation_second_moment: vec![0.5; features],
        model_type: "gpt2".into(),
        model_revision: None,
        dataset_sha256: "a".repeat(64),
        module_type: Some(module_type.into()),
    }
}

/// Padrão determinístico e assimétrico para provar a orientação.
fn logical_pattern(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| {
            let row = index / cols;
            let col = index % cols;
            ((row as f32) * 3.7 + (col as f32) * 0.9) % 2.0 - 1.0
        })
        .collect()
}

fn transposed(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len());
    for col in 0..cols {
        for row in 0..rows {
            out.push(values[row * cols + col]);
        }
    }
    out
}

fn stored_output(file: &SafeTensorFile, plan: &ModelImportPlan) -> ModelReader {
    let mut output = Cursor::new(Vec::new());
    crate::write_model_import_to_hpsi(file, plan, &mut output, TEST_CONFIG, 1).unwrap();
    ModelReader::open(
        Arc::new(MemorySource::new(Arc::<[u8]>::from(output.into_inner()))),
        &ReaderLimits::default(),
    )
    .unwrap()
}

fn tensor_by_name<'a>(reader: &'a ModelReader, name: &str) -> Option<&'a TensorDescriptor> {
    let position = reader
        .metadata
        .tensor_names
        .iter()
        .position(|entry| entry == name)?;
    reader.tensors().get(position)
}

fn indices_bytes(reader: &ModelReader, name: &str) -> Vec<u8> {
    let tensor = tensor_by_name(reader, name).expect("indices tensor must exist");
    let mut bytes = Vec::new();
    for block_id in 0..tensor.block_count {
        let block = reader.read_block(tensor.tensor_id, block_id).unwrap();
        bytes.extend_from_slice(&block.payload);
    }
    bytes
}

fn source_entry(reader: &ModelReader, name: &str) -> serde_json::Value {
    reader.metadata.value["user"]["source_tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == name)
        .cloned()
        .expect("source entry must exist")
}

#[test]
fn linear_import_keeps_original_orientation() {
    let (out_features, in_features) = (24usize, 16usize);
    let logical = logical_pattern(out_features, in_features);
    let source = fixture_file(&[TensorFixture {
        name: "lm_head.weight",
        dtype: "F32",
        shape: vec![out_features as u64, in_features as u64],
        payload: f32_payload(&logical),
    }]);
    let calibrated = ModelImportPlan::build_with_provenance(
        &source,
        &[calibration("lm_head.weight", "linear", in_features)],
        UnsupportedTensorPolicy::Reject,
    )
    .unwrap();
    let reader = stored_output(&source, &calibrated.plan);
    reader.verify_all().unwrap();

    let reference = VqMatrix::encode(&logical, out_features, in_features, TEST_CONFIG).unwrap();
    assert_eq!(
        indices_bytes(&reader, "lm_head.weight.hpsi_vq.indices"),
        reference.packed_indices()
    );
    let source_entry = source_entry(&reader, "lm_head.weight");
    assert_eq!(source_entry["weight_orientation"], "linear_out_in");
    assert_eq!(source_entry["orientation_source"], "calibration");
    assert_eq!(source_entry["logical_shape"], serde_json::json!([24, 16]));
}

#[test]
fn conv1d_import_transposes_to_logical_matrix() {
    let (out_features, in_features) = (16usize, 48usize);
    let logical = logical_pattern(out_features, in_features);
    // Conv1D armazena [in, out]; a calibração declara conv1d + features de entrada.
    let source = fixture_file(&[TensorFixture {
        name: "attn.c_attn.weight",
        dtype: "F32",
        shape: vec![in_features as u64, out_features as u64],
        payload: f32_payload(&transposed(&logical, out_features, in_features)),
    }]);
    let calibrated = ModelImportPlan::build_with_provenance(
        &source,
        &[calibration("attn.c_attn.weight", "conv1d", in_features)],
        UnsupportedTensorPolicy::Reject,
    )
    .unwrap();
    let reader = stored_output(&source, &calibrated.plan);
    reader.verify_all().unwrap();

    let reference = VqMatrix::encode(&logical, out_features, in_features, TEST_CONFIG).unwrap();
    assert_eq!(
        indices_bytes(&reader, "attn.c_attn.weight.hpsi_vq.indices"),
        reference.packed_indices()
    );
    let source_entry = source_entry(&reader, "attn.c_attn.weight");
    assert_eq!(source_entry["weight_orientation"], "conv1d_in_out");
    assert_eq!(source_entry["logical_shape"], serde_json::json!([16, 48]));
}

#[test]
fn conv1d_square_matrix_still_transposes() {
    // Matriz quadrada assimétrica: trocar a orientação muda o resultado.
    let (side, _) = (16usize, ());
    let logical = logical_pattern(side, side);
    let source = fixture_file(&[TensorFixture {
        name: "attn.c_proj.weight",
        dtype: "F32",
        shape: vec![side as u64, side as u64],
        payload: f32_payload(&transposed(&logical, side, side)),
    }]);
    let calibrated = ModelImportPlan::build_with_provenance(
        &source,
        &[calibration("attn.c_proj.weight", "conv1d", side)],
        UnsupportedTensorPolicy::Reject,
    )
    .unwrap();
    let reader = stored_output(&source, &calibrated.plan);
    reader.verify_all().unwrap();

    let reference = VqMatrix::encode(&logical, side, side, TEST_CONFIG).unwrap();
    let naive =
        VqMatrix::encode(&transposed(&logical, side, side), side, side, TEST_CONFIG).unwrap();
    assert_eq!(
        indices_bytes(&reader, "attn.c_proj.weight.hpsi_vq.indices"),
        reference.packed_indices()
    );
    assert_ne!(reference.packed_indices(), naive.packed_indices());
}

#[test]
fn calibration_feature_count_is_checked_against_orientation() {
    let logical = logical_pattern(16, 16);
    let source = fixture_file(&[TensorFixture {
        name: "weight",
        dtype: "F32",
        shape: vec![16, 16],
        payload: f32_payload(&logical),
    }]);
    // Linear: features devem ser shape[1] = 16.
    let mut wrong = calibration("weight", "linear", 15);
    assert!(ModelImportPlan::build(&source, &[wrong], UnsupportedTensorPolicy::Reject).is_err());
    wrong = calibration("weight", "linear", 17);
    assert!(ModelImportPlan::build(&source, &[wrong], UnsupportedTensorPolicy::Reject).is_err());

    // Conv1D: features devem ser shape[0] = 16; shape[1] = 32 é a dimensão de saída.
    let transposed_source = fixture_file(&[TensorFixture {
        name: "weight",
        dtype: "F32",
        shape: vec![16, 32],
        payload: f32_payload(&transposed(&logical_pattern(32, 16), 32, 16)),
    }]);
    let wrong = calibration("weight", "conv1d", 32);
    assert!(
        ModelImportPlan::build(
            &transposed_source,
            &[wrong],
            UnsupportedTensorPolicy::Reject
        )
        .is_err()
    );
}

#[test]
fn missing_or_invalid_orientation_metadata_is_handled_explicitly() {
    // Sem module_type: compatibilidade legada Linear (registrada como legacy_default).
    let mut legacy = calibration("weight", "linear", 16);
    legacy.module_type = None;
    let logical = logical_pattern(16, 16);
    let source = fixture_file(&[TensorFixture {
        name: "weight",
        dtype: "F32",
        shape: vec![16, 16],
        payload: f32_payload(&logical),
    }]);
    let calibrated =
        ModelImportPlan::build_with_provenance(&source, &[legacy], UnsupportedTensorPolicy::Reject)
            .unwrap();
    let entry = calibrated
        .plan
        .tensors
        .iter()
        .find(|tensor| tensor.source_name == "weight")
        .unwrap();
    assert_eq!(entry.weight_orientation, WeightOrientation::LinearOutIn);
    assert_eq!(
        entry.orientation_source,
        WeightOrientationSource::LegacyDefault
    );
    assert_eq!(
        calibrated.calibrations[0].orientation_source,
        "legacy_default"
    );

    // Valores inválidos são rejeitados na calibração.
    let mut invalid = calibration("weight", "linear", 16);
    invalid.module_type = Some("Linear".into());
    assert!(ModelImportPlan::build(&source, &[invalid], UnsupportedTensorPolicy::Reject).is_err());
    let mut invalid = calibration("weight", "linear", 16);
    invalid.module_type = Some("conv1d_in_out".into());
    assert!(ModelImportPlan::build(&source, &[invalid], UnsupportedTensorPolicy::Reject).is_err());
    let mut invalid = calibration("weight", "linear", 16);
    invalid.module_type = Some("".into());
    assert!(ModelImportPlan::build(&source, &[invalid], UnsupportedTensorPolicy::Reject).is_err());
}

#[test]
fn contradictory_orientation_metadata_is_rejected() {
    let (out_features, in_features) = (16usize, 16usize);
    let logical = logical_pattern(out_features, in_features);
    let source = fixture_file(&[TensorFixture {
        name: "weight",
        dtype: "F32",
        shape: vec![in_features as u64, out_features as u64],
        payload: f32_payload(&transposed(&logical, out_features, in_features)),
    }]);
    let calibrated = ModelImportPlan::build_with_provenance(
        &source,
        &[calibration("weight", "conv1d", in_features)],
        UnsupportedTensorPolicy::Reject,
    )
    .unwrap();

    // Proveniência adulterada: orientação invertida.
    let mut forged_provenance = calibrated.clone();
    forged_provenance.calibrations[0].weight_orientation = "linear_out_in".into();
    let mut output = Cursor::new(Vec::new());
    assert!(
        crate::write_model_import_to_hpsi_calibrated(
            &source,
            &forged_provenance,
            &mut output,
            TEST_CONFIG,
            1,
        )
        .is_err()
    );

    // Plano adulterado: tensor marcado como Linear embora a calibração seja Conv1D.
    let mut forged_plan = calibrated.clone();
    forged_plan
        .plan
        .tensors
        .iter_mut()
        .find(|tensor| tensor.source_name == "weight")
        .unwrap()
        .weight_orientation = WeightOrientation::LinearOutIn;
    let mut output = Cursor::new(Vec::new());
    assert!(
        crate::write_model_import_to_hpsi_calibrated(
            &source,
            &forged_plan,
            &mut output,
            TEST_CONFIG,
            1,
        )
        .is_err()
    );

    // Origem da orientação adulterada: declarada como calibração sem calibração.
    let mut unsourced = calibrated.clone();
    unsourced.plan.tensors[0].activation_second_moment = None;
    unsourced.plan.tensors[0].calibration_samples = None;
    let mut output = Cursor::new(Vec::new());
    assert!(
        crate::write_model_import_to_hpsi_calibrated(
            &source,
            &unsourced,
            &mut output,
            TEST_CONFIG,
            1,
        )
        .is_err()
    );
}

#[test]
fn transposed_block_reader_matches_direct_reads() {
    // Vários tiles (bloco interno de 64 linhas) e fronteiras de GROUP_SIZE.
    let (out_features, in_features) = (200usize, 16usize);
    let logical = logical_pattern(out_features, in_features);
    let payload = f32_payload(&transposed(&logical, out_features, in_features));
    let source = fixture_file(&[TensorFixture {
        name: "big.weight",
        dtype: "F32",
        shape: vec![in_features as u64, out_features as u64],
        payload,
    }]);
    let reader = crate::TransposedFloatReader::new(&source, "big.weight").unwrap();
    assert_eq!(reader.logical_shape(), [200, 16]);

    // Leitura inteira e janelas alinhadas ao GROUP_SIZE atravessando tiles.
    let mut full = Vec::with_capacity(logical.len());
    let mut first = 0usize;
    while first < logical.len() {
        let mut window = vec![0.0_f32; GROUP_SIZE];
        reader
            .read_elements_into(first as u64, &mut window)
            .unwrap();
        full.extend_from_slice(&window);
        first += GROUP_SIZE;
    }
    assert_eq!(full, logical);

    let row_zero: Vec<f32> = (0..in_features).map(|i| logical[i]).collect();
    let mut row_buffer = vec![0.0_f32; in_features];
    reader.read_elements_into(0, &mut row_buffer).unwrap();
    assert_eq!(row_buffer, row_zero);

    // Janela que cruza o limite entre tiles (linha 64 → elemento 64*16).
    let mut tile_boundary = vec![0.0_f32; GROUP_SIZE];
    reader
        .read_elements_into(64 * in_features as u64, &mut tile_boundary)
        .unwrap();
    assert_eq!(
        tile_boundary,
        &logical[64 * in_features..64 * in_features + GROUP_SIZE]
    );

    // Pedido maior que GROUP_SIZE em uma única linha: inteiro, mas válido.
    let mut oversized = vec![0.0_f32; in_features + 1];
    assert!(reader.read_elements_into(0, &mut oversized).is_err());
}

#[test]
fn raw_tensors_are_copied_without_any_change() {
    let (out_features, in_features) = (16usize, 16usize);
    let logical = logical_pattern(out_features, in_features);
    let raw_payload: Vec<u8> = (0u8..24).collect();
    let source = fixture_file(&[
        TensorFixture {
            name: "proj.weight",
            dtype: "F32",
            shape: vec![out_features as u64, in_features as u64],
            payload: f32_payload(&logical),
        },
        TensorFixture {
            name: "buffer.norm",
            dtype: "I16",
            shape: vec![12],
            payload: raw_payload.clone(),
        },
    ]);
    let calibrated = ModelImportPlan::build_with_provenance(
        &source,
        &[calibration("proj.weight", "linear", in_features)],
        UnsupportedTensorPolicy::AllowRawPreservation,
    )
    .unwrap();
    let reader = stored_output(&source, &calibrated.plan);
    reader.verify_all().unwrap();
    let raw = tensor_by_name(&reader, "buffer.norm").unwrap();
    assert_eq!(raw.dimensions[0], 12);
    let mut bytes = Vec::new();
    for block_id in 0..raw.block_count {
        let block = reader.read_block(raw.tensor_id, block_id).unwrap();
        bytes.extend_from_slice(&block.payload);
    }
    assert_eq!(bytes, raw_payload);
    let entry = source_entry(&reader, "buffer.norm");
    assert_eq!(entry["disposition"], "preserve_raw");
    assert_eq!(entry["orientation_source"], "legacy_default");
}

#[test]
fn legacy_bundle_without_orientation_matches_linear() {
    let logical = logical_pattern(16, 16);
    let source = fixture_file(&[TensorFixture {
        name: "legacy.weight",
        dtype: "F32",
        shape: vec![16, 16],
        payload: f32_payload(&logical),
    }]);
    let legacy_json = format!(
        r#"{{"format":"hpsi-activation-calibration","version":1,"source":"transformers-forward-pre-hook","model":{{"local_path":"/synthetic","model_type":"gpt2","revision":null}},"dataset":{{"path":"/synthetic","sha256":"{}","processed_sequences":3}},"layer_name":"legacy.weight","shape":[16,16],"sample_count":16,"statistic":"diagonal_second_moment","activation_second_moment":[0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0]}}"#,
        "a".repeat(64)
    );
    let calibration = ActivationCalibration::from_json(&legacy_json).unwrap();
    assert_eq!(calibration.module_type, None);
    assert_eq!(
        calibration.weight_orientation(),
        Ok(WeightOrientation::LinearOutIn)
    );
    let calibrated = ModelImportPlan::build_with_provenance(
        &source,
        &[calibration],
        UnsupportedTensorPolicy::Reject,
    )
    .unwrap();
    let reader = stored_output(&source, &calibrated.plan);
    reader.verify_all().unwrap();
    let entry = source_entry(&reader, "legacy.weight");
    assert_eq!(entry["orientation_source"], "legacy_default");
    assert_eq!(entry["weight_orientation"], "linear_out_in");
    assert_eq!(entry["logical_shape"], serde_json::json!([16, 16]));
}

#[test]
fn reconstructed_imports_match_vq_reference_exactly() {
    // Conv1D: a importação inteira reproduz o encoder de referência sobre a
    // matriz lógica [out, in], e o executor lê as mesmas linhas/colunas.
    let (out_features, in_features) = (16usize, 48usize);
    let logical = logical_pattern(out_features, in_features);
    let source = fixture_file(&[TensorFixture {
        name: "attn.c_attn.weight",
        dtype: "F32",
        shape: vec![in_features as u64, out_features as u64],
        payload: f32_payload(&transposed(&logical, out_features, in_features)),
    }]);
    let calibrated = ModelImportPlan::build_with_provenance(
        &source,
        &[calibration("attn.c_attn.weight", "conv1d", in_features)],
        UnsupportedTensorPolicy::Reject,
    )
    .unwrap();
    let reader = stored_output(&source, &calibrated.plan);
    reader.verify_all().unwrap();

    let reference = VqMatrix::encode(&logical, out_features, in_features, TEST_CONFIG).unwrap();
    let codebook_tensor = tensor_by_name(&reader, "attn.c_attn.weight.hpsi_vq.codebook").unwrap();
    assert_eq!(codebook_tensor.dimensions[..3], [1, 16, GROUP_SIZE as u64]);
    let indices_tensor = tensor_by_name(&reader, "attn.c_attn.weight.hpsi_vq.indices").unwrap();
    assert_eq!(
        indices_tensor.dimensions[0],
        reference.packed_indices().len() as u64
    );

    let input: Vec<f32> = (0..in_features).map(|i| i as f32 * 0.25 + 1.0).collect();
    let mut expected = vec![0.0_f32; out_features];
    reference.view().matvec(&input, &mut expected).unwrap();

    let layout = HpsiVqLayout {
        codebook_tensor_id: codebook_tensor.tensor_id,
        indices_tensor_id: indices_tensor.tensor_id,
        rows: out_features,
        cols: in_features,
        codebook_count: TEST_CONFIG.codebook_count,
    };
    let executor = HpsiVqExecutor::new(&reader, layout).unwrap();
    let mut actual = vec![0.0_f32; out_features];
    executor.matvec_rows(&input, 0, &mut actual).unwrap();
    // Tolerância de 1e-4 no máximo: a ordenação de soma pode diferir por 1 ulp
    // entre o executor e o encoder de referência, mas não por transposição.
    let max_abs_error = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_abs_error <= 1e-4, "max_abs_error = {max_abs_error}");
}

#[test]
fn forged_plans_with_flipped_orientation_are_rejected() {
    let (out_features, in_features) = (16usize, 16usize);
    let logical = logical_pattern(out_features, in_features);
    let source = fixture_file(&[TensorFixture {
        name: "weight",
        dtype: "F32",
        shape: vec![in_features as u64, out_features as u64],
        payload: f32_payload(&transposed(&logical, out_features, in_features)),
    }]);
    let calibrated = ModelImportPlan::build_with_provenance(
        &source,
        &[calibration("weight", "conv1d", in_features)],
        UnsupportedTensorPolicy::Reject,
    )
    .unwrap();
    let mut forged_source = calibrated.clone();
    // Adulta o próprio plano: declara o tensor como Linear.
    forged_source
        .plan
        .tensors
        .iter_mut()
        .find(|tensor| tensor.source_name == "weight")
        .unwrap()
        .weight_orientation = WeightOrientation::LinearOutIn;
    let mut output = Cursor::new(Vec::new());
    assert!(
        crate::write_model_import_to_hpsi_calibrated(
            &source,
            &forged_source,
            &mut output,
            TEST_CONFIG,
            1,
        )
        .is_err()
    );
    // Adulta a origem da orientação: calibração declarada sem calibração.
    let mut forged_source = calibrated.clone();
    forged_source.plan.tensors[0].orientation_source = WeightOrientationSource::LegacyDefault;
    let mut output = Cursor::new(Vec::new());
    assert!(
        crate::write_model_import_to_hpsi_calibrated(
            &source,
            &forged_source,
            &mut output,
            TEST_CONFIG,
            1,
        )
        .is_err()
    );
}
