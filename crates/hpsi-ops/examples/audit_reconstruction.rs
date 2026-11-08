//! Read-only end-to-end numeric audit of an HPSI container against its source Safetensors model.
//!
//! The auditor never writes into the audited artifacts. It:
//!
//! 1. opens the source Safetensors file and the HPSI container;
//! 2. reads the import provenance persisted by the writer (`user.source_tensors`);
//! 3. reconstructs every VQ tensor with the production decode path (`HpsiVqExecutor`);
//! 4. reads every RAW tensor from the container and compares it byte-for-byte;
//! 5. compares the reconstruction against an *architecture oracle*: the
//!    orientation a model-definition manifest (`--architecture`) requires for
//!    the exact tensor name. The manifest is produced from the model's module
//!    classes, never from the container under audit, and orientation is never
//!    inferred from a name pattern or from coinciding dimensions. Without a
//!    manifest this auditor has no architectural reference and says so.
//!
//! It emits a JSON report and a Markdown table.
//!
//! ```text
//! cargo run -p hpsi-ops --example audit_reconstruction -- \
//!   --source .tmp-gpt2-conv1d-test/model/model.safetensors \
//!   --hpsi   .tmp-gpt2-conv1d-test/gpt2-calibrated.hpsi \
//!   --architecture .tmp-gpt2-conv1d-test/architecture.json \
//!   --out    .tmp-gpt2-conv1d-test/audit/rust/calibrated.json \
//!   --md     .tmp-gpt2-conv1d-test/audit/rust/calibrated.md \
//!   --label  calibrated
//! ```

use hpsi_format::{DType, FileSource, ModelReader, ReaderLimits};
use hpsi_ops::{
    ArchitectureExpectation, ArchitectureFacts, HpsiVqExecutor, HpsiVqLayout, OrientationAssurance,
    SafeTensorFile, TensorRole, WeightOrientation,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const COPY_CHUNK_BYTES: usize = 64 * 1024;
const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default)]
struct Metrics {
    n: usize,
    mae: f64,
    rmse: f64,
    relative_l2: f64,
    max_abs: f64,
    l2_reference: f64,
    l2_reconstructed: f64,
}

impl Metrics {
    fn to_json(self) -> Value {
        json!({
            "elements": self.n,
            "mae": self.mae,
            "rmse": self.rmse,
            "relative_l2": self.relative_l2,
            "max_abs": self.max_abs,
            "l2_reference": self.l2_reference,
            "l2_reconstructed": self.l2_reconstructed,
        })
    }
}

fn metrics(reference: &[f32], reconstructed: &[f32]) -> Result<Metrics, String> {
    if reference.len() != reconstructed.len() {
        return Err(format!(
            "metric inputs differ in length: {} vs {}",
            reference.len(),
            reconstructed.len()
        ));
    }
    if reference.is_empty() {
        return Err("metric inputs are empty".to_owned());
    }
    let mut sum_abs = 0.0_f64;
    let mut sum_squared = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut reference_squared = 0.0_f64;
    let mut reconstructed_squared = 0.0_f64;
    for (left, right) in reference.iter().zip(reconstructed.iter()) {
        let reference_value = f64::from(*left);
        let reconstructed_value = f64::from(*right);
        let difference = reconstructed_value - reference_value;
        let absolute = difference.abs();
        sum_abs += absolute;
        sum_squared += difference * difference;
        if absolute > max_abs {
            max_abs = absolute;
        }
        reference_squared += reference_value * reference_value;
        reconstructed_squared += reconstructed_value * reconstructed_value;
    }
    let count = reference.len() as f64;
    let reference_norm = reference_squared.sqrt();
    Ok(Metrics {
        n: reference.len(),
        mae: sum_abs / count,
        rmse: (sum_squared / count).sqrt(),
        relative_l2: if reference_norm == 0.0 {
            f64::NAN
        } else {
            sum_squared.sqrt() / reference_norm
        },
        max_abs,
        l2_reference: reference_norm,
        l2_reconstructed: reconstructed_squared.sqrt(),
    })
}

/// How far the orientation of one source tensor was established, using the
/// same four assurance levels the Rust CLI reports plus the two states only an
/// auditor can name.
///
/// `architecturally_confirmed` is the sole state that means the container's
/// declaration was checked against something outside the container. Nothing
/// here infers orientation from a tensor name or from a dimension that happens
/// to fit: with no manifest a declared matrix is `declared_not_verified`, and a
/// declared square matrix is `square_not_verifiable` because both orientations
/// produce the identical physical layout.
fn orientation_status(
    declared: Option<WeightOrientation>,
    declared_role: Option<TensorRole>,
    oracle: Option<WeightOrientation>,
    oracle_roles: &[&'static str],
    source_shape: &[u64],
) -> &'static str {
    if source_shape.len() != 2 {
        return "not_a_matrix";
    }
    if let Some(role) = declared_role {
        if role == TensorRole::Unknown {
            return "role_unknown";
        }
        if !role.is_projection() {
            if !oracle_roles.is_empty() && !oracle_roles.contains(&role.as_str()) {
                return "role_contradiction";
            }
            return "not_a_projection";
        }
    }
    let Some(declared) = declared else {
        return OrientationAssurance::LegacyUndeclared.as_str();
    };
    if let Some(oracle) = oracle {
        return if declared == oracle {
            OrientationAssurance::ArchitecturallyConfirmed.as_str()
        } else {
            "architecture_contradiction"
        };
    }
    if source_shape[0] == source_shape[1] {
        return OrientationAssurance::SquareNotVerifiable.as_str();
    }
    OrientationAssurance::DeclaredNotVerified.as_str()
}

fn parse_role(value: &str) -> Result<TensorRole, String> {
    TensorRole::parse(value).map_err(|error| error.to_string())
}

fn parse_orientation(value: &str) -> Result<WeightOrientation, String> {
    WeightOrientation::parse(value).map_err(|_| format!("unknown weight orientation {value:?}"))
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

fn matrix_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum::<f64>()
        .sqrt()
}

/// Frobenius asymmetry of a square matrix: `||A - A^T|| / ||A||`.
///
/// A non-trivial value proves that a transposed import would be detectable even
/// when the matrix is square, so squareness alone is never evidence of correct
/// orientation.
fn asymmetry_ratio(values: &[f32], dimension: usize) -> Option<f64> {
    let norm = matrix_norm(values);
    if norm == 0.0 {
        return None;
    }
    let mut squared = 0.0_f64;
    for row in 0..dimension {
        for col in 0..dimension {
            let difference =
                f64::from(values[row * dimension + col]) - f64::from(values[col * dimension + row]);
            squared += difference * difference;
        }
    }
    Some(squared.sqrt() / norm)
}

/// Read the original tensor from Safetensors in the architecturally correct
/// logical `[out_features, in_features]` layout.
///
/// The transpose is performed here, in the auditor, instead of reusing the
/// importer's transposed reader, so the oracle stays independent.
fn original_logical(
    file: &SafeTensorFile,
    name: &str,
    orientation: WeightOrientation,
) -> Result<Vec<f32>, String> {
    let info = file
        .tensors()
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| format!("source tensor {name} is absent from Safetensors"))?;
    let total = info
        .shape
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| format!("source tensor {name} element count overflows"))?;
    let stored = file
        .read_f32_elements(name, 0, total)
        .map_err(|_| format!("could not read source tensor {name} as F32 elements"))?;
    match orientation {
        WeightOrientation::LinearOutIn => Ok(stored),
        WeightOrientation::Conv1DInOut => {
            if info.shape.len() != 2 {
                return Err(format!("Conv1D tensor {name} is not rank 2"));
            }
            let in_features = info.shape[0] as usize;
            let out_features = info.shape[1] as usize;
            Ok(transpose(&stored, in_features, out_features))
        }
    }
}

fn safetensors_dtype_matches(dtype: DType, source_dtype: &str) -> bool {
    matches!(
        (dtype, source_dtype),
        (DType::F32, "F32")
            | (DType::F16, "F16")
            | (DType::Bf16, "BF16")
            | (DType::U8, "U8")
            | (DType::I8, "I8")
            | (DType::I16, "I16")
            | (DType::I32, "I32")
            | (DType::I64, "I64")
            | (DType::F64, "F64")
    )
}

struct ContainerTensor {
    tensor_id: u32,
    name: String,
    dtype: DType,
    shape: Vec<u64>,
    codec: u16,
    logical_nbytes: u64,
    stored_bytes: u64,
}

fn container_inventory(reader: &ModelReader) -> Result<BTreeMap<String, ContainerTensor>, String> {
    let mut inventory = BTreeMap::new();
    for tensor in reader.tensors() {
        let name = reader
            .metadata
            .tensor_names
            .get(tensor.name_ref as usize)
            .ok_or_else(|| format!("tensor {} has an invalid name reference", tensor.tensor_id))?
            .clone();
        let mut stored_bytes = 0_u64;
        for block_id in 0..tensor.block_count {
            let block = reader
                .block(tensor.tensor_id, block_id)
                .map_err(|_| format!("tensor {name} block {block_id} is missing"))?;
            stored_bytes = stored_bytes
                .checked_add(block.payload_length)
                .ok_or_else(|| format!("tensor {name} stored byte count overflows"))?;
        }
        let shape = tensor.dimensions[..usize::from(tensor.rank)].to_vec();
        if inventory
            .insert(
                name.clone(),
                ContainerTensor {
                    tensor_id: tensor.tensor_id,
                    name,
                    dtype: tensor.dtype,
                    shape,
                    codec: tensor.codec,
                    logical_nbytes: tensor.logical_nbytes,
                    stored_bytes,
                },
            )
            .is_some()
        {
            return Err("container has duplicate tensor names".to_owned());
        }
    }
    Ok(inventory)
}

fn read_container_bytes(reader: &ModelReader, tensor: &ContainerTensor) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    let descriptor = reader
        .tensor(tensor.tensor_id)
        .map_err(|_| format!("tensor {} vanished", tensor.name))?;
    for block_id in 0..descriptor.block_count {
        let block = reader
            .read_block(tensor.tensor_id, block_id)
            .map_err(|_| format!("tensor {} block {block_id} is unreadable", tensor.name))?;
        bytes.extend_from_slice(&block.payload);
    }
    Ok(bytes)
}

/// Reconstruct the dense logical matrix using the production VQ decode path.
///
/// `HpsiVqExecutor::matvec_rows` computes selected output rows for a given input
/// vector. Probing with the canonical basis of the input space therefore yields
/// the logical matrix one column at a time, without assuming any block layout.
fn reconstruct_logical(
    reader: &ModelReader,
    codebook: &ContainerTensor,
    indices: &ContainerTensor,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    let codebook_count = usize::try_from(codebook.shape[0])
        .map_err(|_| format!("codebook tensor {} shape is invalid", codebook.name))?;
    let layout = HpsiVqLayout {
        codebook_tensor_id: codebook.tensor_id,
        indices_tensor_id: indices.tensor_id,
        rows,
        cols,
        codebook_count,
    };
    let executor = HpsiVqExecutor::new(reader, layout)
        .map_err(|error| format!("could not prepare VQ executor: {error}"))?;
    let mut dense = vec![0.0_f32; rows * cols];
    let mut input = vec![0.0_f32; cols];
    let mut column = vec![0.0_f32; rows];
    for col in 0..cols {
        input.iter_mut().for_each(|value| *value = 0.0);
        input[col] = 1.0;
        executor
            .matvec_rows(&input, 0, &mut column)
            .map_err(|error| format!("could not reconstruct column {col}: {error}"))?;
        for row in 0..rows {
            dense[row * cols + col] = column[row];
        }
    }
    Ok(dense)
}

fn argument(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|argument| argument == key)
        .and_then(|position| args.get(position + 1))
        .cloned()
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let source_path = argument(&args, "--source").ok_or("--source is required")?;
    let hpsi_path = argument(&args, "--hpsi").ok_or("--hpsi is required")?;
    let out_path = argument(&args, "--out").ok_or("--out is required")?;
    let markdown_path = argument(&args, "--md");
    let label = argument(&args, "--label").unwrap_or_else(|| "container".to_owned());
    let architecture_path = argument(&args, "--architecture");

    let source_file = SafeTensorFile::open(
        Arc::new(FileSource::open(&source_path).map_err(|error| error.to_string())?),
        MAX_HEADER_BYTES,
    )
    .map_err(|error| format!("could not open Safetensors source: {error:?}"))?;
    let reader = ModelReader::open(
        Arc::new(FileSource::open(&hpsi_path).map_err(|error| error.to_string())?),
        &ReaderLimits::default(),
    )
    .map_err(|error| format!("could not open HPSI container: {error:?}"))?;

    let inventory = container_inventory(&reader)?;
    // The architecture oracle is data from outside the container: a manifest
    // generated from the model definition. Without it this auditor cannot tell
    // a faithful container from a transposed import, and says so instead of
    // presenting the container's own claim as confirmation.
    let architecture = match architecture_path.as_deref() {
        Some(path) => Some(
            ArchitectureFacts::from_json(&fs::read_to_string(path).map_err(|error| {
                format!("could not read architecture manifest {path}: {error}")
            })?)
            .map_err(|error| format!("architecture manifest {path}: {error}"))?,
        ),
        None => None,
    };
    let mut status_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let entries = reader
        .metadata
        .value
        .get("user")
        .and_then(|user| user.get("source_tensors"))
        .and_then(Value::as_array)
        .ok_or("HPSI metadata does not carry user.source_tensors provenance")?
        .clone();

    let mut quantized = Vec::new();
    let mut raw = Vec::new();
    let mut findings = Vec::new();

    for entry in &entries {
        let name = entry
            .get("name")
            .and_then(Value::as_str)
            .ok_or("provenance entry lacks a name")?;
        let source_dtype = entry
            .get("source_dtype")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        let source_shape: Vec<u64> = entry
            .get("source_shape")
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_default();
        let declared_orientation = entry.get("weight_orientation").and_then(Value::as_str);
        let declared = declared_orientation.map(parse_orientation).transpose()?;
        let declared_role = entry
            .get("tensor_role")
            .and_then(Value::as_str)
            .map(parse_role)
            .transpose()?;
        let orientation_source = entry
            .get("orientation_source")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        let disposition = entry
            .get("disposition")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        // The oracle is what the manifest requires for this exact name: its
        // layout, stated outright or entailed by the roles and indexing axis it
        // records, plus the roles themselves. A name the manifest does not
        // describe has no expectation; that is absence of evidence, not evidence
        // of agreement.
        let expectation = architecture
            .as_ref()
            .and_then(|facts| facts.expectation(name));
        let oracle = expectation.and_then(ArchitectureExpectation::effective_orientation);
        let oracle_roles: Vec<&'static str> = expectation
            .map(|item| item.roles.iter().map(|role| role.as_str()).collect())
            .unwrap_or_default();
        // Aliases and the alias-to-role pairing are reported beside the roles:
        // which module the model definition reached this parameter under is part
        // of what the architecture claims, and the Python auditor reports the
        // same data for the same tensor.
        let oracle_aliases: Vec<&str> = expectation
            .map(|item| item.aliases.iter().map(String::as_str).collect())
            .unwrap_or_default();
        let role_evidence: Vec<Value> = expectation
            .map(|item| {
                item.role_evidence
                    .iter()
                    .map(|evidence| {
                        json!({
                            "alias": evidence.alias,
                            "module": evidence.module,
                            "module_class": evidence.module_class,
                            "tensor_role": evidence.role.as_str(),
                            "weight_orientation":
                                evidence.orientation.map(WeightOrientation::as_str),
                            "index_axis": evidence.index_axis.map(|axis| axis as u64),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        let status = orientation_status(
            declared,
            declared_role,
            oracle,
            &oracle_roles,
            &source_shape,
        );
        *status_counts.entry(status).or_default() += 1;
        let orientation_matches_oracle = match (declared, oracle) {
            (Some(declared), Some(oracle)) => Some(declared == oracle),
            _ => None,
        };

        if orientation_matches_oracle == Some(false) {
            findings.push(json!({
                "kind": "orientation_disagrees_with_architecture",
                "tensor": name,
                "declared": declared_orientation,
                "oracle": oracle.map(WeightOrientation::as_str),
                "orientation_source": orientation_source,
                "source_shape": source_shape,
            }));
        }
        if let Some(declared_role) = declared_role {
            if !oracle_roles.is_empty() && !oracle_roles.contains(&declared_role.as_str()) {
                findings.push(json!({
                    "kind": "role_disagrees_with_architecture",
                    "tensor": name,
                    "declared": declared_role.as_str(),
                    "oracle_roles": oracle_roles,
                    "note": "a tied tensor may legitimately hold several roles; disagreement is one that the model definition states none of them",
                }));
            }
        }

        if disposition == "preserve_raw" {
            let container_tensor = inventory
                .get(name)
                .ok_or_else(|| format!("RAW tensor {name} is missing from the container"))?;
            let mut original_bytes = Vec::new();
            source_file
                .copy_raw_payload(name, &mut original_bytes, COPY_CHUNK_BYTES)
                .map_err(|error| {
                    format!("could not read raw source payload for {name}: {error:?}")
                })?;
            let container_bytes = read_container_bytes(&reader, container_tensor)?;
            let byte_identical = original_bytes == container_bytes;
            let dtype_ok = safetensors_dtype_matches(container_tensor.dtype, source_dtype);
            let shape_ok = container_tensor.shape == source_shape;
            let size_ok = container_tensor.logical_nbytes == original_bytes.len() as u64
                && container_bytes.len() == original_bytes.len();
            if !byte_identical || !dtype_ok || !shape_ok || !size_ok {
                findings.push(json!({
                    "kind": "raw_preservation_failure",
                    "tensor": name,
                    "byte_identical": byte_identical,
                    "dtype_matches": dtype_ok,
                    "shape_matches": shape_ok,
                    "size_matches": size_ok,
                }));
            }
            raw.push(json!({
                "tensor": name,
                "source_dtype": source_dtype,
                "container_dtype": format!("{:?}", container_tensor.dtype),
                "container_codec": container_tensor.codec,
                "source_shape": source_shape,
                "declared_orientation": declared_orientation,
                "orientation_source": orientation_source,
                "oracle_orientation": oracle.map(WeightOrientation::as_str),
                "orientation_matches_oracle": orientation_matches_oracle,
                "orientation_status": status,
                "declared_role": declared_role.map(TensorRole::as_str),
                "oracle_roles": oracle_roles.clone(),
                "oracle_aliases": oracle_aliases.clone(),
                "role_evidence": role_evidence.clone(),
                "index_axis": entry.get("index_axis").and_then(Value::as_u64),
                // A RAW payload stores the source bytes verbatim, so nothing in
                // the container can falsify its declared orientation: the label
                // only tells a consumer how the bytes are meant to be read.
                "orientation_falsifiable_from_container": false,
                "container_shape": container_tensor.shape,
                "original_bytes": original_bytes.len(),
                "container_bytes": container_bytes.len(),
                "logical_nbytes": container_tensor.logical_nbytes,
                "byte_identical": byte_identical,
                "dtype_matches": dtype_ok,
                "shape_matches": shape_ok,
                "size_matches": size_ok,
            }));
            continue;
        }

        // Quantized tensor.
        let codebook_name = format!("{name}.hpsi_vq.codebook");
        let indices_name = format!("{name}.hpsi_vq.indices");
        let codebook = inventory
            .get(&codebook_name)
            .ok_or_else(|| format!("codebook tensor {codebook_name} is missing"))?;
        let indices = inventory
            .get(&indices_name)
            .ok_or_else(|| format!("index tensor {indices_name} is missing"))?;

        let logical: Vec<u64> = entry
            .get("logical_shape")
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_default();
        if logical.len() != 2 {
            findings.push(json!({
                "kind": "missing_logical_shape",
                "tensor": name,
            }));
            continue;
        }
        let rows = logical[0] as usize;
        let cols = logical[1] as usize;
        let rows_u64 = logical[0];
        let cols_u64 = logical[1];

        // Cross-check the declared logical shape against the stored selector payload.
        let expected_selectors = (cols_u64 / 8)
            .checked_mul(codebook.shape.first().copied().unwrap_or(1))
            .ok_or_else(|| format!("{name} selector count overflows"))?;
        let expected_index_bytes = rows_u64
            .checked_mul(expected_selectors)
            .map(|count| count.div_ceil(2))
            .ok_or_else(|| format!("{name} index byte count overflows"))?;
        let index_size_consistent = expected_index_bytes == indices.logical_nbytes;
        if !index_size_consistent {
            findings.push(json!({
                "kind": "index_payload_inconsistent_with_logical_shape",
                "tensor": name,
                "expected_index_bytes": expected_index_bytes,
                "stored_index_bytes": indices.logical_nbytes,
            }));
        }

        let reconstructed = reconstruct_logical(&reader, codebook, indices, rows, cols)?;
        // The architectural reference exists only when the supplied manifest
        // names this tensor: nothing here substitutes a guess for it.
        let mut oracle_logical: Option<Vec<f32>> = None;
        let mut oracle_shape: Option<(usize, usize)> = None;
        if let Some(orientation) = oracle {
            if let Some([out, inn]) = orientation.logical_shape(&source_shape) {
                oracle_logical = Some(
                    original_logical(&source_file, name, orientation)
                        .map_err(|error| error.to_string())?,
                );
                oracle_shape = Some((out as usize, inn as usize));
            }
        }
        let same_shape = oracle_shape.map(|shape| shape == (rows, cols));
        let transposed_shape = oracle_shape.map(|(out, inn)| (inn, out) == (rows, cols));

        // Orientation-normalized quantization distortion.
        //
        // The container stores a concrete array layout. Which architectural
        // matrix that layout corresponds to is decided by the DECLARED
        // orientation, never by whichever reference happens to fit better: picking
        // the better fit would be circular and would hide a transposed import.
        // This is also what makes square matrices evaluable at all, because their
        // shape cannot disambiguate the orientation.
        //
        // With a manifest the reference is the model-definition matrix, rotated
        // into the declared layout here in the auditor. Without one the
        // comparison can only show that the container encodes the layout it
        // declares for itself, and the basis is labelled to say exactly that.
        let mut distortion: Option<(Vec<f32>, usize, usize)> = None;
        let mut distortion_basis: Option<&'static str> = None;
        if let (Some(oracle_logical), Some((oracle_rows, oracle_cols))) =
            (oracle_logical.as_ref(), oracle_shape)
        {
            distortion_basis = Some("architecture_oracle");
            distortion = Some(if declared == oracle {
                (oracle_logical.clone(), oracle_rows, oracle_cols)
            } else {
                (
                    transpose(oracle_logical, oracle_rows, oracle_cols),
                    oracle_cols,
                    oracle_rows,
                )
            });
        } else if let Some(orientation) = declared {
            if let Some([out, inn]) = orientation.logical_shape(&source_shape) {
                let logical = original_logical(&source_file, name, orientation)
                    .map_err(|error| error.to_string())?;
                distortion_basis = Some("declared_orientation_self_consistency");
                distortion = Some((logical, out as usize, inn as usize));
            }
        }
        if distortion.is_none() {
            findings.push(json!({
                "kind": "no_orientation_basis_for_numeric_comparison",
                "tensor": name,
                "declared_orientation": declared_orientation,
                "oracle_orientation": oracle.map(WeightOrientation::as_str),
                "note": "neither the container nor a model-definition manifest states how these weights are arranged",
            }));
        }
        let mut entry_json = json!({
            "tensor": name,
            "source_dtype": source_dtype,
            "source_shape": source_shape,
            "declared_orientation": declared_orientation,
            "orientation_source": orientation_source,
            "oracle_orientation": oracle.map(WeightOrientation::as_str),
            "orientation_matches_oracle": orientation_matches_oracle,
            "orientation_status": status,
            "declared_role": declared_role.map(TensorRole::as_str),
            "oracle_roles": oracle_roles,
            "oracle_aliases": oracle_aliases,
            "role_evidence": role_evidence,
            "index_axis": entry.get("index_axis").and_then(Value::as_u64),
            "logical_shape": [rows, cols],
            "oracle_logical_shape": oracle_shape.map(|(out, inn)| json!([out, inn])),
            "shape_matches_oracle": same_shape,
            "shape_is_oracle_transpose": transposed_shape,
            "source_shape_is_square": source_shape.first() == source_shape.get(1),
            "quantization_distortion_basis": distortion_basis,
            "codebook_dtype": format!("{:?}", codebook.dtype),
            "codebook_codec": codebook.codec,
            "codebook_shape": codebook.shape,
            "indices_dtype": format!("{:?}", indices.dtype),
            "indices_codec": indices.codec,
            "indices_shape": indices.shape,
            "indices_logical_nbytes": indices.logical_nbytes,
            "expected_index_bytes": expected_index_bytes,
            "index_size_consistent": index_size_consistent,
            "codebook_stored_bytes": codebook.stored_bytes,
            "indices_stored_bytes": indices.stored_bytes,
            "dense_f32_bytes": (rows as u64) * (cols as u64) * 4,
            "vq_stored_bytes": codebook.stored_bytes + indices.stored_bytes,
            "vq_bits_per_weight": (codebook.shape.first().copied().unwrap_or(0) as f64) * 4.0 / 8.0,
        });

        if let Some((reference, reference_rows, reference_cols)) = distortion.as_ref() {
            if (rows, cols) == (*reference_rows, *reference_cols) {
                let normalized = metrics(reference, &reconstructed)?;
                entry_json["quantization_distortion"] = normalized.to_json();
                entry_json["quantization_distortion_relative_l2"] = json!(normalized.relative_l2);
            } else {
                findings.push(json!({
                    "kind": "container_arrangement_does_not_match_declared_orientation",
                    "tensor": name,
                    "reconstructed_shape": [rows, cols],
                    "reference_shape": [*reference_rows, *reference_cols],
                    "declared_orientation": declared_orientation,
                    "oracle_orientation": oracle.map(WeightOrientation::as_str),
                }));
            }
        }

        // Comparison against the architectural matrix is only defined when the
        // manifest supplied one; without it the container's own claim is all
        // there is, so these fields are reported as absent rather than guessed.
        entry_json["architecture_reference_available"] = json!(oracle_shape.is_some());
        match (oracle_shape, oracle_logical.as_ref()) {
            (Some((oracle_rows, oracle_cols)), Some(oracle_logical))
                if (rows, cols) == (oracle_rows, oracle_cols) =>
            {
                let direct = metrics(oracle_logical, &reconstructed)?;
                entry_json["metrics_vs_oracle_logical"] = direct.to_json();
                // Non-square matrices are transposed by shape alone; square matrices
                // need the numeric control to prove transposition would be visible.
                let flipped = transpose(oracle_logical, rows, cols);
                let flipped_metrics = metrics(&flipped, &reconstructed)?;
                entry_json["metrics_vs_oracle_transposed"] = flipped_metrics.to_json();
                entry_json["orientation_evidence_ratio"] =
                    json!(if flipped_metrics.relative_l2 == 0.0 {
                        f64::NAN
                    } else {
                        direct.relative_l2 / flipped_metrics.relative_l2
                    });
                if rows == cols {
                    entry_json["oracle_asymmetry_ratio"] =
                        json!(asymmetry_ratio(oracle_logical, rows));
                } else {
                    entry_json["transpose_detectable_by_shape"] = json!(true);
                }
                // A correctly oriented import must fit the architectural logical
                // matrix strictly better than its transpose. Error magnitude alone is
                // not a defect signal: 0.5 bit/weight is expected to distort heavily.
                if direct.relative_l2 >= flipped_metrics.relative_l2 {
                    findings.push(json!({
                        "kind": "reconstruction_matches_transpose_better_than_logical",
                        "tensor": name,
                        "direct_relative_l2": direct.relative_l2,
                        "transposed_relative_l2": flipped_metrics.relative_l2,
                    }));
                }
            }
            (Some((oracle_rows, oracle_cols)), Some(oracle_logical))
                if (rows, cols) == (oracle_cols, oracle_rows) =>
            {
                let flipped = transpose(oracle_logical, oracle_rows, oracle_cols);
                let flipped_metrics = metrics(&flipped, &reconstructed)?;
                entry_json["metrics_vs_oracle_transposed"] = flipped_metrics.to_json();
                entry_json["metrics_vs_oracle_logical"] = Value::Null;
                findings.push(json!({
                    "kind": "container_stores_transpose_of_architectural_logical_weight",
                    "tensor": name,
                    "declared_orientation": declared_orientation,
                    "oracle_orientation": oracle.map(WeightOrientation::as_str),
                    "logical_shape": [rows, cols],
                    "oracle_logical_shape": [oracle_rows, oracle_cols],
                    "relative_l2_against_transposed_oracle": flipped_metrics.relative_l2,
                }));
            }
            (Some((oracle_rows, oracle_cols)), _) => {
                entry_json["metrics_vs_oracle_logical"] = Value::Null;
                findings.push(json!({
                    "kind": "logical_shape_matches_neither_oracle_nor_transpose",
                    "tensor": name,
                    "logical_shape": [rows, cols],
                    "oracle_logical_shape": [oracle_rows, oracle_cols],
                }));
            }
            (None, _) => {
                entry_json["metrics_vs_oracle_logical"] = Value::Null;
                entry_json["metrics_vs_oracle_transposed"] = Value::Null;
            }
        }
        quantized.push(entry_json);
    }

    // Aggregate metrics over tensors whose shape matched the oracle exactly.
    let mut aggregate_total_absolute = 0.0_f64;
    let mut aggregate_total_squared = 0.0_f64;
    let mut aggregate_reference_squared = 0.0_f64;
    let mut aggregate_reconstructed_squared = 0.0_f64;
    let mut aggregate_elements = 0_u64;
    let mut aggregate_max = 0.0_f64;
    let mut direct_ratios = Vec::new();
    let mut transposed_ratios = Vec::new();
    let mut dense_bytes = 0_u64;
    let mut stored_bytes = 0_u64;
    for entry in &quantized {
        if let Some(ratio) = entry
            .get("metrics_vs_oracle_transposed")
            .and_then(|v| v.get("relative_l2"))
            .and_then(Value::as_f64)
        {
            transposed_ratios.push(ratio);
        }
        dense_bytes += entry
            .get("dense_f32_bytes")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        stored_bytes += entry
            .get("vq_stored_bytes")
            .and_then(Value::as_u64)
            .unwrap_or(0);
    }
    let mut normalized_ratios = Vec::new();
    let mut normalized_squared = 0.0_f64;
    let mut normalized_reference_squared = 0.0_f64;
    let mut normalized_elements = 0_u64;
    for entry in &quantized {
        if let Some(metrics) = entry
            .get("metrics_vs_oracle_logical")
            .and_then(|v| v.as_object())
        {
            let elements = metrics["elements"].as_u64().unwrap_or(0);
            let mae = metrics["mae"].as_f64().unwrap_or(0.0);
            let rmse = metrics["rmse"].as_f64().unwrap_or(0.0);
            let max_abs = metrics["max_abs"].as_f64().unwrap_or(0.0);
            let l2_reference = metrics["l2_reference"].as_f64().unwrap_or(0.0);
            let l2_reconstructed = metrics["l2_reconstructed"].as_f64().unwrap_or(0.0);
            aggregate_total_absolute += mae * elements as f64;
            aggregate_total_squared += rmse * rmse * elements as f64;
            aggregate_reference_squared += l2_reference * l2_reference;
            aggregate_reconstructed_squared += l2_reconstructed * l2_reconstructed;
            aggregate_elements += elements;
            aggregate_max = aggregate_max.max(max_abs);
            direct_ratios.push(metrics["relative_l2"].as_f64().unwrap_or(f64::NAN));
        }
        // Orientation-normalized distortion covers every quantized tensor,
        // including the ones whose stored logical shape is the oracle transpose.
        if let Some(ratio) = entry
            .get("quantization_distortion_relative_l2")
            .and_then(Value::as_f64)
        {
            normalized_ratios.push(ratio);
        }
        let normalized_metrics = entry
            .get("quantization_distortion")
            .and_then(Value::as_object);
        if let Some(metrics) = normalized_metrics {
            let elements = metrics["elements"].as_u64().unwrap_or(0);
            let rmse = metrics["rmse"].as_f64().unwrap_or(0.0);
            let l2_reference = metrics["l2_reference"].as_f64().unwrap_or(0.0);
            normalized_squared += rmse * rmse * elements as f64;
            normalized_reference_squared += l2_reference * l2_reference;
            normalized_elements += elements;
        }
    }
    let mean = |values: &[f64]| {
        if values.is_empty() {
            f64::NAN
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    };
    let aggregate = json!({
        "tensors_quantized": quantized.len(),
        "tensors_raw": raw.len(),
        "source_tensors": entries.len(),
        "elements_in_aggregate": aggregate_elements,
        "mae": if aggregate_elements == 0 { f64::NAN } else { aggregate_total_absolute / aggregate_elements as f64 },
        "rmse": if aggregate_elements == 0 { f64::NAN } else { (aggregate_total_squared / aggregate_elements as f64).sqrt() },
        "relative_l2": if aggregate_reference_squared == 0.0 { f64::NAN } else { aggregate_total_squared.sqrt() / aggregate_reference_squared.sqrt() },
        "max_abs": aggregate_max,
        "l2_reference": aggregate_reference_squared.sqrt(),
        "l2_reconstructed": aggregate_reconstructed_squared.sqrt(),
        "mean_relative_l2_vs_oracle_logical": mean(&direct_ratios),
        "mean_relative_l2_vs_oracle_transposed": mean(&transposed_ratios),
        "elements_in_normalized_aggregate": normalized_elements,
        "mean_quantization_distortion_relative_l2": mean(&normalized_ratios),
        "quantization_distortion_relative_l2": if normalized_reference_squared == 0.0 {
            f64::NAN
        } else {
            normalized_squared.sqrt() / normalized_reference_squared.sqrt()
        },
        "dense_f32_bytes": dense_bytes,
        "vq_stored_bytes": stored_bytes,
        "compression_ratio": if stored_bytes == 0 { f64::NAN } else { dense_bytes as f64 / stored_bytes as f64 },
    });

    let raw_all_identical = raw
        .iter()
        .all(|entry| entry["byte_identical"].as_bool() == Some(true));
    let omitted: Vec<String> = source_file
        .tensors()
        .iter()
        .map(|tensor| tensor.name.clone())
        .filter(|name| {
            !entries
                .iter()
                .any(|entry| entry.get("name").and_then(Value::as_str) == Some(name.as_str()))
        })
        .collect();
    if !omitted.is_empty() {
        findings.push(json!({
            "kind": "source_tensors_omitted_from_container",
            "tensors": omitted,
        }));
    }
    let accounted: u64 = quantized.len() as u64 + raw.len() as u64;
    if accounted != entries.len() as u64 {
        findings.push(json!({
            "kind": "provenance_entries_not_accounted_for",
            "provenance_entries": entries.len(),
            "accounted": accounted,
        }));
    }

    let verified = reader
        .verify_all()
        .map_err(|error| format!("integrity verification failed: {error:?}"))?;
    // The padding budget is the whole file: every padding region must be proven
    // zero, so the audit must not silently stop early on a small budget.
    let padding = reader
        .verify_padding(reader.header.file_size)
        .map_err(|error| format!("padding verification failed: {error:?}"))?;

    let count = |token: &str| status_counts.get(token).copied().unwrap_or(0);
    let unverified = count(OrientationAssurance::DeclaredNotVerified.as_str())
        + count(OrientationAssurance::SquareNotVerifiable.as_str())
        + count(OrientationAssurance::LegacyUndeclared.as_str());
    let architecture_oracle = json!({
        "supplied": architecture.is_some(),
        "manifest": architecture_path,
        "model_type": architecture.as_ref().and_then(|facts| facts.model_type.clone()),
        "tensors_described": architecture.as_ref().map(|facts| facts.expectations.len()),
        "orientation_statuses": status_counts,
        "architecturally_confirmed": count(OrientationAssurance::ArchitecturallyConfirmed.as_str()),
        "architectural_contradictions": count("architecture_contradiction"),
        "unverified": unverified,
        "note": "architecturally_confirmed means the declaration agrees with a manifest produced \
                 from the model definition; that is evidence about the model, not a property of \
                 the container, and no other state carries architectural proof",
    });

    let report = json!({
        "label": label,
        "architecture_oracle": architecture_oracle,
        "source": source_path,
        "container": hpsi_path,
        "container_bytes": fs::metadata(Path::new(&hpsi_path)).map(|m| m.len()).unwrap_or(0),
        "source_bytes": fs::metadata(Path::new(&source_path)).map(|m| m.len()).unwrap_or(0),
        "metadata_bytes": reader.header.metadata.length,
        "payload_region_bytes": reader.header.payload.length,
        "container_tensors": inventory.len(),
        "verified_crc32c_records": verified,
        "padding_records": padding,
        "source_tensor_names": source_file.tensors().iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
        "container_tensor_names": inventory.keys().cloned().collect::<Vec<_>>(),
        "quantized": quantized,
        "raw": raw,
        "raw_all_byte_identical": raw_all_identical,
        "aggregate": aggregate,
        "findings": findings,
    });

    if let Some(parent) = Path::new(&out_path).parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(
        &out_path,
        serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;

    println!("container: {hpsi_path}");
    println!(
        "orientation statuses: {}",
        report["architecture_oracle"]["orientation_statuses"]
    );
    println!(
        "quantized tensors: {}",
        report["aggregate"]["tensors_quantized"]
    );
    println!("raw tensors: {}", report["aggregate"]["tensors_raw"]);
    println!("raw byte-identical: {raw_all_identical}");
    println!("verified CRC32C records: {verified}");
    println!(
        "aggregate relative L2: {}",
        report["aggregate"]["relative_l2"]
    );
    println!(
        "findings: {}",
        report["findings"].as_array().map(Vec::len).unwrap_or(0)
    );
    for finding in report["findings"].as_array().into_iter().flatten() {
        println!("  - {finding}");
    }
    println!("wrote {out_path}");

    if let Some(markdown_path) = markdown_path {
        let mut markdown = String::new();
        markdown.push_str(&format!("# Auditoria de reconstrução — {label}\n\n"));
        markdown.push_str(&format!("- Fonte: `{source_path}`\n"));
        markdown.push_str(&format!("- Container: `{hpsi_path}`\n"));
        markdown.push_str(&format!(
            "- Bytes do container: {}\n",
            report["container_bytes"]
        ));
        let oracle_note = if report["architecture_oracle"]["supplied"].as_bool() == Some(true) {
            format!(
                "`{}`",
                report["architecture_oracle"]["manifest"]
                    .as_str()
                    .unwrap_or("?")
            )
        } else {
            "ausente — nenhuma orientação é confirmada arquiteturalmente".to_owned()
        };
        markdown.push_str(&format!("- Oráculo de arquitetura: {oracle_note}\n"));
        markdown.push_str(&format!(
            "- Estados de orientação: {}\n",
            report["architecture_oracle"]["orientation_statuses"]
        ));
        markdown.push_str(&format!("- Registros CRC32C verificados: {verified}\n"));
        markdown.push_str(&format!(
            "- RAW byte a byte idêntico: {raw_all_identical}\n"
        ));
        markdown.push_str(&format!(
            "- Rel-L2 agregado (tensores com shape compatível): {}\n",
            report["aggregate"]["relative_l2"]
        ));
        markdown.push_str(&format!(
            "- Rel-L2 médio vs. matriz lógica correta: {}\n",
            report["aggregate"]["mean_relative_l2_vs_oracle_logical"]
        ));
        markdown.push_str(&format!(
            "- Rel-L2 médio vs. matriz transposta (controle): {}\n",
            report["aggregate"]["mean_relative_l2_vs_oracle_transposed"]
        ));
        markdown.push_str(&format!(
            "- Compressão denso-F32 → VQ: {} bytes → {} bytes ({}x)\n",
            report["aggregate"]["dense_f32_bytes"],
            report["aggregate"]["vq_stored_bytes"],
            report["aggregate"]["compression_ratio"]
        ));
        markdown.push_str(&format!(
            "- **Distorção de quantização normalizada por orientação** (todos os tensores, comparável entre containers): {} (média por tensor {})\n\n",
            report["aggregate"]["quantization_distortion_relative_l2"],
            report["aggregate"]["mean_quantization_distortion_relative_l2"]
        ));

        markdown.push_str("## Evidência de orientação\n\n");
        markdown.push_str("| Tensor | Shape origem | Orientação declarada | Oráculo | Estado | Rel-L2 correto | Rel-L2 transposto | Razão |\n");
        markdown.push_str("|---|---|---|---|---|---|---|---|\n");
        for entry in &quantized {
            let direct = entry
                .get("metrics_vs_oracle_logical")
                .and_then(|v| v.get("relative_l2"))
                .and_then(Value::as_f64);
            let transposed = entry
                .get("metrics_vs_oracle_transposed")
                .and_then(|v| v.get("relative_l2"))
                .and_then(Value::as_f64);
            markdown.push_str(&format!(
                "| `{}` | {} | {} | {} | {} | {} | {} | {} |\n",
                entry["tensor"].as_str().unwrap_or("?"),
                entry["source_shape"],
                entry["declared_orientation"].as_str().unwrap_or("?"),
                entry["oracle_orientation"].as_str().unwrap_or("—"),
                entry["orientation_status"].as_str().unwrap_or("?"),
                direct.map_or("n/a".to_owned(), |v| format!("{v:.6}")),
                transposed.map_or("n/a".to_owned(), |v| format!("{v:.6}")),
                entry
                    .get("orientation_evidence_ratio")
                    .and_then(Value::as_f64)
                    .map_or("n/a".to_owned(), |v| format!("{v:.3}")),
            ));
        }
        markdown.push('\n');

        for column_set in ["metrics_vs_oracle_logical", "metrics_vs_oracle_transposed"] {
            let heading = if column_set == "metrics_vs_oracle_logical" {
                "## Reconstrução vs. matriz lógica correta (oráculo de arquitetura)"
            } else {
                "## Reconstrução vs. matriz transposta (controle de orientação)"
            };
            markdown.push_str(heading);
            markdown.push_str("\n\n| Tensor | Shape lógico | Orientação | MAE | RMSE | Rel-L2 | Máx abs | ‖orig‖ | ‖recon‖ |\n");
            markdown.push_str("|---|---|---|---|---|---|---|---|---|\n");
            for entry in &quantized {
                let Some(metrics) = entry.get(column_set).and_then(Value::as_object) else {
                    continue;
                };
                markdown.push_str(&format!(
                    "| `{}` | {} | {} | {:.6} | {:.6} | {:.6} | {:.6} | {:.4} | {:.4} |\n",
                    entry["tensor"].as_str().unwrap_or("?"),
                    entry["logical_shape"],
                    entry["declared_orientation"].as_str().unwrap_or("?"),
                    metrics["mae"].as_f64().unwrap_or(f64::NAN),
                    metrics["rmse"].as_f64().unwrap_or(f64::NAN),
                    metrics["relative_l2"].as_f64().unwrap_or(f64::NAN),
                    metrics["max_abs"].as_f64().unwrap_or(f64::NAN),
                    metrics["l2_reference"].as_f64().unwrap_or(f64::NAN),
                    metrics["l2_reconstructed"].as_f64().unwrap_or(f64::NAN),
                ));
            }
            markdown.push('\n');
        }

        markdown.push_str("## Preservação RAW\n\n| Tensor | Bytes origem | Bytes HPSI | dtype | shape | Byte a byte |\n|---|---|---|---|---|---|\n");
        for entry in &raw {
            markdown.push_str(&format!(
                "| `{}` | {} | {} | {} | {} | {} |\n",
                entry["tensor"].as_str().unwrap_or("?"),
                entry["original_bytes"],
                entry["container_bytes"],
                entry["container_dtype"].as_str().unwrap_or("?"),
                entry["container_shape"],
                entry["byte_identical"],
            ));
        }
        markdown.push('\n');

        markdown.push_str("## Achados\n\n");
        if findings.is_empty() {
            markdown.push_str("Nenhum.\n");
        } else {
            for finding in &findings {
                markdown.push_str(&format!(
                    "- `{}` {}\n",
                    finding["kind"].as_str().unwrap_or("?"),
                    finding
                ));
            }
        }

        let markdown_path = PathBuf::from(markdown_path);
        if let Some(parent) = markdown_path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        fs::write(&markdown_path, markdown).map_err(|error| error.to_string())?;
        println!("wrote {}", markdown_path.display());
    }

    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("audit_reconstruction: {message}");
            std::process::ExitCode::FAILURE
        }
    }
}
