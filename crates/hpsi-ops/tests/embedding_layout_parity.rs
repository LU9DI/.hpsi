//! Differential test: the Rust half.
//!
//! Reads the shared normative table in `tests/fixtures/embedding_layout_cases.json`
//! and evaluates every case through the library's own decision function. The
//! Python suite reads the *same file* and evaluates it through an implementation
//! written separately in `tools/audit_hpsi_container.py`; neither implementation
//! calls, imports or shells out to the other, so agreement here is evidence about
//! the contract rather than about shared code (docs/orientation-policy.md).
//!
//! A case is a pair: the metadata and physical facts one entry states, and the
//! violation code the contract must report for it — `null` when the entry is
//! acceptable. Ordering matters and is asserted too, because an entry can break
//! two rules and the contract names only the first.

use hpsi_ops::{EntryLayout, LayoutContext, LayoutViolation, TensorRole, validate_entry_layout};
use serde_json::{Value, json};
use std::path::PathBuf;

fn cases() -> Vec<Value> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/embedding_layout_cases.json");
    let document: Value = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display())),
    )
    .expect("shared layout cases must be valid JSON");
    document["cases"]
        .as_array()
        .expect("`cases` must be an array")
        .clone()
}

fn shape(value: &Value) -> Vec<u64> {
    match value.as_array() {
        Some(values) => values.iter().filter_map(Value::as_u64).collect(),
        None => Vec::new(),
    }
}

fn orientation_token(value: &Value) -> Option<&str> {
    value.get("weight_orientation").and_then(Value::as_str)
}

fn code_of(role_value: &Value, case: &Value) -> Result<Option<String>, (LayoutViolation, String)> {
    let role = match role_value.as_str() {
        None => None,
        Some(token) => Some(TensorRole::parse_token(token)?),
    };
    let context = match case["context"].as_str().unwrap_or("source") {
        "manifest" => LayoutContext::Manifest,
        _ => LayoutContext::SourceEntry,
    };
    let orientation = orientation_token(case)
        .map(|token| {
            hpsi_ops::WeightOrientation::parse(token).map_err(|_| {
                (
                    LayoutViolation::RoleTokenUnknown,
                    format!("unparseable orientation {token:?}"),
                )
            })
        })
        .transpose()?;
    let layout = EntryLayout {
        source_shape: shape(&case["source_shape"]),
        orientation,
        logical_shape: match case["logical_shape"].as_array() {
            Some(values) => {
                let dims = values.iter().filter_map(Value::as_u64).collect::<Vec<_>>();
                Some([dims[0], dims[1]])
            }
            None => None,
        },
        role,
        index_axis: case["index_axis"].as_u64().map(|axis| axis as usize),
        context,
        quantized: case["quantized"].as_bool().unwrap_or(false),
        codebook_count: case["codebook_count"].as_u64().unwrap_or(0),
        stored_index_bytes: case["stored_index_bytes"].as_u64().unwrap_or(0),
        calibrated: case["calibrated"].as_bool().unwrap_or(false),
    };
    match validate_entry_layout(&layout) {
        Ok(()) => Ok(None),
        Err((violation, _)) => Ok(Some(violation.code().to_owned())),
    }
}

#[test]
fn the_shared_table_is_not_empty_and_covers_every_role() {
    let cases = cases();
    assert!(
        cases.len() >= 30,
        "the contract table shrank: {} cases",
        cases.len()
    );
    let roles = cases
        .iter()
        .filter_map(|case| case.get("tensor_role").and_then(Value::as_str))
        .collect::<Vec<_>>();
    for role in [
        "linear_projection",
        "conv1d_projection",
        "embedding_table",
        "other",
        "unknown",
    ] {
        assert!(roles.contains(&role), "{role} is never exercised");
    }
    // Every documented violation code must be reached by at least one case, or
    // the code is dead and the contract is not actually enforced.
    let expected = cases
        .iter()
        .filter_map(|case| case["expect"].as_str().map(str::to_owned))
        .collect::<Vec<_>>();
    for code in [
        "orientation_required",
        "logical_shape_required",
        "role_without_orientation",
        "role_orientation_mismatch",
        "axis_without_role",
        "axis_for_non_indexed_role",
        "table_without_axis",
        "axis_value_unsupported",
        "axis_one_unsupported",
        "table_orientation_mismatch",
        "logical_shape_conflicts_axis",
        "logical_shape_conflicts_orientation",
        "selector_bytes_mismatch",
        "selector_layout_unencodable",
        "table_must_not_be_calibrated",
        "role_token_unknown",
        "logical_shape_on_non_matrix",
        "matrix_role_on_non_matrix",
    ] {
        assert!(
            expected.iter().any(|seen| seen == code),
            "{code} is never reached"
        );
    }
}

#[test]
fn every_case_reports_exactly_the_documented_code() {
    for case in cases() {
        let id = case["id"].as_str().unwrap_or("?").to_owned();
        let expected = case["expect"].as_str().map(str::to_owned);
        let actual = code_of(&case["tensor_role"], &case)
            .unwrap_or_else(|(violation, _)| Some(violation.code().to_owned()));
        assert_eq!(
            actual,
            expected,
            "case {id}: expected {:?}, got {actual:?}",
            case["expect"].as_str()
        );
    }
}

#[test]
fn an_accepted_case_still_carries_no_architectural_claim_by_itself() {
    // The table is a decision function, not a verdict about the model: an
    // acceptable entry is *consistent*, which is not the same as confirmed.
    // This guard exists so nobody reads "no violation" as "orientation proven".
    let case = json!({
        "id": "consistent_is_not_confirmed",
        "source_shape": [32, 16],
        "weight_orientation": "linear_out_in",
        "logical_shape": [32, 16],
        "tensor_role": "unknown",
        "context": "source",
    });
    let role = TensorRole::parse_token("unknown").expect("role parses");
    assert!(
        validate_entry_layout(&EntryLayout {
            source_shape: shape(&case["source_shape"]),
            orientation: Some(hpsi_ops::WeightOrientation::LinearOutIn),
            logical_shape: Some([32, 16]),
            role: Some(role),
            index_axis: None,
            context: LayoutContext::SourceEntry,
            quantized: false,
            codebook_count: 0,
            stored_index_bytes: 0,
            calibrated: false,
        })
        .is_ok(),
        "a consistent entry must be accepted"
    );
    assert!(
        !role.is_projection(),
        "and accepting it must not have promoted it to a projection"
    );
}
