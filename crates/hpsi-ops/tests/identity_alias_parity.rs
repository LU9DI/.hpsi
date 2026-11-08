//! Differential test: the Rust half of the identity and alias contract.
//!
//! `tests/fixtures/identity_alias_cases.json` is read by this suite and, through
//! an implementation written separately in `tools/audit_hpsi_container.py`, by
//! `tests/test_identity_alias_parity.py`. Neither parser calls the other and
//! neither shares code with it, so agreement is evidence about the contract: the
//! same manifests accepted, the same rejections, and reasons that name the same
//! rule (`docs/orientation-policy.md` §8).

use hpsi_ops::{ArchitectureFacts, TensorRole, WeightOrientation};
use serde_json::Value;
use std::path::PathBuf;

fn document() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/identity_alias_cases.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_json::from_str(&text).expect("shared identity cases must be valid JSON")
}

fn cases() -> Vec<Value> {
    document()["cases"]
        .as_array()
        .expect("`cases` must be an array")
        .clone()
}

fn outcome(manifest: &Value) -> Result<String, String> {
    ArchitectureFacts::from_json(&manifest.to_string())
        .map(|facts| facts.expectations.len().to_string())
        .map_err(|error| error.to_string())
}

#[test]
fn the_shared_table_is_substantial_and_covers_each_pairing_rule() {
    let cases = cases();
    assert!(cases.len() >= 15, "only {} identity cases", cases.len());

    let accepted = cases.iter().filter(|case| case["expect"].is_null()).count();
    assert!(accepted >= 6, "{accepted} accepted cases is too few");

    let ids: Vec<&str> = cases
        .iter()
        .filter_map(|case| case["id"].as_str())
        .collect();
    assert_eq!(
        ids.len(),
        ids.iter().collect::<std::collections::HashSet<_>>().len(),
        "case ids must be unique"
    );

    let fragments: Vec<&str> = cases
        .iter()
        .filter_map(|case| case["expect"].as_str())
        .collect();
    for rule in [
        "which `tensor_role` does not declare",
        "with no role_evidence naming the module that evidences it",
        "is not among `aliases`",
        "appears twice in `role_evidence`",
        "is declared twice",
        "without an index_axis",
        "while the entry declares",
        "which that role does not imply",
        "only meaningful for embedding_table",
        "lacks a non-empty `alias`",
        "lacks a non-empty `tensor_role`",
        "role_token_unknown",
        "axis_one_unsupported",
    ] {
        assert!(
            fragments.contains(&rule),
            "no case exercises the rule {rule:?}"
        );
    }
}

#[test]
fn every_case_reports_the_documented_outcome() {
    for case in cases() {
        let expected = &case["expect"];
        let reason = outcome(&case["manifest"]);
        if expected.is_null() {
            assert!(
                reason.is_ok(),
                "{}: expected acceptance, got {:?}",
                case["id"],
                reason.unwrap_err()
            );
            continue;
        }
        let fragment = expected.as_str().expect("string or null");
        let error = match reason {
            Ok(_) => panic!("{}: expected a refusal containing {fragment:?}", case["id"]),
            Err(error) => error,
        };
        assert!(
            error.contains(fragment),
            "{}: expected {fragment:?} in {error}",
            case["id"]
        );
    }
}

#[test]
fn an_accepted_tied_entry_keeps_both_roles_and_the_alias_that_carries_each() {
    let case = cases()
        .into_iter()
        .find(|item| item["id"] == "tied_both_roles_with_pairing")
        .expect("the tied case must exist");
    let facts = ArchitectureFacts::from_json(&case["manifest"].to_string())
        .expect("the tied manifest must parse");
    let entry = facts
        .expectation("transformer.wte.weight")
        .expect("the stored key is the lookup name");

    assert_eq!(
        entry.roles,
        vec![TensorRole::EmbeddingTable, TensorRole::LinearProjection]
    );
    assert_eq!(entry.aliases.len(), 2);
    assert_eq!(entry.role_evidence.len(), 2);
    assert_eq!(entry.index_axis, Some(0));
    assert_eq!(entry.orientation, Some(WeightOrientation::LinearOutIn));
}

#[test]
fn an_accepted_entry_without_aliases_still_reports_the_pairing_it_named() {
    let case = cases()
        .into_iter()
        .find(|item| item["id"] == "pairing_fills_in_missing_aliases")
        .expect("the derived-alias case must exist");
    let facts = ArchitectureFacts::from_json(&case["manifest"].to_string())
        .expect("the manifest must parse");
    let entry = facts.expectation("transformer.wte.weight").expect("entry");
    assert_eq!(
        entry.aliases,
        vec!["transformer.wte.weight".to_owned()],
        "the evidence rows name the parameter, so the alias set is not lost"
    );
}

#[test]
fn a_legacy_entry_states_no_aliases_and_no_pairing() {
    let case = cases()
        .into_iter()
        .find(|item| item["id"] == "legacy_entry_without_new_keys")
        .expect("the legacy case must exist");
    let facts = ArchitectureFacts::from_json(&case["manifest"].to_string())
        .expect("a pre-identity manifest must still parse");

    let orientation_only = facts
        .expectation("a.weight")
        .expect("orientation-only entry");
    assert!(orientation_only.aliases.is_empty());
    assert!(orientation_only.role_evidence.is_empty());
    assert!(orientation_only.roles.is_empty());

    let role_only = facts.expectation("b.weight").expect("role-only entry");
    assert_eq!(role_only.roles, vec![TensorRole::Other]);
    assert!(role_only.aliases.is_empty());
}
