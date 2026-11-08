"""Differential test: the Python half of the shared layout table.

Reads `tests/fixtures/embedding_layout_cases.json` and evaluates every case with
this repository's own implementation in `tools/audit_hpsi_container.py`. The Rust
suite (`crates/hpsi-ops/tests/embedding_layout_parity.rs`) reads the same file and
evaluates it with `hpsi_ops::validate_entry_layout`. The two implementations share
no code, and this test never calls the Rust CLI: agreement is evidence about the
contract, not about a common function (docs/orientation-policy.md).

Written as ``unittest`` cases so both ``python -m pytest tests`` and the CI's
``python -m unittest discover`` collect them.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import audit_hpsi_container as hpsi  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "embedding_layout_cases.json"


def cases() -> list[dict]:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return document["cases"]


def evaluate(case: dict) -> str | None:
    """The violation code this implementation reports, or ``None``."""
    logical = case.get("logical_shape")
    violation = hpsi.entry_layout_violation(
        source_shape=case.get("source_shape") or [],
        weight_orientation=case.get("weight_orientation"),
        logical_shape=list(logical) if logical else None,
        tensor_role=case.get("tensor_role"),
        index_axis=case.get("index_axis"),
        context=case.get("context", "source"),
        quantized=bool(case.get("quantized")),
        codebook_count=int(case.get("codebook_count", 0)),
        stored_index_bytes=int(case.get("stored_index_bytes", 0)),
        calibrated=bool(case.get("calibrated")),
    )
    if violation is None:
        return None
    code, reason = violation
    assert reason, f"case {case['id']} reported {code} without a reason"
    return code


class SharedTableTests(unittest.TestCase):
    def test_the_shared_table_exists_and_is_substantial(self) -> None:
        table = cases()
        self.assertGreaterEqual(len(table), 30)

    def test_every_documented_code_is_reached(self) -> None:
        reached = {case["expect"] for case in cases() if case["expect"]}
        for code in (
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
        ):
            with self.subTest(code=code):
                self.assertIn(code, reached)

    def test_each_case_reports_exactly_the_documented_code(self) -> None:
        for case in cases():
            with self.subTest(case=case["id"]):
                self.assertEqual(
                    evaluate(case),
                    case["expect"],
                    f"{case['id']}: expected {case['expect']!r}",
                )

    def test_case_ids_are_unique(self) -> None:
        ids = [case["id"] for case in cases()]
        self.assertEqual(len(ids), len(set(ids)), "the table has duplicated case ids")


class ContractGuardsTests(unittest.TestCase):
    """Guards against the specific confusions this phase exists to remove."""

    def test_a_physical_fact_is_not_architectural_evidence(self) -> None:
        # An accepted entry is consistent, not confirmed: the grouping token
        # never turns a matrix into a projection.
        self.assertIsNone(
            hpsi.entry_layout_violation(
                source_shape=[32, 16],
                weight_orientation="linear_out_in",
                logical_shape=[32, 16],
                tensor_role="unknown",
            )
        )
        self.assertEqual(
            hpsi.classify_orientation(
                "linear_out_in", None, [32, 16], "unknown", ()
            ),
            "role_unknown",
        )

    def test_embedding_without_orientation_is_rejected_like_rust_does(self) -> None:
        violation = hpsi.entry_layout_violation(
            source_shape=[32, 16],
            weight_orientation=None,
            logical_shape=[32, 16],
            tensor_role="embedding_table",
            index_axis=0,
        )
        self.assertIsNotNone(violation, "the Phase 4.3 divergence must be closed")
        self.assertEqual(violation[0], "orientation_required")

    def test_index_axis_is_checked_against_the_stored_selectors(self) -> None:
        # This is the case Phase 4.3 found unreported on either axis: the shape
        # fits the token, so only the payload says whether the table was encoded
        # the way it claims.
        consistent = hpsi.entry_layout_violation(
            source_shape=[32, 16],
            weight_orientation="linear_out_in",
            logical_shape=[32, 16],
            tensor_role="embedding_table",
            index_axis=0,
            quantized=True,
            codebook_count=1,
            stored_index_bytes=32,
        )
        self.assertIsNone(consistent)
        mismatched = hpsi.entry_layout_violation(
            source_shape=[32, 16],
            weight_orientation="linear_out_in",
            logical_shape=[32, 16],
            tensor_role="embedding_table",
            index_axis=0,
            quantized=True,
            codebook_count=1,
            stored_index_bytes=40,
        )
        self.assertEqual(mismatched[0], "selector_bytes_mismatch")

    def test_legacy_entries_stay_exactly_as_documented(self) -> None:
        for source_shape, orientation, logical in (
            ([32, 16], None, None),          # pre-contract
            ([32, 16], "linear_out_in", [32, 16]),  # orientation only
            ([16], "linear_out_in", None),   # vector provenance, no layout claim
        ):
            with self.subTest(shape=source_shape, orientation=orientation):
                self.assertIsNone(
                    hpsi.entry_layout_violation(
                        source_shape=source_shape,
                        weight_orientation=orientation,
                        logical_shape=logical,
                    )
                )

    def test_axis_one_is_refused_rather_than_resupported(self) -> None:
        violation = hpsi.entry_layout_violation(
            source_shape=[16, 32],
            weight_orientation="conv1d_in_out",
            logical_shape=[32, 16],
            tensor_role="embedding_table",
            index_axis=1,
        )
        self.assertEqual(violation[0], "axis_one_unsupported")
        self.assertIn("No producer path", violation[1])


if __name__ == "__main__":
    unittest.main()
