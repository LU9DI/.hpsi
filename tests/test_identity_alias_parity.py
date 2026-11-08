"""Differential test: the Python half of the identity and alias contract.

Reads `tests/fixtures/identity_alias_cases.json` and feeds each manifest to this
repository's own parser, `hpsi.ArchitectureOracle.load`. The Rust suite
(`crates/hpsi-ops/tests/identity_alias_parity.rs`) reads the same file and feeds it
to `hpsi_ops::ArchitectureFacts::from_json`. Neither implementation calls the
other, and this test never shells out to Rust: agreement is evidence about the
contract (docs/orientation-policy.md §8).

Written as ``unittest`` cases so both ``python -m pytest tests`` and the CI's
``python -m unittest discover`` collect them.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import audit_hpsi_container as hpsi  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "identity_alias_cases.json"

# Each rejection rule, named once by the fragment both implementations produce.
RULE_FRAGMENTS = (
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
)


def cases() -> list[dict]:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return document["cases"]


def case(case_id: str) -> dict:
    for item in cases():
        if item["id"] == case_id:
            return item
    raise AssertionError(f"{case_id} is absent from the shared table")


def evaluate(manifest: dict) -> str | None:
    """The reason this implementation refuses the manifest, or ``None``."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "architecture.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        try:
            hpsi.ArchitectureOracle.load(path)
        except hpsi.HpsiFormatError as error:
            return str(error)
    return None


def oracle(manifest: dict) -> hpsi.ArchitectureOracle:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "architecture.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return hpsi.ArchitectureOracle.load(path)


class IdentityAliasParityTests(unittest.TestCase):
    def test_the_shared_table_is_substantial_and_covers_each_rule(self) -> None:
        rows = cases()
        self.assertGreaterEqual(len(rows), 15)
        self.assertGreaterEqual(
            sum(1 for row in rows if row["expect"] is None),
            6,
        )
        ids = [row["id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)), "case ids must be unique")
        fragments = [row["expect"] for row in rows if row["expect"] is not None]
        for rule in RULE_FRAGMENTS:
            self.assertIn(rule, fragments, f"no case exercises {rule!r}")

    def test_every_case_reports_the_documented_outcome(self) -> None:
        for row in cases():
            with self.subTest(case=row["id"]):
                reason = evaluate(row["manifest"])
                if row["expect"] is None:
                    self.assertIsNone(reason, reason)
                    continue
                self.assertIsNotNone(reason, f"{row['id']} must be refused")
                self.assertIn(row["expect"], reason or "")

    def test_an_accepted_tied_entry_keeps_both_roles_and_the_pairing(self) -> None:
        loaded = oracle(case("tied_both_roles_with_pairing")["manifest"])
        entry = loaded.expectation("transformer.wte.weight")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["tensor_roles"], ("embedding_table", "linear_projection"))
        self.assertEqual(entry["aliases"], ("lm_head.weight", "transformer.wte.weight"))
        self.assertEqual(entry["index_axis"], 0)
        self.assertEqual(entry["weight_orientation"], "linear_out_in")
        self.assertEqual(
            [(row["alias"], row["tensor_role"], row["module_class"]) for row in entry["role_evidence"]],
            [
                ("lm_head.weight", "linear_projection", "linear"),
                ("transformer.wte.weight", "embedding_table", "embedding"),
            ],
        )
        self.assertEqual(entry["effective_orientation"], "linear_out_in")

    def test_the_untied_pair_is_two_entries_with_two_functions(self) -> None:
        loaded = oracle(case("untied_equal_values_stay_two_entries")["manifest"])
        table = loaded.expectation("transformer.wte.weight")
        head = loaded.expectation("lm_head.weight")
        self.assertEqual(table["tensor_roles"], ("embedding_table",))
        self.assertEqual(table["aliases"], ("transformer.wte.weight",))
        self.assertIsNone(table["weight_orientation"])
        self.assertEqual(head["tensor_roles"], ("linear_projection",))
        self.assertEqual(head["aliases"], ("lm_head.weight",))
        self.assertEqual(head["weight_orientation"], "linear_out_in")
        self.assertIsNone(head["index_axis"], "an alias never leaks onto the other identity")

    def test_an_entry_without_aliases_derives_them_from_the_pairing(self) -> None:
        entry = oracle(case("pairing_fills_in_missing_aliases")["manifest"]).expectation(
            "transformer.wte.weight"
        )
        self.assertEqual(entry["aliases"], ("transformer.wte.weight",))
        self.assertEqual(len(entry["role_evidence"]), 1)

    def test_a_legacy_entry_states_no_aliases_and_no_pairing(self) -> None:
        loaded = oracle(case("legacy_entry_without_new_keys")["manifest"])
        orientation_only = loaded.expectation("a.weight")
        self.assertEqual(orientation_only["tensor_roles"], ())
        self.assertEqual(orientation_only["aliases"], ())
        self.assertEqual(orientation_only["role_evidence"], ())
        role_only = loaded.expectation("b.weight")
        self.assertEqual(role_only["tensor_roles"], ("other",))
        self.assertEqual(role_only["aliases"], ())

    def test_the_pairing_may_be_stated_without_a_layout_claim(self) -> None:
        # A conv1d pairing states its own class layout, and a positional table
        # states none: the entry-level orientation stays absent either way.
        conv = oracle(case("conv1d_pairing")["manifest"]).expectation(
            "transformer.h.0.attn.c_attn.weight"
        )
        self.assertEqual(conv["weight_orientation"], "conv1d_in_out")
        position = oracle(case("positional_table_without_orientation")["manifest"]).expectation(
            "transformer.wpe.weight"
        )
        self.assertIsNone(position["weight_orientation"])
        self.assertEqual(position["effective_orientation"], "linear_out_in")

    def test_reasons_carry_the_tensor_name(self) -> None:
        for row in cases():
            if row["expect"] is None:
                continue
            reason = evaluate(row["manifest"])
            self.assertIsNotNone(reason)
            named = row["manifest"]["tensors"][0]["name"]
            self.assertTrue(
                reason.startswith(f"{named}:") or named in (reason or ""),
                f"{row['id']}: {reason!r} must say which tensor it refuses",
            )


if __name__ == "__main__":
    unittest.main()
