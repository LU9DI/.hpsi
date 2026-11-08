"""Tests for the identity-keyed architecture oracle.

These are the Phase 4.2 oracle tests: every case is decided by what the model
definition actually is (parameter object identity and module class), never by a
name pattern. The module under test is torch-free on purpose, so these run in
the plain CI environment as well as under pytest.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import architecture_oracle as ao  # noqa: E402
from architecture_oracle import WeightFact  # noqa: E402


class Linear:
    """Stand-in for ``torch.nn.Linear``."""


class Conv1D:
    """Stand-in for ``transformers.pytorch_utils.Conv1D``."""


class Embedding:
    """Stand-in for ``torch.nn.Embedding``."""


class LayerNorm:
    """A module class the oracle deliberately does not classify."""


class Weight:
    """A parameter object: identity is what matters, values are irrelevant."""

    def __init__(self, shape=(4, 8), data=None) -> None:
        self.ndim = len(shape)
        self.shape = shape
        size = 1
        for dimension in shape:
            size *= dimension
        self.data = data if data is not None else list(range(size))


def build(*facts: WeightFact, model_type: str | None = None) -> ao.ArchitectureOracle:
    return ao.build_architecture_oracle(list(facts), model_type)


class TiedWeightsTests(unittest.TestCase):
    def test_one_object_reached_as_table_and_projection_resolves_under_both_names(self):
        shared = Weight((10, 8))
        oracle = build(
            WeightFact("transformer.wte.weight", id(shared), "embedding_table", None, 0),
            WeightFact("lm_head.weight", id(shared), "linear_projection", "linear_out_in"),
            model_type="gpt2",
        )
        # The name a container stores resolves even though the oracle walked
        # module paths, because both names belong to one identity.
        for name in ("transformer.wte.weight", "lm_head.weight"):
            entry = oracle.expectation(name)
            self.assertIsNotNone(entry, name)
            self.assertEqual(entry.tensor_roles, ("embedding_table", "linear_projection"))
            self.assertEqual(entry.weight_orientation, "linear_out_in")
            self.assertEqual(entry.index_axis, 0)
            self.assertEqual(
                entry.aliases, ("lm_head.weight", "transformer.wte.weight")
            )
        self.assertEqual(
            list(oracle.shared_groups),
            ["lm_head.weight"],
            "the lexicographically first alias is the canonical name",
        )
        self.assertEqual(oracle.to_record()["distinct_identities"], 1)

    def test_container_name_different_from_module_path_still_matches(self):
        """The Phase 4.1 defect: an oracle keyed by module path left the stored
        tensor unmatched."""
        stored = Weight((10, 8))
        oracle = build(
            # Facts as the model's module tree yields them.
            WeightFact("lm_head.weight", id(stored), "linear_projection", "linear_out_in"),
            WeightFact("transformer.wte.weight", id(stored), "embedding_table", None, 0),
        )
        self.assertIsNotNone(oracle.expectation("transformer.wte.weight"))
        self.assertIsNotNone(oracle.expectation("lm_head.weight"))


class UntiedWeightsTests(unittest.TestCase):
    def test_two_objects_stay_two_entries_even_with_identical_values(self):
        """Sharing is an identity relation. Two checkpoints legitimately store a
        duplicate matrix, and value equality must never merge them."""
        left = Weight((10, 8), data=[1] * 80)
        right = Weight((10, 8), data=[1] * 80)
        self.assertEqual(left.data, right.data)
        self.assertNotEqual(id(left), id(right))
        oracle = build(
            WeightFact("transformer.wte.weight", id(left), "embedding_table", None, 0),
            WeightFact("lm_head.weight", id(right), "linear_projection", "linear_out_in"),
        )
        self.assertEqual(oracle.shared_groups, {})
        self.assertEqual(oracle.to_record()["distinct_identities"], 2)
        self.assertEqual(
            oracle.expectation("lm_head.weight").tensor_roles, ("linear_projection",)
        )
        self.assertIsNone(oracle.expectation("transformer.wte.weight").weight_orientation)

    def test_models_without_sharing_keep_one_entry_per_name(self):
        table, head = Weight((10, 8)), Weight((10, 8))
        oracle = build(
            WeightFact("transformer.wte.weight", id(table), "embedding_table", None, 0),
            WeightFact("lm_head.weight", id(head), "linear_projection", "linear_out_in"),
            WeightFact("h.0.attn.c_attn.weight", id(Weight((16, 32))), "conv1d_projection",
                       "conv1d_in_out"),
        )
        self.assertEqual(len(oracle.entries), 3)
        self.assertEqual(oracle.shared_groups, {})
        self.assertEqual(oracle.unresolved, {})


class MultipleAliasesTests(unittest.TestCase):
    def test_three_names_for_one_object_produce_three_resolvable_entries(self):
        one = Weight((6, 8))
        oracle = build(
            WeightFact("zeta.weight", id(one), "linear_projection", "linear_out_in"),
            WeightFact("alpha.weight", id(one), "linear_projection", "linear_out_in"),
            WeightFact("middle.weight", id(one), "linear_projection", "linear_out_in"),
        )
        self.assertEqual(len(oracle.entries), 3)
        for name in ("alpha.weight", "middle.weight", "zeta.weight"):
            self.assertEqual(oracle.expectation(name).canonical_name, "alpha.weight")
            self.assertEqual(
                oracle.expectation(name).aliases, ("alpha.weight", "middle.weight", "zeta.weight")
            )

    def test_canonical_name_is_deterministic_under_input_order(self):
        def facts():
            one, other = Weight((4, 8)), Weight((4, 8))
            return [
                WeightFact("b.weight", id(one), "linear_projection", "linear_out_in"),
                WeightFact("a.weight", id(one), "linear_projection", "linear_out_in"),
                WeightFact("d.weight", id(other), "embedding_table", None, 0),
                WeightFact("c.weight", id(other), "embedding_table", None, 0),
            ]

        forward = ao.build_architecture_oracle(facts())
        backward = ao.build_architecture_oracle(list(reversed(facts())))
        self.assertEqual(
            sorted(forward.entries),
            sorted(backward.entries),
        )
        self.assertEqual(
            {name: entry.canonical_name for name, entry in forward.entries.items()},
            {name: entry.canonical_name for name, entry in backward.entries.items()},
        )
        self.assertEqual(
            {name: entry.aliases for name, entry in forward.entries.items()},
            {name: entry.aliases for name, entry in backward.entries.items()},
        )
        self.assertEqual(forward.entries["a.weight"].canonical_name, "a.weight")
        self.assertEqual(forward.entries["d.weight"].canonical_name, "c.weight")


class UnresolvedTests(unittest.TestCase):
    def test_conflicting_projection_layouts_for_one_object_are_unresolved(self):
        one = Weight((8, 8))
        oracle = build(
            WeightFact("a.weight", id(one), "linear_projection", "linear_out_in"),
            WeightFact("b.weight", id(one), "conv1d_projection", "conv1d_in_out"),
        )
        self.assertEqual(oracle.entries, {})
        self.assertEqual(
            set(oracle.unresolved), {"a.weight", "b.weight"}
        )
        self.assertIn("conflicting layouts", oracle.unresolved["a.weight"].reason)

    def test_one_name_claimed_by_two_objects_is_ambiguous_not_chosen(self):
        first, second = Weight((4, 8)), Weight((4, 8))
        oracle = build(
            WeightFact("same.weight", id(first), "linear_projection", "linear_out_in"),
            WeightFact("same.weight", id(second), "linear_projection", "linear_out_in"),
        )
        self.assertEqual(oracle.entries, {})
        self.assertIn("more than one tensor object", oracle.unresolved["same.weight"].reason)

    def test_conflicting_index_axes_are_unresolved(self):
        one = Weight((4, 8))
        oracle = build(
            WeightFact("a.weight", id(one), "embedding_table", None, 0),
            WeightFact("b.weight", id(one), "embedding_table", None, 1),
        )
        self.assertEqual(oracle.entries, {})
        self.assertIn("conflicting index axes", oracle.unresolved["a.weight"].reason)

    def test_unknown_name_lookup_returns_none_instead_of_a_guess(self):
        oracle = build(
            WeightFact("a.weight", id(Weight()), "linear_projection", "linear_out_in")
        )
        self.assertIsNone(oracle.expectation("a.bias"))
        self.assertIsNone(oracle.expectation("prefix.a.weight"))
        self.assertIsNone(oracle.expectation("weight"))
        self.assertEqual(oracle.aliases_of("a.weight"), ("a.weight",))
        self.assertEqual(oracle.aliases_of("missing.weight"), ())


class FactValidationTests(unittest.TestCase):
    def test_embedding_with_a_projection_orientation_is_refused(self):
        """Role and orientation are separate dimensions: an indexed table cannot
        claim a multiplication layout."""
        oracle = build(WeightFact("e.weight", 1, "embedding_table", "linear_out_in", 0))
        self.assertEqual(oracle.entries, {})
        self.assertIn("indexed, not multiplied", oracle.unresolved["e.weight"].reason)

    def test_embedding_without_index_axis_is_refused(self):
        oracle = build(WeightFact("e.weight", 2, "embedding_table"))
        self.assertIn("requires index_axis", oracle.unresolved["e.weight"].reason)

    def test_projection_roles_require_their_own_orientation(self):
        for role, expected in (
            ("linear_projection", "linear_out_in"),
            ("conv1d_projection", "conv1d_in_out"),
        ):
            missing = build(WeightFact("p.weight", 3, role))
            self.assertIn("requires weight_orientation", missing.unresolved["p.weight"].reason)
            wrong = build(
                WeightFact(
                    "p.weight",
                    4,
                    role,
                    "conv1d_in_out" if expected == "linear_out_in" else "linear_out_in",
                )
            )
            self.assertIn("requires weight_orientation", wrong.unresolved["p.weight"].reason)

    def test_index_axis_is_meaningless_for_non_indexed_roles(self):
        oracle = build(WeightFact("p.weight", 5, "linear_projection", "linear_out_in", 0))
        self.assertIn("only meaningful for", oracle.unresolved["p.weight"].reason)

    def test_other_and_unknown_state_no_orientation_and_are_still_entries(self):
        oracle = build(
            WeightFact("o.weight", 6, "other"),
            WeightFact("u.weight", 7, "unknown"),
        )
        self.assertEqual(oracle.expectation("o.weight").tensor_roles, ("other",))
        self.assertEqual(oracle.expectation("u.weight").tensor_roles, ("unknown",))
        self.assertIsNone(oracle.expectation("u.weight").weight_orientation)
        self.assertNotIn("o.weight", oracle.unresolved)

    def test_unknown_is_not_silently_relabelled_as_other(self):
        oracle = build(WeightFact("u.weight", 8, "unknown"))
        self.assertEqual(oracle.expectation("u.weight").tensor_roles, ("unknown",))

    def test_bad_role_and_orientation_tokens_are_rejected(self):
        oracle = build(
            WeightFact("x.weight", 9, "linear"),
            WeightFact("y.weight", 10, "linear_projection", "out_in"),
            WeightFact("  ", 11, "other"),
        )
        self.assertEqual(oracle.entries, {})
        self.assertEqual(len(oracle.unresolved), 3)


class ClassifyModuleTests(unittest.TestCase):
    CLASSES = {"linear": Linear, "conv1d": Conv1D, "embedding": Embedding}

    def test_recognised_classes_yield_roles_not_name_guesses(self):
        weight = Weight((4, 8))
        cases = [
            (Linear(), "linear_projection", "linear_out_in", None),
            (Conv1D(), "conv1d_projection", "conv1d_in_out", None),
            (Embedding(), "embedding_table", None, 0),
        ]
        for module, role, orientation, axis in cases:
            module.weight = weight
            fact = ao.classify_module_weight("some.path.weight", module, self.CLASSES)
            self.assertIsNotNone(fact, role)
            self.assertEqual(fact.name, "some.path.weight")
            self.assertEqual(fact.identity, id(weight))
            self.assertEqual(fact.tensor_role, role)
            self.assertEqual(fact.weight_orientation, orientation)
            self.assertEqual(fact.index_axis, axis)
            module.weight = None

    def test_classification_ignores_the_name_entirely(self):
        """A tensor named like a Conv1D but owned by a Linear is a Linear."""
        linear = Linear()
        linear.weight = Weight((6, 8))
        fact = ao.classify_module_weight("h.0.attn.c_attn.weight", linear, self.CLASSES)
        self.assertEqual(fact.tensor_role, "linear_projection")
        self.assertEqual(fact.weight_orientation, "linear_out_in")

    def test_unclassified_module_and_non_matrices_produce_no_fact(self):
        norm = LayerNorm()
        norm.weight = Weight((8,))
        self.assertIsNone(ao.classify_module_weight("n.weight", norm, self.CLASSES))
        vector = Embedding()
        vector.weight = Weight((8,))
        self.assertIsNone(ao.classify_module_weight("e.weight", vector, self.CLASSES))
        empty = Linear()
        self.assertIsNone(ao.classify_module_weight("l.weight", empty, self.CLASSES))


class SourceLayoutTests(unittest.TestCase):
    """Layout is decided by evidence of one of two kinds, never by inference."""

    def test_projection_orientations(self):
        self.assertEqual(ao.source_layout("linear_projection", "linear_out_in", None, (6, 8)),
                         ((6, 8), False))
        self.assertEqual(ao.source_layout("conv1d_projection", "conv1d_in_out", None, (6, 8)),
                         ((8, 6), True))

    def test_embedding_index_axis_decides_without_any_orientation(self):
        self.assertEqual(ao.source_layout("embedding_table", None, 0, (10, 8)), ((10, 8), False))
        self.assertEqual(ao.source_layout("embedding_table", None, 1, (10, 8)), ((8, 10), True))

    def test_tied_tensor_prefers_the_projection_evidence(self):
        role, orientation, axis = "embedding_table", "linear_out_in", 0
        self.assertEqual(ao.source_layout(role, orientation, axis, (10, 8)), ((10, 8), False))

    def test_missing_or_wrong_evidence_raises(self):
        with self.assertRaises(ao.LayoutError):
            ao.source_layout("other", None, None, (6, 8))
        with self.assertRaises(ao.LayoutError):
            ao.source_layout("unknown", None, None, (6, 8))
        with self.assertRaises(ao.LayoutError):
            ao.source_layout(None, None, None, (6, 8))
        with self.assertRaises(ao.LayoutError):
            ao.source_layout("embedding_table", None, None, (6, 8))
        with self.assertRaises(ao.LayoutError):
            ao.source_layout("embedding_table", None, 5, (6, 8))
        with self.assertRaises(ao.LayoutError):
            ao.source_layout("linear_projection", "linear_out_in", None, (8,))
        with self.assertRaises(ao.LayoutError):
            ao.source_layout("linear_projection", "unexpected", None, (6, 8))


class RoleEvidenceAssociationTests(unittest.TestCase):
    """Which alias carries which role is knowledge the role set cannot carry."""

    def test_a_tied_parameter_pairs_each_alias_with_the_role_it_evidences(self):
        shared = Weight((10, 8))
        oracle = build(
            WeightFact("transformer.wte.weight", id(shared), "embedding_table", None, 0),
            WeightFact("lm_head.weight", id(shared), "linear_projection", "linear_out_in"),
        )
        self.assertEqual(
            oracle.role_evidence_of("transformer.wte.weight"),
            (
                ("lm_head.weight", "linear_projection"),
                ("transformer.wte.weight", "embedding_table"),
            ),
        )
        record = oracle.expectation("lm_head.weight").as_record()
        self.assertEqual(
            [(row["alias"], row["tensor_role"]) for row in record["role_evidence"]],
            [
                ("lm_head.weight", "linear_projection"),
                ("transformer.wte.weight", "embedding_table"),
            ],
        )

    def test_two_identities_never_share_a_pairing(self):
        table, head = Weight((10, 8)), Weight((10, 8))
        oracle = build(
            WeightFact("transformer.wte.weight", id(table), "embedding_table", None, 0),
            WeightFact("lm_head.weight", id(head), "linear_projection", "linear_out_in"),
        )
        self.assertEqual(
            oracle.role_evidence_of("transformer.wte.weight"),
            (("transformer.wte.weight", "embedding_table"),),
        )
        self.assertEqual(
            oracle.role_evidence_of("lm_head.weight"),
            (("lm_head.weight", "linear_projection"),),
        )
        self.assertEqual(oracle.shared_groups, {})
        self.assertEqual(oracle.to_record()["distinct_identities"], 2)

    def test_an_unresolved_identity_reports_no_pairing(self):
        shared = Weight((8, 16))
        oracle = build(
            WeightFact("a.weight", id(shared), "linear_projection", "linear_out_in"),
            WeightFact("b.weight", id(shared), "conv1d_projection", "conv1d_in_out"),
        )
        self.assertIsNone(oracle.expectation("a.weight"))
        self.assertEqual(oracle.role_evidence_of("a.weight"), ())
        reasons = [item.reason for item in oracle.unresolved.values()]
        self.assertTrue(
            any("conflicting layouts" in reason for reason in reasons), reasons
        )


if __name__ == "__main__":
    unittest.main()
