
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "capture_transformer_activations.py"

SPEC = importlib.util.spec_from_file_location(
    "capture_transformer_activations",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CaptureUtilityTests(unittest.TestCase):
    def test_safe_name_removes_path_and_shell_unfriendly_characters(self):
        self.assertEqual(
            MODULE.safe_name("../model layer:0"),
            "model_layer_0",
        )

    def test_linear_module_maps_to_safetensors_weight_key(self):
        self.assertEqual(
            MODULE.source_tensor_name("model.layers.0.proj"),
            "model.layers.0.proj.weight",
        )
        self.assertEqual(MODULE.source_tensor_name(""), "weight")

    def test_resolves_shared_parameter_to_stored_alias(self):
        parameter = object()
        module = type("FakeModule", (), {"weight": parameter})()

        result = MODULE.resolve_source_tensor_name(
            "lm_head",
            module,
            {id(parameter): ["transformer.wte.weight", "lm_head.weight"]},
            {"transformer.wte.weight"},
        )

        self.assertEqual(result, "transformer.wte.weight")

    def test_prefers_conventional_name_when_physically_stored(self):
        parameter = object()
        module = type("FakeModule", (), {"weight": parameter})()

        result = MODULE.resolve_source_tensor_name(
            "lm_head",
            module,
            {id(parameter): ["transformer.wte.weight", "lm_head.weight"]},
            {"transformer.wte.weight", "lm_head.weight"},
        )

        self.assertEqual(result, "lm_head.weight")

    def test_rejects_projection_without_stored_alias(self):
        parameter = object()
        module = type("FakeModule", (), {"weight": parameter})()

        with self.assertRaisesRegex(ValueError, "no physical Safetensors tensor"):
            MODULE.resolve_source_tensor_name(
                "lm_head",
                module,
                {id(parameter): ["lm_head.weight"]},
                {"transformer.wte.weight"},
            )

    def test_rejects_ambiguous_stored_aliases(self):
        parameter = object()
        module = type("FakeModule", (), {"weight": parameter})()

        with self.assertRaisesRegex(ValueError, "ambiguous physical Safetensors aliases"):
            MODULE.resolve_source_tensor_name(
                "lm_head",
                module,
                {
                    id(parameter): [
                        "transformer.wte.weight",
                        "decoder.embed.weight",
                    ]
                },
                {
                    "transformer.wte.weight",
                    "decoder.embed.weight",
                },
            )

    def test_sha256_file_hashes_exact_dataset_bytes(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            content = b'{"text":"hello"}\n'
            path.write_bytes(content)

            self.assertEqual(
                MODULE.sha256_file(path),
                hashlib.sha256(content).hexdigest(),
            )

    def test_rejects_duplicate_jsonl_object_keys(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            MODULE.parse_jsonl_record(
                '{"text":"first","text":"second"}',
                1,
            )

    def test_rejects_nonstandard_jsonl_numeric_constants(self):
        with self.assertRaisesRegex(ValueError, "non-standard JSON"):
            MODULE.parse_jsonl_record('{"value":NaN}', 1)

    def test_rejects_non_object_jsonl_records_without_attribute_error(self):
        self.assertEqual(
            MODULE.parse_jsonl_record('"hello"', 1),
            "hello",
        )
        self.assertEqual(
            MODULE.parse_jsonl_record('["hello"]', 1),
            ["hello"],
        )

    def test_identifies_linear_input_features(self):
        class FakeLinear:
            in_features = 768

        result = MODULE.supported_projection(
            FakeLinear(), FakeLinear, None
        )
        self.assertEqual(result, ("linear", 768))

    def test_identifies_transformers_conv1d_input_features(self):
        class FakeLinear:
            pass

        class FakeConv1D:
            class Weight:
                shape = (768, 2304)

            weight = Weight()

        result = MODULE.supported_projection(
            FakeConv1D(), FakeLinear, FakeConv1D
        )
        self.assertEqual(result, ("conv1d", 768))

    def test_ignores_unsupported_module(self):
        class Unsupported:
            pass

        self.assertIsNone(
            MODULE.supported_projection(Unsupported(), type, None)
        )

    def test_rejects_conv1d_weight_with_invalid_rank(self):
        class FakeLinear:
            pass

        class FakeConv1D:
            class Weight:
                shape = (768,)

            weight = Weight()

        with self.assertRaisesRegex(ValueError, "rank-2"):
            MODULE.supported_projection(
                FakeConv1D(), FakeLinear, FakeConv1D
            )

    def test_rejects_invalid_linear_feature_count(self):
        class FakeLinear:
            in_features = 0

        with self.assertRaisesRegex(ValueError, "invalid input feature"):
            MODULE.supported_projection(
                FakeLinear(), FakeLinear, None
            )


class FakeWeight:
    """Stand-in for a parameter: a shape, a dtype and an identity, no values."""

    def __init__(self, shape, dtype="torch.float32") -> None:
        self.shape = shape
        self.dtype = dtype

    def __eq__(self, other) -> bool:
        # Equal by shape and dtype on purpose: grouping on equality would merge
        # two distinct parameters, which is the failure these tests look for.
        return isinstance(other, FakeWeight) and (self.shape, self.dtype) == (
            other.shape,
            other.dtype,
        )

    def __hash__(self) -> int:
        return hash((id(self), self.shape, self.dtype))


class FakeLinear:
    def __init__(self, weight, in_features, out_features) -> None:
        self.weight = weight
        self.in_features = in_features
        self.out_features = out_features


class FakeConv1D:
    """Transformers declares `Conv1D(nf, nx)` with a `[nx, nf]` weight."""

    def __init__(self, weight, nf, nx) -> None:
        self.weight = weight
        self.nf = nf
        self.nx = nx


class FakeEmbedding:
    def __init__(self, weight, num_embeddings, embedding_dim) -> None:
        self.weight = weight
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim


class FakeModel:
    def __init__(self, modules: dict[str, object]) -> None:
        self._modules = modules

    def named_modules(self):
        return self._modules.items()


class TensorIdentityCaptureTests(unittest.TestCase):
    """One record per parameter identity, with every alias and role it carries."""

    def views_for(self, modules, aliases, stored):
        return MODULE.architecture_views(
            FakeModel(modules),
            FakeLinear,
            FakeConv1D,
            FakeEmbedding,
            aliases,
            stored,
        )

    def test_embedding_and_head_sharing_one_parameter_make_one_record(self):
        weight = FakeWeight((10, 16))
        modules = {
            "transformer.wte": FakeEmbedding(weight, 10, 16),
            "lm_head": FakeLinear(weight, in_features=16, out_features=10),
        }
        records = MODULE.logical_tensor_records(
            self.views_for(
                modules,
                {id(weight): ["transformer.wte.weight", "lm_head.weight"]},
                {"transformer.wte.weight"},
            )
        )

        self.assertEqual(len(records), 1, records)
        record = records[0]
        self.assertEqual(record["canonical_name"], "transformer.wte.weight")
        self.assertEqual(record["aliases"], ["lm_head.weight", "transformer.wte.weight"])
        self.assertEqual(record["tensor_roles"], ["embedding_table", "linear_projection"])
        self.assertEqual(record["index_axis"], 0)
        self.assertEqual(record["weight_orientation"], "linear_out_in")
        self.assertEqual(record["status"], "resolved")
        self.assertEqual(record["conflicts"], [])
        self.assertEqual(record["shape"], [10, 16])
        self.assertEqual(record["dtype"], "F32")

    def test_the_alias_to_role_association_is_explicit(self):
        weight = FakeWeight((10, 16))
        records = MODULE.logical_tensor_records(
            self.views_for(
                {
                    "transformer.wte": FakeEmbedding(weight, 10, 16),
                    "lm_head": FakeLinear(weight, in_features=16, out_features=10),
                },
                {id(weight): ["transformer.wte.weight", "lm_head.weight"]},
                {"transformer.wte.weight"},
            )
        )

        self.assertEqual(
            [
                (row["alias"], row["tensor_role"], row["module_class"])
                for row in records[0]["evidence"]
            ],
            [
                ("lm_head.weight", "linear_projection", "linear"),
                ("transformer.wte.weight", "embedding_table", "embedding"),
            ],
        )
        # The projection row carries the layout, the table row the indexing axis.
        rows = {row["alias"]: row for row in records[0]["evidence"]}
        self.assertEqual(rows["lm_head.weight"]["weight_orientation"], "linear_out_in")
        self.assertNotIn("index_axis", rows["lm_head.weight"])
        self.assertEqual(rows["transformer.wte.weight"]["index_axis"], 0)
        self.assertNotIn("weight_orientation", rows["transformer.wte.weight"])

    def test_identical_values_under_two_parameters_stay_two_records(self):
        embedding_weight = FakeWeight((10, 16))
        head_weight = FakeWeight((10, 16))
        # Equal as far as a value comparison could tell, distinct as objects.
        self.assertEqual(embedding_weight, head_weight)

        records = MODULE.logical_tensor_records(
            self.views_for(
                {
                    "transformer.wte": FakeEmbedding(embedding_weight, 10, 16),
                    "lm_head": FakeLinear(head_weight, in_features=16, out_features=10),
                },
                {
                    id(embedding_weight): ["transformer.wte.weight"],
                    id(head_weight): ["lm_head.weight"],
                },
                {"transformer.wte.weight", "lm_head.weight"},
            )
        )

        self.assertEqual(len(records), 2, records)
        by_name = {record["canonical_name"]: record for record in records}
        self.assertEqual(by_name["transformer.wte.weight"]["tensor_roles"], ["embedding_table"])
        self.assertIsNone(by_name["transformer.wte.weight"]["weight_orientation"])
        self.assertEqual(by_name["lm_head.weight"]["tensor_roles"], ["linear_projection"])
        self.assertEqual(by_name["lm_head.weight"]["weight_orientation"], "linear_out_in")
        self.assertEqual(by_name["lm_head.weight"]["index_axis"], None)

    def test_embedding_without_a_head_is_a_table_and_no_projection(self):
        weight = FakeWeight((32, 16))
        records = MODULE.logical_tensor_records(
            self.views_for(
                {"transformer.wpe": FakeEmbedding(weight, 32, 16)},
                {id(weight): ["transformer.wpe.weight"]},
                {"transformer.wpe.weight"},
            )
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["tensor_roles"], ["embedding_table"])
        self.assertEqual(records[0]["index_axis"], 0)
        self.assertIsNone(records[0]["weight_orientation"])

    def test_conv1d_layout_comes_from_its_declared_dimensions(self):
        weight = FakeWeight((16, 48))
        records = MODULE.logical_tensor_records(
            self.views_for(
                {"transformer.h.0.attn.c_attn": FakeConv1D(weight, nf=48, nx=16)},
                {id(weight): ["transformer.h.0.attn.c_attn.weight"]},
                {"transformer.h.0.attn.c_attn.weight"},
            )
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["tensor_roles"], ["conv1d_projection"])
        self.assertEqual(records[0]["weight_orientation"], "conv1d_in_out")
        self.assertIsNone(records[0]["index_axis"])

    def test_incompatible_layouts_for_one_parameter_are_a_conflict(self):
        weight = FakeWeight((8, 4))
        records = MODULE.logical_tensor_records(
            self.views_for(
                {
                    "shared.linear": FakeLinear(weight, in_features=4, out_features=8),
                    "shared.conv1d": FakeConv1D(weight, nf=4, nx=8),
                },
                {id(weight): ["shared.linear.weight", "shared.conv1d.weight"]},
                {"shared.linear.weight"},
            )
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["status"], "conflict")
        # Nothing may be declared about a tensor whose two readings disagree.
        self.assertEqual(record["tensor_roles"], [])
        self.assertIsNone(record["weight_orientation"])
        self.assertIsNone(record["index_axis"])
        self.assertTrue(
            any("incompatible layouts" in reason for reason in record["conflicts"]),
            record["conflicts"],
        )

    def test_a_class_that_disagrees_with_the_parameter_claims_no_role(self):
        weight = FakeWeight((16, 10))
        records = MODULE.logical_tensor_records(
            self.views_for(
                {"lm_head": FakeLinear(weight, in_features=16, out_features=10)},
                {id(weight): ["lm_head.weight"]},
                {"lm_head.weight"},
            )
        )

        self.assertEqual(records[0]["status"], "conflict")
        self.assertEqual(records[0]["tensor_roles"], [])
        row = records[0]["evidence"][0]
        self.assertIsNone(row["tensor_role"])
        self.assertTrue(any("refusing to assign" in reason for reason in row["conflicts"]))

    def test_three_aliases_of_one_parameter_make_one_record_with_three_aliases(self):
        weight = FakeWeight((10, 16))
        records = MODULE.logical_tensor_records(
            self.views_for(
                {
                    "transformer.wte": FakeEmbedding(weight, 10, 16),
                    "encoder.embed": FakeEmbedding(weight, 10, 16),
                    "tied.copy": FakeEmbedding(weight, 10, 16),
                },
                {
                    id(weight): [
                        "transformer.wte.weight",
                        "encoder.embed.weight",
                        "tied.copy.weight",
                    ]
                },
                {"transformer.wte.weight"},
            )
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["aliases"],
            ["encoder.embed.weight", "tied.copy.weight", "transformer.wte.weight"],
        )
        self.assertEqual(records[0]["tensor_roles"], ["embedding_table"])

    def test_module_without_a_weight_is_recorded_not_guessed(self):
        class Weightless(FakeLinear):
            def __init__(self) -> None:
                self.in_features = 16
                self.out_features = 10

        records = MODULE.logical_tensor_records(
            self.views_for(
                {"lm_head": Weightless()},
                {},
                {"lm_head.weight"},
            )
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "conflict")
        self.assertIsNone(records[0]["canonical_name"])
        self.assertTrue(
            any("no `weight` parameter" in reason for reason in records[0]["conflicts"])
        )

    def test_unrecognized_modules_produce_no_record(self):
        class LayerNorm:
            def __init__(self) -> None:
                self.weight = FakeWeight((16,))

        self.assertEqual(
            MODULE.logical_tensor_records(
                self.views_for({"transformer.ln_f": LayerNorm()}, {}, {"transformer.ln_f.weight"})
            ),
            [],
        )

    def test_a_parameter_stored_under_two_physical_keys_is_a_conflict(self):
        weight = FakeWeight((10, 16))
        records = MODULE.logical_tensor_records(
            self.views_for(
                {
                    "transformer.wte": FakeEmbedding(weight, 10, 16),
                    "decoder.embed": FakeEmbedding(weight, 10, 16),
                },
                {
                    id(weight): [
                        "transformer.wte.weight",
                        "decoder.embed.weight",
                    ]
                },
                {"transformer.wte.weight", "decoder.embed.weight"},
            )
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "conflict")
        self.assertIsNone(records[0]["canonical_name"])
        self.assertTrue(
            any("several physical keys" in reason for reason in records[0]["conflicts"]),
            records[0]["conflicts"],
        )

    def test_records_are_ordered_deterministically(self):
        first = FakeWeight((4, 4))
        second = FakeWeight((6, 6))
        views = self.views_for(
            {
                "zeta.head": FakeLinear(second, in_features=6, out_features=6),
                "alpha.head": FakeLinear(first, in_features=4, out_features=4),
            },
            {id(first): ["alpha.head.weight"], id(second): ["zeta.head.weight"]},
            {"alpha.head.weight", "zeta.head.weight"},
        )
        self.assertEqual(
            [
                record["canonical_name"]
                for record in MODULE.logical_tensor_records(views)
            ],
            ["alpha.head.weight", "zeta.head.weight"],
        )
        # Reordering the module walk must not reorder the bundle.
        reordered = self.views_for(
            {
                "alpha.head": FakeLinear(first, in_features=4, out_features=4),
                "zeta.head": FakeLinear(second, in_features=6, out_features=6),
            },
            {id(first): ["alpha.head.weight"], id(second): ["zeta.head.weight"]},
            {"alpha.head.weight", "zeta.head.weight"},
        )
        self.assertEqual(
            MODULE.logical_tensor_records(reordered),
            MODULE.logical_tensor_records(views),
        )

    def test_dtype_token_uses_the_safetensors_spelling(self):
        self.assertEqual(MODULE.dtype_token(FakeWeight((2, 2), "torch.float32")), "F32")
        self.assertEqual(MODULE.dtype_token(FakeWeight((2, 2), "torch.bfloat16")), "BF16")
        self.assertEqual(MODULE.dtype_token(FakeWeight((2, 2), "torch.quint8")), "torch.quint8")

    def test_supported_module_class_names_the_dimensions_each_class_declares(self):
        weight = FakeWeight((10, 16))
        self.assertEqual(
            MODULE.supported_module_class(
                FakeLinear(weight, in_features=16, out_features=10),
                FakeLinear,
                FakeConv1D,
                FakeEmbedding,
            ),
            ("linear", {"in_features": 16, "out_features": 10}),
        )
        self.assertEqual(
            MODULE.supported_module_class(
                FakeConv1D(weight, nf=48, nx=16), FakeLinear, FakeConv1D, FakeEmbedding
            ),
            ("conv1d", {"in_features": 16, "out_features": 48}),
        )
        self.assertEqual(
            MODULE.supported_module_class(
                FakeEmbedding(weight, 10, 16), FakeLinear, FakeConv1D, FakeEmbedding
            ),
            ("embedding", {"num_embeddings": 10, "embedding_dim": 16}),
        )

    def test_square_parameter_is_resolved_without_reshaping_anything(self):
        weight = FakeWeight((16, 16))
        records = MODULE.logical_tensor_records(
            self.views_for(
                {"proj.sq": FakeLinear(weight, in_features=16, out_features=16)},
                {id(weight): ["proj.sq.weight"]},
                {"proj.sq.weight"},
            )
        )

        self.assertEqual(records[0]["status"], "resolved")
        self.assertEqual(records[0]["shape"], [16, 16])
        self.assertEqual(records[0]["weight_orientation"], "linear_out_in")


if __name__ == "__main__":
    unittest.main()
