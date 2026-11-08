import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "validate_calibration_bundle.py"
SPEC = importlib.util.spec_from_file_location("hpsi_calibration_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class CalibrationBundleValidationTests(unittest.TestCase):
    def write_bundle(self, root: Path, moments=None):
        moments = [0.25, 1.0, 2.25] if moments is None else moments
        artifact = {
            "format": "hpsi-activation-calibration",
            "version": 1,
            "layer_name": "model.layers.0.mlp.up_proj.weight",
            "module_type": "linear",
            "model": {"model_type": "fixture", "revision": "abc123"},
            "dataset": {"sha256": "a" * 64, "processed_sequences": 2},
            "statistic": "diagonal_second_moment",
            "shape": [8, len(moments)],
            "sample_count": 8,
            "activation_second_moment": moments,
        }
        (root / "0000_layer.json").write_text(json.dumps(artifact), encoding="utf-8")
        manifest = {
            "format": "hpsi-transformer-calibration-manifest",
            "version": 1,
            "model_type": "fixture",
            "model_revision": "abc123",
            "dataset_sha256": "a" * 64,
            "processed_sequences": 2,
            "captured_layers": 1,
            "layers": [{
            "module_name": "model.layers.0.mlp.up_proj",
            "source_tensor_name": artifact["layer_name"],
            "file": "0000_layer.json",
            "module_type": artifact["module_type"],
            "sample_count": 8,
            "features": len(moments),
          }],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_accepts_consistent_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            report = MODULE.validate_bundle(root)
            self.assertTrue(report["valid"])
            self.assertEqual(report["captured_layers"], 1)
            self.assertEqual(report["total_layer_samples"], 8)
            self.assertEqual(report["total_features"], 3)

    def test_rejects_non_finite_or_negative_moments(self):
        cases = [
            ([0.1, float("nan")], "non-standard JSON numeric constant: NaN"),
            ([0.1, -0.2], "finite and non-negative"),
            ([0.1, float("inf")], "non-standard JSON numeric constant: Infinity"),
        ]
        for invalid, expected_error in cases:
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_bundle(root, invalid)
                with self.assertRaisesRegex(ValueError, expected_error):
                    MODULE.validate_bundle(root)

    def test_rejects_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            artifact_path = root / "0000_layer.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["shape"] = [8, 99]
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shape does not match"):
                MODULE.validate_bundle(root)


    def test_rejects_malformed_model_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            path = root / "0000_layer.json"
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["model"] = ["not", "an", "object"]
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model metadata must be an object"):
                MODULE.validate_bundle(root)

    def test_rejects_dataset_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            path = root / "0000_layer.json"
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["dataset"]["sha256"] = "b" * 64
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset SHA-256 mismatch"):
                MODULE.validate_bundle(root)

    def test_rejects_malformed_dataset_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            path = root / "0000_layer.json"
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["dataset"] = None
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset metadata must be an object"):
                MODULE.validate_bundle(root)


    def test_rejects_boolean_layer_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            path = root / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["captured_layers"] = True
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "captured_layers does not match"):
                MODULE.validate_bundle(root)

    def test_rejects_boolean_artifact_sequence_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            path = root / "0000_layer.json"
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["dataset"]["processed_sequences"] = True
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "processed sequence count mismatch"):
                MODULE.validate_bundle(root)

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["layers"][0]["file"] = "../outside.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe calibration filename"):
                MODULE.validate_bundle(root)


    def test_rejects_boolean_manifest_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            path = root / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["version"] = True
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported manifest version"):
                MODULE.validate_bundle(root)

    def test_rejects_boolean_artifact_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            path = root / "0000_layer.json"
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["version"] = True
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported artifact version"):
                MODULE.validate_bundle(root)

    def test_rejects_boolean_manifest_sample_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            path = root / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["layers"][0]["sample_count"] = True
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest sample count mismatch"):
                MODULE.validate_bundle(root)



    def test_rejects_integer_moment_too_large_for_finite_float(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root, [10 ** 309])
            with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                MODULE.validate_bundle(root)



    def test_rejects_duplicate_json_object_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            path = root / "manifest.json"
            raw = path.read_text(encoding="utf-8")
            raw = raw.replace('"version": 1,', '"version": 1, "version": 1,', 1)
            path.write_text(raw, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key: version"):
                MODULE.validate_bundle(root)

    def test_rejects_nonstandard_json_numeric_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            path = root / "0000_layer.json"
            raw = path.read_text(encoding="utf-8")
            raw = raw.replace('"activation_second_moment": [0.25, 1.0, 2.25]',
                              '"activation_second_moment": [0.25, NaN, 2.25]')
            path.write_text(raw, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-standard JSON numeric constant: NaN"):
                MODULE.validate_bundle(root)

    def test_rejects_uppercase_dataset_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            path = root / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["dataset_sha256"] = "A" * 64
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lowercase hexadecimal"):
                MODULE.validate_bundle(root)


class CalibrationIdentityBundleTests(unittest.TestCase):
    """Version 2 adds per-parameter identity records; the validator checks them."""

    LAYER = "transformer.wte.weight"

    def write_identity_bundle(
        self,
        root: Path,
        tensors=None,
        *,
        aliases=None,
        unresolved=None,
    ):
        artifact = {
            "format": "hpsi-activation-calibration",
            "version": 1,
            "layer_name": self.LAYER,
            "module_type": "linear",
            "model": {"model_type": "gpt2", "revision": None},
            "dataset": {"sha256": "a" * 64, "processed_sequences": 3},
            "statistic": "diagonal_second_moment",
            "shape": [8, 16],
            "sample_count": 8,
            "activation_second_moment": [1.0] * 16,
        }
        (root / "0000_layer.json").write_text(json.dumps(artifact), encoding="utf-8")
        if tensors is None:
            tensors = [{
                "canonical_name": self.LAYER,
                "aliases": ["lm_head.weight", self.LAYER],
                "identity": "parameter-object-identity",
                "shape": [10, 16],
                "dtype": "F32",
                "tensor_roles": ["embedding_table", "linear_projection"],
                "index_axis": 0,
                "weight_orientation": "linear_out_in",
                "status": "resolved",
                "conflicts": [],
                "evidence": [
                    {
                        "alias": "lm_head.weight",
                        "module": "lm_head",
                        "module_class": "linear",
                        "tensor_role": "linear_projection",
                        "weight_orientation": "linear_out_in",
                    },
                    {
                        "alias": self.LAYER,
                        "module": "transformer.wte",
                        "module_class": "embedding",
                        "tensor_role": "embedding_table",
                        "index_axis": 0,
                    },
                ],
            }]
        manifest = {
            "format": "hpsi-transformer-calibration-manifest",
            "version": 2,
            "model_type": "gpt2",
            "model_revision": None,
            "dataset_sha256": "a" * 64,
            "processed_sequences": 3,
            "captured_layers": 1,
            "layers": [{
                "module_name": "lm_head",
                "module_type": "linear",
                "source_tensor_name": self.LAYER,
                "file": "0000_layer.json",
                "sample_count": 8,
                "features": 16,
                "aliases": aliases if aliases is not None else ["lm_head.weight", self.LAYER],
            }],
            "tensors": tensors,
            "architecture_tensors": len(tensors),
            "architecture_unresolved": (
                unresolved if unresolved is not None
                else sorted(
                    item["canonical_name"] or item["aliases"][0]
                    for item in tensors
                    if item.get("status") != "resolved"
                )
            ),
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_accepts_a_coherent_identity_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_identity_bundle(root)
            report = MODULE.validate_bundle(root)
            self.assertTrue(report["valid"])
            self.assertEqual(report["bundle_version"], 2)
            self.assertEqual(report["architecture_tensors"], 1)
            self.assertEqual(report["architecture_unresolved"], [])

    def test_rejects_an_alias_claimed_by_two_identities(self):
        def mutate(manifest):
            other = json.loads(json.dumps(manifest["tensors"][0]))
            other["canonical_name"] = "lm_head.weight"
            other["aliases"] = ["lm_head.weight"]
            other["evidence"] = [
                row for row in other["evidence"] if row["alias"] == "lm_head.weight"
            ]
            other["tensor_roles"] = ["linear_projection"]
            other["index_axis"] = None
            manifest["tensors"].append(other)
            manifest["architecture_tensors"] = 2

        self.rejects(mutate, "belongs to more than one identity")

    def test_rejects_a_resolved_identity_with_an_unevidenced_role(self):
        def mutate(manifest):
            record = manifest["tensors"][0]
            record["evidence"] = [
                row for row in record["evidence"] if row["alias"] != "lm_head.weight"
            ]

        self.rejects(mutate, "are declared with no module evidence")

    def test_rejects_a_conflicted_identity_that_still_declares_roles(self):
        def mutate(manifest):
            record = manifest["tensors"][0]
            record["status"] = "conflict"
            record["conflicts"] = ["the layout claims disagree"]

        self.rejects(mutate, "must declare no role, layout or index axis")

    def test_rejects_embedding_evidence_without_an_index_axis(self):
        def mutate(manifest):
            for row in manifest["tensors"][0]["evidence"]:
                if row["tensor_role"] == "embedding_table":
                    row.pop("index_axis")

        self.rejects(mutate, "must state index_axis 0")

    def test_rejects_evidence_claiming_a_layout_its_role_does_not_have(self):
        def mutate(manifest):
            for row in manifest["tensors"][0]["evidence"]:
                if row["tensor_role"] == "embedding_table":
                    row["weight_orientation"] = "conv1d_in_out"

        self.rejects(mutate, "must not claim a layout")

    def test_rejects_a_projection_row_stating_an_index_axis(self):
        def mutate(manifest):
            for row in manifest["tensors"][0]["evidence"]:
                if row["tensor_role"] == "linear_projection":
                    row["index_axis"] = 0

        self.rejects(mutate, "may not state an index_axis")

    def test_rejects_a_layer_that_omits_its_aliases(self):
        self.rejects(
            lambda manifest: manifest["layers"][0].pop("aliases"),
            "must name the aliases",
        )

    def test_rejects_unresolved_accounts_that_do_not_match_the_records(self):
        def mutate(manifest):
            record = manifest["tensors"][0]
            record["status"] = "conflict"
            record["tensor_roles"] = []
            record["index_axis"] = None
            record["weight_orientation"] = None
            record["conflicts"] = ["two modules disagree about this parameter"]
            record["evidence"] = [
                {
                    "alias": row["alias"],
                    "module": row["module"],
                    "module_class": row["module_class"],
                    "tensor_role": None,
                    "conflicts": ["the claim is not supported by this module"],
                }
                for row in record["evidence"]
            ]
            manifest["architecture_unresolved"] = []

        self.rejects(mutate, "architecture_unresolved does not match")

    def test_rejects_a_version_beyond_the_readable_range(self):
        self.rejects(
            lambda manifest: manifest.__setitem__("version", 3),
            "unsupported manifest version",
        )

    def rejects(self, mutate, fragment: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_identity_bundle(root)
            path = root / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            mutate(manifest)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError) as failure:
                MODULE.validate_bundle(root)
            self.assertIn(fragment, str(failure.exception))


if __name__ == "__main__":
    unittest.main()
