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
        for invalid in ([0.1, float("nan")], [0.1, -0.2], [0.1, float("inf")]):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_bundle(root, invalid)
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
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
            self.write_bundle(root, [10 ** 10000])
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


if __name__ == "__main__":
    unittest.main()
