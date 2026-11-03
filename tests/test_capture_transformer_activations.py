import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "capture_transformer_activations.py"
SPEC = importlib.util.spec_from_file_location("hpsi_capture", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class CaptureUtilityTests(unittest.TestCase):
    def test_safe_name_removes_path_and_shell_unfriendly_characters(self):
        self.assertEqual(MODULE.safe_name("../model.layers.0/self_attn:q_proj"), "model.layers.0_self_attn_q_proj")

    def test_linear_module_maps_to_safetensors_weight_key(self):
        self.assertEqual(
            MODULE.source_tensor_name("model.layers.0.self_attn.q_proj"),
            "model.layers.0.self_attn.q_proj.weight",
        )
        self.assertEqual(MODULE.source_tensor_name(""), "weight")

    def test_sha256_file_hashes_exact_dataset_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            payload = b'{"text":"hello"}\n'
            path.write_bytes(payload)
            self.assertEqual(MODULE.sha256_file(path), hashlib.sha256(payload).hexdigest())


    def test_rejects_duplicate_jsonl_object_keys(self):
        with self.assertRaisesRegex(ValueError, "line 7: invalid strict JSON: duplicate JSON object key: text"):
            MODULE.parse_jsonl_record('{"text":"first","text":"second"}', 7)

    def test_rejects_nonstandard_jsonl_numeric_constants(self):
        with self.assertRaisesRegex(ValueError, "line 3: invalid strict JSON: non-standard JSON numeric constant: NaN"):
            MODULE.parse_jsonl_record('{"text":"hello","score":NaN}', 3)

    def test_rejects_non_object_jsonl_records_without_attribute_error(self):
        self.assertEqual(MODULE.parse_jsonl_record('["not", "a", "record"]', 2), ["not", "a", "record"])


if __name__ == "__main__":
    unittest.main()
