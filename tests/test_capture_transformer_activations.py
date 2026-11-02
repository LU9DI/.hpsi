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

    def test_sha256_file_hashes_exact_dataset_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            payload = b'{"text":"hello"}\n'
            path.write_bytes(payload)
            self.assertEqual(MODULE.sha256_file(path), hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()
