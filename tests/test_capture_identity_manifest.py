"""Capture bundle version 2 through the architecture manifesto (Phase 4.5).

The producer chain is: module graph -> identity records (capture) -> bundle index
(`manifest.json`, version 2) -> architecture manifest -> importer. These tests
cover the middle two steps in isolation, with bundles written by hand, so a
regression says which step lost the alias or the role rather than which assertion
happened to fail.

Nothing here reads a container, a model definition by name pattern, or the Rust
code: the manifest is generated from bundle data only.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import architecture_manifest_from_calibration as generator  # noqa: E402
import audit_hpsi_container as hpsi  # noqa: E402

WTE = "transformer.wte.weight"
HEAD = "lm_head.weight"
POSITION = "transformer.wpe.weight"
ATTN = "transformer.h.0.attn.c_attn.weight"


def table_evidence(alias: str, module: str) -> dict:
    return {
        "alias": alias,
        "module": module,
        "module_class": "embedding",
        "tensor_role": "embedding_table",
        "index_axis": 0,
        "conflicts": [],
    }


def head_evidence() -> dict:
    return {
        "alias": HEAD,
        "module": "lm_head",
        "module_class": "linear",
        "tensor_role": "linear_projection",
        "weight_orientation": "linear_out_in",
        "conflicts": [],
    }


def identity_record(
    canonical: str | None,
    aliases: list[str],
    roles: list[str],
    evidence: list[dict],
    *,
    orientation: str | None = None,
    axis: int | None = None,
    status: str = "resolved",
    conflicts: list[str] | None = None,
    shape: list[int] | None = None,
) -> dict:
    return {
        "canonical_name": canonical,
        "aliases": aliases,
        "identity": "parameter-object-identity",
        "shape": shape if shape is not None else [10, 16],
        "dtype": "F32",
        "tensor_roles": roles,
        "index_axis": axis,
        "weight_orientation": orientation,
        "status": status,
        "evidence": evidence,
        "conflicts": conflicts or [],
    }


def bundle(tensors: list[dict], *, version: int = 2, layers: list[dict] | None = None) -> dict:
    return {
        "format": "hpsi-transformer-calibration-manifest",
        "version": version,
        "model_type": "gpt2",
        "model_revision": None,
        "dataset_sha256": "a" * 64,
        "processed_sequences": 3,
        "captured_layers": len(layers or []),
        "layers": layers or [],
        "tensors": tensors,
        "architecture_tensors": len(tensors),
        "architecture_unresolved": sorted(
            item["canonical_name"] or item["aliases"][0]
            for item in tensors
            if item["status"] != "resolved"
        ),
    }


def write_bundle(directory: Path, document: dict) -> Path:
    (directory / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
    return directory


def entry_for(manifest: dict, name: str) -> dict:
    for entry in manifest["tensors"]:
        if entry["name"] == name:
            return entry
    raise AssertionError(f"{name} is absent from {manifest['tensors']}")


class ManifestFromIdentityRecordsTests(unittest.TestCase):
    def test_a_tied_parameter_becomes_one_entry_with_both_roles(self) -> None:
        document = bundle([
            identity_record(
                WTE,
                sorted([HEAD, WTE]),
                ["embedding_table", "linear_projection"],
                [head_evidence(), table_evidence(WTE, "transformer.wte")],
                orientation="linear_out_in",
                axis=0,
            )
        ])
        with tempfile.TemporaryDirectory() as directory:
            manifest = generator.manifest_from_calibration_bundle(
                write_bundle(Path(directory), document)
            )

        entry = entry_for(manifest, WTE)
        self.assertEqual(entry["tensor_role"], ["embedding_table", "linear_projection"])
        self.assertEqual(entry["aliases"], [HEAD, WTE])
        self.assertEqual(entry["index_axis"], 0)
        self.assertEqual(entry["weight_orientation"], "linear_out_in")
        self.assertEqual(
            [(row["alias"], row["tensor_role"], row["module_class"])
             for row in entry["role_evidence"]],
            [
                (HEAD, "linear_projection", "linear"),
                (WTE, "embedding_table", "embedding"),
            ],
            "the association survives the conversion",
        )
        self.assertNotIn("unresolved", manifest)

    def test_a_positional_table_becomes_an_entry_without_an_orientation(self) -> None:
        document = bundle([
            identity_record(
                POSITION,
                [POSITION],
                ["embedding_table"],
                [table_evidence(POSITION, "transformer.wpe")],
                axis=0,
                shape=[32, 16],
            )
        ])
        with tempfile.TemporaryDirectory() as directory:
            manifest = generator.manifest_from_calibration_bundle(
                write_bundle(Path(directory), document)
            )

        entry = entry_for(manifest, POSITION)
        self.assertEqual(entry["tensor_role"], ["embedding_table"])
        self.assertEqual(entry["index_axis"], 0)
        self.assertNotIn(
            "weight_orientation",
            entry,
            "the capture measured no layout for a table, so none is claimed",
        )

    def test_two_identities_of_equal_values_become_two_entries(self) -> None:
        document = bundle([
            identity_record(WTE, [WTE], ["embedding_table"],
                            [table_evidence(WTE, "transformer.wte")], axis=0),
            identity_record(HEAD, [HEAD], ["linear_projection"], [head_evidence()],
                            orientation="linear_out_in"),
            identity_record(ATTN, [ATTN], ["conv1d_projection"], [{
                "alias": ATTN, "module": "transformer.h.0.attn.c_attn",
                "module_class": "conv1d", "tensor_role": "conv1d_projection",
                "weight_orientation": "conv1d_in_out", "conflicts": [],
            }], orientation="conv1d_in_out", shape=[16, 48]),
        ])
        with tempfile.TemporaryDirectory() as directory:
            manifest = generator.manifest_from_calibration_bundle(
                write_bundle(Path(directory), document)
            )

        names = [entry["name"] for entry in manifest["tensors"]]
        self.assertEqual(len(names), len(set(names)), "one entry per identity")
        self.assertEqual(entry_for(manifest, WTE)["aliases"], [WTE])
        self.assertEqual(entry_for(manifest, HEAD)["aliases"], [HEAD])
        self.assertEqual(
            entry_for(manifest, ATTN)["weight_orientation"], "conv1d_in_out"
        )

    def test_a_conflicted_identity_is_reported_and_declares_nothing(self) -> None:
        document = bundle([
            identity_record(
                WTE,
                [WTE],
                [],
                [{"alias": WTE, "module": "transformer.wte", "module_class": "embedding",
                  "tensor_role": None,
                  "conflicts": ["the embedding module declares a weight of [16, 10] "
                                "but its parameter is [10, 16]"]}],
                status="conflict",
                conflicts=[WTE + ": the embedding module declares a weight of [16, 10] "
                                  "but its parameter is [10, 16]"],
            ),
            identity_record(HEAD, [HEAD], ["linear_projection"], [head_evidence()],
                            orientation="linear_out_in"),
        ])
        with tempfile.TemporaryDirectory() as directory:
            manifest = generator.manifest_from_calibration_bundle(
                write_bundle(Path(directory), document)
            )

        self.assertEqual([entry["name"] for entry in manifest["tensors"]], [HEAD])
        self.assertEqual(manifest["unresolved"], [WTE])

    def test_an_identity_with_no_physical_key_is_unresolved(self) -> None:
        document = bundle([
            identity_record(
                None,
                [WTE],
                [],
                [{"alias": WTE, "module": "transformer.wte", "module_class": "embedding",
                  "tensor_role": None,
                  "conflicts": ["no physical Safetensors key"]}],
                status="unresolved",
                conflicts=["no physical Safetensors key was found"],
            )
        ])
        with tempfile.TemporaryDirectory() as directory:
            manifest = generator.manifest_from_calibration_bundle(
                write_bundle(Path(directory), document)
            )

        self.assertEqual(manifest["tensors"], [])
        self.assertEqual(manifest["unresolved"], [WTE])

    def test_a_record_stating_a_role_that_its_evidence_drops_is_not_silently_fixed(self) -> None:
        # The bundle is the producer's own claim; the generator consumes it as
        # stated. A record whose roles and evidence disagree has already been
        # refused by the bundle validator, so the generator only reports it.
        document = bundle([
            identity_record(
                WTE,
                [WTE],
                ["embedding_table", "linear_projection"],
                [table_evidence(WTE, "transformer.wte")],
                orientation="linear_out_in",
                axis=0,
            )
        ])
        with tempfile.TemporaryDirectory() as directory:
            manifest = generator.manifest_from_calibration_bundle(
                write_bundle(Path(directory), document)
            )
        entry = entry_for(manifest, WTE)
        self.assertEqual(len(entry["tensor_role"]), 2)
        self.assertEqual(len(entry["role_evidence"]), 1)
        # Both parsers downstream refuse this pairing rather than repairing it.
        with tempfile.TemporaryDirectory() as second:
            path = Path(second) / "architecture.json"
            path.write_text(
                json.dumps({
                    "format": "hpsi-architecture-manifest",
                    "version": 1,
                    "model_type": "gpt2",
                    "tensors": [entry],
                }),
                encoding="utf-8",
            )
            with self.assertRaises(hpsi.HpsiFormatError) as failure:
                hpsi.ArchitectureOracle.load(path)
        self.assertIn(
            "with no role_evidence naming the module that evidences it",
            str(failure.exception),
        )


class ManifestFromLegacyBundleTests(unittest.TestCase):
    def test_version_one_bundles_still_produce_the_same_entries(self) -> None:
        document = {
            "format": "hpsi-transformer-calibration-manifest",
            "version": 1,
            "model_type": "gpt2",
            "model_revision": None,
            "dataset_sha256": "a" * 64,
            "processed_sequences": 3,
            "captured_layers": 2,
            "layers": [
                {
                    "module_name": "transformer.h.0.attn.c_attn",
                    "module_type": "conv1d",
                    "source_tensor_name": ATTN,
                    "file": "0000.json",
                    "sample_count": 8,
                    "features": 16,
                },
                {
                    "module_name": "lm_head",
                    "module_type": "linear",
                    "source_tensor_name": WTE,
                    "file": "0001.json",
                    "sample_count": 8,
                    "features": 16,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = generator.manifest_from_calibration_bundle(
                write_bundle(Path(directory), document)
            )

        self.assertEqual(
            manifest["tensors"],
            [
                {"name": ATTN, "weight_orientation": "conv1d_in_out",
                 "tensor_role": "conv1d_projection"},
                {"name": WTE, "weight_orientation": "linear_out_in",
                 "tensor_role": "linear_projection"},
            ],
            "a legacy bundle states less, and nothing is invented to fill it in",
        )
        self.assertNotIn("unresolved", manifest)

    def test_an_unsupported_module_class_is_still_an_error(self) -> None:
        document = {
            "format": "hpsi-transformer-calibration-manifest",
            "version": 1,
            "model_type": "gpt2",
            "dataset_sha256": "a" * 64,
            "processed_sequences": 3,
            "captured_layers": 1,
            "layers": [{
                "module_name": "transformer.wte",
                "module_type": "embedding",
                "source_tensor_name": WTE,
                "file": "0000.json",
                "sample_count": 8,
                "features": 16,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no documented weight layout"):
                generator.manifest_from_calibration_bundle(
                    write_bundle(Path(directory), document)
                )

    def test_an_unknown_bundle_version_is_refused(self) -> None:
        document = bundle([
            identity_record(WTE, [WTE], ["embedding_table"],
                            [table_evidence(WTE, "transformer.wte")], axis=0)
        ])
        document["version"] = 3
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unsupported calibration bundle version"):
                generator.manifest_from_calibration_bundle(
                    write_bundle(Path(directory), document)
                )


if __name__ == "__main__":
    unittest.main()
