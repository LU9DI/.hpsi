"""Derive an architecture manifest from a capture bundle.

The capture tool (``tools/capture_transformer_activations.py``) walks the live
model and records, per projection, the *class* of the module that produced the
activations: ``nn.Linear`` or Hugging Face ``Conv1D``. That record is a fact
about the model definition, taken before any container exists, which is exactly
what an orientation oracle needs to be.

This tool converts such a bundle into the ``hpsi-architecture-manifest`` that
``tools/audit_hpsi_container.py --architecture`` and ``hpsi inspect
--architecture`` consume. It never looks at a container, at a tensor name
pattern, or at an importer plan:

    nn.Linear    -> weight is [out_features, in_features] -> linear_out_in
    Conv1D       -> weight is [in_features, out_features] -> conv1d_in_out
    nn.Embedding -> indexed on axis 0, and no layout claim of its own

A version 2 bundle records one identity per *parameter*, with every alias the
model reaches it under and the module class behind each role, so a tied
embedding and output head is recorded once carrying both roles rather than one
role chosen at the other's expense. An identity the bundle could not settle is
reported as unresolved and stays out of the manifest: declaring one reading of a
disagreement would confirm the very misreading the manifest is meant to detect.

Any other module type is an error rather than a guess, because a manifest that
silently defaults a module to ``linear_out_in`` would confirm the very
transposition it is supposed to detect.

Usage::

    python tools/architecture_manifest_from_calibration.py \
        --calibration-dir .tmp-gpt2-conv1d-test/calibration \
        --out .tmp-gpt2-conv1d-test/architecture.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

CALIBRATION_MANIFEST_FORMAT = "hpsi-transformer-calibration-manifest"
# Version 1 recorded one module class per physical key. Version 2 records one
# identity per parameter, with its aliases, its roles and the module behind each
# role. Both are readable: the older bundle simply states less.
CALIBRATION_MANIFEST_VERSIONS = (1, 2)
ARCHITECTURE_MANIFEST_FORMAT = "hpsi-architecture-manifest"
ARCHITECTURE_MANIFEST_VERSION = 1

# The only module classes whose weight layout is defined by the framework.
MODULE_TYPE_ORIENTATIONS = {
    "linear": "linear_out_in",
    "conv1d": "conv1d_in_out",
}

# The same class evidence names the architectural role. A bundle states classes,
# never name patterns, so role and layout come from one source and cannot drift.
MODULE_TYPE_ROLES = {
    "linear": "linear_projection",
    "conv1d": "conv1d_projection",
    # An embedding is reached by index, never by multiplication. It carries a
    # role and an indexing axis, and no orientation of its own: the layout the
    # container records for a table describes how the quantizer grouped it,
    # which the capture cannot claim to have measured.
    "embedding": "embedding_table",
}

# A role that indexes has to say which axis it indexes.
ROLE_INDEX_AXIS = {"embedding_table": 0}


def _load_bundle(directory: Path) -> dict[str, Any]:
    path = directory / "manifest.json"
    if not path.is_file():
        raise ValueError(f"calibration bundle lacks manifest.json: {path}")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("format") != CALIBRATION_MANIFEST_FORMAT:
        raise ValueError(
            f"unsupported calibration bundle format {bundle.get('format')!r}"
        )
    if bundle.get("version") not in CALIBRATION_MANIFEST_VERSIONS:
        raise ValueError(
            f"unsupported calibration bundle version {bundle.get('version')!r}; "
            f"expected one of {list(CALIBRATION_MANIFEST_VERSIONS)}"
        )
    return bundle


def _entries_from_identity_records(bundle: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Turn version-2 identity records into manifest entries.

    The record already carries the role, the alias that carries it and the module
    class behind it, so nothing here looks at a name. A conflicted or unresolved
    identity is returned as an unresolved alias instead of being emitted: the
    alternative would be a manifest declaring whichever half of a disagreement
    this function happened to reach first.
    """
    entries: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for record in bundle.get("tensors") or []:
        if not isinstance(record, dict):
            raise ValueError("calibration bundle identity record must be an object")
        status = record.get("status")
        name = record.get("canonical_name")
        roles = record.get("tensor_roles") or []
        aliases = record.get("aliases") or []
        evidence = record.get("evidence") or []

        if status != "resolved" or not isinstance(name, str) or not name.strip() or not roles:
            unresolved.append(
                name if isinstance(name, str) and name else ", ".join(str(a) for a in aliases)
                or "unnamed identity"
            )
            continue

        index_axis = record.get("index_axis")
        orientation = record.get("weight_orientation")
        entry: dict[str, Any] = {"name": name, "tensor_role": list(roles)}
        if orientation is not None:
            entry["weight_orientation"] = orientation
        if index_axis is not None:
            entry["index_axis"] = index_axis
        entry["aliases"] = sorted({str(alias) for alias in aliases})
        entry["role_evidence"] = [
            {
                key: item[key]
                for key in (
                    "alias",
                    "module",
                    "module_class",
                    "tensor_role",
                    "weight_orientation",
                    "index_axis",
                )
                if isinstance(item, dict) and item.get(key) is not None
            }
            for item in evidence
            if isinstance(item, dict) and item.get("tensor_role") is not None
        ]
        entries.append(entry)

    return entries, unresolved


def _entries_from_module_types(bundle: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Turn version-1 class-per-key rows into manifest entries.

    A version 1 bundle recorded only the projection module whose activations it
    measured, so it cannot state an alias, an indexing axis or a second role: the
    entries it produces are exactly as strong as the record behind them, and no
    table role is inferred from a name to make up the difference.
    """
    layers = bundle.get("layers")
    if not isinstance(layers, list):
        raise ValueError("calibration bundle lacks a `layers` array")

    entries: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("calibration bundle layer must be an object")
        module_type = layer.get("module_type")
        orientation = MODULE_TYPE_ORIENTATIONS.get(module_type)
        if orientation is None:
            raise ValueError(
                f"{layer.get('source_tensor_name')}: module type {module_type!r} has no "
                "documented weight layout; refusing to guess an orientation"
            )
        name = layer.get("source_tensor_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("calibration bundle layer lacks `source_tensor_name`")
        if any(existing["name"] == name for existing in entries):
            raise ValueError(f"{name}: calibration bundle declares it more than once")
        # No `source_shape` is emitted: the bundle records activation geometry,
        # not the parameter's rank-2 shape, and a wrong shape would reject a
        # container for the wrong reason.
        entries.append({
            "name": name,
            "weight_orientation": orientation,
            "tensor_role": MODULE_TYPE_ROLES[module_type],
        })

    return entries, unresolved


def manifest_from_calibration_bundle(directory: Path) -> dict[str, Any]:
    """Build the architecture manifest described by a capture bundle."""
    bundle = _load_bundle(directory)

    if bundle.get("version") == 2:
        entries, unresolved = _entries_from_identity_records(bundle)
    else:
        entries, unresolved = _entries_from_module_types(bundle)

    model_type = bundle.get("model_type")
    manifest = {
        "format": ARCHITECTURE_MANIFEST_FORMAT,
        "version": ARCHITECTURE_MANIFEST_VERSION,
        "model_type": model_type if isinstance(model_type, str) else None,
        "tensors": entries,
        "oracle": "module classes recorded by the activation capture",
    }
    if unresolved:
        # Names the bundle could not settle. Nothing is declared about them, so
        # an auditor reports them as unconfirmed rather than as a contradiction.
        manifest["unresolved"] = sorted(unresolved)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--calibration-dir", required=True, help="capture bundle directory containing manifest.json"
    )
    parser.add_argument("--out", required=True, help="architecture manifest path")
    args = parser.parse_args(argv)

    manifest = manifest_from_calibration_bundle(Path(args.calibration_dir))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path} with {len(manifest['tensors'])} architecture entries")
    for entry in manifest["tensors"]:
        roles = entry["tensor_role"]
        roles = roles if isinstance(roles, list) else [roles]
        summary = f"  {entry['name']} -> roles={','.join(roles)}"
        if entry.get("weight_orientation") is not None:
            summary += f" orientation={entry['weight_orientation']}"
        if entry.get("index_axis") is not None:
            summary += f" index_axis={entry['index_axis']}"
        if entry.get("aliases"):
            summary += f" aliases={','.join(entry['aliases'])}"
        print(summary)
    for name in manifest.get("unresolved", []):
        print(f"  ! unresolved, nothing declared: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
