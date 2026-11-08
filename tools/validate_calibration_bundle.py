#!/usr/bin/env python3
"""Validate a local HPSI transformer-calibration bundle without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{path.name}: cannot read strict UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


TENSOR_ROLES = ("conv1d_projection", "embedding_table", "linear_projection", "other", "unknown")
ROLE_LAYOUTS = {
    "linear_projection": "linear_out_in",
    "conv1d_projection": "conv1d_in_out",
}
IDENTITY_STATUSES = ("resolved", "conflict", "unresolved")


def validate_identity_records(records: Any) -> list[str]:
    """Check the version-2 architecture section of a capture bundle.

    The records are the producer's claim about which parameter is which, so the
    checks are about internal agreement: an alias belongs to one identity, a
    resolved record states the roles its evidence states, and a record that could
    not be settled states no role at all instead of guessing one.
    """
    require(isinstance(records, list), "manifest tensors must be an array")

    seen_canonical: set[str] = set()
    seen_alias: set[str] = set()
    unresolved: list[str] = []

    for record in records:
        require(isinstance(record, dict), "manifest tensor identity record must be an object")
        aliases = record.get("aliases")
        require(
            isinstance(aliases, list)
            and aliases
            and all(isinstance(alias, str) and alias.strip() for alias in aliases)
            and len(set(aliases)) == len(aliases),
            f"identity record {record!r}: `aliases` must be unique non-empty strings",
        )
        canonical = record.get("canonical_name")
        require(
            canonical is None or (isinstance(canonical, str) and canonical.strip()),
            f"identity record: `canonical_name` must be a non-empty string or null",
        )
        name = canonical if isinstance(canonical, str) else aliases[0]
        require(
            all(alias not in seen_alias for alias in aliases),
            f"{name}: an alias belongs to more than one identity; the bundle grouped "
            "parameters by something other than their identity",
        )
        seen_alias.update(aliases)
        require(record.get("identity") == "parameter-object-identity",
                f"{name}: `identity` must be 'parameter-object-identity'")

        status = record.get("status")
        require(status in IDENTITY_STATUSES, f"{name}: unsupported identity status {status!r}")
        roles = record.get("tensor_roles")
        require(
            isinstance(roles, list)
            and all(role in TENSOR_ROLES for role in roles)
            and roles == sorted(set(roles)),
            f"{name}: `tensor_roles` must be unique known roles in sorted order",
        )
        evidence = record.get("evidence")
        require(isinstance(evidence, list), f"{name}: `evidence` must be an array")
        evidenced: set[str] = set()
        for row in evidence:
            require(isinstance(row, dict), f"{name}: evidence rows must be objects")
            alias = row.get("alias")
            require(
                isinstance(alias, str) and alias in aliases,
                f"{name}: evidence alias {alias!r} is not one of the identity's aliases",
            )
            role = row.get("tensor_role")
            if role is None:
                require(
                    isinstance(row.get("conflicts"), list) and row["conflicts"],
                    f"{name}: evidence for {alias!r} states no role and no reason why",
                )
                continue
            require(role in TENSOR_ROLES, f"{name}: evidence role {role!r} is not known")
            require(role in roles, f"{name}: evidence names {role!r}, which the record drops")
            evidenced.add(role)
            if role == "embedding_table":
                require(row.get("index_axis") == 0,
                        f"{name}: embedding_table evidence must state index_axis 0")
                require(row.get("weight_orientation") is None,
                        f"{name}: embedding_table evidence must not claim a layout")
            else:
                require(
                    row.get("weight_orientation") == ROLE_LAYOUTS.get(role),
                    f"{name}: {role} evidence must state layout "
                    f"{ROLE_LAYOUTS.get(role)!r}",
                )
                require(row.get("index_axis") is None,
                        f"{name}: {role} evidence may not state an index_axis")
        if status == "resolved":
            require(isinstance(canonical, str) and canonical.strip(),
                    f"{name}: a resolved identity must name its physical key")
            require(canonical not in seen_canonical, f"{canonical}: declared by two identities")
            seen_canonical.add(canonical)
            require(canonical in aliases,
                    f"{name}: the physical key must be one of the identity's aliases")
            require(bool(roles), f"{name}: a resolved identity states no role")
            require(evidenced == set(roles),
                    f"{name}: roles {sorted(set(roles) - evidenced)} are declared with no "
                    "module evidence behind them")
            require("embedding_table" in roles or record.get("index_axis") is None,
                    f"{name}: index_axis without an embedding_table role")
            require("embedding_table" not in roles or record.get("index_axis") == 0,
                    f"{name}: an embedding table must state its indexing axis")
            projection = next((role for role in roles if role in ROLE_LAYOUTS), None)
            expected = ROLE_LAYOUTS.get(projection) if projection else None
            if projection is None:
                tables = [role for role in roles if role == "embedding_table"]
                require(tables and record.get("weight_orientation") is None,
                        f"{name}: a role set with neither a projection nor a table states no layout "
                        "and no indexing")
            else:
                require(record.get("weight_orientation") == expected,
                        f"{name}: `weight_orientation` must be {expected!r} for {projection}")
        else:
            require(
                not roles and record.get("index_axis") is None
                and record.get("weight_orientation") is None,
                f"{name}: a {status} identity must declare no role, layout or index axis",
            )
            conflicts = record.get("conflicts")
            require(isinstance(conflicts, list) and conflicts,
                    f"{name}: a {status} identity must say what is unresolved")
            unresolved.append(name)

        shape = record.get("shape")
        require(
            shape == [] or (
                isinstance(shape, list)
                and all(isinstance(dimension, int) and not isinstance(dimension, bool)
                        and dimension > 0 for dimension in shape)
            ),
            f"{name}: `shape` must be positive integers or empty",
        )
        dtype = record.get("dtype")
        require(dtype is None or (isinstance(dtype, str) and dtype.strip()),
                f"{name}: `dtype` must be a non-empty string or null")

    return unresolved


def validate_bundle(directory: Path, dataset: Path | None = None) -> dict[str, Any]:
    root = directory.resolve(strict=True)
    require(root.is_dir(), "calibration path must be a directory")
    manifest = load_json(root / "manifest.json")
    require(manifest.get("format") == "hpsi-transformer-calibration-manifest", "unsupported manifest format")
    manifest_version = manifest.get("version")
    require(isinstance(manifest_version, int) and not isinstance(manifest_version, bool)
            and manifest_version in (1, 2), "unsupported manifest version")
    model_type = manifest.get("model_type")
    revision = manifest.get("model_revision")
    require(isinstance(model_type, str) and model_type.strip(), "manifest model_type is missing")
    require(revision is None or isinstance(revision, str), "manifest model_revision must be a string or null")
    dataset_hash = manifest.get("dataset_sha256")
    require(isinstance(dataset_hash, str) and len(dataset_hash) == 64, "manifest dataset_sha256 is invalid")
    require(all(c in "0123456789abcdef" for c in dataset_hash),
            "manifest dataset_sha256 must be lowercase hexadecimal")
    if dataset is not None:
        require(dataset.is_file(), "dataset file does not exist")
        require(sha256_file(dataset) == dataset_hash, "dataset SHA-256 does not match calibration manifest")

    layers = manifest.get("layers")
    require(isinstance(layers, list) and layers, "manifest has no captured layers")
    captured_layers = manifest.get("captured_layers")
    require(isinstance(captured_layers, int) and not isinstance(captured_layers, bool)
            and captured_layers == len(layers), "captured_layers does not match layers length")
    processed_sequences = manifest.get("processed_sequences")
    require(isinstance(processed_sequences, int) and not isinstance(processed_sequences, bool) and processed_sequences > 0,
            "manifest processed_sequences must be a positive integer")
    seen_names: set[str] = set()
    seen_files: set[str] = set()
    total_samples = 0
    for entry in layers:
        require(isinstance(entry, dict), "layer manifest entry must be an object")
        name, filename = entry.get("source_tensor_name"), entry.get("file")
        require(isinstance(name, str) and name.strip(), "layer source_tensor_name is missing")
        require(name not in seen_names, f"duplicate source tensor: {name}")
        seen_names.add(name)
        require(isinstance(filename, str) and filename and Path(filename).name == filename, f"{name}: unsafe calibration filename")
        require(filename not in seen_files, f"duplicate calibration filename: {filename}")
        seen_files.add(filename)
        module_type = entry.get("module_type")
        require(isinstance(module_type, str) and module_type in ("linear", "conv1d"),
                f"{name}: manifest module_type must be 'linear' or 'conv1d'")
        if manifest_version == 2:
            aliases = entry.get("aliases")
            require(
                isinstance(aliases, list)
                and aliases
                and all(isinstance(alias, str) and alias.strip() for alias in aliases)
                and name in aliases,
                f"{name}: a version 2 layer must name the aliases of the parameter it measured",
            )
        artifact_path = (root / filename).resolve(strict=True)
        require(artifact_path.parent == root, f"{name}: calibration file escapes bundle directory")
        artifact = load_json(artifact_path)
        require(artifact.get("format") == "hpsi-activation-calibration", f"{name}: unsupported artifact format")
        artifact_version = artifact.get("version")
        require(isinstance(artifact_version, int) and not isinstance(artifact_version, bool)
                and artifact_version == 1, f"{name}: unsupported artifact version")
        require(artifact.get("layer_name") == name, f"{name}: layer name mismatch")
        require(artifact.get("statistic") == "diagonal_second_moment", f"{name}: unsupported statistic")
        artifact_module_type = artifact.get("module_type")
        if artifact_module_type is not None:
            require(isinstance(artifact_module_type, str) and artifact_module_type in ("linear", "conv1d"),
                    f"{name}: artifact module_type must be 'linear' or 'conv1d'")
            require(artifact_module_type == module_type,
                    f"{name}: artifact module_type does not match the manifest")
        artifact_model = artifact.get("model")
        require(isinstance(artifact_model, dict), f"{name}: model metadata must be an object")
        require(artifact_model.get("model_type") == model_type, f"{name}: model_type mismatch")
        require(artifact_model.get("revision") == revision, f"{name}: model revision mismatch")
        artifact_dataset = artifact.get("dataset")
        require(isinstance(artifact_dataset, dict), f"{name}: dataset metadata must be an object")
        require(artifact_dataset.get("sha256") == dataset_hash, f"{name}: dataset SHA-256 mismatch")
        artifact_sequences = artifact_dataset.get("processed_sequences")
        require(isinstance(artifact_sequences, int) and not isinstance(artifact_sequences, bool)
                and artifact_sequences == processed_sequences,
                f"{name}: processed sequence count mismatch")
        moments = artifact.get("activation_second_moment")
        shape = artifact.get("shape")
        count = artifact.get("sample_count")
        require(isinstance(moments, list) and moments, f"{name}: activation moments are empty")
        require(isinstance(shape, list) and len(shape) == 2
                and all(isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0
                        for dimension in shape), f"{name}: shape must contain two positive integer dimensions")
        require(shape[0] == count and shape[1] == len(moments), f"{name}: shape does not match sample count and feature count")
        require(isinstance(count, int) and not isinstance(count, bool) and count > 0, f"{name}: sample_count must be positive")
        entry_sample_count = entry.get("sample_count")
        require(isinstance(entry_sample_count, int) and not isinstance(entry_sample_count, bool)
                and entry_sample_count == count, f"{name}: manifest sample count mismatch")
        features = entry.get("features")
        require(isinstance(features, int) and not isinstance(features, bool) and features == len(moments),
                f"{name}: manifest feature count mismatch")
        for index, value in enumerate(moments):
            require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name}: moment {index} is not numeric")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            require(finite and value >= 0, f"{name}: moment {index} must be finite and non-negative")
        total_samples += count

    unresolved: list[str] = []
    if manifest_version == 2:
        unresolved = validate_identity_records(manifest.get("tensors"))
        declared_unresolved = manifest.get("architecture_unresolved")
        require(
            isinstance(declared_unresolved, list) and declared_unresolved == sorted(unresolved),
            "architecture_unresolved does not match the identities that are not resolved",
        )

    return {
        "valid": True,
        "bundle_version": manifest_version,
        "model_type": model_type,
        "model_revision": revision,
        "dataset_sha256": dataset_hash,
        "captured_layers": len(layers),
        "architecture_tensors": len(manifest.get("tensors") or []),
        "architecture_unresolved": unresolved,
        "total_layer_samples": total_samples,
        "total_features": sum(int(item["features"]) for item in layers),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_dir", type=Path)
    parser.add_argument("--dataset", type=Path, help="Optionally verify the original JSONL dataset SHA-256")
    args = parser.parse_args()
    try:
        report = validate_bundle(args.calibration_dir, args.dataset)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
