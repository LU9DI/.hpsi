#!/usr/bin/env python3
"""Validate a local HPSI transformer-calibration bundle without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name}: cannot read valid UTF-8 JSON") from exc
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


def validate_bundle(directory: Path, dataset: Path | None = None) -> dict[str, Any]:
    root = directory.resolve(strict=True)
    require(root.is_dir(), "calibration path must be a directory")
    manifest = load_json(root / "manifest.json")
    require(manifest.get("format") == "hpsi-transformer-calibration-manifest", "unsupported manifest format")
    manifest_version = manifest.get("version")
    require(isinstance(manifest_version, int) and not isinstance(manifest_version, bool)
            and manifest_version == 1, "unsupported manifest version")
    model_type = manifest.get("model_type")
    revision = manifest.get("model_revision")
    require(isinstance(model_type, str) and model_type.strip(), "manifest model_type is missing")
    require(revision is None or isinstance(revision, str), "manifest model_revision must be a string or null")
    dataset_hash = manifest.get("dataset_sha256")
    require(isinstance(dataset_hash, str) and len(dataset_hash) == 64, "manifest dataset_sha256 is invalid")
    require(all(c in "0123456789abcdef" for c in dataset_hash.lower()), "manifest dataset_sha256 is not hexadecimal")
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
        artifact_path = (root / filename).resolve(strict=True)
        require(artifact_path.parent == root, f"{name}: calibration file escapes bundle directory")
        artifact = load_json(artifact_path)
        require(artifact.get("format") == "hpsi-activation-calibration", f"{name}: unsupported artifact format")
        artifact_version = artifact.get("version")
        require(isinstance(artifact_version, int) and not isinstance(artifact_version, bool)
                and artifact_version == 1, f"{name}: unsupported artifact version")
        require(artifact.get("layer_name") == name, f"{name}: layer name mismatch")
        require(artifact.get("statistic") == "diagonal_second_moment", f"{name}: unsupported statistic")
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

    return {
        "valid": True,
        "model_type": model_type,
        "model_revision": revision,
        "dataset_sha256": dataset_hash,
        "captured_layers": len(layers),
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
