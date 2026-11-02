#!/usr/bin/env python3
"""Capture diagonal input-activation second moments from a local HF causal LM.

This utility deliberately runs on CPU and never downloads model/tokenizer files.
It emits one HPSI-compatible calibration JSON document per torch.nn.Linear input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:120] or "unnamed"


def source_tensor_name(module_name: str) -> str:
    """Map a Linear module path to its conventional Safetensors weight key."""
    return f"{module_name}.weight" if module_name else "weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Local Hugging Face model directory")
    parser.add_argument("--dataset", required=True, type=Path, help="UTF-8 JSONL file; each line is a string or an object with a text field")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-sequences", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-activation-rows", type=int, default=65536,
                        help="Maximum token rows accumulated independently for each layer")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_sequences < 1 or args.max_length < 1 or args.max_activation_rows < 1:
        raise SystemExit("limits must be positive integers")
    if not args.model.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")
    if not args.dataset.is_file():
        raise SystemExit(f"dataset file does not exist: {args.dataset}")

    # Optional heavyweight dependencies are imported only when capture is requested.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Install the optional capture dependencies: pip install torch transformers"
        ) from exc

    torch.set_num_threads(max(1, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()

    accumulators: dict[str, dict[str, Any]] = {}
    module_names: dict[str, str] = {}
    handles = []

    def make_hook(module_name: str):
        layer_name = source_tensor_name(module_name)
        def capture(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            activation = inputs[0].detach()
            if activation.ndim < 2:
                return
            feature_count = int(activation.shape[-1])
            rows = activation.reshape(-1, feature_count)
            state = accumulators.get(layer_name)
            if state is None:
                state = {
                    "sum_squares": torch.zeros(feature_count, dtype=torch.float64),
                    "sample_count": 0,
                }
                accumulators[layer_name] = state
            if state["sum_squares"].numel() != feature_count:
                raise RuntimeError(f"feature dimension changed within layer {layer_name}")
            remaining = args.max_activation_rows - state["sample_count"]
            if remaining <= 0:
                return
            selected = rows[:remaining].to(device="cpu", dtype=torch.float64)
            state["sum_squares"] += torch.sum(selected * selected, dim=0)
            state["sample_count"] += int(selected.shape[0])
        return capture

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            tensor_name = source_tensor_name(name)
            if tensor_name in module_names:
                raise SystemExit(
                    f"multiple Linear modules map to the same Safetensors key: {tensor_name}"
                )
            module_names[tensor_name] = name
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    if not handles:
        raise SystemExit("no torch.nn.Linear modules found; this first exporter supports Linear layers only")

    dataset_hash = sha256_file(args.dataset)
    processed_sequences = 0
    try:
        with args.dataset.open("r", encoding="utf-8") as stream, torch.no_grad():
            for line_number, line in enumerate(stream, start=1):
                if processed_sequences >= args.max_sequences:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record if isinstance(record, str) else record.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"line {line_number}: expected a non-empty string or object.text")
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                    padding=False,
                )
                if encoded["input_ids"].numel() == 0:
                    continue
                model(
                    input_ids=encoded["input_ids"].to("cpu"),
                    attention_mask=encoded.get("attention_mask", None),
                    use_cache=False,
                )
                processed_sequences += 1
    finally:
        for handle in handles:
            handle.remove()

    if processed_sequences == 0:
        raise SystemExit("dataset contained no usable text sequences")
    if not accumulators:
        raise SystemExit("no activation inputs were captured")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "unknown")
    model_revision = getattr(config, "_commit_hash", None)
    manifest_layers = []
    for index, (layer_name, state) in enumerate(sorted(accumulators.items())):
        count = state["sample_count"]
        if count == 0:
            continue
        moments = (state["sum_squares"] / count).tolist()
        artifact = {
            "format": "hpsi-activation-calibration",
            "version": 1,
            "source": "transformers-forward-pre-hook",
            "model": {
                "local_path": str(args.model.resolve()),
                "model_type": model_type,
                "revision": model_revision,
            },
            "dataset": {
                "path": str(args.dataset.resolve()),
                "sha256": dataset_hash,
                "processed_sequences": processed_sequences,
            },
            "layer_name": layer_name,
            "shape": [count, len(moments)],
            "sample_count": count,
            "statistic": "diagonal_second_moment",
            "activation_second_moment": moments,
            "capture": {
                "device": "cpu",
                "dtype": "float64_accumulator",
                "max_length": args.max_length,
                "max_activation_rows_per_layer": args.max_activation_rows,
            },
        }
        filename = f"{index:04d}_{safe_name(layer_name)}.json"
        destination = args.output_dir / filename
        destination.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        manifest_layers.append({
            "module_name": module_names[layer_name],
            "source_tensor_name": layer_name,
            "file": filename,
            "sample_count": count,
            "features": len(moments),
        })

    manifest = {
        "format": "hpsi-transformer-calibration-manifest",
        "version": 1,
        "model_type": model_type,
        "model_revision": model_revision,
        "dataset_sha256": dataset_hash,
        "processed_sequences": processed_sequences,
        "captured_layers": len(manifest_layers),
        "layers": manifest_layers,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "processed_sequences": processed_sequences,
        "captured_layers": len(manifest_layers),
        "dataset_sha256": dataset_hash,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
