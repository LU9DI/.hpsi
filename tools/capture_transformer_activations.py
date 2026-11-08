
#!/usr/bin/env python3
"""Capture diagonal input-activation second moments from a local HF causal LM.

This utility deliberately runs on CPU and never downloads model/tokenizer files.
It measures projections (`torch.nn.Linear`, Hugging Face Transformers `Conv1D`)
and separately records the architecture of every module that declares a weight
layout, including `torch.nn.Embedding` tables, whose module input is a token
index rather than a feature vector and so has no activation moment to take.

Architecture is grouped by the identity of the parameter object, so one array
reached through two modules — a tied embedding and output head — becomes one
record carrying both aliases and both roles, while two arrays of equal values
stay two records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import operator
import re
from pathlib import Path
from typing import Any, Sequence


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def parse_jsonl_record(line: str, line_number: int) -> Any:
    try:
        return json.loads(
            line,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"line {line_number}: invalid strict JSON: {exc}"
        ) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:120] or "unnamed"


# The Safetensors dtype tokens, so a bundle names a parameter's type in the same
# vocabulary the source file uses instead of a torch-specific spelling.
SAFETENSORS_DTYPES = {
    "torch.float64": "F64",
    "torch.float32": "F32",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.int64": "I64",
    "torch.int32": "I32",
    "torch.int16": "I16",
    "torch.int8": "I8",
    "torch.uint8": "U8",
    "torch.bool": "BOOL",
}


def dtype_token(tensor: Any) -> str:
    """Name a parameter's dtype with the Safetensors token for it."""
    return SAFETENSORS_DTYPES.get(str(tensor.dtype), str(tensor.dtype))


def source_tensor_name(module_name: str) -> str:
    """Map a projection module path to its conventional Safetensors key."""
    return f"{module_name}.weight" if module_name else "weight"


def safetensors_tensor_names(model_dir: Path) -> set[str]:
    """Return physical tensor names from the model Safetensors files."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to inspect model tensor names"
        ) from exc

    model_dir = model_dir.resolve()
    index_path = model_dir / "model.safetensors.index.json"

    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid model.safetensors.index.json") from exc

        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Safetensors index has an invalid weight_map")

        shards: dict[str, set[str]] = {}

        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
                raise ValueError("Safetensors index contains invalid entries")

            shard_path = (model_dir / shard_name).resolve()

            if not shard_path.is_relative_to(model_dir):
                raise ValueError("Safetensors index references a path outside the model directory")

            if not shard_path.is_file():
                raise ValueError(f"Safetensors shard does not exist: {shard_name}")

            shard_key = str(shard_path)

            if shard_key not in shards:
                with safe_open(str(shard_path), framework="pt", device="cpu") as stream:
                    shards[shard_key] = set(stream.keys())

            if tensor_name not in shards[shard_key]:
                raise ValueError(
                    f"Safetensors index references a missing tensor: {tensor_name}"
                )

        return set(weight_map)

    primary = model_dir / "model.safetensors"

    if primary.is_file():
        files = [primary]
    else:
        files = sorted(model_dir.rglob("*.safetensors"))

    if not files:
        raise ValueError("no model Safetensors files found")

    names: set[str] = set()

    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            for name in stream.keys():
                if name in names:
                    raise ValueError(
                        f"duplicate Safetensors tensor name across files: {name}"
                    )
                names.add(name)

    return names


def parameter_names_by_identity(model: Any) -> dict[int, list[str]]:
    """Map each parameter identity to all names exposed by the model."""
    aliases: dict[int, list[str]] = {}

    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)

    return aliases


def resolve_source_tensor_name(
    module_name: str,
    module: Any,
    parameter_aliases: dict[int, list[str]],
    stored_tensor_names: set[str],
) -> str:
    """Resolve a module weight to exactly one physical Safetensors key."""
    conventional_name = source_tensor_name(module_name)

    if conventional_name in stored_tensor_names:
        return conventional_name

    weight = getattr(module, "weight", None)
    if weight is None:
        raise ValueError(
            f"projection has no weight parameter: {module_name}"
        )

    aliases = parameter_aliases.get(id(weight), [])
    candidates = sorted(set(aliases).intersection(stored_tensor_names))

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise ValueError(
            "no physical Safetensors tensor found for projection "
            f"{module_name!r}; expected {conventional_name!r}"
        )

    raise ValueError(
        "ambiguous physical Safetensors aliases for projection "
        f"{module_name!r}: {candidates}"
    )


def supported_projection(
    module: Any,
    linear_type: type,
    conv1d_type: type | None,
) -> tuple[str, int] | None:
    """Return (module_type, input_features) for supported projections."""
    if isinstance(module, linear_type):
        kind = "linear"
        features = getattr(module, "in_features", None)
    elif conv1d_type is not None and isinstance(module, conv1d_type):
        kind = "conv1d"
        weight = getattr(module, "weight", None)
        shape = getattr(weight, "shape", None)

        if shape is None or len(shape) != 2:
            raise ValueError(
                "Transformers Conv1D weight must be a rank-2 tensor"
            )

        # Hugging Face Conv1D stores weights as (in_features, out_features).
        features = shape[0]
    else:
        return None

    if isinstance(features, bool) or features is None:
        raise ValueError(
            f"{kind} projection has an invalid input feature dimension"
        )

    try:
        # Require an integer-like value; do not silently truncate floats.
        features = operator.index(features)
    except TypeError as exc:
        raise ValueError(
            f"{kind} projection has an invalid input feature dimension"
        ) from exc

    if features < 1:
        raise ValueError(
            f"{kind} projection has an invalid input feature dimension"
        )

    return kind, features


def positive_integer(value: Any) -> int | None:
    """Return `value` as an int when it is a positive integer, else None."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        count = operator.index(value)
    except TypeError:
        return None
    return count if count >= 1 else None


def supported_module_class(
    module: Any,
    linear_type: type,
    conv1d_type: type | None,
    embedding_type: type | None,
) -> tuple[str, dict[str, int]] | None:
    """Classify a module by its class and return the dimensions it declares.

    A *projection* is measured; an *embedding* is only identified, because its
    module input is a token index, not a feature vector, so there is no
    activation second moment to take. Both produce architecture facts, and the
    class is the only evidence used: a role is never read off a module name.
    """
    if isinstance(module, linear_type):
        return "linear", {
            "in_features": module.in_features,
            "out_features": module.out_features,
        }
    if conv1d_type is not None and isinstance(module, conv1d_type):
        # Transformers documents Conv1D as `Conv1D(nf, nx)` with `nf` the output
        # features, `nx` the input features and a `[nx, nf]` weight, so the names
        # are read from the class rather than assumed from a Linear-shaped API.
        return "conv1d", {
            "in_features": getattr(module, "nx", None),
            "out_features": getattr(module, "nf", None),
        }
    if embedding_type is not None and isinstance(module, embedding_type):
        return "embedding", {
            "num_embeddings": getattr(module, "num_embeddings", None),
            "embedding_dim": getattr(module, "embedding_dim", None),
        }
    return None


def architecture_view(
    module_name: str,
    module_class: str,
    evidence: dict[str, int],
    weight_shape: Sequence[int],
    weight_dtype: str,
) -> dict[str, Any]:
    """One module's claim about the parameter it holds, checked against it.

    Every class names the layout its weight must have. A class whose attribute
    dimensions disagree with the stored array is not a layout claim at all, so
    the disagreement is recorded as a conflict instead of being emitted as a
    role the model definition cannot support.
    """
    shape = [int(dimension) for dimension in weight_shape]
    view: dict[str, Any] = {
        "alias": source_tensor_name(module_name),
        "module": module_name,
        "module_class": module_class,
        "shape": shape,
        "dtype": weight_dtype,
        "conflicts": [],
    }

    dimensions = {key: positive_integer(value) for key, value in evidence.items()}
    for key, value in dimensions.items():
        if value is None:
            view["conflicts"].append(
                f"{key} is not a positive integer (found {evidence.get(key)!r})"
            )
    if view["conflicts"]:
        view["tensor_role"] = None
        return view

    if len(shape) != 2:
        view["conflicts"].append(
            f"weight is rank {len(shape)} with shape {shape}, but a "
            f"{module_class} module's weight is a matrix"
        )
        view["tensor_role"] = None
        return view

    if module_class == "linear":
        expected = [dimensions["out_features"], dimensions["in_features"]]
        view["tensor_role"] = "linear_projection"
        view["weight_orientation"] = "linear_out_in"
    elif module_class == "conv1d":
        expected = [dimensions["in_features"], dimensions["out_features"]]
        view["tensor_role"] = "conv1d_projection"
        view["weight_orientation"] = "conv1d_in_out"
    elif module_class == "embedding":
        expected = [dimensions["num_embeddings"], dimensions["embedding_dim"]]
        view["tensor_role"] = "embedding_table"
        view["index_axis"] = 0
    else:
        view["conflicts"].append(f"unsupported module class {module_class!r}")
        view["tensor_role"] = None
        return view

    if shape != expected:
        # The class was read from the module, the shape from the parameter: a
        # disagreement means this module is not the architecture the shape says,
        # so no role may be claimed from it.
        view["conflicts"].append(
            f"the {module_class} module declares a weight of {expected} but its "
            f"parameter is {shape}; refusing to assign {view['tensor_role']}"
        )
        view["tensor_role"] = None
        view.pop("weight_orientation", None)
        view.pop("index_axis", None)
    return view


def evidence_row(view: dict[str, Any]) -> dict[str, Any]:
    """The alias-to-role association for one module view, without physical data."""
    row: dict[str, Any] = {
        "alias": view["alias"],
        "module": view["module"],
        "module_class": view["module_class"],
        "tensor_role": view["tensor_role"],
    }
    if view.get("weight_orientation") is not None:
        row["weight_orientation"] = view["weight_orientation"]
    if view.get("index_axis") is not None:
        row["index_axis"] = view["index_axis"]
    if view["conflicts"]:
        row["conflicts"] = list(view["conflicts"])
    return row


def logical_tensor_records(views: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group module views into one record per *parameter identity*.

    Identity is the only grouping key: two parameters with the same values, the
    same shape, the same dtype or names that look related stay two records, and
    one parameter reached through several modules becomes one record with several
    roles. A record whose views disagree is emitted as a conflict that names the
    aliases involved and carries no role, so no downstream manifesto can declare
    a single interpretation of an unresolved tensor.
    """
    grouped: dict[int, list[dict[str, Any]]] = {}
    for view in views:
        grouped.setdefault(view["parameter_identity"], []).append(view)

    records: list[dict[str, Any]] = []
    for group in grouped.values():
        aliases = sorted({str(view["alias"]) for view in group})
        canonical = sorted({
            view["canonical_name"] for view in group if view["canonical_name"]
        })
        shapes = {tuple(view["shape"]) for view in group}
        dtypes = {str(view["dtype"]) for view in group}
        conflicts: list[str] = []
        for view in group:
            for reason in view["conflicts"]:
                conflicts.append(f"{view['alias']}: {reason}")

        if len(canonical) == 0:
            conflicts.append(
                f"no physical Safetensors key was found for the parameter reached "
                f"as {aliases}"
            )
        elif len(canonical) > 1:
            conflicts.append(
                f"one parameter is stored under several physical keys: {canonical}"
            )
        if len(shapes) > 1:
            conflicts.append(
                f"the modules holding this parameter disagree about its shape: "
                f"{sorted(list(shape) for shape in shapes)}"
            )
        if len(dtypes) > 1:
            conflicts.append(
                f"the modules holding this parameter disagree about its dtype: "
                f"{sorted(dtypes)}"
            )

        roles = sorted({
            view["tensor_role"]
            for view in group
            if view["tensor_role"] is not None
        })
        orientations = sorted({
            view["weight_orientation"]
            for view in group
            if view.get("weight_orientation") is not None
        })
        axes = sorted({
            view["index_axis"] for view in group if view.get("index_axis") is not None
        })
        if len(orientations) > 1:
            conflicts.append(
                f"the modules reaching this parameter claim incompatible layouts: "
                f"{orientations}"
            )
        if len(axes) > 1:
            conflicts.append(f"the modules reaching this parameter claim {axes}")
        if "embedding_table" in roles and not axes:
            conflicts.append(
                "an embedding_table role was claimed without an indexing axis"
            )
        if "embedding_table" not in roles and axes:
            conflicts.append(
                f"index_axis {axes} was claimed without an embedding_table role"
            )

        record = {
            "canonical_name": canonical[0] if len(canonical) == 1 else None,
            "aliases": aliases,
            "identity": "parameter-object-identity",
            "shape": sorted(list(shape) for shape in shapes)[0] if shapes else [],
            "dtype": sorted(dtypes)[0] if dtypes else None,
            # A conflicted record states no roles: the absence is the finding,
            # and the reasons say what would have to be settled.
            "tensor_roles": [] if conflicts else roles,
            "index_axis": axes[0] if (not conflicts and len(axes) == 1) else None,
            "weight_orientation": (
                orientations[0] if (not conflicts and len(orientations) == 1) else None
            ),
            "status": "conflict" if conflicts else "resolved",
            # The association the roles alone cannot express: which module, under
            # which name, with which class, evidenced which role.
            "evidence": [evidence_row(view) for view in sorted(
                group, key=lambda item: str(item["alias"])
            )],
            "conflicts": conflicts,
        }
        records.append(record)

    records.sort(key=lambda item: (item["canonical_name"] or "", item["aliases"][0]))
    return records


def architecture_views(
    model: Any,
    linear_type: type,
    conv1d_type: type | None,
    embedding_type: type | None,
    parameter_aliases: dict[int, list[str]],
    stored_tensor_names: set[str],
) -> list[dict[str, Any]]:
    """One view per module that declares a weight layout, projection or table.

    Resolution problems are recorded as conflicts instead of aborting: a table
    whose physical key cannot be established safely must not be guessed at, and
    the reason belongs in the bundle, where an operator can read it.
    """
    views: list[dict[str, Any]] = []

    for name, module in model.named_modules():
        classified = supported_module_class(
            module,
            linear_type,
            conv1d_type,
            embedding_type,
        )
        if classified is None:
            continue

        module_class, evidence = classified
        weight = getattr(module, "weight", None)
        reasons: list[str] = []

        if weight is None:
            views.append({
                **architecture_view(name, module_class, evidence, [], "unknown"),
                "parameter_identity": ("no-parameter", name),
                "canonical_name": None,
                "conflicts": [f"{name}: module exposes no `weight` parameter"],
            })
            continue

        canonical: str | None = None
        try:
            canonical = resolve_source_tensor_name(
                name,
                module,
                parameter_aliases,
                stored_tensor_names,
            )
        except ValueError as exc:
            reasons.append(str(exc))

        view = architecture_view(
            name,
            module_class,
            evidence,
            [int(dimension) for dimension in weight.shape],
            dtype_token(weight),
        )
        view["conflicts"].extend(reasons)
        view["parameter_identity"] = id(weight)
        view["canonical_name"] = canonical
        views.append(view)

    return views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Local Hugging Face model directory",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="UTF-8 JSONL file; each line is a string or an object with a text field",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-sequences", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--max-activation-rows",
        type=int,
        default=65536,
        help="Maximum token rows accumulated independently for each layer",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if (
        args.max_sequences < 1
        or args.max_length < 1
        or args.max_activation_rows < 1
    ):
        raise SystemExit("limits must be positive integers")

    if not args.model.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")

    if not args.dataset.is_file():
        raise SystemExit(f"dataset file does not exist: {args.dataset}")

    # Optional heavyweight dependencies are imported only when capture is requested.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.pytorch_utils import Conv1D as TransformersConv1D
    except ImportError as exc:
        raise SystemExit(
            "Install the optional capture dependencies: pip install torch transformers"
        ) from exc

    torch.set_num_threads(max(1, torch.get_num_threads()))

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model),
        local_files_only=True,
        trust_remote_code=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )

    model.to("cpu")
    model.eval()

    stored_tensor_names = safetensors_tensor_names(args.model)
    parameter_aliases = parameter_names_by_identity(model)

    # Architecture is taken from the module graph and the identity of the
    # parameters it holds, before any activation is measured. This pass also sees
    # embedding tables, which have no input activations to measure and therefore
    # never reach the hooks below.
    records = logical_tensor_records(
        architecture_views(
            model,
            torch.nn.Linear,
            TransformersConv1D,
            torch.nn.Embedding,
            parameter_aliases,
            stored_tensor_names,
        )
    )
    records_by_name = {
        record["canonical_name"]: record
        for record in records
        if record["canonical_name"] is not None
    }

    accumulators: dict[str, dict[str, Any]] = {}
    module_names: dict[str, str] = {}
    module_types: dict[str, str] = {}
    handles = []

    def make_hook(layer_name: str, expected_features: int):

        def capture(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return

            activation = inputs[0].detach()

            if activation.ndim < 2:
                return

            feature_count = int(activation.shape[-1])

            if feature_count != expected_features:
                raise RuntimeError(
                    f"activation feature dimension mismatch for {layer_name}: "
                    f"expected {expected_features}, got {feature_count}"
                )

            rows = activation.reshape(-1, feature_count)
            state = accumulators.get(layer_name)

            if state is None:
                state = {
                    "sum_squares": torch.zeros(
                        feature_count,
                        dtype=torch.float64,
                    ),
                    "sample_count": 0,
                }
                accumulators[layer_name] = state

            if state["sum_squares"].numel() != feature_count:
                raise RuntimeError(
                    f"feature dimension changed within layer {layer_name}"
                )

            remaining = (
                args.max_activation_rows - state["sample_count"]
            )

            if remaining <= 0:
                return

            selected = rows[:remaining].to(
                device="cpu",
                dtype=torch.float64,
            )

            state["sum_squares"] += torch.sum(
                selected * selected,
                dim=0,
            )

            state["sample_count"] += int(selected.shape[0])

        return capture

    # Register hooks for standard Linear and Hugging Face Conv1D projections.
    for name, module in model.named_modules():
        projection = supported_projection(
            module,
            torch.nn.Linear,
            TransformersConv1D,
        )

        if projection is None:
            continue

        module_type, input_features = projection

        try:
            tensor_name = resolve_source_tensor_name(
                name,
                module,
                parameter_aliases,
                stored_tensor_names,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        if tensor_name in module_names:
            raise SystemExit(
                "multiple supported modules map to the same "
                f"Safetensors key: {tensor_name}"
            )

        module_names[tensor_name] = name
        module_types[tensor_name] = module_type

        handles.append(
            module.register_forward_pre_hook(
                make_hook(tensor_name, input_features)
            )
        )

    if not handles:
        raise SystemExit(
            "no supported Linear or Transformers Conv1D modules found"
        )

    dataset_hash = sha256_file(args.dataset)
    processed_sequences = 0

    try:
        with args.dataset.open(
            "r",
            encoding="utf-8",
        ) as stream, torch.no_grad():

            for line_number, line in enumerate(stream, start=1):
                if processed_sequences >= args.max_sequences:
                    break

                if not line.strip():
                    continue

                record = parse_jsonl_record(line, line_number)

                text = (
                    record
                    if isinstance(record, str)
                    else (
                        record.get("text")
                        if isinstance(record, dict)
                        else None
                    )
                )

                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"line {line_number}: expected a non-empty string "
                        "or object.text"
                    )

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

    for index, (layer_name, state) in enumerate(
        sorted(accumulators.items())
    ):
        count = state["sample_count"]

        if count == 0:
            continue

        moments = (
            state["sum_squares"] / count
        ).tolist()

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
            "module_type": module_types[layer_name],
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

        destination.write_text(
            json.dumps(artifact, indent=2) + "\n",
            encoding="utf-8",
        )

        manifest_layers.append({
            "module_name": module_names[layer_name],
            "module_type": module_types[layer_name],
            "source_tensor_name": layer_name,
            "file": filename,
            "sample_count": count,
            "features": len(moments),
            # The statistics row stays exactly as version 1 wrote it; the alias
            # set is added alongside so a reader can link the measurement to the
            # identity record without walking the module graph again.
            "aliases": records_by_name[layer_name]["aliases"]
            if layer_name in records_by_name
            else [layer_name],
        })

    unresolved = sorted(
        record["aliases"][0]
        for record in records
        if record["status"] != "resolved"
    )

    manifest = {
        "format": "hpsi-transformer-calibration-manifest",
        # Version 2 adds the identity records. Version 1 bundles stay readable:
        # the new key is optional, and nothing an older reader relied on moved.
        "version": 2,
        "model_type": model_type,
        "model_revision": model_revision,
        "dataset_sha256": dataset_hash,
        "processed_sequences": processed_sequences,
        "captured_layers": len(manifest_layers),
        "layers": manifest_layers,
        # One entry per parameter, grouped on the parameter object itself: never
        # on equal values, shape, dtype or a name that looks related.
        "tensors": records,
        "architecture_tensors": len(records),
        "architecture_unresolved": unresolved,
    }

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "processed_sequences": processed_sequences,
        "captured_layers": len(manifest_layers),
        "architecture_tensors": len(records),
        "architecture_unresolved": unresolved,
        "dataset_sha256": dataset_hash,
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
