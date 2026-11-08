"""Functional parity check between a source model and its HPSI reconstruction.

This tool does NOT implement an inference runtime. It loads the original
checkpoint with PyTorch/HuggingFace ``transformers`` and rebuilds an identical
architecture from the weights recovered from an HPSI container, then compares
the two models' outputs on exactly the same input.

The consumer implemented here is deliberately *faithful to the container
metadata*: it reconstructs each weight using the orientation the container
declares. That is what a conforming reader is required to do, and it is
therefore the honest way to measure whether the persisted orientation is
correct. A container that declares the wrong orientation yields a transposed
weight and a broken model, and this tool measures that damage.

Scope caveat: the bundled fixture uses randomly initialised weights. Any
perplexity number produced here measures numerical drift only and is NOT
evidence of language-model quality.

Usage::

    python tools/eval_hpsi_parity.py \
        --model-dir .tmp-gpt2-conv1d-test/model \
        --hpsi .tmp-gpt2-conv1d-test/gpt2-calibrated.hpsi \
        --hpsi .tmp-gpt2-conv1d-test/gpt2-uncalibrated.hpsi \
        --out .tmp-gpt2-conv1d-test/audit/functional/parity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import architecture_oracle as oracle  # noqa: E402
import audit_hpsi_container as hpsi  # noqa: E402


def orientation_oracle_from_model(model: Any) -> oracle.ArchitectureOracle:
    """Read each rank-2 weight's role and layout off the live model definition.

    The oracle is keyed by parameter OBJECT identity, so a tied matrix is found
    under every name the model uses for it: ``transformer.wte.weight`` and
    ``lm_head.weight`` are one object reached as an embedding table and as the
    output projection, and a container that stored either name resolves. Names
    are taken from ``named_parameters`` (the registry that produced the
    checkpoint file), roles and layouts from the owning module classes. Nothing
    is inferred from a name pattern, a suffix, a rank or a matching dimension:
    a parameter no recognised class owns is recorded with role ``unknown``, and
    contradictory facts make the tensor unresolved rather than guessed.
    """
    from transformers.pytorch_utils import Conv1D

    classes = {
        "linear": torch.nn.Linear,
        "conv1d": Conv1D,
        "embedding": torch.nn.Embedding,
    }

    # Names by object identity: authoritative for what a checkpoint stores.
    names_by_identity: dict[int, list[str]] = {}
    rank2_identities: set[int] = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if getattr(parameter, "ndim", None) != 2:
            continue
        rank2_identities.add(id(parameter))
        names_by_identity.setdefault(id(parameter), []).append(name)

    facts: list[oracle.WeightFact] = []
    classified: set[int] = set()
    for module_name, module in model.named_modules():
        fact = oracle.classify_module_weight(
            f"{module_name}.weight" if module_name else "weight", module, classes
        )
        if fact is None:
            continue
        classified.add(fact.identity)
        for name in sorted(names_by_identity.get(fact.identity, [fact.name])):
            facts.append(oracle.WeightFact(
                name,
                fact.identity,
                fact.tensor_role,
                fact.weight_orientation,
                fact.index_axis,
            ))

    # A rank-2 parameter no recognised class owns stays in the oracle, with an
    # explicitly unknown role: absence of evidence must not look like absence of
    # the tensor, and it must not look like a projection either.
    for identity in sorted(rank2_identities - classified):
        for name in sorted(names_by_identity.get(identity, [])):
            facts.append(oracle.WeightFact(name, identity, "unknown"))

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return oracle.build_architecture_oracle(facts, model_type)


def source_layout(
    role: str | None,
    orientation: str | None,
    index_axis: int | None,
    source_shape: tuple[int, ...],
) -> tuple[tuple[int, int], bool]:
    """Delegate the layout decision to the torch-free oracle module, where it is
    unit-testable, and surface its refusal as a container-format error so the
    harness records a rejection instead of fabricating a reshaped weight."""
    try:
        return oracle.source_layout(role, orientation, index_axis, source_shape)
    except oracle.LayoutError as error:
        raise hpsi.HpsiFormatError(str(error)) from error


def build_state_dict_from_container(
    container: hpsi.Container,
    orientation_policy: str = "declared",
    architecture_oracle: oracle.ArchitectureOracle | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Rebuild every source tensor from the container.

    ``orientation_policy`` selects which orientation the consumer trusts:

    ``declared``
        Use the orientation persisted in the container metadata. This is what a
        conforming reader must do. It measures whether the artifact is
        self-consistent.
    ``architecture``
        Use the orientation the loaded model's own module classes require and
        ignore the metadata. This models a consumer that already knows the
        architecture (for example ``transformers`` loading a state dict). It
        measures the damage when the persisted orientation is wrong, and it is
        only available when a real model definition supplies the oracle.
    """
    if orientation_policy not in ("declared", "architecture"):
        raise ValueError(f"unknown orientation policy {orientation_policy!r}")
    if orientation_policy == "architecture" and architecture_oracle is None:
        raise ValueError(
            "the architecture policy needs an oracle read from the model definition; "
            "orientation is never inferred from a tensor name"
        )
    provenance = container.metadata["user"]["source_tensors"]
    state: dict[str, torch.Tensor] = {}
    transcript: list[dict[str, Any]] = []

    for entry in provenance:
        name = entry["name"]
        source_shape = tuple(int(dim) for dim in entry["source_shape"])
        declared = entry.get("weight_orientation")
        disposition = entry.get("disposition")

        if disposition == "preserve_raw":
            tensor = container.tensors_by_name[name]
            raw = container.read_tensor_payload(tensor)
            values = np.frombuffer(raw, dtype="<f4").reshape(source_shape)
            transcript.append({
                "tensor": name,
                "path": "raw",
                "source_shape": list(source_shape),
                "recovered_shape": list(values.shape),
                "crc32c": hpsi.crc32c(raw),
            })
        else:
            logical = entry["logical_shape"]
            rows, cols = int(logical[0]), int(logical[1])
            codebook_name = f"{name}.hpsi_vq.codebook"
            indices_name = f"{name}.hpsi_vq.indices"
            dense = hpsi.reconstruct_matrix(
                container,
                container.tensors_by_name[codebook_name],
                container.tensors_by_name[indices_name],
                rows,
                cols,
            )
            # Place the decoded logical matrix back into the source array layout.
            # Which evidence is trusted depends on the policy; either way it is
            # read from an explicit field and never inferred from a name.
            declared_role = entry.get("tensor_role")
            expectation = None
            if orientation_policy == "architecture":
                expectation = architecture_oracle.expectation(name)
                if expectation is None:
                    raise hpsi.HpsiFormatError(
                        f"{name}: the model definition does not describe this tensor under "
                        "any of its parameter names; refusing to guess a layout from its name"
                    )
                if declared_role is not None and declared_role not in expectation.tensor_roles:
                    raise hpsi.HpsiFormatError(
                        f"{name}: container declares role {declared_role!r} but the model "
                        f"definition evidences only {list(expectation.tensor_roles)}"
                    )
                trusted_orientation = expectation.weight_orientation
                trusted_role = declared_role or (
                    "embedding_table" if "embedding_table" in expectation.tensor_roles else None
                )
                trusted_index_axis = expectation.index_axis
            else:
                trusted_orientation = declared
                trusted_role = declared_role
                trusted_index_axis = entry.get("index_axis")
            expected_logical, transpose_to_source = source_layout(
                trusted_role, trusted_orientation, trusted_index_axis, source_shape
            )
            if (rows, cols) != expected_logical:
                raise hpsi.HpsiFormatError(
                    f"{name}: container declares logical shape {(rows, cols)} but the trusted "
                    f"evidence requires {expected_logical} for source shape "
                    f"{list(source_shape)}; refusing to reshape"
                )
            stored = dense.T if transpose_to_source else dense
            values = stored.reshape(source_shape)
            transcript.append({
                "tensor": name,
                "path": "vq",
                "source_shape": list(source_shape),
                "logical_shape": [rows, cols],
                "declared_orientation": declared,
                "declared_role": declared_role,
                "declared_index_axis": entry.get("index_axis"),
                "trusted_orientation": trusted_orientation,
                "trusted_role": trusted_role,
                "trusted_index_axis": trusted_index_axis,
                "layout_evidence": orientation_policy,
                "aliased_names": list(expectation.aliases) if expectation else [],
                "recovered_shape": list(values.shape),
            })

        if values.shape != source_shape:
            raise hpsi.HpsiFormatError(
                f"{name}: recovered shape {values.shape} != source shape {source_shape}"
            )
        state[name] = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))

    return state, transcript


def align_tied_embeddings(
    state: dict[str, torch.Tensor], model: Any
) -> dict[str, torch.Tensor]:
    """Satisfy tied parameters that the source file omits.

    GPT-2 with ``tie_word_embeddings`` shares one matrix between the input
    embedding and the output head, so the checkpoint stores only
    ``transformer.wte.weight``. Binding ``lm_head.weight`` to the same tensor
    object reproduces that sharing rather than duplicating the data.
    """
    expected = model.state_dict()
    for key in expected:
        if key in state:
            continue
        if key == "lm_head.weight" and "transformer.wte.weight" in state:
            state[key] = state["transformer.wte.weight"]
            continue
        raise hpsi.HpsiFormatError(f"cannot satisfy required parameter {key!r}")
    unexpected = [key for key in state if key not in expected]
    if unexpected:
        raise hpsi.HpsiFormatError(f"unexpected parameters: {sorted(unexpected)}")
    return state


def load_models(model_dir: Path) -> tuple[Any, Any]:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config.from_pretrained(str(model_dir))
    original = GPT2LMHeadModel(config)
    state = load_safetensors_state(model_dir / "model.safetensors")
    original.load_state_dict(align_tied_embeddings(state, original), strict=True)
    original.eval()
    return config, original


def load_safetensors_state(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path))


def compare_models(
    original: Any,
    candidate_state: dict[str, torch.Tensor],
    config: Any,
    input_ids: torch.Tensor,
) -> dict[str, Any]:
    from transformers import GPT2LMHeadModel

    candidate = GPT2LMHeadModel(config)
    incompatible = candidate.load_state_dict(
        align_tied_embeddings(dict(candidate_state), candidate), strict=True
    )
    candidate.eval()

    with torch.no_grad():
        base_outputs = original(input_ids=input_ids, output_hidden_states=True)
        cand_outputs = candidate(input_ids=input_ids, output_hidden_states=True)

    base_logits = base_outputs.logits.detach().numpy().astype(np.float64)
    cand_logits = cand_outputs.logits.detach().numpy().astype(np.float64)
    difference = cand_logits - base_logits
    base_norm = float(np.sqrt(np.sum(base_logits * base_logits)))

    base_argmax = base_logits.argmax(axis=-1)
    cand_argmax = cand_logits.argmax(axis=-1)

    hidden = []
    for layer_index, (base_hidden, cand_hidden) in enumerate(
        zip(base_outputs.hidden_states, cand_outputs.hidden_states)
    ):
        base_flat = base_hidden.detach().numpy().astype(np.float64).ravel()
        cand_flat = cand_hidden.detach().numpy().astype(np.float64).ravel()
        denominator = np.linalg.norm(base_flat) * np.linalg.norm(cand_flat)
        hidden.append({
            "layer": layer_index,
            "relative_l2": float(np.linalg.norm(cand_flat - base_flat) / np.linalg.norm(base_flat)),
            "cosine": float(np.dot(base_flat, cand_flat) / denominator) if denominator else float("nan"),
        })

    return {
        "logits": {
            "shape": list(base_logits.shape),
            "max_abs_difference": float(np.max(np.abs(difference))),
            "mean_abs_difference": float(np.mean(np.abs(difference))),
            "relative_l2": float(np.linalg.norm(difference) / base_norm) if base_norm else float("nan"),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "l2_reference": base_norm,
        },
        "argmax_agreement": float(np.mean(base_argmax == cand_argmax)),
        "perplexity": {
            "original": float(np.exp(cross_entropy(base_logits, input_ids))),
            "reconstructed": float(np.exp(cross_entropy(cand_logits, input_ids))),
            "note": (
                "Weights of this fixture are randomly initialised; perplexity here "
                "measures numerical drift only and is NOT a quality claim."
            ),
        },
        "hidden_states": hidden,
        "load_state_dict_missing_keys": list(incompatible.missing_keys),
        "load_state_dict_unexpected_keys": list(incompatible.unexpected_keys),
    }


def cross_entropy(logits: np.ndarray, input_ids: torch.Tensor) -> float:
    """Next-token cross entropy over the sequence, matching positions 0..n-2."""
    ids = input_ids.numpy().astype(np.int64)
    if ids.shape[1] < 2:
        return float("nan")
    logits = logits.reshape(-1, logits.shape[-1])
    targets = ids.reshape(-1)[1:]
    logits = logits[:-1]
    maximum = logits.max(axis=-1, keepdims=True)
    log_sum_exp = maximum + np.log(np.sum(np.exp(logits - maximum), axis=-1, keepdims=True))
    log_probs = logits - log_sum_exp
    picked = log_probs[np.arange(targets.size), targets]
    return float(-np.mean(picked))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HPSI functional parity evaluation")
    parser.add_argument("--model-dir", required=True, help="directory with config.json + model.safetensors")
    parser.add_argument("--hpsi", action="append", required=True, help="HPSI container (repeatable)")
    parser.add_argument("--out", required=True, help="JSON report path")
    parser.add_argument(
        "--tokens",
        default="1,5,3,7,2,9,4,6,3,5,7,1,9,2,4,6",
        help="comma separated deterministic input token ids",
    )
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    config, original = load_models(model_dir)
    architecture_oracle = orientation_oracle_from_model(original)

    vocab = int(getattr(config, "vocab_size"))
    ids = [int(part) for part in args.tokens.split(",") if part.strip()]
    if any(token < 0 or token >= vocab for token in ids):
        raise SystemExit(f"token ids must be within [0, {vocab})")
    input_ids = torch.tensor([ids], dtype=torch.long)

    report: dict[str, Any] = {
        "model_dir": str(model_dir),
        "architecture": config.architectures,
        "orientation_oracle": {
            "source": (
                "parameter object identity of the loaded model definition, classified by "
                "module class (nn.Linear / Conv1D / nn.Embedding)"
            ),
            **architecture_oracle.to_record(),
            "conv1d_weights": sorted(
                name
                for name, entry in architecture_oracle.entries.items()
                if entry.weight_orientation == "conv1d_in_out"
            ),
            "embedding_tables": sorted(
                name
                for name, entry in architecture_oracle.entries.items()
                if "embedding_table" in entry.tensor_roles
            ),
            "unknown_role_weights": sorted(architecture_oracle.unresolved),
        },
        "parameters": int(sum(p.numel() for p in original.parameters())),
        "vocab_size": vocab,
        "n_embd": int(getattr(config, "n_embd", 0)),
        "n_layer": int(getattr(config, "n_layer", 0)),
        "n_head": int(getattr(config, "n_head", 0)),
        "input_ids": ids,
        "note": (
            "Synthetic randomly-initialised GPT-2 fixture. Output parity measures "
            "numerical fidelity of the reconstruction, not model quality."
        ),
        "containers": [],
    }

    # Identity control: rebuild the candidate from the untouched checkpoint. If
    # this does not reproduce the original exactly, the harness itself is wrong
    # and every other number below is meaningless.
    identity_state = load_safetensors_state(model_dir / "model.safetensors")
    report["identity_control"] = compare_models(
        original, identity_state, config, input_ids
    )
    identity_relative = report["identity_control"]["logits"]["relative_l2"]
    if identity_relative > 1e-6:
        raise SystemExit(
            f"identity control failed: logits relative L2 = {identity_relative!r}, "
            "the parity harness is not trustworthy"
        )
    print(f"identity control: logits rel-L2 = {identity_relative:.3e} (must be ~0)")

    for container_path in args.hpsi:
        container = hpsi.open_container(container_path)
        for policy in ("declared", "architecture"):
            try:
                state, transcript = build_state_dict_from_container(
                    container, policy, architecture_oracle
                )
            except hpsi.HpsiFormatError as error:
                # An architecture-aware consumer refuses a self-inconsistent
                # artifact. That refusal is itself the measurement: record it
                # instead of fabricating numbers from a reshaped matrix.
                report["containers"].append({
                    "container": container_path,
                    "orientation_policy": policy,
                    "container_bytes": len(container.raw),
                    "rejected": True,
                    "rejection_reason": str(error),
                })
                print(f"{container_path} [{policy}]: REJECTED — {error}")
                continue
            comparison = compare_models(original, state, config, input_ids)
            report["containers"].append({
                "container": container_path,
                "orientation_policy": policy,
                "container_bytes": len(container.raw),
                "rejected": False,
                "tensors_recovered": len(state),
                "transcript": transcript,
                **comparison,
            })
            print(
                f"{container_path} [{policy}]: "
                f"logits rel-L2 = {comparison['logits']['relative_l2']:.6f}, "
                f"argmax agreement = {comparison['argmax_agreement']:.4f}, "
                f"ppl {comparison['perplexity']['original']:.4f} -> "
                f"{comparison['perplexity']['reconstructed']:.4f}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
