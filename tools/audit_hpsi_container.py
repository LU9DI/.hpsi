"""Independent HPSI v0.1 reader and numeric reconstruction auditor.

This module is a second, from-scratch implementation of the HPSI container
format. It deliberately shares no code with the Rust crates: the point is to
detect a defect that one implementation could hide from the other.

The semantics implemented here were derived by reading the format
specification in ``docs/format-v0.1.md`` and the Rust sources
(``crates/hpsi-format/src/{header,tensor,block,integrity,reader}.rs`` and
``crates/hpsi-ops/src/vq.rs``). Where the Rust code defines an ordering, this
module reproduces it exactly and documents it.

Usage::

    python tools/audit_hpsi_container.py \
        --source .tmp-gpt2-conv1d-test/model/model.safetensors \
        --hpsi   .tmp-gpt2-conv1d-test/gpt2-calibrated.hpsi \
        --out    .tmp-gpt2-conv1d-test/audit/python/calibrated.json \
        --md     .tmp-gpt2-conv1d-test/audit/python/calibrated.md \
        --label  calibrated \
        --architecture .tmp-gpt2-conv1d-test/architecture.json

``--architecture`` is an *external* model-definition manifest (see
``docs/orientation-policy.md``). Without it the auditor performs no
architectural claim: an orientation is then reported as declared-but-unverified,
or as unverifiable for square matrices. Orientation is never inferred from a
tensor name, from a shape coincidence or from the container being audited.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

HEADER_SIZE = 128
TENSOR_ENTRY_SIZE = 128
BLOCK_ENTRY_SIZE = 64
INTEGRITY_ENTRY_SIZE = 32
NONE_U64 = (1 << 64) - 1
NONE_U32 = (1 << 32) - 1

# DType identifiers, from crates/hpsi-format/src/tensor.rs.
DTYPE_NAMES = {
    1: "U8",
    2: "I8",
    3: "F16",
    4: "BF16",
    5: "F32",
    6: "F64",
    7: "I16",
    8: "I32",
    9: "I64",
    10: "PackedU4",
    11: "PackedI4",
    12: "PackedU2",
    13: "PackedI2",
    14: "PackedU1",
    15: "PackedI1",
    16: "Opaque",
}

GROUP_SIZE = 8
CODEBOOK_SIZE = 16
CODEBOOK_VALUES = GROUP_SIZE * CODEBOOK_SIZE

ARCHITECTURE_MANIFEST_FORMAT = "hpsi-architecture-manifest"
ARCHITECTURE_MANIFEST_VERSION = 1
ORIENTATIONS = ("linear_out_in", "conv1d_in_out")

# Architectural roles, as a second dimension to the layout tokens above.
TENSOR_ROLES = (
    "linear_projection",
    "conv1d_projection",
    "embedding_table",
    "other",
    "unknown",
)

# The layout a projection role requires. Roles absent from this mapping state no
# layout claim of their own.
ROLE_ORIENTATIONS = {
    "linear_projection": "linear_out_in",
    "conv1d_projection": "conv1d_in_out",
}


class HpsiFormatError(ValueError):
    """Raised when a container violates the documented v0.1 structure."""


def entailed_orientation(
    roles: tuple[str, ...], index_axis: int | None, stated: str | None
) -> str | None:
    """The layout a set of roles implies, or ``None`` when they imply none.

    A model definition may state only the class a module belongs to; that class
    determines the layout. A container cannot do this: it must record the
    layout it actually encoded. Keeping the two cases apart is what stops a
    role from laundering an unverified orientation into a confirmed one.
    """
    if stated is not None:
        return stated
    for role in roles:
        if role == "linear_projection":
            return "linear_out_in"
        if role == "conv1d_projection":
            return "conv1d_in_out"
        if role == "embedding_table":
            if index_axis == 0:
                return "linear_out_in"
            if index_axis == 1:
                return "conv1d_in_out"
    return None


def orientation_logical_shape(orientation: str, source_shape: Sequence[int]) -> tuple[int, int]:
    """The logical `[out, in]` shape one orientation token arranges."""
    rows, cols = int(source_shape[0]), int(source_shape[1])
    return (rows, cols) if orientation == "linear_out_in" else (cols, rows)


def selector_payload_bytes(rows: int, cols: int, codebook_count: int) -> int | None:
    """Packed selector size a `[rows, cols]` VQ pair must store, or ``None``.

    Derived here from the format's own numbers: the grouped axis must divide by
    the group size, the per-row selector count must be even to stay byte aligned,
    and the result counts *bytes*, not selectors.
    """
    groups, remainder = divmod(cols, GROUP_SIZE)
    if remainder or groups == 0:
        return None
    selectors = groups * codebook_count
    if selectors % 2:
        return None
    return rows * (selectors // 2)


def entry_layout_violation(
    *,
    source_shape: Sequence[int],
    weight_orientation: str | None = None,
    logical_shape: Sequence[int] | None = None,
    tensor_role: str | None = None,
    index_axis: int | None = None,
    context: str = "source",
    quantized: bool = False,
    codebook_count: int = 0,
    stored_index_bytes: int = 0,
    calibrated: bool = False,
) -> tuple[str, str] | None:
    """The layout matrix, implemented a second time on purpose.

    This is the Python auditor's own reading of the same contract the Rust
    library enforces. It shares no code with it and never calls the Rust CLI, yet
    it returns the *same named code* for the same input, so the two implementations
    can be diffed case by case against
    ``tests/fixtures/embedding_layout_cases.json``. Rules are applied in the same
    order as ``validate_entry_layout`` in ``crates/hpsi-ops/src/orientation.rs``,
    because the order decides which reason a doubly-broken entry reports and that
    is part of the contract.

    Nothing here infers a role, an orientation or a shape from a name, a rank or
    a dimension that happens to fit.
    """
    rank = len(source_shape)
    matrix = rank == 2
    source_entry = context == "source"

    if rank and not matrix:
        if logical_shape is not None:
            return (
                "logical_shape_on_non_matrix",
                f"logical_shape is only derivable for rank-2 tensors, found rank {rank}",
            )
        if index_axis is not None:
            return (
                "axis_for_non_indexed_role",
                "index_axis is only meaningful for embedding_table over a rank-2 tensor, "
                f"found rank {rank}",
            )
        if tensor_role is not None and tensor_role not in ("other", "unknown"):
            return (
                "matrix_role_on_non_matrix",
                f"role {tensor_role!r} describes a matrix layout; only 'other' or 'unknown' "
                f"can describe a rank-{rank} tensor",
            )
        return None

    if tensor_role is not None and tensor_role not in TENSOR_ROLES:
        return ("role_token_unknown", f"unknown tensor_role {tensor_role!r}")

    # v0.1 still allows an entry to state nothing about a layout at all: that is a
    # pre-contract artifact, reported as `legacy_undeclared` and read as it always
    # was. What it may not do is declare *half* a layout.
    if source_entry and weight_orientation is None and (
        logical_shape is not None or index_axis is not None or tensor_role == "embedding_table"
    ):
        return (
            "orientation_required",
            "a rank-2 source entry declares logical_shape, index_axis or embedding_table "
            "without weight_orientation: v0.1 requires the grouping token whenever any part "
            "of a layout is stated, for any role. The token is a fact about the encoding, "
            "never evidence that the tensor is a projection",
        )

    if source_entry and weight_orientation is not None and logical_shape is None:
        return (
            "logical_shape_required",
            "logical_shape is required alongside weight_orientation: the declared layout "
            "has to state the shape it arranges the source array into",
        )

    if tensor_role is not None:
        required = ROLE_ORIENTATIONS.get(tensor_role)
        if required is not None:
            if weight_orientation is None:
                return (
                    "role_without_orientation",
                    f"role {tensor_role!r} requires weight_orientation {required!r}",
                )
            if weight_orientation != required:
                return (
                    "role_orientation_mismatch",
                    f"role {tensor_role!r} requires weight_orientation {required!r}, "
                    f"found {weight_orientation!r}",
                )
        if tensor_role != "embedding_table" and index_axis is not None:
            return (
                "axis_for_non_indexed_role",
                f"index_axis is only meaningful for embedding_table, not {tensor_role!r}",
            )
    elif index_axis is not None:
        return ("axis_without_role", "index_axis is only meaningful alongside a tensor_role")

    if tensor_role == "embedding_table":
        if calibrated:
            return (
                "table_must_not_be_calibrated",
                "an embedding table is indexed, so it has no input activations; "
                "diagonal_second_moment does not describe it and must not be attached to it",
            )
        if index_axis is None:
            return ("table_without_axis", "role embedding_table requires index_axis 0 or 1")
        if index_axis not in (0, 1):
            return ("axis_value_unsupported", f"index_axis must be 0 or 1, found {index_axis}")
        if index_axis == 1:
            return (
                "axis_one_unsupported",
                "an embedding table indexed on axis 1 groups its features along the source "
                "rows, which v0.1 can only record as conv1d_in_out with calibration "
                "provenance; and an indexed table must not carry calibration. No producer "
                "path can write this combination, so it is rejected rather than guessed",
            )
        if weight_orientation is not None and weight_orientation != "linear_out_in":
            return (
                "table_orientation_mismatch",
                "an embedding table indexed on axis 0 keeps features on axis 1, so the "
                "grouped axis is the source columns and the recorded orientation must be "
                f"'linear_out_in', found {weight_orientation!r}",
            )
        if matrix and logical_shape is not None:
            arranged = (int(source_shape[0]), int(source_shape[1]))
            if tuple(int(v) for v in logical_shape) != arranged:
                return (
                    "logical_shape_conflicts_axis",
                    f"logical_shape {[int(v) for v in logical_shape]} is not the source array "
                    f"{list(arranged)} with index axis 0 first",
                )
        if quantized and logical_shape is not None:
            rows, cols = int(logical_shape[0]), int(logical_shape[1])
            expected = selector_payload_bytes(rows, cols, codebook_count)
            if expected is None:
                return (
                    "selector_layout_unencodable",
                    f"logical shape {[rows, cols]} is not encodable at group size "
                    f"{GROUP_SIZE} with {codebook_count} codebook(s)",
                )
            if expected != stored_index_bytes:
                return (
                    "selector_bytes_mismatch",
                    f"an indexed table with logical shape {[rows, cols]} stores {expected} "
                    f"selector bytes, the container stores {stored_index_bytes}",
                )
        return None

    if source_entry and weight_orientation is not None and logical_shape is not None and matrix:
        expected = orientation_logical_shape(weight_orientation, source_shape)
        if tuple(int(v) for v in logical_shape) != expected:
            return (
                "logical_shape_conflicts_orientation",
                f"logical_shape {[int(v) for v in logical_shape]} contradicts "
                f"{weight_orientation!r} over source_shape {list(expected)}",
            )
    return None


def role_layout_problem(
    role: str | None, orientation: str | None, index_axis: int | None
) -> str | None:
    """The metadata-only slice of the matrix, as a code-prefixed reason string.

    Kept because a manifest entry and the contract tests reason about this slice
    alone; :func:`entry_layout_violation` is the full table, which additionally
    weighs `logical_shape` against `index_axis` and the stored selector payload.
    """
    violation = entry_layout_violation(
        source_shape=[],
        weight_orientation=orientation,
        tensor_role=role,
        index_axis=index_axis,
        context="manifest",
    )
    if violation is None:
        return None
    code, detail = violation
    return f"{code}: {detail}"


def role_evidence_expectation(name: str, row: Any) -> dict[str, Any]:
    """Validate one `role_evidence` row, written independently of the Rust parser.

    The row is the association a role list alone cannot express: which module,
    under which alias, with which class, evidenced which role. Each class names
    exactly one layout, so a row pairing a class with a layout that class does
    not have is refused here exactly as the Rust manifest parser refuses it.
    """
    if not isinstance(row, dict):
        raise HpsiFormatError(f"{name}: manifest `role_evidence` entries must be objects")

    def text(key: str) -> str:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise HpsiFormatError(
                f"{name}: manifest role_evidence lacks a non-empty `{key}`"
            )
        return value

    alias = text("alias")
    module = text("module")
    module_class = text("module_class")
    role_token = text("tensor_role")
    if role_token not in TENSOR_ROLES:
        raise HpsiFormatError(
            f"{name}: role_token_unknown: unknown tensor_role {role_token!r}"
        )

    stated = row.get("weight_orientation")
    if stated is not None:
        if not isinstance(stated, str):
            raise HpsiFormatError(
                f"{name}: manifest role_evidence `weight_orientation` must be a string"
            )
        if stated not in ORIENTATIONS:
            raise HpsiFormatError(f"{name}: unknown role_evidence orientation {stated!r}")

    axis = row.get("index_axis")
    if axis is not None and (isinstance(axis, bool) or axis not in (0, 1)):
        raise HpsiFormatError(
            f"{name}: manifest role_evidence `index_axis` must be 0 or 1"
        )

    required = {
        "linear_projection": "linear_out_in",
        "conv1d_projection": "conv1d_in_out",
        "embedding_table": {0: "linear_out_in", 1: "conv1d_in_out"}.get(axis),
    }.get(role_token)
    if stated is not None and required != stated:
        raise HpsiFormatError(
            f"{name}: role_evidence for alias {alias!r} pairs role {role_token} with "
            f"orientation {stated!r}, which that role does not imply"
        )
    if axis is not None and role_token != "embedding_table":
        raise HpsiFormatError(
            f"{name}: role_evidence `index_axis` is only meaningful for embedding_table, "
            f"not {role_token}"
        )
    if role_token == "embedding_table" and axis is None:
        raise HpsiFormatError(
            f"{name}: role_evidence for alias {alias!r} names embedding_table "
            "without an index_axis"
        )

    return {
        "alias": alias,
        "module": module,
        "module_class": module_class,
        "tensor_role": role_token,
        "weight_orientation": stated,
        "index_axis": axis,
    }


class ArchitectureOracle:
    """What the model definition, from outside the container, requires.

    ``nn.Linear`` stores ``[out_features, in_features]`` and
    ``transformers.pytorch_utils.Conv1D`` stores ``[in_features, out_features]``.
    That fact belongs to the model, so it is supplied as a manifest produced
    from the model definition and parsed here independently of the Rust code
    under audit: sharing the Rust decision function would make the cross-check
    circular.

    The manifest is authoritative only for the exact names it lists. Nothing is
    guessed from a name pattern, a matching dimension, or the orientation the
    importer happened to record — each of those would let a wrong transposition
    validate itself.
    """

    def __init__(
        self,
        expectations: dict[str, dict[str, Any]],
        model_type: str | None,
        path: Path,
    ) -> None:
        self.expectations = expectations
        self.model_type = model_type
        self.path = path

    @classmethod
    def load(cls, path: str | Path) -> "ArchitectureOracle":
        file_path = Path(path)
        try:
            root = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise HpsiFormatError(f"architecture manifest is not valid JSON: {error}") from error
        if not isinstance(root, dict):
            raise HpsiFormatError("architecture manifest must be a JSON object")
        if root.get("format") != ARCHITECTURE_MANIFEST_FORMAT:
            raise HpsiFormatError(
                f"unsupported architecture manifest format {root.get('format')!r}"
            )
        if root.get("version") != ARCHITECTURE_MANIFEST_VERSION:
            raise HpsiFormatError(
                f"unsupported architecture manifest version {root.get('version')!r}"
            )
        model_type = root.get("model_type")
        if model_type is not None and not isinstance(model_type, str):
            raise HpsiFormatError("architecture manifest `model_type` must be a string")
        entries = root.get("tensors")
        if not isinstance(entries, list):
            raise HpsiFormatError("architecture manifest lacks a `tensors` array")
        expectations: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise HpsiFormatError("architecture manifest tensor entry must be an object")
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                raise HpsiFormatError("architecture manifest entry lacks a name")
            if name in expectations:
                raise HpsiFormatError(f"{name}: architecture manifest declares it twice")
            token = entry.get("weight_orientation")
            if token is not None and token not in ORIENTATIONS:
                raise HpsiFormatError(
                    f"{name}: manifest `weight_orientation` must be one of {ORIENTATIONS}"
                )
            raw_roles = entry.get("tensor_role")
            if raw_roles is None:
                roles: tuple[str, ...] = ()
            elif isinstance(raw_roles, str):
                roles = (raw_roles,)
            elif isinstance(raw_roles, list) and all(
                isinstance(item, str) for item in raw_roles
            ):
                roles = tuple(raw_roles)
            else:
                raise HpsiFormatError(
                    f"{name}: manifest `tensor_role` must be a string or an array of strings"
                )
            roles = tuple(sorted(set(roles)))
            index_axis = entry.get("index_axis")
            if index_axis is not None and index_axis not in (0, 1):
                raise HpsiFormatError(f"{name}: manifest `index_axis` must be 0 or 1")
            if token is None and not roles:
                raise HpsiFormatError(
                    f"{name}: manifest entry states neither `weight_orientation` nor "
                    "`tensor_role`; orientation is never inferred from a name and neither is a role"
                )
            effective = entailed_orientation(roles, index_axis, token)
            for role in roles:
                problem = role_layout_problem(
                    role,
                    effective,
                    index_axis if role == "embedding_table" else None,
                )
                if problem is not None:
                    raise HpsiFormatError(f"{name}: {problem}")
            shape = entry.get("source_shape")
            if shape is not None:
                if not (
                    isinstance(shape, list)
                    and len(shape) == 2
                    and all(
                        isinstance(value, int) and not isinstance(value, bool) and value > 0
                        for value in shape
                    )
                ):
                    raise HpsiFormatError(
                        f"{name}: manifest `source_shape` must be two positive integers"
                    )
                shape = (shape[0], shape[1])
            raw_aliases = entry.get("aliases")
            aliases: list[str] = []
            if raw_aliases is not None:
                if not (
                    isinstance(raw_aliases, list)
                    and all(isinstance(alias, str) and alias.strip() for alias in raw_aliases)
                ):
                    raise HpsiFormatError(
                        f"{name}: manifest `aliases` must be an array of non-empty strings"
                    )
                aliases = sorted(set(raw_aliases))
            raw_evidence = entry.get("role_evidence")
            evidence: list[dict[str, Any]] = []
            if raw_evidence is not None:
                if not isinstance(raw_evidence, list):
                    raise HpsiFormatError(f"{name}: manifest `role_evidence` must be an array")
                for row in raw_evidence:
                    parsed = role_evidence_expectation(name, row)
                    if parsed["alias"] in {item["alias"] for item in evidence}:
                        raise HpsiFormatError(
                            f"{name}: alias {parsed['alias']!r} appears twice in `role_evidence`"
                        )
                    evidence.append(parsed)
                evidence.sort(key=lambda row: row["alias"])
                # The pairing and the role list are two views of one claim, so
                # they must say the same thing.
                for row in evidence:
                    if row["tensor_role"] not in roles:
                        raise HpsiFormatError(
                            f"{name}: role_evidence names {row['tensor_role']} for alias "
                            f"{row['alias']!r}, which `tensor_role` does not declare"
                        )
                    if aliases and row["alias"] not in aliases:
                        raise HpsiFormatError(
                            f"{name}: role_evidence alias {row['alias']!r} is not among `aliases`"
                        )
                for role in roles:
                    if role not in {row["tensor_role"] for row in evidence}:
                        raise HpsiFormatError(
                            f"{name}: `tensor_role` names {role} with no role_evidence naming "
                            "the module that evidences it"
                        )
                # A row may be less specific than the entry, never different from
                # it: an axis or a layout disagreeing with the entry's would be a
                # second, unresolved claim about the same parameter.
                for row in evidence:
                    if row["index_axis"] is not None and row["index_axis"] != index_axis:
                        raise HpsiFormatError(
                            f"{name}: role_evidence for alias {row['alias']!r} claims "
                            f"index_axis {row['index_axis']!r} while the entry declares "
                            f"{index_axis!r}"
                        )
                    if row["weight_orientation"] is not None and row["weight_orientation"] != token:
                        raise HpsiFormatError(
                            f"{name}: role_evidence for alias {row['alias']!r} declares "
                            f"orientation {row['weight_orientation']!r} while the entry "
                            f"declares {token!r}"
                        )
            if not aliases:
                # When the entry omits `aliases`, the pairing is authoritative:
                # the names it states are exactly the names its evidence uses.
                aliases = sorted({row["alias"] for row in evidence})
            # An alias names one parameter, so two entries claiming it claim the
            # same array twice: the manifest-side counterpart of grouping by
            # identity, where nothing may merge or split parameters by name.
            for alias in aliases:
                for other_name, other in expectations.items():
                    if alias in other["aliases"]:
                        raise HpsiFormatError(
                            f"{name}: alias {alias!r} is declared twice, by {name!r} and "
                            f"{other_name!r}; one name belongs to one parameter"
                        )
            expectations[name] = {
                "weight_orientation": token,
                "source_shape": shape,
                "tensor_roles": roles,
                "index_axis": index_axis,
                "effective_orientation": effective,
                "aliases": tuple(aliases),
                "role_evidence": tuple(evidence),
            }
        return cls(expectations, model_type, file_path)

    def expectation(self, name: str) -> dict[str, Any] | None:
        """Exact-name lookup only: no prefixes, suffixes or fuzzy matching."""
        return self.expectations.get(name)


# ---------------------------------------------------------------------------
# CRC32C (Castagnoli), table driven, no third-party dependency.
# ---------------------------------------------------------------------------

def _crc32c_table() -> list[int]:
    table = []
    for index in range(256):
        crc = index
        for _ in range(8):
            crc = (crc >> 1) ^ 0x82F63B78 if crc & 1 else crc >> 1
        table.append(crc)
    return table


_CRC32C_TABLE = _crc32c_table()


def crc32c(data: bytes) -> int:
    """Return the CRC32C (Castagnoli) checksum of ``data``."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Container structures.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Region:
    name: str
    offset: int
    length: int


@dataclass(frozen=True)
class Header:
    format_major: int
    format_minor: int
    flags: int
    file_size: int
    metadata: Region
    tensor_directory_offset: int
    tensor_count: int
    block_directory_offset: int
    block_count: int
    payload: Region
    integrity_directory_offset: int
    integrity_count: int
    artifact_id: bytes


@dataclass(frozen=True)
class TensorDescriptor:
    tensor_id: int
    rank: int
    dtype: int
    codec: int
    block_count: int
    first_block_index: int
    logical_nbytes: int
    dimensions: tuple[int, ...]
    name_ref: int
    representation_ref: int | None

    @property
    def dtype_name(self) -> str:
        return DTYPE_NAMES.get(self.dtype, f"<unknown:{self.dtype}>")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.dimensions[: self.rank]


@dataclass(frozen=True)
class BlockDescriptor:
    tensor_id: int
    block_id: int
    codec: int
    logical_offset: int
    logical_length: int
    payload_offset: int
    payload_length: int
    auxiliary_ref: int | None
    checksum_index: int | None


@dataclass(frozen=True)
class IntegrityRecord:
    target_kind: int
    target_index: int
    crc32c: int

    @property
    def is_metadata(self) -> bool:
        return self.target_kind == 1


@dataclass
class Container:
    path: Path
    raw: bytes
    header: Header
    metadata: dict[str, Any]
    tensor_names: list[str]
    tensors: list[TensorDescriptor]
    blocks: list[BlockDescriptor]
    integrity: list[IntegrityRecord]
    tensors_by_name: dict[str, TensorDescriptor] = field(default_factory=dict)

    @property
    def tensor_name_by_id(self) -> dict[int, str]:
        return {tensor.tensor_id: self.tensor_names[tensor.name_ref] for tensor in self.tensors}

    def blocks_of(self, tensor: TensorDescriptor) -> list[BlockDescriptor]:
        return [self.blocks[tensor.first_block_index + i] for i in range(tensor.block_count)]

    def read_block_payload(self, block: BlockDescriptor) -> bytes:
        end = block.payload_offset + block.payload_length
        if end > len(self.raw):
            raise HpsiFormatError("block payload extends past end of file")
        return self.raw[block.payload_offset:end]

    def read_tensor_payload(self, tensor: TensorDescriptor) -> bytes:
        parts = [self.read_block_payload(block) for block in self.blocks_of(tensor)]
        return b"".join(parts)


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def parse_header(raw: bytes) -> Header:
    if len(raw) < HEADER_SIZE:
        raise HpsiFormatError(f"file shorter than the {HEADER_SIZE}-byte header")
    if raw[0:4] != b"HPSI":
        raise HpsiFormatError("bad magic")
    major, minor = _u16(raw, 4), _u16(raw, 6)
    if (major, minor) != (0, 1):
        raise HpsiFormatError(f"unsupported version {major}.{minor}")
    if _u32(raw, 8) != HEADER_SIZE:
        raise HpsiFormatError("header_size must be 128")
    if _u32(raw, 12) != 0:
        raise HpsiFormatError("flags must be zero")
    file_size = _u64(raw, 16)
    if file_size != len(raw):
        raise HpsiFormatError(
            f"declared file size {file_size} differs from actual {len(raw)}"
        )
    metadata = Region("metadata", _u64(raw, 24), _u64(raw, 32))
    tensor_count = _u32(raw, 48)
    tensor_directory_offset = _u64(raw, 40)
    if _u32(raw, 52) != TENSOR_ENTRY_SIZE:
        raise HpsiFormatError("unexpected tensor entry size")
    block_count = _u32(raw, 64)
    block_directory_offset = _u64(raw, 56)
    if _u32(raw, 68) != BLOCK_ENTRY_SIZE:
        raise HpsiFormatError("unexpected block entry size")
    payload = Region("payload", _u64(raw, 72), _u64(raw, 80))
    integrity_count = _u32(raw, 96)
    integrity_directory_offset = _u64(raw, 88)
    integrity_entry_size = _u32(raw, 100)
    if integrity_count and integrity_entry_size != INTEGRITY_ENTRY_SIZE:
        raise HpsiFormatError("unexpected integrity entry size")
    if raw[120:128] != b"\x00" * 8:
        raise HpsiFormatError("reserved header bytes must be zero")
    if tensor_count == 0 or block_count == 0:
        raise HpsiFormatError("v0.1 requires at least one tensor and one block")
    if metadata.length == 0 or payload.length == 0:
        raise HpsiFormatError("metadata and payload regions must be non-empty")
    if metadata.offset + metadata.length > len(raw):
        raise HpsiFormatError("metadata region exceeds file size")
    if payload.offset + payload.length > len(raw):
        raise HpsiFormatError("payload region exceeds file size")
    return Header(
        format_major=major,
        format_minor=minor,
        flags=0,
        file_size=file_size,
        metadata=metadata,
        tensor_directory_offset=tensor_directory_offset,
        tensor_count=tensor_count,
        block_directory_offset=block_directory_offset,
        block_count=block_count,
        payload=payload,
        integrity_directory_offset=integrity_directory_offset,
        integrity_count=integrity_count,
        artifact_id=raw[104:120],
    )


def parse_tensor(data: bytes, offset: int) -> TensorDescriptor:
    entry = data[offset : offset + TENSOR_ENTRY_SIZE]
    if len(entry) != TENSOR_ENTRY_SIZE:
        raise HpsiFormatError("truncated tensor entry")
    rank = entry[4]
    if rank > 8:
        raise HpsiFormatError("tensor rank exceeds 8")
    dtype = entry[5]
    if dtype not in DTYPE_NAMES:
        raise HpsiFormatError(f"unknown dtype id {dtype}")
    if _u32(entry, 8) != 0 or _u32(entry, 100) != 0:
        raise HpsiFormatError("reserved tensor fields must be zero")
    representation = _u64(entry, 104)
    return TensorDescriptor(
        tensor_id=_u32(entry, 0),
        rank=rank,
        dtype=dtype,
        codec=_u16(entry, 6),
        block_count=_u32(entry, 12),
        first_block_index=_u64(entry, 16),
        logical_nbytes=_u64(entry, 24),
        dimensions=struct.unpack_from("<8Q", entry, 32),
        name_ref=_u32(entry, 96),
        representation_ref=None if representation == NONE_U64 else representation,
    )


def parse_block(data: bytes, offset: int) -> BlockDescriptor:
    entry = data[offset : offset + BLOCK_ENTRY_SIZE]
    if len(entry) != BLOCK_ENTRY_SIZE:
        raise HpsiFormatError("truncated block entry")
    if _u32(entry, 8) != 0 or _u16(entry, 14) != 0 or _u32(entry, 60) != 0:
        raise HpsiFormatError("unsupported block flags, layout, or reserved field")
    auxiliary = _u64(entry, 48)
    checksum = _u32(entry, 56)
    return BlockDescriptor(
        tensor_id=_u32(entry, 0),
        block_id=_u32(entry, 4),
        codec=_u16(entry, 12),
        logical_offset=_u64(entry, 16),
        logical_length=_u64(entry, 24),
        payload_offset=_u64(entry, 32),
        payload_length=_u64(entry, 40),
        auxiliary_ref=None if auxiliary == NONE_U64 else auxiliary,
        checksum_index=None if checksum == NONE_U32 else checksum,
    )


def parse_integrity(data: bytes, offset: int) -> IntegrityRecord:
    entry = data[offset : offset + INTEGRITY_ENTRY_SIZE]
    if len(entry) != INTEGRITY_ENTRY_SIZE:
        raise HpsiFormatError("truncated integrity entry")
    kind, index, algorithm, flags = struct.unpack_from("<IIHH", entry, 0)
    if algorithm != 1 or flags != 0 or entry[16:28] != b"\x00" * 12:
        raise HpsiFormatError("invalid integrity algorithm, flags, or reserved bytes")
    if _u32(entry, 28) != 0:
        raise HpsiFormatError("invalid integrity reserved word")
    if not ((kind == 1 and index == 0) or kind == 2):
        raise HpsiFormatError("invalid integrity target")
    return IntegrityRecord(kind, index, _u32(entry, 12))


def open_container(path: str | Path) -> Container:
    path = Path(path)
    raw = path.read_bytes()
    header = parse_header(raw)
    metadata_bytes = raw[header.metadata.offset : header.metadata.offset + header.metadata.length]
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    if metadata.get("format") != "hpsi" or metadata.get("metadata_version") != 1:
        raise HpsiFormatError("metadata must declare format=hpsi, metadata_version=1")
    tensor_names = metadata.get("tensor_names")
    if not isinstance(tensor_names, list) or not all(isinstance(n, str) for n in tensor_names):
        raise HpsiFormatError("metadata tensor_names must be a list of strings")
    tensors = [
        parse_tensor(raw, header.tensor_directory_offset + i * TENSOR_ENTRY_SIZE)
        for i in range(header.tensor_count)
    ]
    blocks = [
        parse_block(raw, header.block_directory_offset + i * BLOCK_ENTRY_SIZE)
        for i in range(header.block_count)
    ]
    integrity = [
        parse_integrity(raw, header.integrity_directory_offset + i * INTEGRITY_ENTRY_SIZE)
        for i in range(header.integrity_count)
    ] if header.integrity_count else []
    container = Container(
        path=path,
        raw=raw,
        header=header,
        metadata=metadata,
        tensor_names=tensor_names,
        tensors=tensors,
        blocks=blocks,
        integrity=integrity,
    )
    for tensor in tensors:
        if tensor.name_ref >= len(tensor_names):
            raise HpsiFormatError(f"tensor {tensor.tensor_id} has invalid name reference")
        container.tensors_by_name[tensor_names[tensor.name_ref]] = tensor
    return container


def verify_integrity(container: Container) -> tuple[int, list[str]]:
    """Verify every CRC32C record. Returns (verified_count, failures)."""
    failures: list[str] = []
    verified = 0
    for record in container.integrity:
        if record.is_metadata:
            payload = container.raw[
                container.header.metadata.offset :
                container.header.metadata.offset + container.header.metadata.length
            ]
            label = "metadata"
        else:
            if record.target_index >= len(container.blocks):
                failures.append(f"integrity record targets unknown block {record.target_index}")
                continue
            block = container.blocks[record.target_index]
            payload = container.read_block_payload(block)
            label = f"block[{record.target_index}]"
        actual = crc32c(payload)
        verified += 1
        if actual != record.crc32c:
            failures.append(
                f"{label}: stored 0x{record.crc32c:08x} != computed 0x{actual:08x}"
            )
    return verified, failures


# ---------------------------------------------------------------------------
# Safetensors.
# ---------------------------------------------------------------------------

@dataclass
class SafeTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int


class SafeTensorFile:
    """Minimal Safetensors reader: u64-LE header length, JSON header, payload."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.raw = self.path.read_bytes()
        if len(self.raw) < 8:
            raise HpsiFormatError("safetensors file is shorter than 8 bytes")
        header_length = struct.unpack_from("<Q", self.raw, 0)[0]
        if header_length == 0 or 8 + header_length > len(self.raw):
            raise HpsiFormatError("safetensors header length is invalid")
        header = json.loads(self.raw[8 : 8 + header_length].decode("utf-8"))
        self.metadata = header.pop("__metadata__", None)
        self.tensors: dict[str, SafeTensor] = {}
        base = 8 + header_length
        for name, entry in header.items():
            start, end = entry["data_offsets"]
            self.tensors[name] = SafeTensor(
                name=name,
                dtype=entry["dtype"],
                shape=tuple(entry["shape"]),
                data_start=base + start,
                data_end=base + end,
            )

    def raw_payload(self, name: str) -> bytes:
        tensor = self.tensors[name]
        return self.raw[tensor.data_start : tensor.data_end]

    def f32_values(self, name: str) -> np.ndarray:
        tensor = self.tensors[name]
        if tensor.dtype != "F32":
            raise HpsiFormatError(f"safetensors tensor {name} is {tensor.dtype}, expected F32")
        values = np.frombuffer(self.raw_payload(name), dtype="<f4")
        return values.astype(np.float64)

    def shape_of(self, name: str) -> tuple[int, ...]:
        return self.tensors[name].shape


# ---------------------------------------------------------------------------
# VQ reconstruction.
# ---------------------------------------------------------------------------

def unpack_selectors(packed: bytes, selector_count: int) -> np.ndarray:
    """Unpack 4-bit selectors exactly as crates/hpsi-ops/src/vq.rs does.

    The global selector index is ``(row * groups_per_row + group) * codebook_count + stage``.
    Even indices live in the low nibble of a byte, odd indices in the high nibble.
    """
    needed_bytes = (selector_count + 1) // 2
    if len(packed) < needed_bytes:
        raise HpsiFormatError(
            f"selector stream holds {len(packed)} bytes, needs {needed_bytes}"
        )
    buffer = np.frombuffer(packed, dtype=np.uint8, count=needed_bytes).astype(np.uint16)
    selectors = np.empty(selector_count, dtype=np.uint16)
    selectors[0::2] = buffer[: (selector_count + 1) // 2] & 0x0F
    if selector_count > 1:
        selectors[1::2] = buffer[: selector_count // 2] >> 4
    return selectors


def reconstruct_matrix(
    container: Container,
    codebook_tensor: TensorDescriptor,
    indices_tensor: TensorDescriptor,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Dense f32 reconstruction of a VQ matrix, shape ``(rows, cols)``."""
    codebook_count = int(codebook_tensor.shape[0])
    if codebook_tensor.shape != (codebook_count, CODEBOOK_SIZE, GROUP_SIZE):
        raise HpsiFormatError(f"codebook shape {codebook_tensor.shape} is not [c,16,8]")
    if codebook_count < 1 or codebook_count > 4:
        raise HpsiFormatError(f"codebook count {codebook_count} outside 1..=4")
    if cols % GROUP_SIZE:
        raise HpsiFormatError(f"column count {cols} is not a multiple of {GROUP_SIZE}")
    codebook = np.frombuffer(
        container.read_tensor_payload(codebook_tensor), dtype="<f4"
    ).astype(np.float64)
    expected_codebook = codebook_count * CODEBOOK_VALUES
    if codebook.size != expected_codebook:
        raise HpsiFormatError(
            f"codebook holds {codebook.size} values, expected {expected_codebook}"
        )
    codebook = codebook.reshape(codebook_count, CODEBOOK_SIZE, GROUP_SIZE)

    groups_per_row = cols // GROUP_SIZE
    selectors_per_row = groups_per_row * codebook_count
    selector_count = rows * selectors_per_row
    packed = container.read_tensor_payload(indices_tensor)
    selectors = unpack_selectors(packed, selector_count)
    selectors = selectors.reshape(rows, groups_per_row, codebook_count)

    dense = np.zeros((rows, cols), dtype=np.float64)
    for stage in range(codebook_count):
        codes = selectors[:, :, stage]
        # codebook[stage][code] is a (CODEBOOK_SIZE, GROUP_SIZE) table.
        gathered = codebook[stage][codes]  # (rows, groups_per_row, GROUP_SIZE)
        dense += gathered.reshape(rows, cols)
    return dense


# ---------------------------------------------------------------------------
# Orientation oracle and metrics.
# ---------------------------------------------------------------------------

def classify_orientation(
    declared: str | None,
    oracle: str | None,
    source_shape: Sequence[int],
    declared_role: str | None = None,
    oracle_roles: tuple[str, ...] = (),
) -> str:
    """Five-state orientation assurance, written independently of the Rust audit.

    ``architecture_confirmed`` is the only state that means the orientation was
    checked against something outside the container. Structural consistency, a
    plausible shape or a matching name never reaches it.

    Roles are a separate dimension and are reported by :func:`classify_role`;
    here they only *subtract* claims: a tensor that is positively known not to be
    a projection has no orientation question to answer, and a tensor the producer
    labelled ``unknown`` keeps its claim explicitly unverified instead of being
    grouped with pre-role artifacts.
    """
    if len(source_shape) != 2:
        # Only matrices have an orientation. The importer still records tokens
        # for rank-1 and higher-rank tensors; those are provenance, not a
        # layout claim, and must not be reported as verified or as violated.
        return "not_a_matrix"
    if declared_role == "unknown":
        return "role_unknown"
    if declared_role is not None and declared_role not in ROLE_ORIENTATIONS:
        if oracle_roles and declared_role not in oracle_roles:
            return "role_contradiction"
        return "not_a_projection"
    if declared not in ORIENTATIONS:
        return "legacy_undeclared"
    if oracle is not None:
        return "architecture_confirmed" if declared == oracle else "architecture_contradiction"
    if len(source_shape) == 2 and source_shape[0] == source_shape[1]:
        # Both orientations produce the identical matrix shape and, for a VQ
        # pair, the identical stored byte count: nothing in the container
        # discriminates them.
        return "square_not_verifiable"
    return "declared_not_verified"


def classify_role(
    declared_role: str | None, oracle_roles: tuple[str, ...] = ()
) -> str:
    """How far the architectural role of one tensor has been established.

    ``role_confirmed`` requires the model definition to state the same role. A
    role that contradicts it is a finding, never a softer status, and the
    absence of any role is kept distinct from a role the producer could not
    establish: conflating them would let a new import look like a legacy artifact.
    """
    if declared_role is None:
        return "role_not_declared"
    if declared_role not in TENSOR_ROLES:
        return "role_invalid"
    if not oracle_roles:
        return "role_declared_not_confirmed"
    if declared_role in oracle_roles:
        return "role_confirmed"
    return "role_contradiction"


def logical_from_source(
    source: SafeTensorFile, name: str, orientation: str
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return the architecturally correct logical ``[out, in]`` matrix."""
    if orientation not in ORIENTATIONS:
        raise HpsiFormatError(f"unknown orientation {orientation!r} for {name}")
    stored = source.f32_values(name).reshape(source.shape_of(name))
    if orientation == "linear_out_in":
        return stored, stored.shape
    if stored.ndim != 2:
        raise HpsiFormatError(f"Conv1D tensor {name} is not rank 2")
    transposed = stored.T
    return transposed, transposed.shape


def metrics(reference: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64).ravel()
    reconstructed = np.asarray(reconstructed, dtype=np.float64).ravel()
    if reference.shape != reconstructed.shape:
        raise ValueError(f"metric inputs differ: {reference.shape} vs {reconstructed.shape}")
    if reference.size == 0:
        raise ValueError("metric inputs are empty")
    difference = reconstructed - reference
    l2_reference = float(np.sqrt(np.sum(reference * reference)))
    return {
        "elements": int(reference.size),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "relative_l2": float(np.sqrt(np.sum(difference * difference)) / l2_reference)
        if l2_reference
        else float("nan"),
        "max_abs": float(np.max(np.abs(difference))),
        "l2_reference": l2_reference,
        "l2_reconstructed": float(np.sqrt(np.sum(reconstructed * reconstructed))),
    }


# ---------------------------------------------------------------------------
# Audit driver.
# ---------------------------------------------------------------------------

def audit(
    source_path: str | Path,
    container_path: str | Path,
    label: str = "container",
    architecture_path: str | Path | None = None,
) -> dict[str, Any]:
    source = SafeTensorFile(source_path)
    container = open_container(container_path)
    oracle = ArchitectureOracle.load(architecture_path) if architecture_path else None
    verified, integrity_failures = verify_integrity(container)

    provenance = (
        container.metadata.get("user", {}).get("source_tensors")
        if isinstance(container.metadata.get("user"), dict)
        else None
    )
    if not isinstance(provenance, list):
        raise HpsiFormatError("metadata lacks user.source_tensors provenance")
    provenance_by_name = {}
    for entry in provenance:
        name = entry.get("name")
        if not isinstance(name, str):
            raise HpsiFormatError("provenance entry lacks a name")
        provenance_by_name[name] = entry

    findings: list[dict[str, Any]] = []
    quantized: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    layout_violations: dict[str, int] = {}

    for name, entry in provenance_by_name.items():
        disposition = entry.get("disposition")
        declared = entry.get("weight_orientation")
        declared = declared if isinstance(declared, str) else None
        orientation_source = entry.get("orientation_source")
        declared_role = entry.get("tensor_role")
        declared_role = declared_role if isinstance(declared_role, str) else None
        declared_axis = entry.get("index_axis")
        expectation = oracle.expectation(name) if oracle is not None else None
        oracle_token = (
            expectation["effective_orientation"] if expectation is not None else None
        )
        oracle_roles = expectation["tensor_roles"] if expectation is not None else ()
        oracle_aliases = expectation["aliases"] if expectation is not None else ()
        oracle_evidence = (
            [dict(row) for row in expectation["role_evidence"]]
            if expectation is not None
            else []
        )
        source_shape = list(source.shape_of(name))
        orientation_status = classify_orientation(
            declared,
            oracle_token,
            source_shape,
            declared_role,
            oracle_roles,
        )
        role_status = classify_role(declared_role, oracle_roles)
        orientation_matches = (
            declared == oracle_token if oracle_token is not None else None
        )
        status_counts[orientation_status] = status_counts.get(orientation_status, 0) + 1
        role_counts[role_status] = role_counts.get(role_status, 0) + 1
        manifest_shape = expectation["source_shape"] if expectation is not None else None

        # The layout matrix, applied to the metadata *and* to the physical pair,
        # so an index axis that contradicts the stored selectors is reported even
        # when the CRC was recomputed. Same codes as the Rust library.
        pair_shape = (
            [int(v) for v in entry["logical_shape"]]
            if isinstance(entry.get("logical_shape"), list)
            and len(entry["logical_shape"]) == 2
            else None
        )
        codebook_count, stored_index_bytes = 0, 0
        if disposition == "quantize_vq":
            codebook = container.tensors_by_name.get(f"{name}.hpsi_vq.codebook")
            indices = container.tensors_by_name.get(f"{name}.hpsi_vq.indices")
            if (
                codebook is not None
                and indices is not None
                and len(codebook.shape) == 3
                and indices.rank == 1
            ):
                codebook_count = int(codebook.shape[0])
                stored_index_bytes = int(indices.logical_nbytes)
        violation = entry_layout_violation(
            source_shape=source_shape,
            weight_orientation=declared,
            logical_shape=pair_shape,
            tensor_role=declared_role,
            index_axis=declared_axis,
            context="source",
            quantized=disposition == "quantize_vq",
            codebook_count=codebook_count,
            stored_index_bytes=stored_index_bytes,
            calibrated=entry.get("calibration") is not None,
        )
        if violation is not None:
            code, reason = violation
            layout_violations[code] = layout_violations.get(code, 0) + 1
            findings.append({
                "kind": "entry_layout_violation",
                "code": code,
                "tensor": name,
                "reason": reason,
                "declared_role": declared_role,
                "weight_orientation": declared,
                "index_axis": declared_axis,
                "logical_shape": pair_shape,
            })
        if role_status == "role_contradiction":
            findings.append({
                "kind": "role_disagrees_with_architecture",
                "tensor": name,
                "declared_role": declared_role,
                "oracle_roles": list(oracle_roles),
            })

        if orientation_status == "architecture_contradiction":
            findings.append({
                "kind": "orientation_disagrees_with_architecture",
                "tensor": name,
                "declared": declared,
                "oracle": oracle_token,
                "orientation_source": orientation_source,
                "source_shape": source_shape,
            })
        if (
            manifest_shape is not None
            and len(source_shape) == 2
            and tuple(source_shape) != manifest_shape
        ):
            findings.append({
                "kind": "source_shape_disagrees_with_architecture",
                "tensor": name,
                "source_shape": source_shape,
                "manifest_source_shape": list(manifest_shape),
            })

        if disposition == "preserve_raw":
            tensor = container.tensors_by_name.get(name)
            if tensor is None:
                findings.append({"kind": "raw_tensor_missing", "tensor": name})
                continue
            original = source.raw_payload(name)
            recovered = container.read_tensor_payload(tensor)
            identical = original == recovered
            dtype_ok = tensor.dtype_name == source.tensors[name].dtype
            shape_ok = tuple(tensor.shape) == tuple(source.shape_of(name))
            size_ok = (
                tensor.logical_nbytes == len(original)
                and len(recovered) == len(original)
            )
            if not (identical and dtype_ok and shape_ok and size_ok):
                findings.append({
                    "kind": "raw_preservation_failure",
                    "tensor": name,
                    "byte_identical": identical,
                    "dtype_matches": dtype_ok,
                    "shape_matches": shape_ok,
                    "size_matches": size_ok,
                })
            raw.append({
                "tensor": name,
                "source_dtype": source.tensors[name].dtype,
                "container_dtype": tensor.dtype_name,
                "container_codec": tensor.codec,
                "source_shape": source_shape,
                "declared_orientation": declared,
                "orientation_source": orientation_source,
                "oracle_orientation": oracle_token,
                "orientation_matches_oracle": orientation_matches,
                "orientation_status": orientation_status,
                "declared_role": declared_role,
                "index_axis": declared_axis,
                "oracle_roles": list(oracle_roles),
                "oracle_aliases": list(oracle_aliases),
                "role_evidence": oracle_evidence,
                "role_status": role_status,
                # A RAW payload stores the source bytes verbatim, so nothing in
                # the container can falsify its declared orientation: the label
                # only tells a consumer how the bytes are meant to be read.
                "orientation_falsifiable_from_container": False,
                "container_shape": list(tensor.shape),
                "original_bytes": len(original),
                "container_bytes": len(recovered),
                "logical_nbytes": tensor.logical_nbytes,
                "byte_identical": identical,
                "dtype_matches": dtype_ok,
                "shape_matches": shape_ok,
                "size_matches": size_ok,
                "crc32c_original": crc32c(original),
                "crc32c_recovered": crc32c(recovered),
            })
            continue

        # Quantized tensor.
        codebook_name = f"{name}.hpsi_vq.codebook"
        indices_name = f"{name}.hpsi_vq.indices"
        codebook_tensor = container.tensors_by_name.get(codebook_name)
        indices_tensor = container.tensors_by_name.get(indices_name)
        if codebook_tensor is None or indices_tensor is None:
            findings.append({"kind": "vq_tensor_missing", "tensor": name})
            continue
        logical = entry.get("logical_shape")
        if not (isinstance(logical, list) and len(logical) == 2):
            findings.append({"kind": "missing_logical_shape", "tensor": name})
            continue
        rows, cols = int(logical[0]), int(logical[1])

        codebook_count = int(codebook_tensor.shape[0])
        expected_index_bytes = (
            rows * (cols // GROUP_SIZE) * codebook_count + 1
        ) // 2
        index_size_consistent = expected_index_bytes == indices_tensor.logical_nbytes
        if not index_size_consistent:
            findings.append({
                "kind": "index_payload_inconsistent_with_logical_shape",
                "tensor": name,
                "expected_index_bytes": expected_index_bytes,
                "stored_index_bytes": indices_tensor.logical_nbytes,
            })

        reconstructed = reconstruct_matrix(
            container, codebook_tensor, indices_tensor, rows, cols
        )
        if oracle_token is not None:
            oracle_logical, oracle_shape = logical_from_source(source, name, oracle_token)
        else:
            oracle_logical, oracle_shape = None, None

        record: dict[str, Any] = {
            "tensor": name,
            "source_dtype": source.tensors[name].dtype,
            "source_shape": source_shape,
            "declared_orientation": declared,
            "orientation_source": orientation_source,
            "oracle_orientation": oracle_token,
            "orientation_matches_oracle": orientation_matches,
            "orientation_status": orientation_status,
            "declared_role": declared_role,
            "index_axis": declared_axis,
            "oracle_roles": list(oracle_roles),
            "oracle_aliases": list(oracle_aliases),
            "role_evidence": oracle_evidence,
            "role_status": role_status,
            "logical_shape": [rows, cols],
            "oracle_logical_shape": list(oracle_shape) if oracle_shape else None,
            "shape_matches_oracle": (
                (rows, cols) == tuple(oracle_shape) if oracle_shape else None
            ),
            "shape_is_oracle_transpose": (
                (rows, cols) == tuple(reversed(oracle_shape)) if oracle_shape else None
            ),
            "source_shape_is_square": len(source_shape) == 2 and source_shape[0] == source_shape[1],
            "codebook_dtype": codebook_tensor.dtype_name,
            "codebook_codec": codebook_tensor.codec,
            "codebook_shape": list(codebook_tensor.shape),
            "indices_dtype": indices_tensor.dtype_name,
            "indices_codec": indices_tensor.codec,
            "indices_shape": list(indices_tensor.shape),
            "indices_logical_nbytes": indices_tensor.logical_nbytes,
            "expected_index_bytes": expected_index_bytes,
            "index_size_consistent": index_size_consistent,
            "dense_f32_bytes": rows * cols * 4,
            "vq_stored_bytes": sum(b.payload_length for b in container.blocks_of(codebook_tensor))
            + sum(b.payload_length for b in container.blocks_of(indices_tensor)),
            "vq_bits_per_weight": codebook_count * 4.0 / GROUP_SIZE,
        }

        # Orientation-normalized distortion: compare the container's own array
        # layout against the architecturally correct matrix for that same
        # layout. The layout is chosen from the DECLARED orientation,
        # never from which reference fits better (that would be circular). This
        # is what makes square matrices evaluable: shape alone cannot disambiguate
        # them, and a transposed square import is numerically detectable.
        if oracle_logical is not None:
            reference = (
                oracle_logical.T if declared != oracle_token else oracle_logical
            )
            record["quantization_distortion_basis"] = "architecture_oracle"
        elif declared in ORIENTATIONS:
            # No external definition: the distortion measures whether the
            # container faithfully encodes the layout IT declares. It says
            # nothing about whether that declaration matches the model.
            reference, _ = logical_from_source(source, name, declared)
            record["quantization_distortion_basis"] = "declared_orientation_self_consistency"
        else:
            reference = None
            record["quantization_distortion_basis"] = None
            findings.append({
                "kind": "no_orientation_basis_for_numeric_comparison",
                "tensor": name,
                "declared_orientation": declared,
                "oracle_orientation": oracle_token,
                "note": "neither the container nor a model-definition manifest states how these weights are arranged",
            })
        if reference is not None:
            if reference.shape == reconstructed.shape:
                normalized = metrics(reference, reconstructed)
                record["quantization_distortion"] = normalized
                record["quantization_distortion_relative_l2"] = normalized["relative_l2"]
            else:
                findings.append({
                    "kind": "container_arrangement_does_not_match_declared_orientation",
                    "tensor": name,
                    "reconstructed_shape": [rows, cols],
                    "reference_shape": list(reference.shape),
                    "declared_orientation": declared,
                    "oracle_orientation": oracle_token,
                })

        # Direct comparison is only defined when the container's logical shape
        # equals the architectural one.
        record["architecture_reference_available"] = oracle_shape is not None
        if oracle_shape is None:
            # Without a model-definition manifest the container's own claim is
            # all there is: report it, and state that nothing confirms it.
            record["metrics_vs_oracle_logical"] = None
            record["metrics_vs_oracle_transposed"] = None
        elif (rows, cols) == tuple(oracle_shape):
            direct = metrics(oracle_logical, reconstructed)
            flipped = metrics(oracle_logical.T, reconstructed)
            record["metrics_vs_oracle_logical"] = direct
            record["metrics_vs_oracle_transposed"] = flipped
            if rows == cols:
                record["oracle_asymmetry_ratio"] = float(
                    np.linalg.norm(oracle_logical - oracle_logical.T)
                    / np.linalg.norm(oracle_logical)
                ) if np.linalg.norm(oracle_logical) else float("nan")
            if direct["relative_l2"] >= flipped["relative_l2"]:
                findings.append({
                    "kind": "reconstruction_matches_transpose_better_than_logical",
                    "tensor": name,
                    "direct_relative_l2": direct["relative_l2"],
                    "transposed_relative_l2": flipped["relative_l2"],
                })
        elif (rows, cols) == tuple(reversed(oracle_shape)):
            record["metrics_vs_oracle_logical"] = None
            record["metrics_vs_oracle_transposed"] = metrics(
                oracle_logical.T, reconstructed
            )
            findings.append({
                "kind": "container_stores_transpose_of_architectural_logical_weight",
                "tensor": name,
                "logical_shape": [rows, cols],
                "oracle_logical_shape": list(oracle_shape),
                "relative_l2_against_transposed_oracle": record[
                    "metrics_vs_oracle_transposed"
                ]["relative_l2"],
            })
        else:
            findings.append({
                "kind": "logical_shape_matches_neither_oracle_nor_transpose",
                "tensor": name,
                "logical_shape": [rows, cols],
                "oracle_logical_shape": list(oracle_shape),
            })
        quantized.append(record)

    omitted = [name for name in source.tensors if name not in provenance_by_name]
    if omitted:
        findings.append({"kind": "source_tensors_omitted_from_container", "tensors": omitted})
    accounted = len(quantized) + len(raw)
    if accounted != len(provenance):
        findings.append({
            "kind": "provenance_entries_not_accounted_for",
            "provenance_entries": len(provenance),
            "accounted": accounted,
        })

    normalized_entries = [
        record["quantization_distortion"] for record in quantized
        if "quantization_distortion" in record
    ]
    pooled_elements = sum(item["elements"] for item in normalized_entries)
    pooled_sse = sum(item["rmse"] ** 2 * item["elements"] for item in normalized_entries)
    pooled_reference = sum(item["l2_reference"] ** 2 for item in normalized_entries)
    direct_entries = [
        record["metrics_vs_oracle_logical"] for record in quantized
        if record.get("metrics_vs_oracle_logical")
    ]
    transposed_entries = [
        record["metrics_vs_oracle_transposed"] for record in quantized
        if record.get("metrics_vs_oracle_transposed")
    ]

    def mean(values: Iterable[float]) -> float:
        values = list(values)
        return float(np.mean(values)) if values else float("nan")

    dense_bytes = sum(record["dense_f32_bytes"] for record in quantized)
    stored_bytes = sum(record["vq_stored_bytes"] for record in quantized)

    return {
        "implementation": "python-independent",
        "label": label,
        "source": str(source_path),
        "container": str(container_path),
        "container_bytes": len(container.raw),
        "source_bytes": len(source.raw),
        "format_version": f"{container.header.format_major}.{container.header.format_minor}",
        "artifact_id": container.header.artifact_id.hex(),
        "metadata_bytes": container.header.metadata.length,
        "payload_region_bytes": container.header.payload.length,
        "tensor_directory_bytes": container.header.tensor_count * TENSOR_ENTRY_SIZE,
        "block_directory_bytes": container.header.block_count * BLOCK_ENTRY_SIZE,
        "integrity_directory_bytes": container.header.integrity_count * INTEGRITY_ENTRY_SIZE,
        "container_tensors": len(container.tensors),
        "container_blocks": len(container.blocks),
        "verified_crc32c_records": verified,
        "integrity_failures": integrity_failures,
        "architecture_oracle": {
            "supplied": oracle is not None,
            "manifest": str(oracle.path) if oracle is not None else None,
            "model_type": oracle.model_type if oracle is not None else None,
            "tensors_described": len(oracle.expectations) if oracle is not None else 0,
            "orientation_statuses": status_counts,
            "role_statuses": role_counts,
            "layout_violations": layout_violations,
            "roles_declared": sum(
                count for status, count in role_counts.items() if status != "role_not_declared"
            ),
            "roles_confirmed": role_counts.get("role_confirmed", 0),
            "architecturally_confirmed": status_counts.get("architecture_confirmed", 0),
            "architectural_contradictions": status_counts.get("architecture_contradiction", 0),
            "unverified": sum(
                count
                for status, count in status_counts.items()
                if status
                in (
                    "declared_not_verified",
                    "square_not_verifiable",
                    "legacy_undeclared",
                )
            ),
            "note": (
                "architecture_confirmed means the declaration agrees with a manifest produced "
                "from the model definition; that is evidence about the model, not a property "
                "of the container, and no other state carries architectural proof"
            ),
            "role_note": (
                "role statuses describe the architectural function of a tensor and are "
                "reported apart from orientation: a confirmed role is no evidence about a "
                "layout, and a confirmed layout is no evidence about a role. role_not_declared "
                "means the artifact predates the role contract and is read exactly as before"
            ),
        },
        "source_tensor_names": sorted(source.tensors),
        "container_tensor_names": sorted(container.tensors_by_name),
        "quantized": quantized,
        "raw": raw,
        "raw_all_byte_identical": all(item["byte_identical"] for item in raw),
        "aggregate": {
            "tensors_quantized": len(quantized),
            "tensors_raw": len(raw),
            "source_tensors": len(provenance),
            "elements_in_normalized_aggregate": pooled_elements,
            "quantization_distortion_relative_l2": float(
                np.sqrt(pooled_sse) / np.sqrt(pooled_reference)
            ) if pooled_reference else float("nan"),
            "mean_quantization_distortion_relative_l2": mean(
                record["quantization_distortion_relative_l2"] for record in quantized
                if "quantization_distortion_relative_l2" in record
            ),
            "mean_relative_l2_vs_oracle_logical": mean(
                item["relative_l2"] for item in direct_entries
            ),
            "mean_relative_l2_vs_oracle_transposed": mean(
                item["relative_l2"] for item in transposed_entries
            ),
            "mae_pooled": float(
                sum(item["mae"] * item["elements"] for item in normalized_entries) / pooled_elements
            ) if pooled_elements else float("nan"),
            "rmse_pooled": float(np.sqrt(pooled_sse / pooled_elements)) if pooled_elements else float("nan"),
            "max_abs": max((item["max_abs"] for item in normalized_entries), default=float("nan")),
            "dense_f32_bytes": dense_bytes,
            "vq_stored_bytes": stored_bytes,
            "compression_ratio": dense_bytes / stored_bytes if stored_bytes else float("nan"),
        },
        "findings": findings,
    }


def compare_with_rust(python_report: dict[str, Any], rust_report_path: str | Path) -> list[dict[str, Any]]:
    """Diff per-tensor claims against the Rust harness; flag metrics > 1e-5 relative.

    Two things are compared on purpose. The reconstruction metrics say the two
    implementations read the same payload the same way; the architecture claims
    say they read the same *model definition* the same way, which is the half no
    shared code could verify.
    """
    path = Path(rust_report_path)
    if not path.exists():
        return []
    rust_report = json.loads(path.read_text(encoding="utf-8"))
    discrepancies: list[dict[str, Any]] = []

    def rust_entries(key: str) -> dict[str, dict[str, Any]]:
        return {item["tensor"]: item for item in rust_report.get(key, [])}

    def evidence_rows(value: Any) -> list[tuple[Any, ...]]:
        rows = value or []
        return sorted(
            (
                row.get("alias"),
                row.get("module"),
                row.get("module_class"),
                row.get("tensor_role"),
                row.get("weight_orientation"),
                row.get("index_axis"),
            )
            for row in rows
        )

    claims = (
        "declared_role",
        "index_axis",
        "oracle_roles",
        "oracle_aliases",
    )
    for section in ("quantized", "raw"):
        rust_by_name = rust_entries(section)
        for item in python_report.get(section, []):
            other = rust_by_name.get(item["tensor"])
            if other is None:
                continue
            for key in claims:
                mine, theirs = item.get(key), other.get(key)
                if isinstance(mine, (list, tuple)):
                    mine = sorted(mine)
                if isinstance(theirs, (list, tuple)):
                    theirs = sorted(theirs)
                if mine != theirs:
                    discrepancies.append({
                        "tensor": item["tensor"],
                        "kind": "architecture_claim_differs",
                        "field": key,
                        "python": mine,
                        "rust": theirs,
                    })
            if evidence_rows(item.get("role_evidence")) != evidence_rows(
                other.get("role_evidence")
            ):
                discrepancies.append({
                    "tensor": item["tensor"],
                    "kind": "architecture_claim_differs",
                    "field": "role_evidence",
                    "python": evidence_rows(item.get("role_evidence")),
                    "rust": evidence_rows(other.get("role_evidence")),
                })

    rust_by_name = rust_entries("quantized")
    for item in python_report["quantized"]:
        other = rust_by_name.get(item["tensor"])
        if other is None:
            discrepancies.append({"tensor": item["tensor"], "kind": "missing_in_rust"})
            continue
        for key in ("mae", "rmse", "relative_l2", "max_abs", "l2_reference", "l2_reconstructed"):
            mine = (item.get("quantization_distortion") or {}).get(key)
            theirs = (other.get("quantization_distortion") or {}).get(key)
            if theirs is None:
                theirs = (other.get("metrics_vs_oracle_logical") or {}).get(key)
            if mine is None or theirs is None:
                continue
            denominator = max(abs(theirs), 1e-12)
            relative = abs(mine - theirs) / denominator
            if relative > 1e-5:
                discrepancies.append({
                    "tensor": item["tensor"],
                    "metric": key,
                    "python": mine,
                    "rust": theirs,
                    "relative_difference": relative,
                })
    return discrepancies


def render_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        f"# Auditoria independente (Python) — {report['label']}",
        "",
        f"- Fonte: `{report['source']}`",
        f"- Container: `{report['container']}`",
        f"- Versão do formato: {report['format_version']}",
        f"- Bytes do container: {report['container_bytes']}",
        f"- Tensores no container: {report['container_tensors']}",
        f"- Blocos: {report['container_blocks']}",
        f"- Registros CRC32C verificados: {report['verified_crc32c_records']}",
        f"- Falhas de integridade: {len(report['integrity_failures'])}",
        (
            f"- Oráculo arquitetural: `{report['architecture_oracle']['manifest']}` "
            f"({report['architecture_oracle']['tensors_described']} tensores), "
            f"{report['architecture_oracle']['architecturally_confirmed']} confirmados, "
            f"{report['architecture_oracle']['unverified']} não verificados"
            if report["architecture_oracle"]["supplied"]
            else (
                "- Oráculo arquitetural: não fornecido; nenhuma orientação foi confirmada "
                f"({report['architecture_oracle']['unverified']} tensores não verificados)"
            )
        ),
        f"- Estados de orientação: {json.dumps(report['architecture_oracle']['orientation_statuses'], sort_keys=True)}",
        f"- Estados de papel: {json.dumps(report['architecture_oracle']['role_statuses'], sort_keys=True)}",
        f"- RAW byte a byte idêntico: {report['raw_all_byte_identical']}",
        f"- Distorção de quantização normalizada (pooled): {aggregate['quantization_distortion_relative_l2']}",
        f"- Distorção média por tensor: {aggregate['mean_quantization_distortion_relative_l2']}",
        f"- Rel-L2 médio correto/transposto: {aggregate['mean_relative_l2_vs_oracle_logical']} / {aggregate['mean_relative_l2_vs_oracle_transposed']}",
        f"- Compressão denso-F32 → VQ: {aggregate['dense_f32_bytes']} → {aggregate['vq_stored_bytes']} bytes ({aggregate['compression_ratio']:.3f}x)",
        "",
        "## Métricas por tensor",
        "",
        "| Tensor | Shape lógico | Orientação declarada | Oráculo | Estado | Papel | Elementos | MAE | RMSE | Rel-L2 | Máx abs | ‖orig‖ | ‖recon‖ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in report["quantized"]:
        distortion = item.get("quantization_distortion") or {}
        lines.append(
            "| `{}` | {} | {} | {} | `{}` | `{}` | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.4f} | {:.4f} |".format(
                item["tensor"],
                item["logical_shape"],
                item["declared_orientation"],
                item["oracle_orientation"] or "—",
                item["orientation_status"],
                item.get("role_status", "role_not_declared"),
                distortion.get("elements", ""),
                distortion.get("mae", float("nan")),
                distortion.get("rmse", float("nan")),
                distortion.get("relative_l2", float("nan")),
                distortion.get("max_abs", float("nan")),
                distortion.get("l2_reference", float("nan")),
                distortion.get("l2_reconstructed", float("nan")),
            )
        )
    lines += [
        "",
        "## Preservação RAW",
        "",
        "| Tensor | Bytes origem | Bytes HPSI | dtype | shape | Byte a byte | CRC32C origem | CRC32C HPSI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in report["raw"]:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | 0x{:08x} | 0x{:08x} |".format(
                item["tensor"],
                item["original_bytes"],
                item["container_bytes"],
                item["container_dtype"],
                item["container_shape"],
                item["byte_identical"],
                item["crc32c_original"],
                item["crc32c_recovered"],
            )
        )
    lines += ["", "## Achados", ""]
    if not report["findings"]:
        lines.append("Nenhum.")
    else:
        for finding in report["findings"]:
            lines.append(f"- `{finding['kind']}` `{json.dumps(finding, ensure_ascii=False)}`")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent HPSI container auditor")
    parser.add_argument("--source", required=True, help="source .safetensors file")
    parser.add_argument("--hpsi", required=True, help="HPSI container to audit")
    parser.add_argument("--out", required=True, help="JSON report path")
    parser.add_argument("--md", help="optional Markdown report path")
    parser.add_argument("--label", default="container", help="label for the report")
    parser.add_argument(
        "--architecture",
        help=(
            "architecture manifest (model definition) used as the orientation oracle; "
            "without it no tensor is reported as architecturally confirmed"
        ),
    )
    parser.add_argument(
        "--compare-rust",
        help="optional Rust harness JSON to cross-check against",
    )
    args = parser.parse_args(argv)

    report = audit(args.source, args.hpsi, args.label, args.architecture)
    if args.compare_rust:
        report["cross_check_vs_rust"] = {
            "rust_report": args.compare_rust,
            "discrepancies": compare_with_rust(report, args.compare_rust),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.md:
        md_path = Path(args.md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"container: {args.hpsi}")
    oracle_report = report["architecture_oracle"]
    print(
        "architecture oracle: "
        + (
            f"manifest {oracle_report['manifest']} ({oracle_report['tensors_described']} tensors)"
            if oracle_report["supplied"]
            else "not supplied; no orientation is architecturally confirmed"
        )
    )
    print(
        "orientation statuses: "
        + json.dumps(oracle_report["orientation_statuses"], sort_keys=True)
    )
    print(
        "role statuses: "
        + json.dumps(oracle_report.get("role_statuses", {}), sort_keys=True)
    )
    print(
        "layout violation codes: "
        + json.dumps(oracle_report.get("layout_violations", {}), sort_keys=True)
    )
    print(f"quantized tensors: {report['aggregate']['tensors_quantized']}")
    print(f"raw tensors: {report['aggregate']['tensors_raw']}")
    print(f"raw byte-identical: {report['raw_all_byte_identical']}")
    print(f"verified CRC32C records: {report['verified_crc32c_records']}")
    print(f"normalized distortion: {report['aggregate']['quantization_distortion_relative_l2']}")
    print(f"findings: {len(report['findings'])}")
    for finding in report["findings"]:
        print(f"  - {json.dumps(finding, ensure_ascii=False)}")
    if args.compare_rust:
        discrepancies = report["cross_check_vs_rust"]["discrepancies"]
        print(f"cross-check vs Rust: {len(discrepancies)} discrepancies")
        for item in discrepancies:
            print(f"  ! {json.dumps(item, ensure_ascii=False)}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
