"""Tests for the independent HPSI reader in ``tools/audit_hpsi_container.py``.

These are known-answer tests: every decoding path is checked against a value
computed by hand, not merely against "it did not raise". That matters because a
second implementation only has audit value if its arithmetic is pinned down.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import audit_hpsi_container as hpsi  # noqa: E402
import architecture_manifest_from_calibration as arch  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / ".tmp-gpt2-conv1d-test"
CALIBRATED = ARTIFACT_DIR / "gpt2-calibrated.hpsi"
UNCALIBRATED = ARTIFACT_DIR / "gpt2-uncalibrated.hpsi"
SOURCE = ARTIFACT_DIR / "model" / "model.safetensors"
CALIBRATION_BUNDLE = ARTIFACT_DIR / "calibration"
RUST_CLI = next(
    (
        candidate
        for candidate in (
            REPO_ROOT / "target" / "debug" / "hpsi.exe",
            REPO_ROOT / "target" / "debug" / "hpsi",
        )
        if candidate.exists()
    ),
    None,
)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------

def _descriptor(
    tensor_id: int,
    rank: int,
    dtype: int,
    codec: int,
    first_block_index: int,
    dimensions: tuple[int, ...],
    logical_nbytes: int,
    name_ref: int,
) -> hpsi.TensorDescriptor:
    padded = tuple(dimensions) + (0,) * (8 - len(dimensions))
    return hpsi.TensorDescriptor(
        tensor_id=tensor_id,
        rank=rank,
        dtype=dtype,
        codec=codec,
        block_count=1,
        first_block_index=first_block_index,
        logical_nbytes=logical_nbytes,
        dimensions=padded,
        name_ref=name_ref,
        representation_ref=None,
    )


def _container_with_vq(
    codebook: np.ndarray, packed: bytes, cols: int
) -> hpsi.Container:
    """Build an in-memory Container holding one VQ matrix from known bytes."""
    codebook_bytes = np.asarray(codebook, dtype="<f4").tobytes()
    raw = codebook_bytes + packed
    codebook_count = int(np.asarray(codebook).shape[0])
    codebook_tensor = _descriptor(
        0, 3, 5, 1, 0, tuple(np.asarray(codebook).shape), len(codebook_bytes), 0
    )
    indices_tensor = _descriptor(
        1, 1, 10, 2, 1, (len(packed),), len(packed), 1
    )
    blocks = [
        hpsi.BlockDescriptor(
            tensor_id=0,
            block_id=0,
            codec=1,
            logical_offset=0,
            logical_length=len(codebook_bytes),
            payload_offset=0,
            payload_length=len(codebook_bytes),
            auxiliary_ref=None,
            checksum_index=None,
        ),
        hpsi.BlockDescriptor(
            tensor_id=1,
            block_id=0,
            codec=2,
            logical_offset=0,
            logical_length=len(packed),
            payload_offset=len(codebook_bytes),
            payload_length=len(packed),
            auxiliary_ref=None,
            checksum_index=None,
        ),
    ]
    header = hpsi.Header(
        format_major=0,
        format_minor=1,
        flags=0,
        file_size=len(raw),
        metadata=hpsi.Region("metadata", 0, 0),
        tensor_directory_offset=0,
        tensor_count=2,
        block_directory_offset=0,
        block_count=2,
        payload=hpsi.Region("payload", 0, len(raw)),
        integrity_directory_offset=0,
        integrity_count=0,
        artifact_id=b"\x00" * 16,
    )
    return hpsi.Container(
        path=Path("<memory>"),
        raw=raw,
        header=header,
        metadata={"codebook_count": codebook_count},
        tensor_names=["w.hpsi_vq.codebook", "w.hpsi_vq.indices"],
        tensors=[codebook_tensor, indices_tensor],
        blocks=blocks,
        integrity=[],
        tensors_by_name={
            "w.hpsi_vq.codebook": codebook_tensor,
            "w.hpsi_vq.indices": indices_tensor,
        },
    )


def _codebook(codebook_count: int) -> np.ndarray:
    """codebook[stage, code, dim] = 100*stage + 10*code + dim.

    Distinct, exactly representable in f32, so reconstruction is checkable by eye.
    """
    values = np.zeros((codebook_count, hpsi.CODEBOOK_SIZE, hpsi.GROUP_SIZE), dtype=np.float64)
    for stage in range(codebook_count):
        for code in range(hpsi.CODEBOOK_SIZE):
            for dim in range(hpsi.GROUP_SIZE):
                values[stage, code, dim] = 100 * stage + 10 * code + dim
    return values


# ---------------------------------------------------------------------------
# CRC32C.
# ---------------------------------------------------------------------------

def test_crc32c_known_vector():
    # Standard CRC-32C check value for the ASCII string "123456789".
    assert hpsi.crc32c(b"123456789") == 0xE3069283


def test_crc32c_matches_bitwise_reference():
    def reference(data: bytes) -> int:
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ 0x82F63B78 if crc & 1 else crc >> 1
        return crc ^ 0xFFFFFFFF

    for payload in (b"", b"\x00", b"hpsi", bytes(range(256))):
        assert hpsi.crc32c(payload) == reference(payload)


# ---------------------------------------------------------------------------
# Selector unpacking.
# ---------------------------------------------------------------------------

def test_unpack_selectors_low_nibble_first():
    # 0x21 -> low nibble 1 then high nibble 2; 0xF0 -> low 0 then high 15.
    assert list(hpsi.unpack_selectors(b"\x21\xf0", 4)) == [1, 2, 0, 15]


def test_unpack_selectors_rejects_short_stream():
    with pytest.raises(hpsi.HpsiFormatError):
        hpsi.unpack_selectors(b"\x01", 4)


def test_unpack_selectors_odd_count_ignores_high_nibble():
    # One selector needs one byte; only the low nibble is meaningful.
    assert list(hpsi.unpack_selectors(b"\x07", 1)) == [7]


# ---------------------------------------------------------------------------
# VQ reconstruction, checked against hand-computed values.
# ---------------------------------------------------------------------------

def test_reconstruct_matrix_single_codebook_known_answer():
    cols = 16  # two groups of eight
    # Selectors: group 0 -> code 3, group 1 -> code 5.
    packed = bytes([3 | (5 << 4)])
    container = _container_with_vq(_codebook(1), packed, cols)

    dense = hpsi.reconstruct_matrix(
        container,
        container.tensors_by_name["w.hpsi_vq.codebook"],
        container.tensors_by_name["w.hpsi_vq.indices"],
        rows=1,
        cols=cols,
    )
    expected = np.array([30 + dim for dim in range(8)] + [50 + dim for dim in range(8)])
    assert dense.shape == (1, cols)
    np.testing.assert_allclose(dense[0], expected, rtol=0, atol=0)


def test_reconstruct_matrix_two_codebooks_are_additive():
    cols = 16
    # Four selectors for two groups x two stages, low nibble first.
    # group0: stage0=1, stage1=2 ; group1: stage0=4, stage1=7
    selectors = [1, 2, 4, 7]
    packed = bytes([selectors[0] | (selectors[1] << 4), selectors[2] | (selectors[3] << 4)])
    container = _container_with_vq(_codebook(2), packed, cols)

    dense = hpsi.reconstruct_matrix(
        container,
        container.tensors_by_name["w.hpsi_vq.codebook"],
        container.tensors_by_name["w.hpsi_vq.indices"],
        rows=1,
        cols=cols,
    )
    group0 = np.array([10 + dim for dim in range(8)]) + np.array([100 + 20 + dim for dim in range(8)])
    group1 = np.array([40 + dim for dim in range(8)]) + np.array([100 + 70 + dim for dim in range(8)])
    expected = np.concatenate([group0, group1])
    np.testing.assert_allclose(dense[0], expected, rtol=0, atol=0)


def test_reconstruct_matrix_row_major_selector_stride():
    cols = 16
    rows = 2
    # Row 0 -> codes (1, 2); row 1 -> codes (6, 9).
    packed = bytes([1 | (2 << 4), 6 | (9 << 4)])
    container = _container_with_vq(_codebook(1), packed, cols)
    dense = hpsi.reconstruct_matrix(
        container,
        container.tensors_by_name["w.hpsi_vq.codebook"],
        container.tensors_by_name["w.hpsi_vq.indices"],
        rows=rows,
        cols=cols,
    )
    np.testing.assert_allclose(dense[0, :8], [10 + d for d in range(8)], rtol=0, atol=0)
    np.testing.assert_allclose(dense[0, 8:], [20 + d for d in range(8)], rtol=0, atol=0)
    np.testing.assert_allclose(dense[1, :8], [60 + d for d in range(8)], rtol=0, atol=0)
    np.testing.assert_allclose(dense[1, 8:], [90 + d for d in range(8)], rtol=0, atol=0)


def test_reconstruct_matrix_rejects_non_multiple_of_group_size():
    container = _container_with_vq(_codebook(1), b"\x00", cols=12)
    with pytest.raises(hpsi.HpsiFormatError):
        hpsi.reconstruct_matrix(
            container,
            container.tensors_by_name["w.hpsi_vq.codebook"],
            container.tensors_by_name["w.hpsi_vq.indices"],
            rows=1,
            cols=12,
        )


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------

def test_metrics_known_answer():
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    reconstructed = np.array([1.0, 2.0, 5.0, 4.0])
    result = hpsi.metrics(reference, reconstructed)
    assert result["elements"] == 4
    assert result["mae"] == pytest.approx(0.5)
    # A single deviation of 2 over 4 elements: mean square = 4/4 = 1, so RMSE = 1.
    assert result["rmse"] == pytest.approx(1.0)
    assert result["max_abs"] == pytest.approx(2.0)
    assert result["l2_reference"] == pytest.approx(np.sqrt(30.0))
    assert result["l2_reconstructed"] == pytest.approx(np.sqrt(46.0))
    assert result["relative_l2"] == pytest.approx(2.0 / np.sqrt(30.0))


def test_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        hpsi.metrics(np.zeros(3), np.zeros(4))


# ---------------------------------------------------------------------------
# Orientation oracle: an external model definition, never a name pattern.
# ---------------------------------------------------------------------------

def test_orientation_is_never_inferred_from_tensor_names():
    """The Phase 3 name-suffix heuristic must not come back.

    ``c_attn``/``c_proj``/``c_fc`` suffix matching is exactly the rule that
    cannot tell a square transposition apart, so the auditor must not own it.
    """
    assert not hasattr(hpsi, "oracle_orientation")
    assert not hasattr(hpsi, "CONV1D_WEIGHT_SUFFIXES")


@pytest.mark.parametrize(
    "declared, oracle, source_shape, expected",
    [
        ("linear_out_in", "linear_out_in", [16, 32], "architecture_confirmed"),
        ("conv1d_in_out", "conv1d_in_out", [8, 16], "architecture_confirmed"),
        ("linear_out_in", "conv1d_in_out", [16, 32], "architecture_contradiction"),
        ("linear_out_in", None, [16, 32], "declared_not_verified"),
        ("conv1d_in_out", None, [8, 16], "declared_not_verified"),
        ("linear_out_in", None, [16, 16], "square_not_verifiable"),
        ("conv1d_in_out", None, [16, 16], "square_not_verifiable"),
        (None, None, [16, 32], "legacy_undeclared"),
        (None, "conv1d_in_out", [8, 16], "legacy_undeclared"),
        ("linear_out_in", None, [16], "not_a_matrix"),
        ("linear_out_in", "linear_out_in", [2, 3, 4], "not_a_matrix"),
    ],
)
def test_classify_orientation_states(declared, oracle, source_shape, expected):
    assert hpsi.classify_orientation(declared, oracle, source_shape) == expected


def test_classified_confirmation_is_the_only_state_that_claims_it():
    confirmed = [
        declared
        for declared, oracle, shape in [
            ("linear_out_in", "linear_out_in", [16, 32]),
            ("linear_out_in", "conv1d_in_out", [16, 32]),
            ("linear_out_in", None, [16, 32]),
            ("linear_out_in", None, [16, 16]),
            (None, None, [16, 32]),
        ]
        if hpsi.classify_orientation(declared, oracle, shape) == "architecture_confirmed"
    ]
    assert confirmed == ["linear_out_in"]


def _write_manifest(tmp_path: Path, entries: list[dict], **overrides) -> Path:
    document = {
        "format": "hpsi-architecture-manifest",
        "version": 1,
        "model_type": "fixture-model",
        "tensors": entries,
    }
    document.update(overrides)
    path = tmp_path / "architecture.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_architecture_oracle_reads_the_manifest_and_matches_exact_names(tmp_path):
    path = _write_manifest(
        tmp_path,
        [
            {"name": "transformer.h.0.mlp.c_proj.weight", "weight_orientation": "conv1d_in_out"},
            {"name": "transformer.wte.weight", "weight_orientation": "linear_out_in"},
        ],
    )
    oracle = hpsi.ArchitectureOracle.load(path)
    assert oracle.model_type == "fixture-model"
    assert oracle.expectation("transformer.h.0.mlp.c_proj.weight")["weight_orientation"] == (
        "conv1d_in_out"
    )
    # Nothing prefixes, suffixes or otherwise fuzzes into a match.
    assert oracle.expectation("mlp.c_proj.weight") is None
    assert oracle.expectation("transformer.h.1.mlp.c_proj.weight") is None
    assert oracle.expectation("transformer.wte.bias") is None


@pytest.mark.parametrize(
    "entries, message",
    [
        ([{"name": "a.weight"}], "orientation is never inferred from a name"),
        ([{"name": "a.weight", "weight_orientation": "out_in"}], "must be one of"),
        ([{"name": "a.weight", "weight_orientation": "linear_out_in"},
          {"name": "a.weight", "weight_orientation": "conv1d_in_out"}], "declares it twice"),
        ([{"weight_orientation": "linear_out_in"}], "lacks a name"),
        ([{"name": "a.weight", "weight_orientation": "linear_out_in",
           "source_shape": [4]}], "two positive integers"),
    ],
)
def test_architecture_oracle_refuses_incomplete_manifests(tmp_path, entries, message):
    with pytest.raises(hpsi.HpsiFormatError, match=message):
        hpsi.ArchitectureOracle.load(_write_manifest(tmp_path, entries))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"format": "something-else"}, "unsupported architecture manifest format"),
        ({"version": 99}, "unsupported architecture manifest version"),
    ],
)
def test_architecture_oracle_refuses_unknown_manifest_versions(tmp_path, overrides, message):
    with pytest.raises(hpsi.HpsiFormatError, match=message):
        hpsi.ArchitectureOracle.load(_write_manifest(tmp_path, [], **overrides))


# ---------------------------------------------------------------------------
# Roles: a second dimension to orientation (Phase 4.2).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "role, orientation, axis, expected",
    [
        # Projections require exactly their own layout.
        ("linear_projection", "linear_out_in", None, None),
        ("conv1d_projection", "conv1d_in_out", None, None),
        ("linear_projection", "conv1d_in_out", None, "role_orientation_mismatch"),
        ("conv1d_projection", "linear_out_in", None, "role_orientation_mismatch"),
        ("linear_projection", None, None, "role_without_orientation"),
        # An indexed table names its axis; only axis 0 has a producer path in
        # v0.1, and the reason is the payload chain, not the type of the field.
        ("embedding_table", None, 0, None),
        ("embedding_table", "linear_out_in", 0, None),
        ("embedding_table", "conv1d_in_out", 1, "axis_one_unsupported"),
        ("embedding_table", "linear_out_in", 1, "axis_one_unsupported"),
        ("embedding_table", "conv1d_in_out", 0, "table_orientation_mismatch"),
        ("embedding_table", None, None, "table_without_axis"),
        ("embedding_table", None, 5, "axis_value_unsupported"),
        # Roles that claim no projection may not borrow an index axis.
        ("other", None, None, None),
        ("other", "linear_out_in", None, None),
        ("unknown", "linear_out_in", None, None),
        ("other", None, 0, "axis_for_non_indexed_role"),
        ("unknown", None, 0, "axis_for_non_indexed_role"),
        # No role at all: nothing to check, which is what keeps legacy artifacts
        # reading exactly as they did - except an axis with nothing to name.
        (None, "linear_out_in", None, None),
        (None, None, 0, "axis_without_role"),
        ("linear", "linear_out_in", None, "role_token_unknown"),
    ],
)
def test_role_layout_matrix(role, orientation, axis, expected):
    """The metadata slice of the matrix, asserted by *code*, not by prose.

    Codes are what the Rust and Python auditors can be diffed on; reason strings
    may be reworded, the codes are the contract
    (`tests/fixtures/embedding_layout_cases.json`).
    """
    violation = hpsi.entry_layout_violation(
        source_shape=[],
        weight_orientation=orientation,
        tensor_role=role,
        index_axis=axis,
        context="manifest",
    )
    if expected is None:
        assert violation is None, violation
    else:
        assert violation is not None, f"{role}, {orientation}, {axis} was accepted"
        assert violation[0] == expected, violation
    problem = hpsi.role_layout_problem(role, orientation, axis)
    if expected is None:
        assert problem is None
    else:
        assert problem.startswith(f"{expected}: "), problem


@pytest.mark.parametrize(
    "roles, axis, stated, expected",
    [
        ((), None, None, None),
        (("linear_projection",), None, None, "linear_out_in"),
        (("conv1d_projection",), None, None, "conv1d_in_out"),
        (("embedding_table",), 0, None, "linear_out_in"),
        (("embedding_table",), 1, None, "conv1d_in_out"),
        (("embedding_table",), None, None, None),
        (("other",), None, None, None),
        (("unknown",), None, None, None),
        # A stated layout is never overridden by an entailed one.
        (("linear_projection",), None, "linear_out_in", "linear_out_in"),
        # A tied tensor states both functions; the projection names the layout.
        (("conv1d_projection", "embedding_table"), 0, None, "conv1d_in_out"),
    ],
)
def test_entailed_orientation_only_from_roles(roles, axis, stated, expected):
    assert hpsi.entailed_orientation(roles, axis, stated) == expected


@pytest.mark.parametrize(
    "declared_role, oracle_roles, orientation, expected",
    [
        (None, (), "linear_out_in", "declared_not_verified"),
        ("unknown", (), "linear_out_in", "role_unknown"),
        ("unknown", ("unknown",), "linear_out_in", "role_unknown"),
        ("embedding_table", (), "linear_out_in", "not_a_projection"),
        ("other", (), "linear_out_in", "not_a_projection"),
        ("embedding_table", ("other",), "linear_out_in", "role_contradiction"),
        ("embedding_table", ("embedding_table",), None, "not_a_projection"),
        ("linear_projection", ("linear_projection",), "linear_out_in", "declared_not_verified"),
        ("linear_projection", ("linear_projection",), None, "legacy_undeclared"),
    ],
)
def test_orientation_status_respects_roles(
    declared_role, oracle_roles, orientation, expected
):
    status = hpsi.classify_orientation(
        orientation, None, [16, 32], declared_role, oracle_roles
    )
    assert status == expected


def test_a_projection_role_does_not_shield_a_wrong_layout_from_the_oracle():
    """Role confirmation and layout confirmation stay separate verdicts: the
    definition agreeing that a tensor is a Linear layer does not stop its
    recorded layout from being contradicted by the same definition."""
    assert hpsi.classify_orientation(
        "linear_out_in", "conv1d_in_out", [16, 32], "linear_projection", ("linear_projection",)
    ) == "architecture_contradiction"
    assert hpsi.classify_role(
        "linear_projection", ("linear_projection",)
    ) == "role_confirmed"


def test_square_matrix_still_needs_the_oracle_even_with_a_projection_role():
    """A role does not make a square claim verifiable: it says nothing about which
    axis was grouped."""
    assert hpsi.classify_orientation(
        "linear_out_in", None, [16, 16], "linear_projection", ()
    ) == "square_not_verifiable"
    assert hpsi.classify_orientation(
        "linear_out_in", "linear_out_in", [16, 16], "linear_projection", ("linear_projection",)
    ) == "architecture_confirmed"


@pytest.mark.parametrize(
    "declared_role, oracle_roles, expected",
    [
        (None, (), "role_not_declared"),
        (None, ("linear_projection",), "role_not_declared"),
        ("linear_projection", (), "role_declared_not_confirmed"),
        ("linear_projection", ("other",), "role_contradiction"),
        ("embedding_table", ("embedding_table", "linear_projection"), "role_confirmed"),
        ("nonsense", (), "role_invalid"),
    ],
)
def test_role_status_states_its_evidence(declared_role, oracle_roles, expected):
    assert hpsi.classify_role(declared_role, oracle_roles) == expected


def test_manifest_accepts_role_only_entries_and_rejects_contradictions(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        [
            {"name": "embed.table", "tensor_role": "embedding_table", "index_axis": 0},
            {"name": "proj.linear", "tensor_role": ["linear_projection"]},
        ],
    )
    oracle = hpsi.ArchitectureOracle.load(manifest)
    table = oracle.expectation("embed.table")
    assert table["tensor_roles"] == ("embedding_table",)
    assert table["weight_orientation"] is None
    # The layout is entailed by the class, so a consumer can still check it.
    assert table["effective_orientation"] == "linear_out_in"
    assert oracle.expectation("proj.linear")["effective_orientation"] == "linear_out_in"

    contradictory = _write_manifest(
        tmp_path,
        [
            {
                "name": "embed.table",
                "tensor_role": "embedding_table",
                "index_axis": 0,
                "weight_orientation": "conv1d_in_out",
            }
        ],
    )
    with pytest.raises(hpsi.HpsiFormatError, match="indexed on axis 0"):
        hpsi.ArchitectureOracle.load(contradictory)

    silent = _write_manifest(tmp_path, [{"name": "whatever"}])
    with pytest.raises(hpsi.HpsiFormatError, match="states neither"):
        hpsi.ArchitectureOracle.load(silent)


@pytest.mark.skipif(
    not (CALIBRATED.exists() and SOURCE.exists()),
    reason="synthetic GPT-2 artifacts are not present",
)
def test_end_to_end_a_pre_role_container_is_read_unchanged():
    """The additive dimension must not shift a single status of the fixtures."""
    report = hpsi.audit(SOURCE, CALIBRATED, "calibrated")
    records = report["quantized"] + report["raw"]
    assert records, "expected audited tensors"
    for item in records:
        assert item["declared_role"] is None, item["tensor"]
        assert item["role_status"] == "role_not_declared", item["tensor"]
    oracle = report["architecture_oracle"]
    assert oracle["roles_declared"] == 0
    assert oracle["roles_confirmed"] == 0
    # Orientation statuses are the ones Phase 4 reported, before roles existed.
    assert oracle["orientation_statuses"]["square_not_verifiable"] == 1


def test_architecture_manifest_is_built_from_module_classes_not_names(tmp_path):
    """The generator maps the recorded module type, never the tensor name."""
    bundle = tmp_path / "calibration"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "format": "hpsi-transformer-calibration-manifest",
                "version": 1,
                "model_type": "gpt2",
                "layers": [
                    {
                        "module_name": "transformer.h.0.attn.c_proj",
                        "module_type": "conv1d",
                        "source_tensor_name": "transformer.h.0.attn.c_proj.weight",
                    },
                    {
                        "module_name": "lm_head",
                        "module_type": "linear",
                        "source_tensor_name": "lm_head.weight",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = arch.manifest_from_calibration_bundle(bundle)
    assert manifest["format"] == "hpsi-architecture-manifest"
    assert manifest["model_type"] == "gpt2"
    assert manifest["tensors"] == [
        {
            "name": "transformer.h.0.attn.c_proj.weight",
            "weight_orientation": "conv1d_in_out",
            "tensor_role": "conv1d_projection",
        },
        {
            "name": "lm_head.weight",
            "weight_orientation": "linear_out_in",
            "tensor_role": "linear_projection",
        },
    ]
    # The role is named by the class evidence, in the same step as the layout: a
    # tensor called `lm_head.weight` is a projection because the bundle says the
    # module was `linear`, not because of its name.
    assert all(
        entry["tensor_role"] in arch.MODULE_TYPE_ROLES.values() for entry in manifest["tensors"]
    )
    # The generated manifest is readable by the auditor's own parser.
    path = tmp_path / "architecture.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    oracle = hpsi.ArchitectureOracle.load(path)
    assert oracle.expectation("lm_head.weight")["weight_orientation"] == "linear_out_in"


def test_architecture_manifest_refuses_an_unknown_module_type(tmp_path):
    bundle = tmp_path / "calibration"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "format": "hpsi-transformer-calibration-manifest",
                "version": 1,
                "layers": [
                    {
                        "module_type": "embedding",
                        "source_tensor_name": "transformer.wte.weight",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="refusing to guess an orientation"):
        arch.manifest_from_calibration_bundle(bundle)


def test_reconstructed_matrix_is_oriented_by_the_supplied_token():
    """`logical_from_source` takes the orientation as data; it has no name table."""
    stored = np.arange(12.0).reshape(3, 4)
    source = _SourceStub({"a.weight": stored})
    linear, linear_shape = hpsi.logical_from_source(source, "a.weight", "linear_out_in")
    conv, conv_shape = hpsi.logical_from_source(source, "a.weight", "conv1d_in_out")
    assert linear_shape == (3, 4) and conv_shape == (4, 3)
    assert np.array_equal(linear, stored) and np.array_equal(conv, stored.T)
    with pytest.raises(hpsi.HpsiFormatError, match="unknown orientation"):
        hpsi.logical_from_source(source, "c_proj.weight", "out_in")


class _SourceStub:
    """Minimal stand-in for SafeTensorFile used by the orientation helpers."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self._arrays = arrays

    def shape_of(self, name: str) -> tuple[int, ...]:
        return self._arrays[name].shape

    def f32_values(self, name: str) -> np.ndarray:
        return self._arrays[name].astype(np.float64)


# ---------------------------------------------------------------------------
# Header validation.
# ---------------------------------------------------------------------------

def _minimal_header_bytes(size: int = 400) -> bytearray:
    data = bytearray(size)
    data[0:4] = b"HPSI"
    struct.pack_into("<H", data, 4, 0)
    struct.pack_into("<H", data, 6, 1)
    struct.pack_into("<I", data, 8, 128)
    struct.pack_into("<Q", data, 16, size)
    struct.pack_into("<Q", data, 24, 128)
    struct.pack_into("<Q", data, 32, 16)
    struct.pack_into("<Q", data, 40, 144)
    struct.pack_into("<I", data, 48, 1)
    struct.pack_into("<I", data, 52, 128)
    struct.pack_into("<Q", data, 56, 272)
    struct.pack_into("<I", data, 64, 1)
    struct.pack_into("<I", data, 68, 64)
    struct.pack_into("<Q", data, 72, 336)
    struct.pack_into("<Q", data, 80, 64)
    return data


def test_parse_header_accepts_valid_header():
    raw = bytes(_minimal_header_bytes())
    header = hpsi.parse_header(raw)
    assert header.tensor_count == 1
    assert header.block_count == 1
    assert header.payload.length == 64


def test_parse_header_rejects_bad_magic():
    raw = bytearray(_minimal_header_bytes())
    raw[0] = ord("X")
    with pytest.raises(hpsi.HpsiFormatError, match="bad magic"):
        hpsi.parse_header(bytes(raw))


def test_parse_header_rejects_file_size_mismatch():
    raw = bytes(_minimal_header_bytes(400))
    with pytest.raises(hpsi.HpsiFormatError, match="declared file size"):
        hpsi.parse_header(raw + b"\x00")


def test_parse_header_rejects_truncation():
    with pytest.raises(hpsi.HpsiFormatError, match="shorter than"):
        hpsi.parse_header(b"HPSI")


def test_parse_header_rejects_nonzero_reserved_bytes():
    raw = bytearray(_minimal_header_bytes())
    raw[127] = 1
    with pytest.raises(hpsi.HpsiFormatError, match="reserved header bytes"):
        hpsi.parse_header(bytes(raw))


# ---------------------------------------------------------------------------
# End-to-end over the real synthetic artifacts, when present.
# ---------------------------------------------------------------------------

requires_artifacts = pytest.mark.skipif(
    not (CALIBRATED.exists() and SOURCE.exists()),
    reason="synthetic GPT-2 artifacts are not present",
)

requires_bundle = pytest.mark.skipif(
    not (CALIBRATION_BUNDLE / "manifest.json").is_file(),
    reason="activation capture bundle is not present",
)


def _architecture_manifest(tmp_path: Path) -> Path:
    """Model-definition facts, taken from the capture bundle's module classes."""
    manifest = arch.manifest_from_calibration_bundle(CALIBRATION_BUNDLE)
    path = tmp_path / "architecture.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


@requires_artifacts
@requires_bundle
def test_end_to_end_calibrated_container(tmp_path):
    report = hpsi.audit(SOURCE, CALIBRATED, "calibrated", _architecture_manifest(tmp_path))
    assert report["aggregate"]["tensors_quantized"] == 6
    assert report["aggregate"]["tensors_raw"] == 10
    assert report["raw_all_byte_identical"] is True
    assert report["verified_crc32c_records"] == 23
    assert report["integrity_failures"] == []
    # Orientation is declared from calibration and matches the architecture.
    assert report["findings"] == []
    for item in report["quantized"]:
        assert item["quantization_distortion_relative_l2"] > 0.0
        if item["oracle_orientation"] is None:
            # The model definition does not describe this projection, so the
            # auditor must say so instead of confirming it.
            assert item["orientation_matches_oracle"] is None
            assert item["orientation_status"] in ("declared_not_verified", "square_not_verifiable")
        else:
            assert item["orientation_matches_oracle"] is True
            assert item["orientation_status"] == "architecture_confirmed"
    oracle = report["architecture_oracle"]
    assert oracle["supplied"] is True
    assert oracle["architecturally_confirmed"] >= 4
    assert oracle["architectural_contradictions"] == 0


@requires_artifacts
def test_end_to_end_without_a_manifest_nothing_is_confirmed():
    report = hpsi.audit(SOURCE, CALIBRATED, "calibrated")
    oracle = report["architecture_oracle"]
    assert oracle["supplied"] is False
    assert oracle["architecturally_confirmed"] == 0
    for item in report["quantized"]:
        assert item["oracle_orientation"] is None
        assert item["orientation_matches_oracle"] is None
        assert item["architecture_reference_available"] is False
        assert item["orientation_status"] in ("declared_not_verified", "square_not_verifiable")
        # Without the model definition the distortion only proves the container
        # encodes the layout it claims for itself.
        assert item["quantization_distortion_basis"] == "declared_orientation_self_consistency"


@requires_artifacts
@requires_bundle
@pytest.mark.skipif(not UNCALIBRATED.exists(), reason="uncalibrated container is not present")
def test_end_to_end_uncalibrated_container_flags_conv1d_orientation(tmp_path):
    report = hpsi.audit(SOURCE, UNCALIBRATED, "uncalibrated", _architecture_manifest(tmp_path))
    kinds = [finding["kind"] for finding in report["findings"]]
    assert "orientation_disagrees_with_architecture" in kinds
    flagged = {
        finding["tensor"]
        for finding in report["findings"]
        if finding["kind"] == "orientation_disagrees_with_architecture"
    }
    assert flagged == {
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.attn.c_proj.weight",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_proj.weight",
    }
    assert report["architecture_oracle"]["architectural_contradictions"] == len(flagged)
    # The square Conv1D matrix cannot be detected by shape; only the numeric
    # control exposes it. This is the regression that pins that behaviour down.
    assert "reconstruction_matches_transpose_better_than_logical" in kinds


@requires_artifacts
@requires_bundle
@pytest.mark.skipif(not UNCALIBRATED.exists(), reason="uncalibrated container is not present")
def test_square_conv1d_normalized_distortion_uses_declared_orientation(tmp_path):
    """Regression for the square-matrix normalization rule.

    For a square matrix the logical shape matches both the architectural matrix
    and its transpose, so shape cannot disambiguate. The normalized distortion
    must therefore be measured against the layout implied by the *declared*
    orientation, not against whichever reference fits better.
    """
    report = hpsi.audit(SOURCE, UNCALIBRATED, "uncalibrated", _architecture_manifest(tmp_path))
    by_name = {item["tensor"]: item for item in report["quantized"]}
    square = by_name["transformer.h.0.attn.c_proj.weight"]
    direct = square["metrics_vs_oracle_logical"]["relative_l2"]
    transposed = square["metrics_vs_oracle_transposed"]["relative_l2"]
    normalized = square["quantization_distortion_relative_l2"]
    # The declared orientation is wrong for this tensor, so the container's
    # layout is the transposed one.
    assert transposed < direct
    assert normalized == pytest.approx(transposed)
    assert square["orientation_status"] == "architecture_contradiction"


@requires_artifacts
@requires_bundle
@pytest.mark.skipif(RUST_CLI is None, reason="the Rust CLI has not been built")
@pytest.mark.skipif(not UNCALIBRATED.exists(), reason="uncalibrated container is not present")
@pytest.mark.parametrize("container", [CALIBRATED, UNCALIBRATED])
def test_rust_and_python_reach_the_same_orientation_conclusions(tmp_path, container):
    """The two implementations must not disagree about the same artifact.

    They share no code: the Rust audit lives in ``crates/hpsi-ops`` and this
    decoder is a second implementation. Agreement here is the point of having
    both, so the comparison uses the counts each one reports, not a re-run of
    either decision function.
    """
    manifest = _architecture_manifest(tmp_path)
    python_report = hpsi.audit(SOURCE, container, "cross-check", manifest)
    completed = subprocess.run(
        [str(RUST_CLI), "inspect", str(container), "--architecture", str(manifest)],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("orientation summary:")
    )
    rust = {
        part.split("=")[0]: int(part.split("=")[1])
        for part in summary.removeprefix("orientation summary:").split()
    }
    statuses = python_report["architecture_oracle"]["orientation_statuses"]
    assert rust["confirmed"] == statuses.get("architecture_confirmed", 0)
    assert rust["declared_not_verified"] == statuses.get("declared_not_verified", 0)
    assert rust["square_not_verifiable"] == statuses.get("square_not_verifiable", 0)
    assert rust["legacy_undeclared"] == statuses.get("legacy_undeclared", 0)
    rust_flagged = {
        line.split("orientation finding: ")[1].split(":")[0]
        for line in completed.stdout.splitlines()
        if line.startswith("orientation finding: ")
    }
    python_flagged = {
        finding["tensor"]
        for finding in python_report["findings"]
        if finding["kind"] == "orientation_disagrees_with_architecture"
    }
    assert rust_flagged == python_flagged
    # Only matrices have orientation semantics: Rust keeps non-rank-2 tensors
    # out of the audit entirely, Python counts them as `not_a_matrix`. Rust
    # reports a contradicted matrix as a finding instead of as a status, so the
    # two counts add up to the same population the Python auditor classifies.
    assert rust["rank2"] + len(rust_flagged) == sum(
        count for status, count in statuses.items() if status != "not_a_matrix"
    )
    assert len(rust_flagged) == python_report["architecture_oracle"]["architectural_contradictions"]


@requires_artifacts
def test_end_to_end_raw_bytes_are_identical_and_typed():
    report = hpsi.audit(SOURCE, CALIBRATED, "calibrated")
    assert report["raw"], "expected preserved RAW tensors"
    for item in report["raw"]:
        assert item["byte_identical"] is True
        assert item["dtype_matches"] is True
        assert item["shape_matches"] is True
        assert item["size_matches"] is True
        assert item["crc32c_original"] == item["crc32c_recovered"]
