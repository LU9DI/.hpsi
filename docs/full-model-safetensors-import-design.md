# Full-model Safetensors import: design and invariants

Status: the multi-tensor import pipeline described here is implemented in the current HPSI ops crate. This document records its present contract and the remaining boundaries; it is not a claim of general model-runtime compatibility.

## Public API

The public API includes a compatibility-preserving legacy path and a provenance-aware path:

- `ModelImportPlan::build(source, calibrations, policy)` creates a deterministic plan covering every source tensor.
- `write_model_import_to_hpsi(source, plan, output, config, block_rows)` writes one multi-tensor HPSI container to a seekable output.
- `write_model_import_to_hpsi_path(source, plan, destination, config, block_rows)` writes, reopens, verifies, and publishes a filesystem artifact without replacing an existing destination.
- `ModelImportPlan::build_with_provenance(source, calibrations, policy)` returns a `CalibratedModelImportPlan` that pairs the legacy plan with compact provenance records without duplicating the activation moment vectors.
- `write_model_import_to_hpsi_calibrated` and `write_model_import_to_hpsi_path_calibrated` persist that provenance. The path variant retains reopen, integrity verification, and no-replace publication.

The original plan and writer APIs remain available and retain their previous metadata behavior.

The single-tensor `SafeTensorFile::encode_f32_tensor_to_hpsi` convenience API remains available. It is not necessary to invoke it repeatedly to create a multi-tensor artifact; the model importer owns that orchestration.

## Current import contract

Planning sorts source tensors by exact name and requires complete, unique source coverage. HPSI import currently accepts non-scalar tensors of rank 1 through 8 with non-zero dimensions. Safetensors scalar and zero-sized tensors are rejected because the current HPSI tensor descriptor/import contract does not represent them.

The current VQ-quantizable set is deliberately narrow:

- rank-2 tensors;
- source dtype `F32`, `F16`, or `BF16`;
- feature dimension divisible by the VQ group size;
- for the selected VQ configuration, the number of packed selectors per row (`(features / group_size) * codebook_count`) must be even so each independently emitted row is byte-aligned. The writer validates this before encoding.

All other eligible-to-preserve tensors follow the explicit policy:

- `UnsupportedTensorPolicy::Reject` fails planning if any source tensor cannot be VQ-quantized.
- `UnsupportedTensorPolicy::AllowRawPreservation` preserves each non-quantized tensor's original payload bytes without decoding, casting, or omitting it.

This policy name does not mean every raw-preserved tensor is an unsupported dtype: rank, shape, or dtype can make a tensor ineligible for VQ.

## Dtype and provenance mapping

Dtypes with native HPSI representations retain their native HPSI dtype: `U8`, `I8`, `I16`, `F16`, `BF16`, `I32`, `F32`, `I64`, and `F64`.

Safetensors types without a corresponding native HPSI scalar dtype are stored as `OPAQUE` byte streams. This currently includes `BOOL`, `F4`, `F6_E2M3`, `F6_E3M2`, the supported FP8 variants, `U16`, `U32`, `U64`, and `C64`. Their bytes are preserved exactly; this does not provide native arithmetic or semantic decoding for those source types.

For an `OPAQUE` output tensor, the HPSI dimensions describe the preserved byte stream (currently a one-dimensional byte length). The source dtype and original shape remain available in `metadata.user.source_tensors` as `source_dtype` and `source_shape`. Native HPSI dtypes retain the original source dimensions. Each provenance entry also records the source name and disposition (`quantize_vq` or `preserve_raw`).

The format contract defines `OPAQUE` as a byte stream without dimension-derived element sizing. It may be used for lossless source-format preservation when descriptive metadata records the external type and shape. Consumers must not infer numeric semantics from `OPAQUE`.

## Streaming and publication

The importer stages intermediate payloads in a managed disk-backed spool, builds the complete tensor and block descriptors, then calls the streaming HPSI writer once. Source reads and spool transfers use bounded buffers; the full dense source model is not materialized in RAM by this orchestration. VQ training samples, one source row, codebooks, directory metadata, and bounded transfer buffers still consume memory.

The path API creates temporary files with exclusive creation, verifies the completed HPSI artifact by reopening it and running `ModelReader::verify_all`, then publishes without overwriting an existing destination. Publication uses a same-directory hard link where supported. A failure from the generic `Write + Seek` API may leave a partial artifact in the caller's output; callers requiring all-or-nothing filesystem publication should use the path API.

## Calibration

Calibration is matched by exact source tensor name and feature count. Duplicate or unknown calibration names, invalid dimensions, non-finite or negative moments, and calibration for non-quantizable tensors are rejected. The calibration moments influence VQ encoding.

The provenance-aware plan and writers persist a `calibration` object on each calibrated source-tensor entry in `metadata.user.source_tensors`. It contains `model_type`, optional `model_revision`, `dataset_sha256`, `sample_count`, `features`, and `statistic` (currently `diagonal_second_moment`). Calibration moments are intentionally not copied into the manifest: they are retained in the plan only as the numeric input to encoding. Raw-preserved tensors never receive a calibration object. The legacy writer APIs continue to omit this object, preserving their existing behavior.

## Validation coverage and remaining limits

The implementation includes tests for deterministic planning, complete source coverage, name collisions, mixed VQ/raw tensors, calibration validation, multi-tensor output, raw payload preservation across all 22 currently supported Safetensors dtypes, HPSI reopening, and integrity verification. CI checks formatting, Clippy, and tests on Ubuntu, Windows, and macOS.

These tests establish implementation behavior for their fixtures; they do not establish compatibility with every model architecture or production-scale memory/performance claims. Representative-model testing, model-specific tensor eligibility policies, and broader native dtype support remain future work.