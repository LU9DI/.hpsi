# Streaming Safetensors to VQ

The HPSI ops crate exposes bounded Safetensors reads and a two-pass VQ encoder.
Together, these APIs let callers quantize a rank-1 or rank-2 floating tensor
without materializing its dense payload in memory.

## Read bounded windows

`SafeTensorFile::read_f32_elements(name, first_element, element_count)`
reads a contiguous element range from F32, F16, or BF16 tensors and returns
F32 values. It checks the requested range, uses checked offset arithmetic,
and rejects non-finite values. The allocation is proportional to the requested
window, not the complete tensor.

`read_f32_rows` is the rank-2 row-window convenience API and delegates to the
same implementation.

## Two-pass encoding

`VqMatrix::encode_from_reader(rows, cols, config, read_elements)` accepts a
random-access closure with the signature:

```rust
FnMut(u64, u64) -> Result<Vec<f32>, VqError>
```

The closure receives a first-element offset and element count. The encoder:

1. Selects deterministic, evenly spaced 8-element groups and reads them to
   train the additive codebooks.
2. Reads the source again, one matrix row at a time, and writes each row's
   packed selectors into the final compressed selector buffer.

The source-side dense working set is therefore bounded by one row, plus the
training sample set and codebooks. The packed selector output and codebooks
remain resident in memory in this API.

For calibrated encoding, use
`VqMatrix::encode_from_reader_activation_aware`. It applies the same
per-position activation second-moment scaling used by the in-memory
activation-aware encoder and restores the codebook scale before returning.

## Safetensors example

```rust
use hpsi_ops::{SafeTensorFile, VqConfig, VqMatrix};

let matrix = VqMatrix::encode_from_reader(
    rows,
    cols,
    VqConfig::default(),
    |first, count| {
        safetensors
            .read_f32_elements("model.layers.0.self_attn.q_proj.weight", first, count)
            .map_err(|_| hpsi_ops::VqError("Safetensors window read failed"))
    },
)?;
```

The caller should derive `rows` and `cols` from the selected tensor's
validated `SafeTensorInfo::shape`. The column count must be divisible by the
8-element VQ group size, and each encoded row must have a byte-aligned selector
count for independently addressable HPSI row blocks.

## Current boundary

This encoder removes the full dense source matrix from memory, but it still
retains the compressed selector stream. The existing `ModelWriter::write`
API accepts in-memory `TensorInput` payload blocks. A separate seekable,
streaming writer is still required to stream selectors directly to the HPSI
artifact without retaining the complete compressed index tensor.


## Streaming HPSI writer

The format crate now exposes `StreamingTensorInput` and
`ModelWriter::write_streaming`. Supply tensor metadata and the exact length
of each payload block up front; a callback then writes each block directly to
the seekable output. The writer plans offsets before emission and computes
CRC32C incrementally as bytes pass through. It patches each integrity record
after the corresponding block has been written, without buffering the block.

For filesystem output, `ModelWriter::write_path_streaming` writes to a
same-directory temporary file, syncs it, reopens and verifies the complete
artifact, and publishes it with a non-overwriting hard link. The callback
must write exactly the declared block length. The older in-memory writer API
remains available and unchanged.


## Incremental VQ row emission and HPSI export

`VqMatrix::encode_from_reader_to` now emits one byte-aligned selector row
at a time through a callback and returns only the trained codebook. The
activation-aware counterpart applies the same group-position scaling as the
buffered encoder while preserving the row-at-a-time output contract.

`VqMatrix::encode_from_reader_to_hpsi` connects this encoder to
`ModelWriter::write_streaming`. It writes selector rows into caller-provided
seekable spool storage, then streams the codebook and bounded selector blocks
into the HPSI container. This avoids retaining the complete packed selector
matrix in RAM. The spool is caller-owned so applications can choose an in-memory
cursor for small inputs or a temporary file for large models. Dense source
weights are still read through bounded windows.


## Direct Safetensors-to-HPSI import

`SafeTensorFile::encode_f32_tensor_to_hpsi` now connects a rank-2
`F32`, `F16`, or `BF16` Safetensors tensor directly to the VQ encoder
and HPSI streaming writer. The caller supplies the tensor name, VQ settings,
and selector block row count; no custom reader closure or spool management
is required.

The convenience path creates a unique disk-backed selector spool with
exclusive file creation and removes it after use. The spool is explicitly
closed before cleanup for Windows compatibility. Source windows remain
bounded, selector transfer uses a 64 KiB buffer, and the HPSI writer rejects
callbacks that attempt to write beyond a block's declared length.

The convenience API above remains a single-tensor path. Whole-model conversion is provided separately by the multi-tensor importer described below.

## Lossless raw payload preservation

`SafeTensorFile::copy_raw_payload(name, output, chunk_bytes)` copies a
validated tensor's original byte range to any `Write` sink without decoding,
casting, or changing its source dtype. Reads are performed in chunks no larger
than `chunk_bytes` and is additionally capped at 64 KiB even if a caller
requests a larger buffer; the method returns the exact number of bytes copied
and rejects a zero chunk size. This is the low-level path required for tensors that
the VQ importer does not quantize (for example, biases, normalization
parameters, integer tensors, and embeddings).

The API deliberately copies payload bytes only. The multi-tensor orchestration is implemented by the public `ModelImportPlan`, `write_model_import_to_hpsi`, and `write_model_import_to_hpsi_path` APIs. Planning covers every source tensor in deterministic name order. Rank-2 F32/F16/BF16 tensors with a feature dimension divisible by the VQ group size are quantized; other non-scalar, non-empty tensors are either rejected by `UnsupportedTensorPolicy::Reject` or copied losslessly with `AllowRawPreservation`. The writer stages payloads on disk, builds one HPSI container, and records each source tensor's name, original dtype, original shape, and import disposition in `metadata.user.source_tensors`. Dtypes without a native HPSI scalar representation use `OPAQUE`; this preserves bytes, not native numeric semantics. For filesystem output, the path API reopens and verifies the artifact before no-replace publication. The generic seekable-writer API may leave partial output on error.

For persisted calibration provenance, use `ModelImportPlan::build_with_provenance` followed by `write_model_import_to_hpsi_calibrated` or `write_model_import_to_hpsi_path_calibrated`. Each calibrated source tensor receives a compact `calibration` object with model type, optional revision, dataset SHA-256, sample count, feature count, and statistic identifier. The numeric moment vector is not copied into metadata. Raw-preserved tensors have no calibration object. The original `build` and writer APIs remain available and do not persist calibration provenance.
