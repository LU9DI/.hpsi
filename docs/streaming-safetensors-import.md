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
