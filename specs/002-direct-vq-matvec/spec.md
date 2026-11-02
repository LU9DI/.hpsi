# Feature 002 — Direct vector-quantized matrix-vector inference

## Problem

The container reader and RAW writer establish selective access to HPSI blocks, but no neural-network operation yet executes on a compressed representation. This feature introduces the first measured compute primitive without reconstructing a dense weight matrix.

## Chosen experimental representation

- Matrix weights are grouped into vectors of eight adjacent F32 values.
- A shared codebook contains 16 vectors of eight F32 values.
- Each weight group is represented by one 4-bit codebook selector; two selectors are packed per byte.
- The codebook is fitted deterministically with bounded-sample k-means.
- The primary operation is y = W_hat x, where each output accumulates dot products between an input group and the selected codebook vector.
- Runtime multiplication reads packed selectors and codebook values directly. It must not allocate or reconstruct a dense W_hat.

This is a research baseline, not a quality guarantee. The theoretical selector cost is 0.5 bits per weight before the shared codebook, but actual quality and end-to-end storage/latency must be measured. No claim is made that it preserves a production model's accuracy.

## HPSI representation

The initial exporter emits two tensors:
1. hpsi.vq.codebook: F32 tensor with shape [16, 8].
2. hpsi.vq.indices: Opaque byte tensor containing packed 4-bit selectors.

Both tensors are stored through the existing HPSI writer and can be loaded with selective block reads. The first implementation keeps the selected codebook and packed index stream in memory for a matrix-vector call; row-streamed execution is a later optimization.

## Acceptance criteria

- Encoding rejects invalid shapes, non-finite values, and invalid training configuration.
- Packed indices are deterministic and have a documented nibble order.
- VqMatrixView::matvec computes directly from selectors and codebook values.
- HPSI export and readback reproduce the same direct result as the in-memory representation.
- Tests report complete representation bytes, not just index bytes.
- Numerical quality is evaluated against dense F32 on held-out input vectors before any quality claims are made.
- Benchmarks report encode time, direct matvec latency, dense baseline latency, stored bytes, and numerical error.

## Non-goals for this first experiment

- No GPU kernel.
- No production model-quality guarantee.
- No general tensor codec framework.
- No automatic claim of 40:1 compression for arbitrary models.
