# Feature 002 plan

1. Establish a deterministic 8-value / 16-centroid vector-quantization baseline.
2. Implement a borrowed runtime view that consumes 4-bit selectors directly.
3. Export codebook and selector tensors into HPSI and prove readback execution.
4. Add held-out numerical evaluation and storage accounting.
5. Add reproducible CPU benchmarks before optimizing kernels.
6. Only after measurement, evaluate residual/additive codebooks and row-streamed block execution.
