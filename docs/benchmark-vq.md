# Direct VQ CPU benchmark

Run the deterministic synthetic baseline in release mode:

```sh
cargo run -p hpsi-ops --example vq_benchmark --release
```

The report includes:
- encode time;
- dense F32 bytes, VQ codebook-plus-index bytes, and complete HPSI artifact bytes;
- artifact compression ratio including HPSI metadata and directories;
- relative L2 and maximum absolute output error on 16 deterministic held-out input vectors;
- p50/p95 dense and direct-VQ matvec latency over 100 repetitions.

This is a microbenchmark on a synthetic 256 × 256 matrix. It is not representative of a production transformer layer and must not be used to claim model-quality preservation. Run on an otherwise idle machine and record CPU, OS, compiler, and power settings when comparing results.


## Recorded CI sample — 2026-11-02

Run: [GitHub Actions #51](https://github.com/LU9DI/.hpsi/actions/runs/41369735956)

| Metric | Result |
|---|---:|
| Matrix | 256 × 256 |
| Dense F32 tensor | 262,144 bytes |
| VQ codebook + packed selectors | 4,608 bytes |
| Complete HPSI artifact | 5,312 bytes |
| Artifact compression vs. dense F32 | 49.35× |
| Codebook encode time | 7.519 ms |
| Held-out relative L2 error | 0.763030 |
| Held-out maximum absolute error | 1.384189 |
| Dense matvec p50 / p95 | 0.030270 / 0.030525 ms |
| Direct VQ matvec p50 / p95 | 0.039741 / 0.041976 ms |
| Held-out input vectors | 16 |
| Latency repetitions per path | 100 |

**Interpretation:** the 49.35× figure includes the HPSI header, metadata, directories, codebook, and packed selectors. Despite that storage result, the relative L2 error is high and the direct CPU kernel is approximately 31% slower at p50 than the dense F32 reference in this run. The benchmark ran on a GitHub-hosted Ubuntu runner; its exact CPU model and power state were not recorded, so the latency numbers are illustrative rather than portable performance claims.

The result argues against treating 4-bit selectors per 8-weight group as a final representation. The next experiment should compare additive/residual codebooks (two 4-bit selectors per group), then measure quality and latency again before considering GPU work.
