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

The result argues against treating 4-bit selectors per 8-weight group as a final representation. The next experiment compared additive/residual codebooks and expanded the sweep to one through four codebooks (4, 8, 12, and 16 selector bits per 8-weight group). The updated benchmark prints artifact size, held-out error, encode time, and p50/p95 latency for each rate. New measurements are pending CI; no quality or speedup conclusion is made for three- and four-codebook modes yet.


## Additive residual-codebook comparison — CI run #53

Run: [GitHub Actions #53](https://github.com/LU9DI/.hpsi/actions/runs/41370076016)

Both modes used the same deterministic 256 × 256 synthetic matrix and the same 16 held-out input vectors.

| Metric | Single codebook | Additive (2 codebooks) |
|---|---:|---:|
| Selector bits per 8-weight group | 4 | 8 |
| Encode time | 6.848 ms | 13.828 ms |
| Codebook + selector bytes | 4,608 | 9,216 |
| Complete HPSI artifact | 5,312 bytes | 9,920 bytes |
| Artifact compression vs. dense F32 | 49.35× | 26.43× |
| Held-out relative L2 error | 0.763030 | 0.589400 |
| Held-out maximum absolute error | 1.384189 | 1.202029 |
| Direct matvec p50 | 0.048693 ms | 0.098207 ms |
| Direct matvec p95 | 0.048833 ms | 0.104376 ms |

Dense F32 reference in the same run: p50 **0.043665 ms**, p95 **0.049184 ms**. The additive mode reduces relative L2 error by approximately **22.7%** versus single-codebook mode, at the cost of roughly halving the compression ratio and more than doubling direct-matvec p50 latency versus dense F32.

The additive codebook reserves a zero residual centroid, so its weight-domain per-group reconstruction error cannot exceed the single-codebook residual error. The held-out matrix-vector error also improved in this sample, but remains large (0.589 relative L2). The experiment therefore supports further research into residual/additive representations, not a claim of acceptable neural-network quality.


## Multi-stage residual-codebook sweep

The encoder now supports one through four additive codebooks. Each residual stage is trained against the residual left by all preceding stages and reserves a zero centroid. Selectors are nibble-packed at four bits each; the row-streamed HPSI path reads only blocks intersecting the requested output rows.

The deterministic benchmark reports all four rates on the same matrix and held-out inputs. It measures a 32-row window through `ModelReader` over a `MemorySource` using a prepared `HpsiVqExecutor`: the shared codebook is loaded once at executor construction, then each timed operation reads only the selected index block(s) and computes only the requested rows. Initialization latency is reported separately from repeated row-window p50/p95 and payload bytes per window. This isolates container parsing, block selection, and direct computation without claiming physical-storage latency. The in-memory VQ operator, prepared HPSI row-window path, and dense F32 reference are reported separately.

### Recorded four-rate, prepared-executor, and dot-product-reuse sweep — CI run #87

Run: [GitHub Actions #87](https://github.com/LU9DI/.hpsi/actions/runs/41372570663)

All four modes used the same deterministic synthetic 256 × 256 F32 matrix and the same 16 held-out input vectors. Latency uses 100 repetitions per path. The CPU kernel now precomputes each input-group/codebook-centroid dot product once and reuses it across output rows. The HPSI path uses a prepared `HpsiVqExecutor`: codebook initialization is measured separately, and repeated row-window calls reuse that codebook.

| Metric | Dense F32 | 1 codebook | 2 codebooks | 3 codebooks | 4 codebooks |
|---|---:|---:|---:|---:|---:|
| Selector bits per 8-weight group | — | 4 | 8 | 12 | 16 |
| Encode time (ms) | — | 7.461 | 14.775 | 22.375 | 30.642 |
| Complete HPSI artifact (bytes) | 262,144* | 5,984 | 10,592 | 15,200 | 19,808 |
| Compression vs. dense F32 | 1× | 43.81× | 24.75× | 17.25× | 13.23× |
| Held-out relative L2 error | 0 | 0.763030 | 0.589400 | 0.448215 | 0.346410 |
| Held-out max absolute error | 0 | 1.384189 | 1.202029 | 0.836721 | 0.701543 |
| In-memory direct matvec p50 (ms) | 0.029566 | 0.014797 | 0.024744 | 0.035724 | 0.046901 |
| In-memory direct matvec p95 (ms) | 0.032205 | 0.015151 | 0.024786 | 0.035932 | 0.052354 |

*Dense F32 tensor size, not a container artifact.

The HPSI prepared row-window path was measured separately for a 32-row window (one eighth of the matrix rows), using `ModelReader` over `MemorySource`:

| Metric | 1 codebook | 2 codebooks | 3 codebooks | 4 codebooks |
|---|---:|---:|---:|---:|
| Executor initialization (ms) | 0.001907 | 0.000807 | 0.001062 | 0.001188 |
| HPSI row-window p50 (ms) | 0.003559 | 0.006474 | 0.009436 | 0.012467 |
| HPSI row-window p95 (ms) | 0.003633 | 0.006591 | 0.009610 | 0.012596 |
| Codebook payload loaded at initialization (bytes) | 512 | 1,024 | 1,536 | 2,048 |
| Index payload bytes read per 32-row window | 512 | 1,024 | 1,536 | 2,048 |

The executor reads the codebook once during initialization. Each timed row-window operation reads only the selector payload for the selected 32 rows; it does not reload the codebook. These row-window timings are not directly comparable to the full 256-row dense matvec: the HPSI operation computes only 32 output rows. The source is in-memory, not physical storage.

**Interpretation:** reusing centroid dot products materially changes the CPU result. In this runner sample, one codebook is about 2.00× faster at p50 than dense F32, and two codebooks are about 1.19× faster. Three codebooks are about 1.21× slower, and four are about 1.59× slower. The best storage rate (43.81×) still has high held-out error (0.763); the best error (0.346) still costs 13.23× artifact compression. This is a useful algorithmic improvement, not yet evidence of a general speedup: the sample is a synthetic 256 × 256 matrix on a GitHub-hosted runner whose exact CPU model and power state were not recorded. A pinned local CPU run, larger transformer-shaped matrices, and file-backed I/O measurements remain necessary.
