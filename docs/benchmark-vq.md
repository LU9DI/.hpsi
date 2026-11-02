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

### Recorded four-rate, prepared-executor, dot-product-reuse, and row-scaling sweep — CI run #93

Run: [GitHub Actions #87](https://github.com/LU9DI/.hpsi/actions/runs/41372889727)

All four modes used the same deterministic synthetic 256 × 256 F32 matrix and the same 16 held-out input vectors. Latency uses 100 repetitions per path. The CPU kernel now precomputes each input-group/codebook-centroid dot product once and reuses it across output rows. The HPSI path uses a prepared `HpsiVqExecutor`: codebook initialization is measured separately, and repeated row-window calls reuse that codebook.

| Metric | Dense F32 | 1 codebook | 2 codebooks | 3 codebooks | 4 codebooks |
|---|---:|---:|---:|---:|---:|
| Selector bits per 8-weight group | — | 4 | 8 | 12 | 16 |
| Encode time (ms) | — | 5.068 | 10.166 | 15.352 | 20.969 |
| Complete HPSI artifact (bytes) | 262,144* | 5,984 | 10,592 | 15,200 | 19,808 |
| Compression vs. dense F32 | 1× | 43.81× | 24.75× | 17.25× | 13.23× |
| Held-out relative L2 error | 0 | 0.763030 | 0.589400 | 0.448215 | 0.346410 |
| Held-out max absolute error | 0 | 1.384189 | 1.202029 | 0.836721 | 0.701543 |
| In-memory direct matvec p50 (ms) | 0.023756 | 0.009114 | 0.018047 | 0.026830 | 0.037306 |
| In-memory direct matvec p95 (ms) | 0.023786 | 0.009134 | 0.018087 | 0.026941 | 0.037376 |

*Dense F32 tensor size, not a container artifact.

The HPSI prepared row-window path was measured separately for a 32-row window (one eighth of the matrix rows), using `ModelReader` over `MemorySource`:

| Metric | 1 codebook | 2 codebooks | 3 codebooks | 4 codebooks |
|---|---:|---:|---:|---:|
| Executor initialization (ms) | 0.001492 | 0.000631 | 0.000560 | 0.000781 |
| HPSI row-window p50 (ms) | 0.002313 | 0.004367 | 0.006470 | 0.008803 |
| HPSI row-window p95 (ms) | 0.002323 | 0.004377 | 0.006490 | 0.008864 |
| Codebook payload loaded at initialization (bytes) | 512 | 1,024 | 1,536 | 2,048 |
| Index payload bytes read per 32-row window | 512 | 1,024 | 1,536 | 2,048 |

The executor reads the codebook once during initialization. Each timed row-window operation reads only the selector payload for the selected 32 rows; it does not reload the codebook. These row-window timings are not directly comparable to the full 256-row dense matvec: the HPSI operation computes only 32 output rows. The source is in-memory, not physical storage.

**Interpretation:** reusing centroid dot products materially changes the CPU result. In this CI sample, one codebook is about 2.60× faster at p50 than dense F32, two codebooks are about 1.32× faster, while three and four codebooks are about 1.13× and 1.57× slower. The best storage rate (43.81×) still has high held-out error (0.763); the best error (0.346) still costs 13.23× artifact compression.

### Row-count crossover — CI run #93

The same optimized kernel was measured at four row counts with 256 columns, using a separate deterministic synthetic matrix per shape. This helps expose when codebook-dot reuse amortizes its fixed per-input work:

| Rows × 256 | Dense F32 p50 (ms) | 1 codebook p50 (ms) | 4 codebooks p50 (ms) |
|---|---:|---:|---:|
| 16 × 256 | 0.001843 | 0.001743 | 0.006530 |
| 64 × 256 | 0.006320 | 0.003205 | 0.012649 |
| 256 × 256 | 0.025108 | 0.008994 | 0.036965 |
| 1,024 × 256 | 0.099743 | 0.032180 | 0.134677 |

The one-codebook mode improves p50 by roughly 1.06×, 1.97×, 2.79×, and 3.10× as rows increase. Four-codebook mode is slower at all four shapes in this sample. This shows that the current dot-table kernel benefits most from low-rate codebooks and larger row counts; it does **not** show that all VQ rates outperform dense F32. The CI runner's exact CPU model and power state were not recorded, so these timings are indicative only. Pinned local CPU runs and larger transformer-shaped matrices remain necessary.

### File-backed row-window reads — CI run #97

Run: [GitHub Actions #97](https://github.com/LU9DI/.hpsi/actions/runs/41373098140)

The benchmark writes each HPSI artifact to a temporary file, opens it through `FileSource` (OS random-access reads), prepares the executor, then measures repeated 32-row windows. The process does not drop the OS page cache, so these are **warm page-cache file-backed** timings, not cold physical-disk latency.

| Metric | 1 codebook | 2 codebooks | 3 codebooks | 4 codebooks |
|---|---:|---:|---:|---:|
| Executor initialization p50 (ms) | 0.001914 | 0.001503 | 0.001693 | 0.001744 |
| File-backed row-window p50 (ms) | 0.004357 | 0.007294 | 0.010559 | 0.013685 |
| File-backed row-window p95 (ms) | 0.004499 | 0.007384 | 0.010860 | 0.013765 |
| Index payload bytes per 32-row window | 512 | 1,024 | 1,536 | 2,048 |

The file-backed path returns the same output as the in-memory reference and reads only the index block selected for the requested rows after the codebook has been loaded. This validates the random-access file path end to end, but does not establish cold-storage performance or large-model throughput.
