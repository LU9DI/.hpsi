# Feature 002 tasks

- [x] T001 Define the experimental VQ representation and bit packing.
- [x] T002 Implement deterministic bounded-sample codebook fitting.
- [x] T003 Implement direct matvec over packed selectors without dense reconstruction.
- [x] T004 Export codebook and selector tensors to HPSI.
- [x] T005 Test HPSI readback and direct computation parity.
- [x] T006 Add held-out numerical quality report against dense F32.
- [x] T007 Add reproducible CPU benchmark for encode and matvec.
- [x] T008 Add row-streamed execution over independently addressable HPSI blocks.
- [x] T009 Evaluate residual/additive quantization against the single-codebook baseline.
- [x] T010 Extend the CPU experiment to one through four additive residual codebooks and test monotonic weight-domain error.
- [x] T011 Record the four-rate CI benchmark and separate prepared-executor initialization from repeated HPSI row-window latency; pinned local CPU and file-backed storage measurements remain open.
- [x] T012 Add a prepared HPSI VQ row executor that caches the shared codebook, validates a contiguous row-aligned block map once, and reads/computes only the requested row subranges on repeated calls.
- [x] T013 Test partial row windows, repeated calls without codebook reload, and windows crossing index-block boundaries.
- [x] T014 Precompute input-group/codebook-centroid dot products and reuse them across output rows; validate the optimized kernel against an independent scalar reference for all four rates.
- [x] T015 Benchmark the optimized kernel across 16, 64, 256, and 1,024 output rows at 256 columns; record the row-count crossover and rate-dependent tradeoffs.
- [x] T016 Add a `FileSource` benchmark using real temporary HPSI files and warm page-cache random-access reads; verify selected-row output parity and report index bytes read.
- [ ] T017 Run pinned local CPU benchmarks on transformer-shaped matrices and cold/warm file-backed HPSI artifacts.
- [ ] T018 Add a GPU kernel only after pinned CPU profiling and storage measurements.

- [x] T019 Evaluate beam-search additive assignment across widths 1, 2, 4, 8, and 16; select default width 4 based on the best held-out error in the recorded sweep and encode-time tradeoff.
- [ ] T020 Repeat beam-width sweeps across multiple seeds and real model weight tensors; evaluate activation-weighted layer output error and downstream model quality.

- [x] T021 Add activation-aware VQ fitting using diagonal activation second moments, validate malformed calibration vectors, and benchmark weighted error at unchanged representation size on deterministic synthetic activations.
- [ ] T022 Capture calibration activations from real transformer layers; evaluate layer-output error, perplexity/task quality, and per-layer importance strategies beyond diagonal shared-position averages.
