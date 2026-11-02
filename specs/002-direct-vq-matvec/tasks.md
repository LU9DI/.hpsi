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
- [ ] T014 Add a GPU kernel only after CPU correctness and profiling.
