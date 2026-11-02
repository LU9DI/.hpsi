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
- [ ] T011 Record the four-rate benchmark on a pinned, documented CPU environment; compare the HPSI row-streamed path separately from in-memory VQ matvec.
- [ ] T012 Add GPU kernel only after CPU correctness and profiling.
