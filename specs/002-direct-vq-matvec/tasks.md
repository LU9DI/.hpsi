# Feature 002 tasks

- [x] T001 Define the experimental VQ representation and bit packing.
- [x] T002 Implement deterministic bounded-sample codebook fitting.
- [x] T003 Implement direct matvec over packed selectors without dense reconstruction.
- [x] T004 Export codebook and selector tensors to HPSI.
- [x] T005 Test HPSI readback and direct computation parity.
- [x] T006 Add held-out numerical quality report against dense F32.
- [x] T007 Add reproducible CPU benchmark for encode and matvec.
- [ ] T008 Add row-streamed execution over independently addressable HPSI blocks.
- [x] T009 Evaluate residual/additive quantization against the single-codebook baseline.
- [ ] T010 Add GPU kernel only after CPU correctness and profiling.
