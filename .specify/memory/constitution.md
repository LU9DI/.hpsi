# HPSI Constitution

**Status:** Draft for the first engineering baseline  
**Version:** 0.1.0  
**Scope:** All HPSI repositories, libraries, file formats, runtimes, tools, and benchmarks.

## 1. Evidence over claims

1. No compression ratio, speedup, memory reduction, quality preservation, or scalability claim is accepted without a reproducible experiment.
2. Every benchmark must identify the exact input artifact, source revision, compiler/runtime versions, hardware, parameters, warm/cold state, and measurement method.
3. Report artifact size including headers, indexes, codebooks, lookup tables, residuals, alignment, and integrity metadata. Never report payload-only size as total model size.
4. Distinguish bytes stored, bytes read from storage, bytes transferred to host memory, bytes transferred to accelerator memory, bytes decoded, and bytes used by arithmetic.
5. Separate measured results, analytical estimates, hypotheses, and future targets.

## 2. Direct-compute is a verifiable contract

1. HPSI's central objective is to execute supported mathematical operators from compressed representations without materializing the complete original dense tensor.
2. A kernel may decode small tiles, codebook entries, or bounded working sets when mathematically necessary. It must not silently reconstruct and cache the entire dense source tensor.
3. Every direct-compute implementation must expose instrumentation proving which representations and blocks were accessed and the maximum temporary/dense allocation.
4. A fallback that reconstructs a dense tensor must be explicit, separately named, instrumented, and excluded from direct-compute performance claims.
5. Random access, memory mapping, and selective decompression are enabling mechanisms, not proof of direct computation.

## 3. Numerical quality is a first-class requirement

1. Every lossy representation must define its error model, supported numeric types, calibration procedure, and quality evaluation.
2. Compare against a clearly defined reference implementation and report absolute error, relative error, RMSE, maximum error, and task-level metrics where applicable.
3. Evaluate lookup tables and learned/codebook approximations on held-out inputs; never validate only on calibration samples.
4. Refinement paths must state when residuals are accessed and how the error/latency trade-off is controlled.
5. No target such as 40:1 compression is a guarantee. Quality and performance determine whether a configuration is useful.

## 4. Format safety and forward compatibility

1. Treat every .hpsi file as untrusted input.
2. Validate magic, version, flags, counts, offsets, lengths, alignment, arithmetic overflow, references, checksums, and resource limits before using payload data.
3. Never trust offsets, dimensions, allocation sizes, compression metadata, or tensor names from a file.
4. Unknown mandatory features must fail closed with a typed error; unknown optional metadata may be skipped only when the format explicitly permits it.
5. The format must be versioned, deterministic where practical, documented byte-for-byte, and tested with malformed/truncated/fuzzed inputs.
6. No model file may trigger arbitrary code execution, dynamic library loading, shell commands, or unsafe object deserialization.

## 5. Small, explicit, composable components

1. Prefer a library-first architecture with narrow, documented interfaces.
2. Separate format parsing, storage access, mathematical representation, execution planning, caching, kernels, and evaluation.
3. Keep the CPU reference implementation simple and authoritative before adding SIMD, GPU, or distributed execution.
4. Every optimization must preserve a correct reference path and have a benchmark demonstrating its value.
5. New dependencies require a written justification covering maintenance, license, security, binary size, and alternatives.

## 6. Reproducible engineering

1. Each feature must include tests, fixtures, and commands to reproduce validation.
2. Correctness tests must be deterministic and independent of benchmark timing.
3. Performance tests must not be used as flaky pass/fail unit tests; use explicit regression budgets only after stable baselines exist.
4. CI must test supported operating systems and architectures as they are introduced, not claim portability from a single machine.
5. No implementation is complete until documentation, error behavior, compatibility implications, and test evidence are updated.

## 7. Resource-bounded execution

1. Parsers, loaders, caches, and runtimes must expose explicit limits for file size, tensor count, block count, dimensions, allocation, cache capacity, and concurrent work.
2. Use checked arithmetic and bounded allocations; avoid unbounded decompression and adversarial resource amplification.
3. I/O and compute scheduling must support cancellation and report failures without corrupting shared state.
4. Cache policy must be observable and bounded; cache hits must not conceal full-tensor reconstruction.

## 8. Compatibility and change control

1. Public format fields, APIs, and semantics are contracts. Breaking changes require a versioning decision and migration notes.
2. Never reinterpret an existing field or codec identifier silently.
3. Keep experimental codecs and operators clearly marked as experimental until conformance tests and independent benchmarks pass.
4. Preserve repository history and existing work. Do not force-push, rewrite history, merge, or deploy without explicit authorization.
5. Work in feature branches and keep changes reviewable.

## 9. Definition of done

A feature is done only when:
- its behavior and non-goals are specified;
- its implementation matches the specification;
- correctness, malformed-input, and resource-limit tests pass;
- direct-compute claims are backed by instrumentation where applicable;
- benchmarks include full costs and reproducibility metadata;
- public documentation and compatibility notes are updated;
- remaining limitations are explicit.

## 10. Initial technical posture

The first milestone is a safe, portable, versioned container and bounded random-access reader. The first compute milestone is a CPU reference operator for a compressed matrix representation. The initial implementation must not assume a particular transformer architecture, MoE routing scheme, GPU vendor, or proprietary model format.
