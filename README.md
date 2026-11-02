# HPSI — Hybrid Predictive Sparse Inference

**HPSI is an engineering project to build a compact model format and inference runtime that computes directly on compressed representations of neural-network weights.** The goal is not merely to compress a model and decompress it on demand. HPSI aims to keep weights in a mathematical representation that the runtime can operate on without materializing the full original tensors.

> **Status:** Early-stage implementation project. This repository is being established around a measurable proof of concept. The architecture and targets below are design goals, not claims of already achieved performance or quality.

## The problem

Large models are expensive to store, transfer, load, and serve. Conventional quantization reduces storage, but inference still requires kernels that interpret quantized tensors. Many compression pipelines also expand or reconstruct data into a compute-friendly representation.

HPSI explores a different storage-and-execution boundary:

- Encode weights as compact codes, codebooks, factors, and selectively stored residuals.
- Index those structures so the runtime can access required data without scanning or decoding the entire model.
- Execute supported operations directly against the compact representation.
- Spend additional bytes or computation only where measured output error justifies it.
- Treat storage traffic, memory footprint, latency, throughput, and model quality as first-class engineering metrics.

**Primary design constraint: no full-tensor reconstruction on the inference path.**

## What HPSI is building

HPSI is intended to deliver four connected components:

1. **The .hpsi model container** — a versioned binary format for model metadata, tensor descriptors, quantized blocks, codebooks, mathematical indexes, residuals, and integrity data.
2. **Conversion and compression pipeline** — tools that inspect source models, select per-tensor representations, encode them, and validate the resulting artifact.
3. **Direct Compute runtime** — kernels and scheduling logic that execute supported operations over encoded representations instead of rebuilding full-precision tensors first.
4. **Evaluation suite** — reproducible tests for compression, numerical error, model quality, random access, storage I/O, memory use, and inference performance.

The longer-term direction includes sparse/MoE-aware execution: routing identifies relevant experts and the runtime accesses only the required expert data. MoE support is a later milestone, not a prerequisite for the first proof of concept.

## Core technical approach

HPSI will evaluate and combine established techniques rather than assume one compression method is optimal for every tensor:

- **Quantization and codebooks:** low-bit scalar, vector, product, and additive quantization.
- **Low-rank representations:** factorize suitable matrices and compute through the factors.
- **Tensor decomposition:** investigate decompositions where measured savings justify runtime costs.
- **Outlier handling:** store exceptional values separately when doing so improves the size/quality trade-off.
- **Residual refinement:** encode correction data and consult it selectively when the base approximation exceeds an error budget.
- **Mathematical indexes and lookup tables:** precompute useful transforms or partial products where storage costs and generalization behavior are justified.

For example, a matrix may be represented approximately as W ≈ AB, allowing an input operation to be evaluated as (XA)B without reconstructing W. A codebook-based representation may replace portions of conventional matrix multiplication with operations over compact codes and precomputed tables. These are candidate execution paths to benchmark, not universal wins.

### Three execution levels

1. **Encoded-domain lookup:** use indexes or precomputed structures when the operation and error budget permit it.
2. **Direct compact computation:** run kernels against codes, codebooks, or factors without materializing the original tensor.
3. **Selective refinement:** access residual blocks only when needed to meet a configured accuracy target.

All three levels must preserve the no-full-reconstruction design goal. Selective refinement must not silently turn into whole-model decompression.

## Proposed .hpsi container

The format is expected to use a versioned header and independently addressable sections:

| Section | Responsibility |
|---|---|
| Header | Magic bytes, format version, flags, model metadata location |
| Tensor directory | Tensor IDs, shapes, logical types, codecs, block maps |
| Mathematical directory | Codebook, factor, projection, lookup, and residual indexes |
| Data blocks | Independently addressable encoded tensor data |
| Shared assets | Reusable codebooks and other shared structures |
| Integrity directory | Checksums, lengths, offsets, and validation metadata |

The format must support bounds-checked parsing, explicit endianness, offset/length validation, corruption detection, and forward-compatible versioning. Block size will be selected by measurement, balancing random-access overhead, I/O efficiency, and compute-kernel requirements.

## Initial proof of concept

The first milestone is narrower than full language-model inference: demonstrate that a matrix operation can be executed directly from an HPSI-encoded representation.

**Reference workload:** a 4096 × 4096 matrix, with fixed and held-out input vectors/batches.

Compare at least:

- FP16 reference computation.
- INT4 and INT2 baselines.
- A codebook/vector-quantized representation.
- A direct encoded-domain or lookup-assisted computation path.

The test must use inputs that were not used to construct lookup tables or fit the representation. It must verify that the implementation does not materialize the complete FP16 matrix during measured HPSI execution.

### Proof-of-concept acceptance criteria

- Encode and reopen an HPSI artifact with validated metadata and integrity checks.
- Address and read individual tensor blocks without scanning the complete artifact.
- Execute the target matrix operation from the compact representation.
- Demonstrate with instrumentation that the full reference tensor is not reconstructed on the HPSI path.
- Report artifact size including indexes, codebooks, scales, outliers, and residuals.
- Report numerical error and latency against the FP16 reference and quantized baselines.
- Include repeatable commands, test-data generation, environment details, and raw benchmark output.

No compression ratio or quality target is considered achieved until reproduced by the benchmark suite. An extreme target such as 400 GB to 10 GB is a research objective, not a guaranteed result; quality, metadata, and runtime costs must be measured together.

## Metrics and validation

Every optimization must be evaluated across the same workload and hardware configuration.

| Area | Measurements |
|---|---|
| Storage | Artifact bytes, compression ratio, index/codebook/residual overhead |
| I/O | Bytes read, read amplification, random-access latency |
| Memory | Peak resident RAM and VRAM; temporary allocations |
| Runtime | Operation latency, throughput, kernel time |
| Numerical fidelity | Weight error, layer-output error, cosine similarity, relative error |
| Model quality | Logit divergence, perplexity, task-specific evaluation where applicable |
| Reliability | Corruption handling, malformed offsets, deterministic decoding, repeatability |

Comparisons should include relevant established methods and runtimes where applicable, such as AQLM, QuIP-family methods, GGUF/llama.cpp, and optimized low-bit kernels. HPSI must not claim superiority without reproducible, like-for-like evidence.

## Architecture direction

The initial implementation separates format/storage concerns from mathematical encoding and execution:

- **Core and format:** binary schema, parser, validation, tensor directory, block access.
- **Indexer:** build and query mathematical metadata and block maps.
- **Compression:** quantization, codebooks, low-rank factors, outlier and residual encoding.
- **Direct compute:** CPU reference kernels first, followed by optimized GPU kernels.
- **Runtime and scheduler:** tensor access, caching, execution planning, and later expert-aware routing.
- **Evaluation:** correctness, quality, performance, and reproducibility.

### Candidate repository layout

    .hpsi/
    ├── .github/
    ├── .specify/                 # Spec Kit project configuration
    ├── specs/                    # Feature specifications and implementation plans
    ├── crates/
    │   ├── hpsi-core/
    │   ├── hpsi-format/
    │   ├── hpsi-indexer/
    │   ├── hpsi-loader/
    │   ├── hpsi-cache/
    │   └── hpsi-runtime/
    ├── compression/
    │   ├── quantization/
    │   ├── codebooks/
    │   ├── low_rank/
    │   ├── tensor_decomposition/
    │   └── outlier_encoding/
    ├── kernels/
    │   ├── cpu/
    │   ├── cuda/
    │   └── vector_lookup/
    ├── benchmarks/
    │   ├── compression/
    │   ├── inference/
    │   └── quality/
    ├── tests/
    ├── tools/
    │   ├── inspect_model/
    │   ├── convert_model/
    │   └── evaluate_quality/
    └── docs/
        ├── architecture.md
        ├── hpsi_format.md
        └── benchmark_protocol.md

This is a target layout, not a statement that all directories or components already exist.

## Technology direction

The initial design favors:

- **Rust** for the format, safe parsing, indexing, block I/O, scheduling, and runtime orchestration.
- **C++/CUDA** for performance-critical GPU kernels when justified by benchmarks.
- **Python and PyTorch** for experiments, quantizer/codebook training, conversion prototypes, and quality evaluation.
- **CPU reference implementations** as correctness oracles before GPU optimization.
- **Memory mapping and positional reads** for efficient file access. Memory mapping is an I/O mechanism, not decompression and not a substitute for direct-compute kernels.

Final choices remain subject to implementation and benchmark results.

## Development workflow — Spec Kit

HPSI is an implementation project, not a paper-only research exercise. Work should be driven by explicit specifications, dependency-ordered tasks, tests, and convergence checks.

We use [GitHub Spec Kit](https://github.com/github/spec-kit) for Spec-Driven Development (SDD). The intended workflow is:

1. **/speckit-constitution** — establish project principles and engineering constraints.
2. **/speckit-specify** — define observable requirements and acceptance criteria.
3. **/speckit-clarify** — resolve important ambiguities.
4. **/speckit-plan** — define architecture and implementation choices.
5. **/speckit-checklist** — validate requirement quality.
6. **/speckit-tasks** — produce dependency-ordered implementation tasks.
7. **/speckit-analyze** — detect inconsistencies and gaps.
8. **/speckit-implement** — implement in reviewed increments.
9. **/speckit-converge** — verify the implementation against the spec and tasks.

Install the Specify CLI:

    uv tool install specify-cli

Then initialize Spec Kit for this existing repository using the current [Spec Kit documentation](https://github.github.com/spec-kit/). Choose the integration for the coding agent used in this workspace and review generated files before committing them.

Useful skills directory: [Skill Issue](https://skill-issue.in/) — a curated library of agent skills. Skills should be reviewed for provenance, permissions, and relevance before installation or execution.

## Roadmap

- [ ] **Milestone 0 — Engineering foundation:** repository conventions, Spec Kit constitution, initial specifications, CI, and test strategy.
- [ ] **Milestone 1 — HPSI Core:** versioned file header, tensor directory, safe parser, independent blocks, random-access reader, and integrity tests.
- [ ] **Milestone 2 — Mathematical Engine:** baseline quantizers/codebooks and a CPU direct-compute prototype.
- [ ] **Milestone 3 — Proof-of-concept benchmark:** FP16/INT4/INT2 comparisons, held-out inputs, quality and I/O instrumentation.
- [ ] **Milestone 4 — Runtime optimization:** caching, scheduling, kernel optimization, and GPU execution.
- [ ] **Milestone 5 — Model integration:** conversion and end-to-end inference on a small supported model.
- [ ] **Milestone 6 — Sparse/MoE execution:** expert-aware indexing and selective expert access.
- [ ] **Milestone 7 — Advanced compression:** adaptive per-tensor codecs and more aggressive rate/distortion experiments.

A milestone is complete only when its tests, acceptance criteria, and benchmark evidence are recorded.

## Engineering principles

- Correctness and measurable quality take priority over headline compression ratios.
- Never report estimated or theoretical numbers as measured results.
- Never reconstruct the complete original model on the primary HPSI inference path.
- Keep the format versioned, documented, bounds-checked, and corruption-aware.
- Separate experimental algorithms from stable format/runtime contracts.
- Preserve reproducibility: record datasets, seeds, hardware, software versions, and commands.
- Benchmark against strong baselines; do not present known techniques as HPSI inventions.
- Do not add model-specific assumptions, including MoE structure, without inspecting the actual model configuration.

## Current status and contribution

This repository is at the project-foundation stage. The first deliverable is a tested HPSI Core and a matrix-level direct-compute proof of concept. No model compatibility, compression ratio, inference speedup, or quality guarantee is claimed yet.

Contributions should begin with or update the relevant specification, include tests, and provide benchmark evidence for performance-related changes. Avoid breaking format changes without a versioning and migration plan.

---

**HPSI — store less, compute directly, measure everything.**
