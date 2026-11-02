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

---

# Engineering Design Notes

This section makes the architecture concrete. It is a design baseline for the first implementation, not a frozen binary specification. Decisions that affect compatibility must be finalized in a versioned format specification before implementation.

## 1. Non-negotiable system requirements

HPSI is not successful merely because it produces a smaller file. A valid implementation must satisfy all of the following:

1. **Encoded-domain execution:** supported operators consume the compact representation directly.
2. **No hidden reconstruction:** the primary path must not allocate a full FP16/FP32 copy of the original weight tensor.
3. **Random access:** the runtime locates and reads required blocks without scanning the entire model file.
4. **Bounded approximation:** every lossy representation declares its codec parameters and exposes measurable error.
5. **Progressive refinement:** optional residual data improves accuracy without requiring full tensor reconstruction.
6. **Reproducibility:** compression and inference results can be recreated from documented inputs, versions, seeds, and commands.
7. **Safe parsing:** malformed files, offsets, lengths, dimensions, and indexes fail safely rather than cause out-of-bounds reads or uncontrolled allocation.

A codec that only decompresses a tensor into a conventional dense buffer may be useful as a compatibility fallback, but it does not satisfy the HPSI Direct Compute objective by itself.

## 2. Mathematical execution model

### 2.1 Low-rank execution

For a weight matrix W with dimensions m × n, a low-rank approximation may represent:

W ≈ A B

where A has dimensions m × r, B has dimensions r × n, and r is substantially smaller than min(m,n).

For an input X, the output is approximated as:

X Wᵀ ≈ (X Bᵀ) Aᵀ

The runtime operates on the stored factors. It must not multiply them together first to recreate W. This method is attractive when the matrix has sufficiently low effective rank; it can be slower or less compact when r is too large.

### 2.2 Codebook-based execution

Partition the input dimension into groups. For a group j, approximate the corresponding weight subvector using a codebook entry:

wⱼ ≈ cⱼ,qⱼ

A simple quantized representation stores the code index qⱼ and the codebook. More advanced schemes may use multiple codebooks or additive codes:

wⱼ ≈ Σₖ cₖ,j,qₖ

The compute kernel accumulates contributions from selected code vectors. The representation must define exactly how indices, scales, group boundaries, and accumulation precision are interpreted.

### 2.3 Lookup-assisted partial products

For a fixed weight block Wⱼ and a codebook C, a converter may precompute:

Tⱼ,q = Cq Wⱼ

At inference, a selected code q retrieves a partial contribution rather than multiplying a reconstructed weight block. This can trade arithmetic for storage bandwidth and lookup cost.

Important constraints:

- Tables can become larger than the compressed weights they are intended to replace.
- The lookup key must represent relevant runtime input information; arbitrary continuous activations cannot be covered by a finite exact-answer cache.
- Interpolation or quantization introduces approximation error that must be measured on held-out inputs.
- Table construction cost and table I/O must be included in total cost.
- A lookup optimization is accepted only if complete artifact size and end-to-end latency improve the baseline.

### 2.4 Residual refinement

Represent a tensor as a base approximation plus an encoded correction:

W ≈ Wbase + R

The base representation is always usable. The residual is divided into independently addressable blocks. A runtime may access selected residual blocks when an error estimator or configured policy indicates that refinement is needed.

The first implementation should use explicit deterministic refinement policies. A learned confidence predictor may be explored later, but it must be calibrated and validated on held-out data.

### 2.5 Numerical policy

Each codec must declare:

- logical tensor shape and source data type;
- quantization group size and bit width, where applicable;
- codebook dimensions and codebook precision;
- scale and zero-point interpretation;
- accumulation type;
- rounding and saturation rules;
- residual format and refinement semantics;
- expected error metrics and measured error distributions.

The runtime must not silently change numerical semantics between CPU and GPU implementations.

## 3. Proposed HPSI file organization

The HPSI file is a container with explicit offsets and independently addressable payloads. The following is a logical layout; exact field widths and byte offsets belong in the formal format specification.

| Region | Contents | Access pattern |
|---|---|---|
| Fixed header | Magic, major/minor version, flags, directory offsets | Read at open |
| Model metadata | Architecture identifier, tokenizer/config references, provenance | Read at open or on demand |
| Tensor directory | Tensor IDs, shapes, logical dtype, codec IDs | Indexed lookup |
| Mathematical directory | Codebooks, factors, lookup-table descriptors, residual maps | Indexed lookup |
| Block directory | Tensor-to-block mapping, offsets, compressed/uncompressed lengths | Random access |
| Payload region | Encoded codes, factors, codebooks, residual blocks | Read on demand |
| Integrity records | Checksums and integrity algorithm identifiers | Validate at open/read |

### 3.1 Directory invariants

- Every offset and length is validated using checked arithmetic before use.
- Every referenced region must be inside the file and must not overlap protected metadata.
- Counts and dimensions have configured upper bounds before allocation.
- Unknown mandatory features cause a clear unsupported-version error.
- Unknown optional metadata can be skipped using its declared length.
- Directories are deterministic and have stable ordering for reproducible builds.
- Checksums cover the exact bytes specified by the format version.
- The file records the codec implementation/version needed to interpret each payload.

### 3.2 Block independence

Tensor payloads are divided into independently addressable blocks. A block descriptor should contain, at minimum:

- tensor identifier;
- logical block coordinate or range;
- payload offset and stored length;
- codec identifier and codec-parameters reference;
- optional residual-block reference;
- integrity checksum.

A block should be independently readable and decodable by its declared codec. The first benchmark should compare several block sizes rather than hard-code a universal value. The initial exploration range is 256 KiB to 4 MiB, subject to actual tensor shapes, storage devices, and kernel requirements.

### 3.3 Versioning

- A major version changes when old readers cannot safely interpret the new layout or semantics.
- A minor version may add optional structures that old readers can skip safely.
- A reader must reject unsupported mandatory features.
- A writer must never silently emit a format version its own reader cannot validate.

The first public format should be intentionally small. Do not freeze speculative fields simply because they appear in this README.

## 4. Runtime execution pipeline

A typical operator execution should follow this sequence:

1. **Open model:** validate header, version, file length, and required metadata.
2. **Resolve tensor:** map the operator's logical tensor ID to its descriptor.
3. **Plan operation:** select a supported codec-specific kernel and determine required block coordinates.
4. **Resolve dependencies:** locate codebooks, low-rank factors, lookup tables, and optional residual blocks.
5. **Schedule reads:** issue bounded positional reads or mapped access for selected regions.
6. **Populate cache:** retain reusable encoded blocks and shared codebooks under a memory budget.
7. **Execute directly:** run the codec-aware operator without constructing the complete source tensor.
8. **Refine if requested:** access only residual blocks required by the selected accuracy policy.
9. **Record metrics:** bytes read, cache hits, temporary memory, kernel time, total latency, and error telemetry.

The scheduler must distinguish storage bytes read from bytes decoded and bytes transferred to the GPU. These are different quantities and must not be conflated.

## 5. Initial operator contract

The first direct-compute target is matrix multiplication:

Y = X Wᵀ

The initial interface should separate the logical operation from the physical representation:

- Input: X, tensor descriptor for W, execution policy, output buffer.
- Dispatch: select a codec-specific kernel based on the descriptor.
- Output: Y plus optional execution statistics.
- Forbidden behavior on the HPSI direct path: materialize the complete dense W as an intermediate.

The CPU implementation is the correctness reference. GPU kernels are introduced only after the CPU path has tests verifying numerical behavior and the no-reconstruction invariant.

## 6. Benchmark protocol

### 6.1 Required baselines

Use the same source matrix, input batches, hardware, and measurement method for:

- FP16 dense reference;
- INT4 quantized baseline;
- INT2 quantized baseline;
- codebook or additive-quantized representation;
- low-rank representation where applicable;
- HPSI lookup-assisted/direct-compute candidate.

Do not compare a CPU HPSI result with a GPU dense baseline and call the timing a speedup. Report hardware and execution paths separately.

### 6.2 Input splits

Use three distinct input sets:

- **Calibration set:** may be used to fit quantizers, codebooks, or approximation parameters.
- **Validation set:** used to choose codec settings and refinement thresholds.
- **Held-out test set:** never used to construct or tune the representation.

For lookup-based methods, the held-out set is mandatory. It helps detect systems that merely memorize calibration inputs.

### 6.3 Reported measurements

Every benchmark report should include:

- total artifact size and per-section size;
- compression ratio relative to the declared source dtype;
- file bytes read and number of read operations;
- bytes transferred to CPU memory and GPU memory, where measurable;
- peak resident RAM and VRAM;
- temporary allocations and whether a dense source tensor was created;
- encode time and index-build time;
- cold-cache and warm-cache latency;
- median and tail latency (p50, p95, and p99 where sample counts allow);
- throughput at declared batch sizes;
- numerical error distribution, not only a single average;
- model-level quality metrics for end-to-end model experiments.

Report warm-cache and cold-cache runs separately. Include repeated trials and disclose variance. Do not hide conversion time, table size, index size, or residual overhead from the compression ratio.

### 6.4 Correctness gates

A benchmark result is invalid if:

- the HPSI path reconstructs the complete source weight tensor;
- lookup tables were evaluated only on calibration inputs;
- metadata or shared codebooks are excluded from the file-size total;
- the baseline uses different input data or precision without disclosure;
- correctness is assessed only by comparing compressed weights rather than operator outputs;
- timing excludes required I/O or initialization without clearly labeling the scope.

## 7. Security and robustness

Model files are untrusted input. The parser and loader must:

- validate all offsets, lengths, shapes, counts, and multiplication results for integer overflow;
- cap allocations and reject unreasonable dimensions before allocation;
- reject overlapping or cyclic references unless explicitly permitted by the format;
- validate checksums before handing payloads to compute kernels;
- avoid executing code embedded in model artifacts;
- avoid unsafe deserialization of arbitrary objects;
- provide deterministic errors for truncated, malformed, or unsupported files;
- fuzz the parser and block-directory reader;
- test truncated files, corrupted checksums, duplicate IDs, invalid codec parameters, and out-of-range offsets.

A model converter should also record provenance and source model metadata. It must not silently discard tensors or alter model configuration.

## 8. Suggested Rust crate responsibilities

| Crate | Responsibility | Must not own |
|---|---|---|
| hpsi-core | Shared IDs, errors, scalar types, validated descriptors | GPU kernel implementation |
| hpsi-format | Header, directories, binary parser/writer, versioning | Model-specific inference policy |
| hpsi-indexer | Build/query tensor and mathematical indexes | Hidden tensor reconstruction |
| hpsi-loader | Positional reads, block loading, integrity verification | Codec training |
| hpsi-cache | Bounded caches for encoded blocks and shared assets | Unbounded model-wide materialization |
| hpsi-runtime | Operator dispatch, execution plans, metrics | File-format parsing details |

Python tools may initially produce experimental payloads, but the Rust reader must independently validate them. C++/CUDA kernels should implement a documented codec/operator contract shared with the CPU reference.

## 9. First implementation sequence

The first feature should be deliberately small and independently testable.

### Feature 001 — HPSI container and block reader

- Define a minimal versioned header.
- Define tensor and block descriptors.
- Implement a writer for a synthetic tensor.
- Implement a reader with strict bounds checks.
- Add checksum validation.
- Add tests for malformed offsets, truncated files, unsupported versions, and random block reads.
- Add a small inspection CLI that prints metadata without loading tensor payloads.

### Feature 002 — Baseline encoded matrix operator

- Generate a deterministic 4096 × 4096 matrix and fixed random seeds.
- Implement FP16 reference and INT4/INT2 baselines.
- Record exact source and calibration/test splits.
- Add a CPU codebook representation and direct-compute kernel.
- Instrument allocations to prove no complete dense source tensor is materialized.
- Compare output error, latency, and total artifact size.

### Feature 003 — Mathematical lookup experiment

- Define one explicit lookup-assisted encoding scheme.
- Calculate and report table size before running inference.
- Evaluate on held-out input vectors.
- Compare lookup time and storage traffic against direct codebook computation.
- Keep or reject the method based on measured end-to-end results.

Do not begin MoE routing or whole-model conversion before these features produce a reliable baseline.

## 10. Spec Kit workflow for this repository

Spec Kit commands are invoked in the coding agent's chat, not typed as terminal commands. The CLI is used to install and initialize the tooling. For an existing repository, follow the official existing-project adoption guide rather than treating this repository as a new empty project.

Recommended project-level constitution topics:

- direct computation over encoded representations is the product's defining constraint;
- no hidden dense reconstruction on the primary HPSI path;
- every lossy codec must declare and measure its error;
- correctness tests precede optimization;
- performance claims require reproducible benchmark artifacts;
- file parsing is bounds-checked and fuzz-tested;
- format compatibility is versioned;
- no destructive Git operations, merges, or deployments are implicit in implementation tasks.

Recommended first Spec Kit feature:

> Implement the minimal HPSI v0 container and a safe random-access block reader. Define a versioned header, tensor directory, block directory, and checksum contract. Provide a writer, reader, inspection CLI, and tests for malformed/truncated files. Do not implement model quantization or inference kernels in this feature.

Then create separate specifications for the direct-compute matrix operator and lookup experiment. This keeps file-format correctness independent from algorithm experiments.

Official references:

- [Spec Kit repository](https://github.com/github/spec-kit)
- [Spec Kit quickstart](https://github.github.com/spec-kit/quickstart)
- [Spec Kit existing-project adoption](https://github.github.com/spec-kit/)
- [Skill Issue](https://skill-issue.in/)

## 11. Explicit non-goals for the first release

- Guaranteeing a 40:1 compression ratio or any fixed quality level.
- Claiming every neural-network operator can be evaluated directly from every compressed representation.
- Replacing all existing model formats or inference runtimes immediately.
- Implementing a complete MoE runtime before the basic operator contract works.
- Using precomputed lookup tables as a substitute for generalization tests.
- Treating memory mapping as a compression algorithm or proof of direct computation.
- Reporting theoretical storage estimates as measured model-quality results.

The first release succeeds when it proves one carefully scoped operation, from a real HPSI artifact, with validated random access, bounded memory, no full source-tensor reconstruction, and independently reproducible quality/performance measurements.
