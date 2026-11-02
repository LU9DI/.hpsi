# Implementation Plan — HPSI Core Container

**Feature:** 001-hpsi-core-container  
**Status:** Proposed; implementation has not started  
**Target:** Rust workspace, HPSI format 0.1

## 1. Technical context

The repository currently contains project documentation only. There is no existing language toolchain, workspace, CI configuration, coding convention, or test framework to preserve. This plan therefore proposes a minimal Rust-first foundation and records the assumptions that must be verified on the first implementation machine.

### Proposed baseline

- Language: Rust.
- Edition: 2024.
- Minimum supported Rust version (MSRV): 1.85.0.
- Workspace resolver: 3.
- Primary target: portable CPU implementation on Windows, Linux, and macOS.
- CLI: `hpsi`.
- Initial implementation: safe Rust; no unsafe blocks and no memory mapping.
- Initial release: parser, writer, inspect, verify, fixtures, fuzz target, and benchmarks for file access.

If CI or the selected build environment cannot support this baseline, record the evidence and adjust the plan before coding rather than silently lowering guarantees.

## 2. Architecture

### 2.1 Initial workspace

```text
.
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml
├── crates/
│   └── hpsi-format/
│       ├── Cargo.toml
│       ├── src/
│       │   ├── lib.rs
│       │   ├── error.rs
│       │   ├── header.rs
│       │   ├── metadata.rs
│       │   ├── tensor.rs
│       │   ├── block.rs
│       │   ├── integrity.rs
│       │   ├── limits.rs
│       │   ├── source.rs
│       │   ├── reader.rs
│       │   ├── writer.rs
│       │   └── stats.rs
│       └── tests/
│           ├── fixtures/
│           ├── format_roundtrip.rs
│           ├── malformed_files.rs
│           └── random_access.rs
├── apps/
│   └── hpsi-cli/
│       ├── Cargo.toml
│       └── src/main.rs
├── fuzz/
│   └── fuzz_targets/
│       ├── parse_header.rs
│       └── open_model.rs
├── benches/
│   └── reader_bench.rs
├── docs/
│   └── format-v0.1.md
└── specs/
    └── 001-hpsi-core-container/
```

Do not create all future crates now. Split them only when a real dependency boundary exists. The first public library is `hpsi-format`; the CLI is a thin client.

### 2.2 Module responsibilities

| Module | Responsibility | Must not do |
|---|---|---|
| error | Stable typed errors and context | Panic for malformed input |
| header | Decode/validate fixed 128-byte header | Read payload |
| metadata | Bounded UTF-8/JSON parse and schema checks | Execute metadata |
| tensor | Tensor descriptor and dtype rules | Decode codec payload |
| block | Block descriptor and logical coverage validation | Allocate tensor data |
| integrity | CRC32C calculation and target mapping | Claim authenticity |
| limits | Explicit resource policy | Silently raise caller limits |
| source | Random-access source abstraction | Assume mmap |
| reader | Directory indexing and bounded block reads | Reconstruct dense tensors |
| writer | Canonical serialization and safe replacement | Modify unrelated paths |
| stats | Read/verification counters | Hide cache/read behavior |

## 3. Core abstractions

### 3.1 Random-access source

Define a small `ReadAt`-style trait with:
- `len() -> Result<u64>`;
- `read_exact_at(offset, buffer) -> Result<()> `.

Implement:
- file-backed source using platform-supported positional reads or a safe seek/read implementation with synchronization;
- in-memory source for tests.

The contract must permit concurrent reads safely. If a seek/read adapter uses a shared cursor, serialize access internally and document this. Do not expose a mutable file cursor to parser code.

### 3.2 ModelReader

Opening performs:
1. read exactly 128 bytes;
2. decode header without unsafe casts;
3. compare declared and actual file size;
4. validate region ranges and resource limits;
5. read bounded metadata and directories;
6. validate tensor, block, and integrity relationships;
7. build lookup indexes;
8. publish a usable reader only after all checks pass.

Opening must not read payload ranges. Block reads allocate no more than the configured per-block limit and return the opaque payload with its descriptor.

### 3.3 Writer

Writer accepts validated model descriptors and opaque payload blocks. It:
- validates IDs, ranges, shapes, lengths, and references before touching destination;
- calculates directory positions and checks all arithmetic;
- writes canonical order with zero padding;
- calculates integrity values;
- flushes and reopens through the reference reader;
- replaces only the requested destination after successful validation.

The writer should use a temporary sibling filename created securely. Avoid predictable shared temp names and avoid deleting a temp file not created by this invocation.

## 4. Parsing and validation sequence

The order matters to avoid resource exhaustion:

1. Obtain actual source length.
2. Read the fixed header into a stack-sized buffer.
3. Validate magic/version/header size/flags/reserved bytes.
4. Validate declared file size against actual size and caller maximum.
5. Validate counts and fixed entry sizes.
6. Calculate directory lengths using checked multiplication.
7. Validate every top-level region using checked addition, alignment, file bounds, and non-overlap.
8. Validate configured aggregate directory limits before allocating.
9. Read metadata and directory bytes.
10. Parse metadata with strict limits.
11. Parse tensor descriptors and validate dimensions/dtypes/name refs.
12. Parse block descriptors and validate grouping, ownership, logical coverage, payload ranges, and codec-specific structural rules.
13. Parse integrity records and validate references.
14. Validate auxiliary references and reject cycles.
15. Build indexes and publish ModelReader.

No step may allocate from an unchecked untrusted number. Use fallible reservation where applicable and translate allocation failures into typed errors.

## 5. Lookup structures

- Tensor IDs: hash map or sorted index with O(1)/O(log n) lookup.
- Block lookup: use each tensor's `first_block_index + block_id` after verifying contiguous directory grouping; no per-request scan.
- Integrity records: direct indexed vector; validate uniqueness by target.
- Name lookup: optional map from name to tensor ID; duplicate names rejected.

Do not retain copies of payload bytes when opening a model.

## 6. Error model

Expose a non-exhaustive public error enum with variants equivalent to:
- Io;
- BadMagic;
- UnsupportedVersion;
- UnsupportedFeature;
- InvalidHeader;
- InvalidRegion;
- InvalidDirectory;
- InvalidMetadata;
- InvalidTensor;
- InvalidBlock;
- InvalidReference;
- LimitExceeded;
- ChecksumMismatch;
- UnsupportedCodec;
- AllocationFailed.

Errors should include a field/record index and byte offset when meaningful. Avoid embedding full paths or untrusted metadata strings in errors by default.

## 7. Dependency policy

Proposed direct dependencies, subject to license/security review:
- `serde` + `serde_json` for bounded metadata parsing/serialization;
- `crc32c` for CRC32C;
- `clap` for CLI argument parsing;
- `thiserror` for typed errors (optional if handwritten errors remain clearer).

Do not add Zstandard decoding to Feature 001 unless a concrete test fixture and output-size enforcement are implemented. RAW is sufficient to prove the container contract. If ZSTD_FRAME is implemented, use a maintained decoder and enforce exact output size and maximum window/memory settings.

Development dependencies may include `tempfile`, `proptest`, `criterion`, and `cargo-fuzz` tooling. Each addition must be pinned by Cargo.lock and reviewed.

## 8. Testing strategy

### Unit tests
- Header byte offsets, endianness, flags, reserved bytes.
- Dtype storage width and checked dimension multiplication.
- Region arithmetic, alignment, overlap and file bounds.
- JSON limits and tensor-name validation.
- CRC32C known vectors.
- Logical block coverage and ownership.
- Auxiliary reference graph and cycle detection.

### Integration tests
- Write/read round-trip.
- Minimal RAW tensor.
- Multiple tensors and multiple blocks.
- Verify-all success and deliberate corruption.
- Inspect does not read payload.
- Read-block reads exactly one requested range.
- File-backed and in-memory source produce identical results.

### Adversarial tests
- Every truncation boundary in header and directories.
- Maximum integers, overflow, giant counts, zero lengths.
- Overlapping regions/payloads.
- Invalid UTF-8/JSON and excessive nesting.
- Invalid tensor/block/integrity references.
- Unknown mandatory flags and codec IDs.
- Repeated randomized mutations and fuzzing.

### Cross-platform
Run `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` on Windows, Ubuntu, and macOS. Add a 32-bit target only if it remains a supported target and CI resources allow it; all format arithmetic remains u64 regardless.

## 9. Benchmark plan

Benchmarks are informational initially, not hard CI gates.

Measure separately:
- open/header/directory parse latency;
- inspect latency and bytes read;
- lookup latency for 1, 1k, 100k, and 1M blocks;
- random block read latency across block sizes;
- sequential block read throughput;
- verify-all throughput;
- peak resident memory on open and block read;
- allocations and read operation count.

Every benchmark output must include:
- source revision;
- OS and kernel;
- CPU model and memory;
- Rust version and build profile;
- file fixture size and block distribution;
- cache state (cold/warm where meaningful);
- repetitions and percentile calculation.

Do not mix OS page-cache effects with decoder time. Do not claim storage savings from synthetic RAW payloads.

## 10. Security review checklist

- No unsafe code in initial format crate.
- No unchecked offset arithmetic.
- No allocation based on raw count before cap checks.
- No decompression during parsing.
- No codec/plugin dynamic loading.
- No path traversal or archive extraction.
- No arbitrary metadata execution.
- No symlink-following surprises during output replacement.
- Temp files are uniquely created and cleaned only when owned.
- Fuzz targets never assume input validity.

## 11. Complexity tracking

| Decision | Cost | Reason | Alternative rejected |
|---|---|---|---|
| Rust 2024/MSRV 1.85 | Requires modern toolchain | Strong checked-memory model and portability | C++ first would add memory-safety burden to parser |
| JSON metadata | Parsing dependency and some overhead | Human-inspectable, bounded descriptive metadata | Custom binary metadata is premature |
| Fixed-size directories | Some reserved space | Direct indexing and simple validation | Variable-length entries complicate random access |
| CRC32C | Detects accidental corruption only | Fast and widely implementable | Cryptographic authenticity is not needed for initial local format |
| No mmap initially | May leave some performance untapped | Simpler cross-platform correctness and explicit reads | Add mmap only after measured benefit |
| RAW first | No compression benefit | Isolates container correctness from codec correctness | Implementing codecs simultaneously increases failure surface |

## 12. Exit criteria

Feature 001 can be declared complete only when every task in `tasks.md` is checked, all validation commands pass on the declared platforms, fixtures are documented, and the inspect/verify/read-block instrumentation demonstrates the required I/O behavior.
