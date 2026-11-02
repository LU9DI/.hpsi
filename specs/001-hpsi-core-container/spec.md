# Feature 001 — HPSI Core Container and Safe Random-Access Reader

**Status:** Proposed engineering contract  
**Branch:** `feat/hpsi-core-container`  
**Priority:** P0 — foundation  
**Target milestone:** HPSI 0.1  
**Constitution:** `.specify/memory/constitution.md`

## 1. Problem statement

HPSI needs a portable artifact that can store tensors as independently addressable compressed or mathematically encoded blocks. A runtime must be able to locate only the blocks and auxiliary structures needed for an operation, validate them, and hand bounded byte ranges to a decoder or direct-compute kernel.

A conventional archive that requires sequential extraction is insufficient. A memory-mapped compressed file is still compressed bytes; mapping it does not make its values directly usable by arithmetic. The container must make addressing explicit and must not imply that every codec supports direct computation.

## 2. Goal

Define and implement the first safe, versioned `.hpsi` container and a reader that:
- parses a fixed-size header and bounded directories;
- identifies tensors and their logical shapes/types;
- maps logical tensor ranges to independently addressable blocks;
- locates payload and auxiliary representation data without scanning the whole file;
- validates structural invariants and integrity records;
- reads requested blocks using bounded random-access I/O;
- rejects malformed or unsupported files deterministically.

The first release is a storage/format foundation. It does not yet promise a neural-network runtime or a particular compression ratio.

## 3. User stories

### P1 — Inspect an artifact
As a developer, I can inspect a .hpsi file and see its version, tensor names, shapes, codecs, block counts, stored bytes, and logical bytes without loading tensor payloads.

**Acceptance**
- The inspect command succeeds for a valid fixture.
- It does not read payload blocks unless checksum verification is explicitly requested.
- Output distinguishes logical bytes from stored bytes.

### P2 — Read a tensor block
As a runtime developer, I can request a tensor block by tensor ID and block index and receive exactly the validated payload range plus its descriptor.

**Acceptance**
- Only the requested block's byte range is read.
- Returned data length matches the descriptor.
- Invalid tensor IDs and block indexes return typed errors.
- Reader memory use is bounded by configured limits and requested block size.

### P3 — Reject hostile or corrupt files
As an operator, I need malformed artifacts to fail safely before oversized allocation, out-of-bounds access, or partial state mutation.

**Acceptance**
- Truncated headers/directories, integer overflow, overlapping or out-of-file regions, invalid references, unsupported mandatory flags, and checksum mismatch are rejected.
- No panic, unchecked allocation, or arbitrary code execution occurs for malformed inputs.
- Errors identify the failing structural element without leaking sensitive host paths.

### P4 — Preserve format evolution
As a format author, I need explicit version and feature negotiation so newer files cannot be silently misread by older readers.

**Acceptance**
- Major-version mismatch is rejected.
- Unknown mandatory feature flags are rejected.
- Unknown optional metadata is skipped only where explicitly allowed.
- Reserved fields must be zero in v0.1.

## 4. Functional requirements

- **FR-001:** The file starts with the exact 128-byte v0.1 header defined in `docs/format-v0.1.md`.
- **FR-002:** All fixed-width integers are unsigned little-endian unless a field explicitly states otherwise.
- **FR-003:** The header declares total file size and the offset/length or offset/count/entry-size of each top-level region.
- **FR-004:** The reader validates all region ranges using checked arithmetic before reading them.
- **FR-005:** The metadata region is UTF-8 JSON with a strict maximum size and nesting/field limits; it contains descriptive metadata only, never executable instructions.
- **FR-006:** Tensor directory entries define stable tensor IDs, rank, dtype, codec, dimensions, logical byte count, block range, and metadata name reference.
- **FR-007:** Block directory entries define tensor ownership, logical range, payload range, codec/layout identifiers, and integrity reference.
- **FR-008:** Blocks are independently addressable. A block payload is not required to be independently decodable unless its codec contract says so; v0.1 reader reports codec capability and does not infer it.
- **FR-009:** The reader exposes a random-access source abstraction supporting file-backed reads and an in-memory test source.
- **FR-010:** The reader supports configurable caps for file size, metadata size, tensor count, block count, rank, dimensions, payload size per block, and total directory bytes.
- **FR-011:** The reader validates tensor/block ownership, block ordering, logical range coverage rules, payload bounds, auxiliary references, and directory consistency.
- **FR-012:** Integrity records support CRC32C for accidental corruption detection in v0.1. Cryptographic authenticity/signatures are out of scope.
- **FR-013:** The CLI supports `hpsi inspect <file>` and `hpsi verify <file>`; verify reads and checks every integrity-covered region.
- **FR-014:** Writer output is deterministic for identical logical input and options, excluding any explicitly documented random identifier.
- **FR-015:** The writer writes to a temporary sibling file, flushes and validates it, then atomically replaces the destination where the platform supports this behavior. It must not leave a partially valid final artifact after failure.
- **FR-016:** Reader errors use a stable typed error taxonomy (I/O, bad magic, unsupported version, invalid header, invalid directory, limit exceeded, invalid reference, checksum mismatch, unsupported mandatory feature).

## 5. Non-functional requirements

- **NFR-001 Safety:** No untrusted length or dimension may directly drive an allocation before limits and checked arithmetic are applied.
- **NFR-002 Bounded memory:** Opening a file reads header and directories only; payload data is fetched on demand.
- **NFR-003 Portability:** Core format logic must be platform-independent; file I/O adapters must work on Windows, Linux, and macOS.
- **NFR-004 Observability:** Reader statistics include metadata bytes read, directory bytes read, payload bytes read, read operation count, and verification count.
- **NFR-005 Performance:** Directory lookup must be O(1) or O(log n) after opening, not a linear scan for every block request.
- **NFR-006 Correctness:** A valid file round-trips through writer and reader with identical descriptors and payload bytes.
- **NFR-007 Compatibility:** v0.1 is experimental. Any incompatible change before 1.0 must still increment the format minor/major version according to documented rules and update fixtures.

## 6. Proposed format model

The normative byte-level definition is in `docs/format-v0.1.md`. It defines:
- 128-byte fixed header;
- UTF-8 JSON metadata region;
- fixed-size tensor directory entries (128 bytes);
- fixed-size block directory entries (64 bytes);
- payload region containing opaque codec payloads;
- fixed-size integrity entries (32 bytes);
- all offsets measured from the beginning of the file;
- no implicit padding or alignment assumptions beyond declared fields.

The reader must treat payload as opaque. It must not decompress or execute a codec while parsing the container.

## 7. Explicitly out of scope

- Quantization algorithms and quality calibration.
- Direct matrix multiplication over codebooks.
- Tensor encoding/decoding beyond opaque payload pass-through.
- CUDA, Metal, Vulkan, distributed execution, or multi-GPU.
- Transformer architecture, tokenization, attention, KV cache, MoE routing, or model conversion.
- Compression ratio guarantees (including 40:1).
- Cryptographic signatures, encryption, and remote streaming.
- mmap as a performance requirement.
- Loading arbitrary plugin code from model files.

## 8. Validation strategy

1. Golden byte fixtures for valid minimal, multi-tensor, multi-block, and auxiliary-reference files.
2. Round-trip writer/reader tests.
3. Boundary tests for zero-length and maximum allowed values.
4. Mutation tests for every header/directory field.
5. Truncation tests at every structural boundary.
6. Fuzzing of header, metadata JSON, tensor directory, block directory, and cross-references.
7. Cross-platform CI for Windows, Linux, and macOS.
8. CLI integration tests proving inspect avoids payload reads and verify reads all covered payloads.
9. Resource-limit tests using deliberately huge declared sizes with tiny actual files.
10. Benchmarks separating open/parse time, directory lookup time, random block-read latency, bytes read, and peak memory.

## 9. Definition of done

- [ ] Normative format document reviewed and internally consistent.
- [ ] Core crate and CLI compile on the supported toolchain.
- [ ] Valid fixtures pass parse, inspect, verify, and round-trip tests.
- [ ] Malformed fixtures fail with typed errors and no panic.
- [ ] Fuzz target runs in CI or a documented local command.
- [ ] Instrumentation proves payload bytes are not read during ordinary open/inspect.
- [ ] Windows, Linux, and macOS test results recorded.
- [ ] Limitations and experimental status are visible in documentation.

## 10. Open design decisions to resolve before implementation

1. **Language baseline:** Rust is the proposed format/runtime implementation language; confirm MSRV and edition in the plan after checking available toolchain/CI.
2. **Metadata schema:** choose a minimal schema and strict JSON parser limits; avoid embedding arbitrary user-defined executable or deeply nested objects.
3. **Integrity scope:** decide whether CRC32C covers payload blocks only or also header/directories; the format document proposes explicit target records.
4. **Block logical coverage:** decide whether tensor blocks must be contiguous and exhaustive or may represent sparse/optional regions; v0.1 should choose one rule and enforce it.
5. **Auxiliary objects:** define how codebooks/factors/lookup tables are referenced without circular references; first format can model them as typed tensors/objects but not execute them.
6. **Atomic replacement:** document platform differences and durability guarantees for Windows vs POSIX.

## 11. Research and design references

- HPSI project rationale and initial architecture: root `README.md`.
- Byte-level container contract: `docs/format-v0.1.md`.
- Project principles: `.specify/memory/constitution.md`.
- Spec Kit existing-project adoption guidance: https://github.com/github/spec-kit/blob/main/docs/guides/existing-projects.md
