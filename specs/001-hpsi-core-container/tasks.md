# Tasks — HPSI Core Container

**Feature:** 001-hpsi-core-container  
**Status:** In progress  
**Rule:** A task is complete only with code, tests, and recorded evidence. Documentation-only completion does not satisfy implementation tasks.

## Phase 0 — Repository and toolchain baseline

- [ ] T001 Record current branch, HEAD, and clean/dirty state before implementation.
- [ ] T002 Install/verify Rust 1.85+ and Cargo; record `rustc -Vv` and `cargo -V`.
- [ ] T003 Confirm Windows/Linux/macOS CI target matrix and select pinned toolchain.
- [ ] T004 Create minimal Cargo workspace and lockfile.
- [ ] T005 Add repository-wide formatting, lint, and test commands to CONTRIBUTING or developer docs.
- [ ] T006 Add license and dependency policy review before selecting third-party crates.

## Phase 1 — Format primitives

- [ ] T007 Implement byte-oriented little-endian read helpers without unsafe casts.
- [ ] T008 Implement the 128-byte header parser and serializer.
- [ ] T009 Implement header validation: magic, version, size, flags, reserved bytes, file size.
- [ ] T010 Implement checked region arithmetic, 8-byte alignment, bounds, and overlap checks.
- [ ] T011 Implement resource-limit configuration and safe defaults.
- [ ] T012 Implement dtype identifiers and checked logical byte calculations, including packed dtypes.
- [ ] T013 Implement typed error enum and stable error context.
- [ ] T014 Add unit tests for every header field, boundary, and overflow path.

## Phase 2 — Metadata and directories

- [ ] T015 Implement bounded UTF-8 JSON metadata parser.
- [ ] T016 Validate required metadata keys, tensor_names, uniqueness, and name constraints.
- [ ] T017 Implement tensor directory entry parser/serializer (128 bytes).
- [ ] T018 Validate tensor IDs, ranks, dimensions, dtypes, logical byte counts, and name references.
- [ ] T019 Implement block directory entry parser/serializer (64 bytes).
- [ ] T020 Validate tensor block ranges, ownership, block IDs, logical coverage, and payload bounds.
- [ ] T021 Implement integrity directory entry parser/serializer (32 bytes).
- [ ] T022 Validate checksum uniqueness, target mapping, and block checksum_index references.
- [ ] T023 Validate auxiliary references and reject missing references/cycles.
- [ ] T024 Add golden byte fixtures for valid one-tensor and multi-tensor files.
- [ ] T025 Add malformed fixtures for invalid metadata, directories, and cross-references.

## Phase 3 — Random-access source and reader

- [ ] T026 Define thread-safe random-access source trait.
- [ ] T027 Implement file-backed source with exact positional-read semantics.
- [ ] T028 Implement in-memory source with read-range instrumentation for tests.
- [ ] T029 Implement ModelReader::open using the prescribed validation order.
- [ ] T030 Ensure open reads header, metadata, and directories only — never payload.
- [ ] T031 Build tensor and block indexes with O(1)/O(log n) lookup.
- [ ] T032 Implement tensor descriptor and block descriptor lookup.
- [ ] T033 Implement read_block with exact-range reads and per-block allocation limits.
- [ ] T034 Implement ReaderStats for metadata, directory, payload bytes and operation counts.
- [ ] T035 Implement verify_all with bounded buffers and CRC32C.
- [ ] T036 Add tests proving exact requested ranges are read and unrelated payload is untouched.
- [ ] T037 Add concurrent read tests and verify source synchronization semantics.

## Phase 4 — Writer

- [x] T038 Define writer input model for metadata, tensor descriptors, blocks, and opaque payloads.
- [x] T039 Validate all writer inputs before creating/replacing destination.
- [ ] T040 Calculate canonical region layout with checked arithmetic and zero padding.
- [x] T041 Serialize metadata, tensor directory, block directory, payload, and integrity records.
- [ ] T042 Implement securely created temporary sibling output file.
- [ ] T043 Flush/synchronize where supported, reopen with reference reader, and verify output.
- [ ] T044 Replace destination atomically where supported; document platform guarantees.
- [ ] T045 Test that failed writes preserve pre-existing destination and clean only owned temporary files.
- [x] T046 Add deterministic writer round-trip fixtures.

## Phase 5 — CLI

- [x] T047 Create thin `hpsi` CLI binary.
- [ ] T048 Implement `hpsi inspect <file>` with tensor names/shapes/codecs/block counts and logical/stored sizes.
- [ ] T049 Ensure inspect output labels stored bytes vs logical bytes and reports no compression claim.
- [x] T050 Implement `hpsi verify <file>` with per-region verification summary.
- [x] T051 Add exit codes and actionable errors for invalid files, unsupported versions/codecs, and I/O failures.
- [ ] T052 Add CLI integration tests, including proof that inspect does not read payload.

## Phase 6 — Adversarial validation and fuzzing

- [ ] T053 Add truncation tests at every structural boundary.
- [ ] T054 Add integer-overflow, oversized-count, oversized-dimension, and allocation-limit tests.
- [ ] T055 Add overlapping-region and overlapping-payload tests.
- [ ] T056 Add corrupted metadata/payload CRC32C tests.
- [ ] T057 Add fuzz target for header parser.
- [ ] T058 Add fuzz target for complete ModelReader::open.
- [ ] T059 Run fuzzing for a documented minimum duration and preserve crash regressions as fixtures.
- [ ] T060 Verify malformed inputs never panic and do not trigger unbounded allocation.

## Phase 7 — CI, benchmarks, and docs

- [ ] T061 Add CI matrix for Windows, Ubuntu, and macOS.
- [ ] T062 Run fmt, clippy with warnings denied, and all workspace tests in CI.
- [ ] T063 Add reader benchmarks for open, lookup, random read, sequential read, and verify.
- [ ] T064 Record benchmark environment and distinguish cold/warm cache.
- [ ] T065 Update format document for every implementation-discovered ambiguity.
- [ ] T066 Add crate/API docs and examples for reading a block.
- [ ] T067 Add threat model and supported/unsupported codec documentation.
- [ ] T068 Run full acceptance checklist and record results.

## Release gate

- [ ] G001 All tasks T001–T068 complete or explicitly deferred with rationale.
- [ ] G002 No parser panic on the fuzz/regression corpus.
- [ ] G003 Reader open performs zero payload reads.
- [ ] G004 read_block reads exactly the selected block range.
- [ ] G005 verify_all validates every integrity record with bounded memory.
- [ ] G006 Writer output reopens and verifies using the reference reader.
- [ ] G007 Windows, Linux, and macOS CI green.
- [ ] G008 No compression ratio or direct-compute performance claim is made by this feature.
