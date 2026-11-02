# HPSI Container Format v0.1

**Status:** Proposed normative draft for Feature 001  
**Endianness:** Little-endian  
**Addressing:** All file offsets are absolute byte offsets from byte zero  
**Magic:** ASCII `HPSI` (`48 50 53 49`)

This document is the byte-level contract for the first HPSI container. Implementations must not infer unspecified fields or silently accept invalid layouts. The v0.1 format is experimental and not yet a stable interchange format.

## 1. Design constraints

- Opening a file must not require reading tensor payloads.
- Directories must be sufficient to locate any block without scanning preceding payload data.
- Payload bytes are opaque to the container layer.
- A valid file must be verifiable without materializing logical tensors.
- Every untrusted count, offset, length, dimension, and reference is validated before use.
- The writer emits one canonical region order; the reader may accept other non-overlapping region orders only if all declared invariants hold.
- No compression codec or mathematical representation is implied by the file extension.

## 2. Canonical region order

The canonical writer layout is:

1. fixed header;
2. metadata JSON;
3. tensor directory;
4. block directory;
5. payload region;
6. integrity directory.

Every region starts at an 8-byte-aligned offset. Padding bytes between regions must be zero. The header is exactly 128 bytes. The file size in the header must equal the actual file length.

Regions must not overlap. The reader must use checked addition for every `offset + length` calculation and verify the result is within the actual file size. Region ordering and alignment must be validated independently of any canonical-writer assumption.

For an empty optional region, its offset, length/count, and entry size must be zero. The tensor and block directories are required for any non-empty model.

## 3. Fixed header — 128 bytes

All offsets below are decimal byte offsets from the beginning of the file.

| Offset | Size | Type | Field | v0.1 rule |
|---:|---:|---|---|---|
| 0 | 4 | bytes | magic | Exactly `HPSI` |
| 4 | 2 | u16 | format_major | Exactly 0 |
| 6 | 2 | u16 | format_minor | Exactly 1 |
| 8 | 4 | u32 | header_size | Exactly 128 |
| 12 | 4 | u32 | flags | Unknown set bits are mandatory/unsupported and rejected |
| 16 | 8 | u64 | file_size | Exact actual file length |
| 24 | 8 | u64 | metadata_offset | Absolute offset; 8-byte aligned |
| 32 | 8 | u64 | metadata_length | Bytes in metadata region |
| 40 | 8 | u64 | tensor_dir_offset | Absolute offset; 8-byte aligned |
| 48 | 4 | u32 | tensor_count | Must be within configured cap |
| 52 | 4 | u32 | tensor_entry_size | Exactly 128 |
| 56 | 8 | u64 | block_dir_offset | Absolute offset; 8-byte aligned |
| 64 | 4 | u32 | block_count | Must be within configured cap |
| 68 | 4 | u32 | block_entry_size | Exactly 64 |
| 72 | 8 | u64 | payload_offset | Absolute offset; 8-byte aligned when non-empty |
| 80 | 8 | u64 | payload_length | Stored payload bytes |
| 88 | 8 | u64 | integrity_offset | Absolute offset; 8-byte aligned when non-empty |
| 96 | 4 | u32 | integrity_count | Must be within configured cap |
| 100 | 4 | u32 | integrity_entry_size | Exactly 32 when count > 0; otherwise 0 |
| 104 | 16 | bytes | artifact_id | Opaque 128-bit identifier; writer may generate random bytes |
| 120 | 8 | bytes | reserved | All zero in v0.1 |

### 3.1 Header flags

All v0.1 header flag bits are reserved. A v0.1 writer must write zero. A v0.1 reader must reject any non-zero bit as an unsupported mandatory feature. Future optional capabilities must be introduced through a versioned extension mechanism rather than silently consuming reserved bits.

## 4. Metadata region

The metadata region is UTF-8 JSON without a byte-order mark. It must be a JSON object and must include:

- `format`: exactly `"hpsi"`;
- `metadata_version`: integer `1`;
- `tensor_names`: array of strings; each string is a non-empty tensor name;
- optional `model`: descriptive object;
- optional `producer`: descriptive object;
- optional `user`: descriptive object.

Tensor names are referenced by `name_ref` in the tensor directory. Duplicate names are forbidden in v0.1. Tensor names must be valid UTF-8, contain no NUL, and be at most 1024 UTF-8 bytes each.

The parser must enforce configured maximum metadata bytes, JSON nesting depth, string length, array length, and object member count. Unknown JSON properties are ignored only in descriptive objects. Unknown required top-level properties are not allowed in v0.1. Metadata is data, never executable code.

The writer should emit deterministic JSON (stable key ordering and no insignificant whitespace) to support reproducible artifacts.

## 5. Tensor directory — 128-byte entries

Entries are sorted by strictly increasing `tensor_id`. IDs are unique. A tensor's blocks occupy one contiguous range in the block directory.

| Offset | Size | Type | Field | Rule |
|---:|---:|---|---|---|
| 0 | 4 | u32 | tensor_id | Unique stable ID |
| 4 | 1 | u8 | rank | 0 through 8 |
| 5 | 1 | u8 | dtype | Defined below |
| 6 | 2 | u16 | codec | Default codec ID for blocks |
| 8 | 4 | u32 | flags | Must be zero in v0.1 |
| 12 | 4 | u32 | block_count | At least 1 |
| 16 | 8 | u64 | first_block_index | Index into block directory |
| 24 | 8 | u64 | logical_nbytes | Exact logical tensor byte count |
| 32 | 64 | u64[8] | dimensions | First `rank` dimensions are non-zero; unused entries are zero |
| 96 | 4 | u32 | name_ref | Index into metadata `tensor_names` |
| 100 | 4 | u32 | reserved0 | Zero |
| 104 | 8 | u64 | representation_ref | `0xffffffffffffffff` means none; otherwise references tensor_id |
| 112 | 16 | bytes | reserved1 | All zero |

### 5.1 Dtype identifiers

| ID | Name | Bytes per element |
|---:|---|---:|
| 1 | U8 | 1 |
| 2 | I8 | 1 |
| 3 | F16 | 2 |
| 4 | BF16 | 2 |
| 5 | F32 | 4 |
| 6 | F64 | 8 |
| 7 | I16 | 2 |
| 8 | I32 | 4 |
| 9 | I64 | 8 |
| 10 | PACKED_U4 | 1/2 byte per element |
| 11 | PACKED_I4 | 1/2 byte per element |
| 12 | PACKED_U2 | 1/4 byte per element |
| 13 | PACKED_I2 | 1/4 byte per element |
| 14 | PACKED_U1 | 1/8 byte per element |
| 15 | PACKED_I1 | 1/8 byte per element |
| 16 | OPAQUE | No dimension-derived byte calculation; logical_nbytes is authoritative |

Packed dtypes use low-order element first within each byte. The first logical element occupies the least significant bits. The final byte may contain unused high bits, which must be zero when emitted by the canonical writer. For non-OPAQUE types, the product of dimensions and dtype storage width must equal `logical_nbytes`; multiplication uses checked arithmetic. Rank zero represents a scalar with one element.

`OPAQUE` represents a byte stream whose semantic element type is not represented by a native HPSI dtype. It may be used for codec-defined structures and for lossless preservation of tensors from a source format when HPSI has no equivalent native dtype. For source-format preservation, the importer must retain the original source dtype and shape in descriptive metadata; the HPSI dimensions do not replace that provenance. `logical_nbytes` is authoritative, and readers must not infer element width, numeric interpretation, or permission to execute data from `OPAQUE`. It is not a license to execute embedded code.

## 6. Block directory — 64-byte entries

Entries are grouped by tensor directory order. Within a tensor, `block_id` starts at zero and increments by one. Logical ranges are contiguous, non-overlapping, and cover exactly `logical_nbytes` for the tensor. A v0.1 tensor may not contain sparse logical holes.

| Offset | Size | Type | Field | Rule |
|---:|---:|---|---|---|
| 0 | 4 | u32 | tensor_id | Must identify owning tensor |
| 4 | 4 | u32 | block_id | Zero-based within tensor |
| 8 | 4 | u32 | flags | Zero in v0.1 |
| 12 | 2 | u16 | codec | 0 means inherit tensor codec; otherwise overrides |
| 14 | 2 | u16 | layout | 0 means opaque/codec-defined layout |
| 16 | 8 | u64 | logical_offset | Offset in logical tensor byte stream |
| 24 | 8 | u64 | logical_length | Logical bytes represented by block; > 0 |
| 32 | 8 | u64 | payload_offset | Absolute file offset |
| 40 | 8 | u64 | payload_length | Stored payload bytes; > 0 |
| 48 | 8 | u64 | auxiliary_ref | `0xffffffffffffffff` means none; otherwise references tensor_id |
| 56 | 4 | u32 | checksum_index | Integrity directory index or `0xffffffff` if absent |
| 60 | 4 | u32 | reserved | Zero |

A block's payload range must be wholly contained within the declared payload region. Payload ranges must not overlap. A block descriptor does not assert that payload length equals logical length. A codec-specific implementation defines how opaque payload maps to logical bytes.

The reader validates that every block in a tensor's declared block range has the matching tensor ID and that the block logical ranges are contiguous from zero to the tensor's logical byte count.

## 7. Codec identifiers

Codec IDs are namespaced 16-bit identifiers. The container parser does not decode codecs.

| ID | Name | v0.1 meaning |
|---:|---|---|
| 0 | INHERIT | Block inherits tensor codec; invalid as tensor default |
| 1 | RAW | Payload bytes are the uncompressed logical byte stream for that block |
| 2 | ZSTD_FRAME | One independently decodable Zstandard frame per block; decoder must enforce output limits |
| 3 | HPSI_CODEBOOK_V1 | Reserved experimental codebook payload; no production decoder in Feature 001 |
| 4 | HPSI_LOW_RANK_V1 | Reserved experimental factor representation |
| 5 | HPSI_LOOKUP_V1 | Reserved experimental lookup-assisted representation |
| 6 | HPSI_RESIDUAL_V1 | Reserved experimental residual representation |

Unknown codec IDs are not automatically a malformed container. The parser may inspect and report them, but a runtime must return `UnsupportedCodec` when asked to decode or compute with an unknown codec. No codec may be loaded as executable code from the artifact.

For RAW, `payload_length == logical_length`. For ZSTD_FRAME, the frame must be independently decodable and decompressed output must equal `logical_length`; decompression must be bounded by configured limits.

## 8. Integrity directory — 32-byte entries

| Offset | Size | Type | Field | Rule |
|---:|---:|---|---|---|
| 0 | 4 | u32 | target_kind | 1 = metadata region; 2 = block payload |
| 4 | 4 | u32 | target_index | 0 for metadata; block directory index for payload |
| 8 | 2 | u16 | algorithm | 1 = CRC32C (Castagnoli) |
| 10 | 2 | u16 | flags | Zero |
| 12 | 16 | bytes | digest | CRC32C little-endian in bytes 0..3; bytes 4..15 zero |
| 28 | 4 | u32 | reserved | Zero |

Integrity records are optional in v0.1. The integrity directory may be empty, and metadata or individual block payloads may have no checksum record. At most one integrity record may target a given metadata region or block payload. A block's `checksum_index` is either `0xffffffff` (no checksum) or an index into the integrity directory; when present, it must refer to a record whose target kind is block payload and whose target index equals that block's directory index. A metadata checksum record uses target index zero. Every integrity record must be referenced consistently by its target; duplicate targets and orphaned records are invalid.

`verify_all()` verifies every declared integrity record and reports the number successfully checked. It does not imply that unrecorded regions were checksummed. Applications requiring full integrity coverage must check that metadata and every block have a checksum record before treating verification as complete.

CRC32C detects accidental corruption; it does not authenticate a file or protect against a malicious producer who can rewrite the file and checksum. Cryptographic hashes, signatures, and authenticity are out of scope for v0.1.

## 9. Structural invariants

A reader must reject a file when any of the following is true:

- magic, major/minor version, or fixed entry sizes are unsupported;
- actual file length differs from header `file_size`;
- header reserved bytes or v0.1 flags are non-zero;
- a required region is missing, misaligned, outside the file, or overlaps another region;
- region-size multiplication or offset addition overflows;
- a count or byte size exceeds configured limits;
- metadata is invalid UTF-8, invalid JSON, violates the schema, or exceeds parser limits;
- tensor IDs/names are duplicated, name references are invalid, rank/dimensions/dtype are invalid, or logical byte calculations disagree;
- block ranges are invalid, unsorted, overlapping, incomplete, or refer to the wrong tensor;
- payload ranges escape the payload region or overlap;
- auxiliary references point to missing tensor IDs or form a cycle in the v0.1 reference graph;
- checksum references are inconsistent or a verified checksum fails;
- a reserved field is non-zero;
- a codec-specific invariant (such as RAW length equality) is violated.

All validation must occur before exposing a parsed model as usable. Parsing failure must not leave a partially initialized public model object.

## 10. Resource limits

The library must expose a configurable `ReaderLimits` structure. Initial safe defaults (subject to benchmark and fuzz review) are:

| Limit | Default |
|---|---:|
| Maximum file size | 1 TiB |
| Maximum metadata bytes | 16 MiB |
| Maximum tensors | 1,000,000 |
| Maximum blocks | 16,000,000 |
| Maximum rank | 8 |
| Maximum tensor logical bytes | 1 TiB |
| Maximum payload bytes per block | 64 MiB |
| Maximum total directory bytes | 2 GiB |
| Maximum JSON nesting depth | 32 |
| Maximum JSON string bytes | 1 MiB |
| Maximum tensor name bytes | 1024 |

Applications may lower these limits. Raising them requires an explicit caller choice. Limits must be checked before allocation and before decompression.

## 11. Random-access reader API contract

The initial API is language-specific in implementation, but its semantics are fixed:

- `open(source, limits) -> ModelReader`: read and validate header, metadata, and directories; do not read payload.
- `model_info() -> ModelInfo`: return version, artifact ID, counts, and metadata summary.
- `tensor(id) -> TensorDescriptor`: resolve tensor in O(1) or O(log n).
- `block(tensor_id, block_id) -> BlockDescriptor`: resolve block without scanning payload.
- `read_block(tensor_id, block_id) -> BlockData`: read exactly the declared payload range and return descriptor plus bytes.
- `verify_all() -> VerificationReport`: stream through every declared integrity record using bounded buffers; the result must not be interpreted as full artifact coverage when records are optional or absent.
- `verify_padding(max_bytes) -> Result<u64>`: explicitly verify zero-filled gaps between the fixed header and declared top-level regions, refusing to scan when the total gap length exceeds the caller-supplied byte budget. Payload regions and trailing bytes after the last declared region are not read.
- `stats() -> ReaderStats`: expose metadata/directory/payload bytes read and read-operation counts.

The reader must support an injectable random-access source abstraction so tests can assert exactly which ranges were read.

## 12. Writer contract

- Validate logical input and all references before writing.
- Write canonical region order and zero-filled alignment padding.
- Create a uniquely named temporary file in the destination directory using exclusive creation; never follow or overwrite an existing temporary path.
- Flush buffered writes, synchronize file contents where supported, reopen with the reference reader, and verify the artifact.
- Publish without replacing an existing destination. The current implementation uses a same-directory hard link, which provides atomic no-replace publication on filesystems that support hard links; unsupported filesystems return an I/O error.
- Synchronize the parent directory where supported. If this fails after publication, report that the destination may already be visible while crash durability is uncertain.
- On ordinary errors and panic unwinding, clean up only the temporary file created by this operation. Cleanup is best-effort.
- Never delete or rewrite unrelated files.

### 12.1 Abandoned temporary files

Temporary files use the name pattern `.hpsi-<process-id>-<sequence>.tmp`. A process crash, forced termination, or power loss can prevent the in-process cleanup guard from running. The library deliberately does not remove old files automatically: age and process identifiers alone cannot prove that a file is orphaned, and an automatic sweep could delete a file belonging to an active writer or a process whose identifier was reused.

Operational cleanup must therefore be performed only when no HPSI writer can be active in the target directory. Inspect each candidate and remove only files matching the exact HPSI temporary-file naming pattern; do not follow symbolic links or remove arbitrary neighboring files. Applications that cannot guarantee writer quiescence should leave such files untouched and surface them for operator review.

## 13. Conformance fixtures

The repository must include:
- minimal one-tensor RAW artifact;
- one tensor split into multiple contiguous blocks;
- multiple tensors with non-sequential IDs;
- metadata checksum and per-block checksum examples;
- ZSTD frame fixture if the decoder dependency is selected;
- invalid magic/version/header-size fixtures;
- invalid offsets, overflow, overlap, truncation, invalid UTF-8/JSON, bad name refs;
- logical coverage gap/overlap, wrong tensor ownership, payload overlap/out-of-bounds;
- invalid checksum references and corrupted payload;
- oversized counts and dimensions proving limits are checked before allocation.

Each fixture must have a documented expected outcome and be stable across platforms.
