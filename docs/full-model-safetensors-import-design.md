# Full-model Safetensors import: design and invariants

Status: implementation plan for the next importer layer. The current tensor-level importer remains the only production API; this document does not claim that complete-model import is implemented.

## Current boundary

`SafeTensorFile::open` validates the Safetensors header, tensor byte ranges, element counts, and supported source dtypes. `encode_f32_tensor_to_hpsi` currently accepts one rank-2 `F32`, `F16`, or `BF16` tensor and writes a two-tensor VQ representation (`hpsi.vq.codebook` and `hpsi.vq.indices`). The writer requires the complete tensor directory before payload emission and requires strictly increasing tensor IDs and unique names.

Consequently, invoking the single-tensor API repeatedly against one output is not a valid multi-tensor strategy: each invocation expects an empty output and creates its own container. Concatenating those containers would not create a valid HPSI model.

## Required model-import contract

A future `encode_model_to_hpsi` API should accept a source file, VQ configuration, block sizing, and an explicit policy for tensors that cannot be VQ encoded. It must produce exactly one HPSI container and preserve, for every source tensor:

- original tensor name (or a reversible, collision-checked mapping);
- original shape and source dtype in model-level provenance;
- deterministic source ordering independent of JSON object iteration;
- an explicit outcome: quantized, copied losslessly, or skipped with a machine-readable reason.

The importer must not silently drop unsupported tensors. The default policy should fail closed when a tensor would otherwise be omitted. A caller may opt into lossless raw preservation for unsupported tensors; skipping tensors is not an accepted import policy.

## Streaming architecture

The current writer needs all `StreamingTensorInput` descriptors before writing the header and directories. The safe multi-tensor design is therefore a two-phase pipeline:

1. **Plan and stage:** enumerate source tensors in deterministic order; validate names, shapes, dtypes, dimensions, and policy; encode each supported matrix once; stage only payload streams in a managed seekable spool; record each staged range, block lengths, generated tensor descriptors, and source-to-output mapping.
2. **Publish:** assign globally unique, strictly increasing tensor IDs; construct the complete descriptor list; call `ModelWriter::write_streaming` once; serve each requested output block from its staged range using bounded buffers.

Staging must be bounded by disk capacity rather than model size in RAM. Each staged range must use checked arithmetic, exact byte lengths, and an ownership guard that closes the file before cleanup (including on Windows). The destination should be published atomically only after the writer has completed and the resulting container passes `ModelReader::verify_all`.

## Tensor naming and identity

Generated VQ tensors must be namespaced by the original tensor identity, for example `<source-name>.hpsi_vq.codebook` and `<source-name>.hpsi_vq.indices`. The implementation must reject collisions with source names and with other generated names before creating the destination. IDs should be allocated from a single monotonic counter across the entire output; per-tensor reuse of IDs 0 and 1 is invalid.

The metadata should include a versioned import manifest mapping source tensor names to generated tensor names, source shape/dtype, and the selected policy. The reader must continue to treat unknown manifest fields as optional metadata; the format version should not be bumped merely to add an optional manifest field.

## Calibration integration

Activation calibration is keyed by exact layer/tensor name and declares a feature count. Before activation-aware quantization, the importer must require an exact name match and require `calibration.features == source_tensor.shape[1]`. It must reject duplicate calibration entries, mismatched dimensions, non-finite values, and ambiguous aliases. A missing calibration may use the explicitly selected uncalibrated path; it must never be represented as a zero-valued calibration.

The activation-aware streaming encoder should share the same staging and HPSI writer path as ordinary VQ. Calibration provenance (sample count, model revision, dataset digest, statistic version) belongs in the import manifest, while the full per-feature moment vector need not be copied into the model unless a later runtime feature requires it.

## Validation gates

Before describing complete-model import as available, add tests for:

- two or more quantized source tensors in one HPSI container, with unique names and monotonically increasing IDs;
- deterministic output planning when Safetensors header keys arrive in a different order;
- duplicate generated-name and source-name collision rejection;
- exact calibration-name and feature-count matching;
- mixed supported and unsupported dtypes under fail-closed and lossless raw-preservation policies;
- bounded spool reads/writes across block boundaries;
- failure during staging and during final writing leaves no published partial destination and cleans temporary files;
- `ModelReader::verify_all` succeeds for the final artifact and all manifest mappings resolve.

The tests should assert metadata and payload semantics, not only tensor counts. Performance and peak-RSS claims require execution on representative model files and must not be inferred from unit tests.
