# Capturing Transformer activation statistics

`tools/capture_transformer_activations.py` captures the inputs to supported projection modules in a local Hugging Face causal language model and writes one HPSI-compatible diagonal second-moment JSON file per captured layer.

The currently supported module types are:

- `torch.nn.Linear`
- Hugging Face Transformers `Conv1D` (`transformers.pytorch_utils.Conv1D`), used by architectures such as GPT-2.

## Scope and safety

- CPU-only capture; no CUDA dependency is required.
- The model and tokenizer must already exist in a local directory.
- `local_files_only=True` prevents model/tokenizer downloads; `trust_remote_code=False` prevents executing repository-supplied model code.
- The dataset is a UTF-8 JSONL file. Each non-empty line is either a JSON string or an object with a `text` field.
- The exporter explicitly supports `torch.nn.Linear` and Hugging Face Transformers `Conv1D`. Architectures implemented with other custom projection types need an explicit adapter before their layers are covered.
- Input feature dimensions are determined according to the module type:
  - `Linear`: `in_features`.
  - Transformers `Conv1D`: `weight.shape[0]`, because its weight layout is `(in_features, out_features)`.
- The exporter does not transpose or modify model weights. It captures the input activations and uses the corresponding input feature dimension for validation.
- Each captured module is mapped to the conventional Safetensors parameter key `<module path>.weight` (or `weight` for a root projection). This matches common Hugging Face checkpoints but is not a universal architecture mapping; inspect the checkpoint keys before importing.
- Statistics are accumulated online as float64 sums of squares. Full activation matrices are not retained.
- Each layer has an independent token-row cap. The captured statistic is `E[x_j^2]`, not centered variance or a full covariance matrix.
- Unsupported module types are skipped. The exporter fails if it finds no supported projection modules.

## Install optional dependencies

Use an isolated Python environment and install a PyTorch build suitable for the host CPU:

```bash
python -m venv .venv-capture
# Windows:
.venv-capture\Scripts\activate
# Linux/macOS:
source .venv-capture/bin/activate
python -m pip install torch transformers
```

## Run against a local model

```bash
python tools/capture_transformer_activations.py \
  --model ./models/local-causal-lm \
  --dataset ./calibration.jsonl \
  --output-dir ./artifacts/calibration \
  --max-sequences 128 \
  --max-length 512 \
  --max-activation-rows 65536
```

The script does not download weights. If the model is not already present locally, it fails rather than fetching it.

## Outputs and provenance

The output directory contains:

- `manifest.json`: model type/revision when available, dataset SHA-256, sequence count, layer-to-file mapping, and (from bundle version 2) one architecture record per **parameter**.
- One JSON file per captured projection layer, with `format=hpsi-activation-calibration`, `version=1`, `layer_name` (the mapped Safetensors weight key), `sample_count`, `shape`, and `activation_second_moment`.
- The manifest records both `module_name` and `source_tensor_name` so the mapping can be audited.
- The manifest also records `module_type`, identifying each *measured* module as `linear` or `conv1d`. Only projections appear here, because only projections have input activations to measure.
- `module_type` drives the `weight_orientation` persisted by the importer (`linear` → `linear_out_in`, layout `[out, in]`; `conv1d` → `conv1d_in_out`, layout `[in, out]`). The importer never infers orientation from tensor names or shapes. Calibration features are validated against the **input** dimension of the declared orientation (Linear: `shape[1]`, Conv1D: `shape[0]` — Transformers declares `Conv1D(nf, nx)` with an `[nx, nf]` weight, and the capture reads `nx`/`nf` rather than assuming a Linear-shaped API).
- The same class evidence also names the architectural role, recorded in the container as `tensor_role` beside the orientation. A tensor with no class evidence is recorded as `tensor_role: unknown`, never silently as a projection. See `docs/format-v0.1.md` §4.2 and `docs/orientation-policy.md`.

### Bundle version 2: one record per parameter

Version 1 recorded one module class per physical Safetensors key. That was enough
for orientation and not enough for identity: a tied embedding and output head is
one array reached through two module classes, and the older record could keep only
one of them — which turned the table into a `linear_projection` and dropped its
`index_axis`. Version 2 adds a `tensors` array to `manifest.json` with one record
per parameter **object**, grouped on `id(parameter)` and on nothing else:

```json
{
  "canonical_name": "transformer.wte.weight",
  "aliases": ["lm_head.weight", "transformer.wte.weight"],
  "identity": "parameter-object-identity",
  "shape": [10, 16],
  "dtype": "F32",
  "tensor_roles": ["embedding_table", "linear_projection"],
  "index_axis": 0,
  "weight_orientation": "linear_out_in",
  "status": "resolved",
  "conflicts": [],
  "evidence": [
    {"alias": "lm_head.weight", "module": "lm_head", "module_class": "linear",
     "tensor_role": "linear_projection", "weight_orientation": "linear_out_in"},
    {"alias": "transformer.wte.weight", "module": "transformer.wte",
     "module_class": "embedding", "tensor_role": "embedding_table", "index_axis": 0}
  ]
}
```

- `aliases` are the parameter names the model exposes for that one object. Two
  parameters holding equal values, equal shapes or names that look related stay
  two records: equality is never a grouping key.
- `evidence` is the association the role list cannot express: which module, under
  which alias, with which class, evidences which role.
- `canonical_name` is the single physical key the parameter is stored under. A
  parameter that resolves to no key, or to several, is `status: "unresolved"` or
  `"conflict"` with the reasons in `conflicts`, and states **no** role, layout or
  index axis: an unsettled identity is reported, not guessed.
- A class whose declared dimensions disagree with the parameter it holds
  evidences nothing. The record says so in `conflicts` instead of assigning the
  role that class would otherwise imply.
- `architecture_unresolved` lists the identities that produced no declaration, and
  `layers[].aliases` links each measured row back to its record.

**Embeddings are measured by no hook and identified by one record.** The hook reads
a module's input activations, and the input of an `nn.Embedding` is token ids, so no
feature-wise second moment of an embedding table exists; the capture must not invent
one, and `diagonal_second_moment` must not be reused to look like one. What the table
does need is architecture: `classify`-by-class gives it `tensor_role:
embedding_table` and `index_axis: 0` (Hugging Face stores `[num_embeddings,
embedding_dim]` and indexes rows), which is a documented property of the class rather
than an inference from the matrix. A table that is *also* a measured head keeps both
roles in one record.

**Compatibility.** `tools/validate_calibration_bundle.py` and
`tools/architecture_manifest_from_calibration.py` accept bundle version 1 and 2 and
refuse anything else. A version 1 bundle states less — one class per key, no alias,
no second role, no index axis — and the manifest entries derived from it are exactly
as strong as that record: nothing is inferred from a name to fill the gap. The
architecture manifest keeps `format=hpsi-architecture-manifest`, `version=1`;
`aliases` and `role_evidence` are optional keys that older readers ignore, and a
manifest that pairs a role with an alias or a layout it does not imply is refused by
both auditors.
- Legacy bundles without `module_type` are accepted with the documented legacy default (`linear_out_in`, recorded as `orientation_source: legacy_default`); they are not silently reinterpreted. The container-format contract is documented in `docs/format-v0.1.md` §4.1.

The second-moment field is directly compatible with the vector expected by `VqMatrix::encode_activation_aware`. The exporter emits calibration data; it does not quantize weights or perform HPSI inference.

## Import the checkpoint into HPSI

After checking that the manifest's `source_tensor_name` values match the keys reported by `hpsi safetensors-inspect`, run:

```bash
cargo run -p hpsi-cli -- safetensors-import ./model.safetensors ./model.hpsi --calibration-dir ./artifacts/calibration
```

The importer reads each calibration JSON in the directory (excluding `manifest.json`), validates it, and matches it to the exact source tensor name. The filesystem writer reopens the output, verifies all integrity records, and publishes without overwriting an existing destination. By default, tensors not eligible for VQ are preserved byte-for-byte. Add `--reject-raw` to fail instead when any source tensor cannot be quantized. Without `--calibration-dir`, the same command imports without activation calibration.

## Reproducibility and limitations

Record the exact local model snapshot, tokenizer files, Python/PyTorch/Transformers versions, dataset contents, and command-line limits alongside experimental results. Dataset SHA-256 identifies the exact JSONL bytes, but does not prove the dataset is representative.

The model is loaded in float32 on CPU, which can require substantial RAM. Start with a small local model and a small calibration corpus.

The current hook supports standard `torch.nn.Linear` and Hugging Face Transformers `Conv1D` projections. It does not yet cover every Transformer architecture, custom projection classes, fused attention implementations, quantized modules, or multimodal models. The Safetensors name mapping is conventional and must be checked against the actual checkpoint.

Unit tests cover module classification and input-feature dimension handling. They do not replace an integration run against a real local model. No real-model quality or perplexity claim is made until layer-output and end-task evaluations are run.

## Import calibration into the Rust quantizer

The `hpsi-ops` crate now exposes `ActivationCalibration::from_json`. It validates the format/version, statistic name, sample/feature shape, moment vector, and dataset SHA-256 provenance before exposing the per-feature values. Zero second moments are accepted: they can represent inactive features, and the activation-aware encoder applies a small positive numerical floor internally when deriving its group-position scale.

The multi-tensor Safetensors importer accepts validated calibration records through `ModelImportPlan::build`, or through `ModelImportPlan::build_with_provenance` when the calibration provenance must be persisted. Records match exact source tensor names and their feature count must equal the tensor's input dimension. Eligible rank-2 F32/F16/BF16 weights are quantized with activation-aware VQ; other tensors follow the selected raw-preservation or reject policy. Use `write_model_import_to_hpsi_calibrated` or `write_model_import_to_hpsi_path_calibrated` with the provenance-aware plan to store model type, optional revision, dataset SHA-256, sample count, feature count, and statistic identifier in the HPSI manifest. The path writer reopens and verifies the artifact before no-replace publication. The numeric moment vector is not copied into metadata. Real-model layer-output and task-quality evaluation remain separate work.

## Inspecting local Safetensors weights

The Rust `hpsi-ops` crate now includes a bounded-header, random-access Safetensors reader. It parses tensor names, dtypes, shapes and byte ranges without loading the weight payloads. The reader validates all 22 Safetensors dtypes currently supported by the format parser without loading tensor payloads. For rank-2 F32, F16 and BF16 tensors, `SafeTensorFile::read_f32_rows` reads only a requested row window and converts the selected values to F32; `copy_raw_payload` preserves any supported dtype's original bytes without conversion.

The CLI exposes metadata inspection:

```bash
cargo run -p hpsi-cli -- safetensors-inspect ./model.safetensors
```

The `safetensors-inspect` CLI command reports metadata only. The library also exposes a single-tensor VQ convenience path and a complete multi-tensor import pipeline: `ModelImportPlan::build` plans every tensor, while `write_model_import_to_hpsi` emits one HPSI container and `write_model_import_to_hpsi_path` verifies and publishes a file without replacing an existing destination. For persistent calibration provenance, use `build_with_provenance` and the calibrated writer variants. VQ is currently limited to eligible rank-2 F32/F16/BF16 tensors; other non-scalar, non-empty tensors can be preserved byte-for-byte or rejected by policy. Sharded model indexes and automatic calibration-name mapping across architectures remain follow-up work.