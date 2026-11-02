# Capturing Transformer activation statistics

`tools/capture_transformer_activations.py` captures the inputs to each `torch.nn.Linear` module in a local Hugging Face causal language model and writes one HPSI-compatible diagonal second-moment JSON file per layer.

## Scope and safety

- CPU-only capture; no CUDA dependency is required.
- The model and tokenizer must already exist in a local directory.
- `local_files_only=True` prevents model/tokenizer downloads; `trust_remote_code=False` prevents executing repository-supplied model code.
- The dataset is a UTF-8 JSONL file. Each non-empty line is either a JSON string or an object with a `text` field.
- The first version hooks `torch.nn.Linear` modules only. Architectures implemented with custom projections (for example, non-Linear module types) need an explicit adapter before their layers are covered.
- Each captured module is mapped to the conventional Safetensors parameter key `<module path>.weight` (or `weight` for a root Linear). This matches common Hugging Face checkpoints but is not a universal architecture mapping; inspect the checkpoint keys before importing.
- Statistics are accumulated online as float64 sums of squares. Full activation matrices are not retained.
- Each layer has an independent token-row cap. The captured statistic is `E[x_j^2]`, not centered variance or a full covariance matrix.

## Install optional dependencies

Use an isolated Python environment and install a PyTorch build suitable for the host CPU:

```bash
python -m venv .venv-capture
# Windows:
.venv-capture\\Scripts\\activate
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
- `manifest.json`: model type/revision when available, dataset SHA-256, sequence count, and layer-to-file mapping.
- One JSON file per captured Linear layer, with `format=hpsi-activation-calibration`, `version=1`, `layer_name` (the mapped Safetensors weight key), `sample_count`, `shape`, and `activation_second_moment`.
- The manifest records both `module_name` and `source_tensor_name` so the mapping can be audited.

The second-moment field is directly compatible with the vector expected by `VqMatrix::encode_activation_aware`. The exporter emits calibration data; it does not quantize weights or perform HPSI inference.

## Reproducibility and limitations

Record the exact local model snapshot, tokenizer files, Python/PyTorch/Transformers versions, dataset contents, and command-line limits alongside experimental results. Dataset SHA-256 identifies the exact JSONL bytes, but does not prove the dataset is representative.

The model is loaded in float32 on CPU, which can require substantial RAM. Start with a small local model and a small calibration corpus. The current hook targets standard Linear modules and does not yet cover every Transformer architecture, fused attention implementation, convolution-style projection, or multimodal model. No real-model quality or perplexity claim is made until layer-output and end-task evaluations are run.

## Import calibration into the Rust quantizer

The `hpsi-ops` crate now exposes `ActivationCalibration::from_json`. It validates the
format/version, statistic name, sample/feature shape, moment vector, and dataset SHA-256
provenance before exposing the per-feature values. Zero second moments are accepted:
they can represent inactive features, and the activation-aware encoder applies a small
positive numerical floor internally when deriving its group-position scale.

The multi-tensor Safetensors importer accepts validated calibration records through
`ModelImportPlan::build`, or through `ModelImportPlan::build_with_provenance` when
the calibration provenance must be persisted. Records match exact source tensor names
and their feature count must equal the tensor's input dimension. Eligible rank-2
F32/F16/BF16 weights are quantized with activation-aware VQ; other tensors follow the
selected raw-preservation or reject policy. Use
`write_model_import_to_hpsi_calibrated` or
`write_model_import_to_hpsi_path_calibrated` with the provenance-aware plan to store
model type, optional revision, dataset SHA-256, sample count, feature count, and
statistic identifier in the HPSI manifest. The path writer reopens and verifies the
artifact before no-replace publication. The numeric moment vector is not copied into
metadata. Real-model layer-output and task-quality evaluation remain separate work.

## Inspecting local Safetensors weights

The Rust `hpsi-ops` crate now includes a bounded-header, random-access Safetensors reader.
It parses tensor names, dtypes, shapes and byte ranges without loading the weight payloads.
The reader validates all 22 Safetensors dtypes currently supported by the format parser without loading tensor payloads. For rank-2 F32, F16 and BF16 tensors, `SafeTensorFile::read_f32_rows` reads only a requested row window and converts the selected values to F32; `copy_raw_payload` preserves any supported dtype's original bytes without conversion.
The CLI exposes metadata inspection:

```bash
cargo run -p hpsi-cli -- safetensors-inspect ./model.safetensors
```

The `safetensors-inspect` CLI command reports metadata only. The library also exposes
a single-tensor VQ convenience path and a complete multi-tensor import pipeline:
`ModelImportPlan::build` plans every tensor, while `write_model_import_to_hpsi` emits
one HPSI container and `write_model_import_to_hpsi_path` verifies and publishes a file
without replacing an existing destination. For persistent calibration provenance, use
`build_with_provenance` and the calibrated writer variants. VQ is currently limited to eligible rank-2
F32/F16/BF16 tensors; other non-scalar, non-empty tensors can be preserved byte-for-byte
or rejected by policy. Sharded model indexes and automatic calibration-name mapping across
architectures remain follow-up work.
