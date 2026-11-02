# Capturing Transformer activation statistics

`tools/capture_transformer_activations.py` captures the inputs to each `torch.nn.Linear` module in a local Hugging Face causal language model and writes one HPSI-compatible diagonal second-moment JSON file per layer.

## Scope and safety

- CPU-only capture; no CUDA dependency is required.
- The model and tokenizer must already exist in a local directory.
- `local_files_only=True` prevents model/tokenizer downloads; `trust_remote_code=False` prevents executing repository-supplied model code.
- The dataset is a UTF-8 JSONL file. Each non-empty line is either a JSON string or an object with a `text` field.
- The first version hooks `torch.nn.Linear` modules only. Architectures implemented with custom projections (for example, non-Linear module types) need an explicit adapter before their layers are covered.
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
- One JSON file per captured Linear layer, with `format=hpsi-activation-calibration`, `version=1`, `layer_name`, `sample_count`, `shape`, and `activation_second_moment`.

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

A caller can pass the validated vector directly to
`VqMatrix::encode_activation_aware(weights, rows, cols, config,
&calibration.activation_second_moment)`. The matrix column count must equal
`calibration.features`. This is a library integration point; a complete source-model
weight importer and layer-by-layer end-to-end evaluation are still separate work.

## Inspecting local Safetensors weights

The Rust `hpsi-ops` crate now includes a bounded-header, random-access Safetensors reader.
It parses tensor names, dtypes, shapes and byte ranges without loading the weight payloads.
For rank-2 F32 tensors, `SafeTensorFile::read_f32_rows` reads only a requested row window.
The CLI exposes metadata inspection:

```bash
cargo run -p hpsi-cli -- safetensors-inspect ./model.safetensors
```

This first importer stage deliberately supports reading F32 row windows only; F16/BF16
and other supported metadata dtypes can be identified but are not silently converted.
The current CLI command reports metadata and does not yet quantize tensors or write an
HPSI model. Sharded model indexes, automatic layer/calibration matching, and end-to-end
weight conversion remain follow-up work.
