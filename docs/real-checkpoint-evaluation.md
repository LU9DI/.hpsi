# Real-checkpoint calibration and evaluation

This protocol turns a locally available Hugging Face checkpoint and a fixed JSONL
corpus into a provenance-checked HPSI calibration bundle. It does not claim that
activation-aware VQ preserves model quality: that must be measured separately.

## 1. Prepare a reproducible input

- Use a local Hugging Face model directory containing the model and tokenizer.
- Keep the evaluation corpus in UTF-8 JSONL. Each non-empty line must be either
  a JSON string or an object with a non-empty `text` field.
- Keep the corpus fixed between ordinary-VQ and activation-aware runs.
- Record the checkpoint identity, corpus origin, and any preprocessing.

The capture utility is CPU-first and uses `local_files_only=True`; it will not
download missing model files. It currently captures inputs to `torch.nn.Linear`
modules only.

## 2. Capture activation statistics

Install the optional dependencies in an isolated Python environment, then run:

```sh
python tools/capture_transformer_activations.py \
  --model /path/to/local-checkpoint \
  --dataset /path/to/evaluation.jsonl \
  --output-dir ./calibration-output \
  --max-sequences 128 \
  --max-length 512
```

The exporter records diagonal input-activation second moments, the model type
and revision exposed by the local Transformers config, the dataset SHA-256,
processed sequence count, and per-layer feature dimensions. The row cap is
applied independently to each layer.

## 3. Validate provenance before importing

```sh
python tools/validate_calibration_bundle.py ./calibration-output \
  --dataset /path/to/evaluation.jsonl
```

Do not proceed unless the command reports `"valid": true`. Supplying
`--dataset` verifies the corpus bytes against the manifest. The validator also
checks that every layer artifact agrees with the manifest's model identity,
dataset hash, processed-sequence count, layer name, shape, and feature count.

Treat a failed validation as a hard stop. Do not edit hashes or metadata merely
to make a bundle pass; recapture from the intended model and corpus instead.

## 4. Import into HPSI

Use the repository's Safetensors import command with the calibration directory:

```sh
hpsi safetensors-import /path/to/model.safetensors output.hpsi \
  --calibration-dir ./calibration-output
```

The current activation-aware import path is limited to supported rank-2
F32/F16/BF16 tensors. Other source tensors follow the importer's raw-preserve
or rejection policy; a successful container import is not proof that every
tensor was activation-aware quantized. Inspect the import report and the
container metadata to determine which tensors were quantized, preserved, or
rejected.

## 5. Required evaluation before making quality claims

For each checkpoint, report at minimum:

- exact model identifier/revision and Safetensors shard set;
- corpus SHA-256, processed sequences, tokenizer, maximum sequence length;
- per-layer count of eligible, calibrated, quantized, and raw-preserved tensors;
- artifact bytes, dense reference bytes, and compression ratio including
  container metadata and indexes;
- weight-domain reconstruction error and held-out layer-output error;
- activation-weighted output error using held-out activations;
- peak host memory, encode time, load time, and inference latency p50/p95;
- end-to-end task metric such as perplexity on a held-out corpus, where the
  model/runtime path supports a valid measurement.

Compare ordinary VQ and activation-aware VQ with identical model weights,
corpus, held-out inputs, codebook configuration, and hardware. Repeat with
multiple seeds where the encoder is stochastic. Preserve raw machine-readable
results and environment details.

A calibration bundle passing validation proves only structural and provenance
consistency. It does not establish lower output error, preserved perplexity,
faster inference, or acceptable model quality. Current synthetic VQ benchmarks
must not be presented as real-checkpoint results.
