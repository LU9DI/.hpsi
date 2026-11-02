# Activation calibration input

The CLI derives diagonal activation second moments from a rank-2, row-major F32 tensor already stored in an HPSI artifact.

```bash
cargo run -p hpsi-cli -- calibrate activations.hpsi 42 > layer-0-calibration.json
```

The tensor ID is the numeric ID shown by `hpsi inspect`. The command currently accepts only the RAW codec (codec ID 1), F32 dtype, and rank-2 tensors with shape `[samples, input_features]`.

## Output

The JSON document uses the `hpsi-activation-calibration` format, version 1. Its `activation_second_moment` array has one value per input feature:

`E[x_j^2] = (1 / samples) * sum_i(x[i, j]^2)`

The document records the source tensor ID/name, shape, sample count, and source payload I/O counters. It is designed to feed the `VqMatrix::encode_activation_aware` API after validation and any caller-selected flooring of zero-variance dimensions.

## Streaming and correctness properties

- Reads one HPSI payload block at a time; it does not assemble the full activation matrix.
- Carries partial F32 values across block boundaries, including boundaries that are not 4-byte aligned.
- Verifies contiguous logical offsets and RAW payload lengths while reading.
- Rejects non-finite activation values and shape/payload mismatches.
- Reports the exact payload bytes and read operations used for calibration.

## Important limitations

This command does **not** run a Transformer, hook model layers, or capture activations from a framework. It processes activation tensors that have already been exported into HPSI. It computes only diagonal second moments; it does not estimate a covariance matrix or capture correlations between input dimensions. Calibration quality still depends on representative input samples and the layer/tensor chosen.

A separate CPU-first Transformer capture utility is now documented in [Transformer activation capture](transformer-activation-capture.md). It exports per-layer calibration records for supported local Hugging Face causal language models; its framework coverage and real-model quality remain unvalidated. The multi-tensor Safetensors importer can consume matching calibration records by exact tensor name and feature count. Calibration provenance is validated and used during VQ encoding, but is not yet persisted in the resulting HPSI import metadata. Activation-weighted layer-output error and downstream perplexity/task quality still require evaluation on representative real model data.
