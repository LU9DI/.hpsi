# Direct VQ CPU benchmark

Run the deterministic synthetic baseline in release mode:

```sh
cargo run -p hpsi-ops --example vq_benchmark --release
```

The report includes:
- encode time;
- dense F32 bytes, VQ codebook-plus-index bytes, and complete HPSI artifact bytes;
- artifact compression ratio including HPSI metadata and directories;
- relative L2 and maximum absolute output error on 16 deterministic held-out input vectors;
- p50/p95 dense and direct-VQ matvec latency over 100 repetitions.

This is a microbenchmark on a synthetic 256 × 256 matrix. It is not representative of a production transformer layer and must not be used to claim model-quality preservation. Run on an otherwise idle machine and record CPU, OS, compiler, and power settings when comparing results.
