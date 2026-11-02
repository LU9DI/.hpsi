# HPSI fuzzing

The fuzz package is isolated from the stable workspace and uses cargo-fuzz/libFuzzer with nightly Rust. The ordinary Rust test matrix does not run fuzzers; a green unit-test matrix is not evidence of sustained fuzzing.

## Targets

- `fuzz_header`: exercises the fixed-header parser with arbitrary byte slices.
- `fuzz_model_reader`: exercises complete container opening and, for accepted inputs, bounded CRC32C verification.

## Run locally

```sh
rustup toolchain install nightly
cargo install cargo-fuzz --locked
cargo +nightly fuzz run fuzz_header -- -max_total_time=600
cargo +nightly fuzz run fuzz_model_reader -- -max_total_time=600
```

The repository also provides a manual GitHub Actions workflow named **Fuzz**. Choose **Actions → Fuzz → Run workflow** and set the duration per target. It runs both targets and uploads crash artifacts if any are produced.

To seed a run, place valid and malformed `.hpsi` examples in `fuzz/corpus/<target-name>/`. Keep minimized crashers as regression fixtures in the normal test suite before fixing the underlying issue.

Generated corpora, crash artifacts, and fuzz build output are local working data; do not commit them wholesale. Commit only small, intentional regression fixtures.
