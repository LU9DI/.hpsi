# HPSI fuzzing

The fuzz package is intentionally isolated from the stable workspace and uses
cargo-fuzz/libFuzzer with nightly Rust. The ordinary cross-platform CI does not
run fuzzers; a green unit-test matrix is not evidence of a sustained fuzz run.

## Targets

- `fuzz_header`: exercises the fixed-header parser with arbitrary byte slices.
- `fuzz_model_reader`: exercises complete container opening and, for accepted
  inputs, bounded CRC32C verification.

## Run locally

Install nightly Rust and cargo-fuzz:

```sh
rustup toolchain install nightly
cargo install cargo-fuzz --locked
```

Run each target for at least ten minutes and preserve any crash inputs as
regression fixtures:

```sh
cargo +nightly fuzz run fuzz_header -- -max_total_time=600
cargo +nightly fuzz run fuzz_model_reader -- -max_total_time=600
```

To seed a run, place valid and malformed `.hpsi` examples in the matching
`fuzz/corpus/<target-name>/` directory. Keep minimized crashers in the normal
test suite before fixing the underlying issue.

Generated corpora, crash artifacts, and fuzz build output are local working
data; do not commit them wholesale. Commit only small, intentional regression
fixtures.
