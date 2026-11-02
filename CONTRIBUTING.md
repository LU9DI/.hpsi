# Contributing to HPSI

HPSI is an experimental systems project. Correctness, explicit resource bounds, and reproducible evidence take priority over feature count.

## Toolchain

The repository pins Rust 1.85.0 in `rust-toolchain.toml`. Install the toolchain with the rustup toolchain manager, then run:

```sh
rustc --version
cargo --version
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
```

## Development rules

- Work on a feature branch; do not push directly to `main`.
- Do not use `unsafe` in the format/parser crate.
- Treat every model file as untrusted input.
- Validate integer arithmetic, offsets, lengths, references, and limits before allocating.
- Do not read tensor payloads while opening or inspecting directory metadata.
- Do not claim compression, direct-compute, or performance gains without reproducible measurements.
- Keep the CPU reference behavior authoritative before adding accelerator-specific implementations.

## Current implementation status

Feature 001 is in progress. The initial workspace, typed errors, resource limits, fixed-header parser, region validation, and random-access source abstraction are being established. The CLI is currently only a shell; inspect/verify are not implemented yet.
