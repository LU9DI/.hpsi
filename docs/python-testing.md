# Running HPSI Python utility tests

The Python utilities in `tools/` use the standard library for their validation
and helper tests. They do not require Aider, PyTorch, Transformers, or packages
from the global Python installation.

Run the tests from the repository root and scope discovery to this repository's
`tests` directory. Do not use an unrestricted discovery command in a Python
environment that has unrelated packages installed.

## Windows PowerShell

```powershell
py -3.14 -m venv .venv-hpsi
.\.venv-hpsi\Scripts\python.exe -m unittest discover -s tests -p "test_validate_calibration_bundle.py" -v
.\.venv-hpsi\Scripts\python.exe -m unittest discover -s tests -p "test_capture_transformer_activations.py" -v
.\.venv-hpsi\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

If Python 3.14 is not installed, replace `py -3.14` with an installed
supported interpreter. The venv commands above isolate installed third-party
packages; no `pip install` is needed for these standard-library tests.

## Rust workspace

From the repository root, run:

```powershell
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Run the Rust commands with the toolchain specified by the repository's
`rust-toolchain.toml`, if present. Clippy with `-D warnings` is an additional
strict check and may expose pre-existing warnings beyond the test suite.

## Interpreting discovery errors

An import path such as `...site-packages\aider\tests\test_coder.py` means
Python loaded a third-party package's tests, not HPSI's tests. Missing modules
such as `tiktoken`, `rich`, or an incompatible `openai.error` import are
therefore not evidence of an HPSI regression.

A `KeyboardInterrupt` while importing Python's `ssl` module indicates that
the process was interrupted during import; it is not a test assertion failure.
Rerun the scoped commands and retain the complete output and exit code.

The calibration-bundle tests validate metadata and provenance. The activation
capture tests exercise helper functions only; they do not load a checkpoint or
prove model-level quality. Full checkpoint evaluation remains a separate step.
