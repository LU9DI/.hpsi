# HPSI — Etapa C2d: validação integral pré-release

**Data:** 2026-11-07
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**Commit-base:** `99b967c27ae6e7c5e609711eef1543c6635930a0`
**Escopo:** reexecutar os gates completos do HPSI Toolkit v0.1 no estado exato do working tree e produzir evidências atuais de prontidão. Nenhuma alteração funcional de produto. O `fuzz/Cargo.lock` foi versionado em commit próprio (reprodutibilidade), separado deste relatório.

---

## 1. Estado Git inicial e final

| Item | Início | Fim |
|---|---|---|
| Branch | `feat/hpsi-core-container` | `feat/hpsi-core-container` |
| HEAD (base C2d) | `99b967c27ae6e7c5e609711eef1543c6635930a0` | `ae737d9…` (lockfile) + commit deste relatório |
| Índice | vazio | vazio após o commit |
| Trabalho pré-existente | 15 ` M` + 31 `??` | 15 ` M` + 30 `??` (o `fuzz/Cargo.lock` passou a rastreado) |
| Hunk `pytest` em `ci.yml` | preservado | preservado (não tocado) |
| `crates/hpsi-format/` | sem diff | sem diff |

Nenhum `reset`, `clean`, `checkout`, `restore`, rebase, push, merge, tag ou deploy.

## 2. Ferramentas e versões

| Ferramenta | Versão |
|---|---|
| Rust (pinned por `rust-toolchain.toml`) | `rustc 1.85.0`, `cargo 1.85.0` (`1.85.0-x86_64-pc-windows-msvc`) |
| Nightly (fuzzing) | `1.101.0-nightly` |
| `cargo-fuzz` | 0.13.2 |
| Python | 3.14.6 |
| pytest / numpy | 8.4.2 / 2.5.1 |
| torch (global) | **ausente** |
| torch (`.venv-capture`) | 2.15.0+cpu |
| SO | Windows 10 (10.0.19045) |

## 3. Gates Rust

| Gate | Comando | Exit | Duração | Resultado |
|---|---|---:|---:|---|
| Formatação | `cargo fmt --all -- --check` | **0** | 20 s | sem divergências |
| Lint | `cargo clippy --workspace --all-targets -- -D warnings` | **0** | 97 s | 0 avisos |
| Testes | `cargo test --workspace --all-targets` | **0** | 37 s | **278 passed / 0 failed / 0 ignored** (17 suítes) |

Distribuição (soma 278): bin 4 · cli 15 · format lib 46 · reader 14 · source 3 · writer 9 · ops lib 78 · direct_vq 11 · embedding_contract_e2e 11 · embedding_layout_parity 3 · identity_alias_contract 16 · identity_alias_parity 5 · orientation_contract 47 · reconstruction_regression 13 · vq_quality 3 · 2 exemplos com 0.

Logs: `.tmp-phase4/phaseC2d/10-fmt.txt`, `11-clippy.txt`, `12-test.txt`.

## 4. Gates Python

| Gate | Comando | Exit | Duração | Resultado |
|---|---|---:|---:|---|
| Compilação | `python -m compileall -q tools tests` | **0** | 1 s | sem saída |
| unittest | `python -m unittest discover -s tests -p "test_*.py"` | **0** | 10 s | **Ran 109 tests — OK** |
| pytest | `python -m pytest tests -q` | **0** | 16 s | **205 passed** |

**Dependências:** os testes coletados precisam de `pytest` e `numpy` (presentes). `torch` **não** é importado pela suíte; é exigido apenas por `tools/eval_hpsi_parity.py` (ferramenta manual, não teste). Nenhum teste foi omitido por falta de dependência.

Logs: `.tmp-phase4/phaseC2d/20-compileall.txt`, `21-unittest.txt`, `22-pytest.txt`.

## 5. Benchmark e fuzz-smoke

| Gate | Comando | Exit | Duração | Resultado |
|---|---|---:|---:|---|
| VQ benchmark | `cargo run -p hpsi-ops --example vq_benchmark --release` | **0** | 82 s | ver números abaixo |
| Fuzz build | `cargo +nightly fuzz build` (ambos) | **0** | 83 s | `fuzz_header` + `fuzz_model_reader` compilados |
| Smoke header | `... fuzz run fuzz_header -- -max_total_time=20` | **0** | 21 s | 618.423 runs, ~29.448 exec/s, 0 crashes |
| Smoke reader | `... fuzz run fuzz_model_reader -- -max_total_time=20` | **0** | 76 s | 629 runs (replay do corpus de 627 arquivos), 0 crashes |

Números do VQ benchmark (release, run atual): `beam_16_heldout_relative_l2 = 0.309849`, `beam_16_artifact_bytes = 18432`, `beam_16_matvec_p50_ms = 0.0858`; `beam_8_heldout_relative_l2 = 0.312106`. São medições **sintéticas** (256×256), não qualidade de modelo real.
Logs: `.tmp-phase4/phaseC2d/30-vq-benchmark.txt`, `31-fuzz-build.txt`, `32-smoke-header.txt`, `33-smoke-reader.txt`.

> Observação: o smoke do `fuzz_model_reader` levou 76 s porque primeiro reprocessa todo o corpus acumulado (627 arquivos, ~9 MB); não é uma campanha de 1 h. As campanhas completas de 3600 s estão na Etapa C2c.

## 6. CI e documentação

Validação do workflow **commitado** (`git show HEAD:.github/workflows/ci.yml`), via PyYAML e inspeção:

| Verificação | Resultado |
|---|---|
| Sintaxe YAML | **OK** |
| `permissions` | `{contents: read}` |
| Jobs | `python-capture-tools`, `rust`, `vq-benchmark`, `fuzz-smoke` |
| Job `format` (auto-commit) | **ausente** |
| `needs: format` | **ausente** |
| `cargo fmt --all -- --check` | presente |
| `cargo clippy --workspace --all-targets -- -D warnings` | presente |
| `cargo test --workspace --all-targets` | presente |
| `vq-benchmark` / `fuzz-smoke` | presentes |
| Comandos `git add/commit/push/config` | **nenhum** |

O **hunk local pré-existente de `pytest`** (no job `python-capture-tools`) permanece **apenas no working tree** e não foi incorporado a esta etapa. Log: `.tmp-phase4/phaseC2d/40-ci-validation.txt`.

## 7. Reprodutibilidade do fuzzing (lockfile)

- `fuzz/Cargo.lock` (versão 4, **22 pacotes**) foi gerado no build da C2c e confere com as versões resolvidas na campanha (`libfuzzer-sys 0.4.13`, `arbitrary 1.5.0`, `cc 1.11.1`, `shlex 2.0.1`, `jobserver 0.1.35`, `libc 0.2.191`, `getrandom 0.4.3`, `r-efi 6.0.0`, além das deps do workspace).
- SHA-256 do lockfile: `f40da9f2f2c2264a79475f545838949fc87e8a76710c2196178035193c81f28e`.
- **Versionado em commit próprio** `ae737d9ec1c2b34842fc95a9eb29aa0554352312` (`build(fuzz): pin fuzzing dependency lockfile`), separado deste relatório.
- Sem esse pin, a campanha C2c **não** era integralmente reproduzível (requisito `libfuzzer-sys = "0.4"`). Com o lockfile versionado, as versões ficam fixas.

## 8. Comparação com resultados históricos

| Gate | Fase 4.8 (histórico) | C2d (atual) | Situação |
|---|---|---|---|
| `cargo fmt` | 0 | 0 | igual |
| `clippy -D warnings` | 0 | 0 | igual |
| Rust tests | 278 | **278** | igual |
| pytest | 205 | **205** | igual |
| unittest | 109 | **109** | igual |
| compileall | 0 | 0 | igual |
| VQ benchmark | exit 0 | exit 0 | igual (medições sintéticas, variam por runner) |
| Fuzz build | compila | compila | igual |
| Campanha fuzzing | não executada | C2c: 0 crashes | ver C2c |

## 9. Gates aprovados, reprovados e não executados

**Aprovados (exit 0):** fmt, clippy, cargo test, compileall, unittest, pytest, vq_benchmark, fuzz build, smoke header, smoke reader, YAML/CI.

**Reprovados:** nenhum.

**Não executados (e por quê):**
- Campanhas de 3600 s — já documentadas na C2c; não repetidas (instrução explícita).
- `cargo test --workspace --doc` — não é gate declarado da CI; não executado nesta etapa.
- `tools/eval_hpsi_parity.py` — requer `torch` e um par de artefatos; é ferramenta manual, não teste.
- CI no GitHub Actions — exigiria push (proibido).
- Cobertura LLVM — indisponível (sem `llvm-tools`), conforme C2c.

## 10. Limitações

1. Validação executada em **um host** (Windows 10); a matriz Linux/macOS é feita pela CI (não executada aqui por ausência de push).
2. Gates Rust usam a toolchain pinada `1.85.0-msvc`; o fuzzing usa `1.101.0-nightly` — duas toolchains, registradas.
3. VQ benchmark é **sintético**; não mede qualidade de modelo real.
4. `eval_hpsi_parity.py` não foi exercitado (sem torch no Python global).
5. O smoke do reader reprocessa o corpus acumulado (627 arquivos), o que explica a taxa baixa de exec/s.

## 11. Pendências de release

- Reexecutar os gates na CI (Linux/macOS/Windows) via push — **fora do escopo** e proibido nesta etapa.
- Decidir sobre o hunk `pytest` de `ci.yml` (deve entrar num commit de CI próprio).
- Empacotamento/instalação verificável (C2e).
- Escopo público do v0.1 (toolkit, sem inferência).

## 12. Preservação

- 15 ` M` + 30 `??` preservados após o commit do lockfile (31 → 30 porque `fuzz/Cargo.lock` passou a rastreado). `fuzz/coverage/` permanece como scratch não versionado.
- Hunk `pytest` de `ci.yml` **não** foi incorporado; permanece no working tree.
- `crates/hpsi-format/` sem diff; nenhum arquivo descartado/movido/sobrescrito.

## 13. Ausência de push, merge, tag e deploy

Confirmado: nenhum `push`, `merge`, `tag` ou `deploy`. Ambos os commits (lockfile e este relatório) são locais.

---

*Relatório da Etapa C2d. Todos os gates executáveis aprovados (exit 0); nenhum reprovado.*
