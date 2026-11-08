# HPSI — Etapa C2f: correções de distribuição e preparação da release v0.1

**Data:** 2026-11-07
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**HEAD-base:** `503d13c90d8b50a3ba21a7621f85fe73b745c094`
**Escopo:** corrigir os bloqueios de empacotamento da auditoria C2e e revalidar a distribuição local dos três crates. Nenhuma publicação, push, tag, release ou deploy.

---

## 1. HEAD e estado Git

| Momento | HEAD | M / ?? |
|---|---|---|
| Início C2f | `503d13c90d8b50a3ba21a7621f85fe73b745c094` | 15 / 30 |
| Após A | `9726e66a5b9a21b822b8c6f4f7ef5407f49591a1` | 15 / 30 |
| Após B | `a948469d7f9dd96e1518c2f959b1a6ea5eda4451` | 15 / 30 |
| Após D | `3fdfa4d88b670880b6e43ec1707feb663c095220` | 15 / 30 |
| Final (relatório) | commit deste relatório | 15 / 30 |

- Índice sempre vazio antes de cada commit; `git diff --cached` conferido arquivo a arquivo.
- Os **15 ` M` preexistentes** foram preservados; o hunk `pytest` de `ci.yml` e o hunk §4.1/§4.2 de `docs/format-v0.1.md` continuam apenas no working tree.
- Nenhum `reset`, `clean`, `checkout`, `restore`, rebase, push, merge, tag, release ou deploy.

## 2. Correções e commits

### C2f-A — `9726e66` `fix: declare versioned HPSI workspace dependencies`

- `Cargo.toml`: novo `[workspace.dependencies]` com `hpsi-format`/`hpsi-ops` (`path` + `version = "0.1.0"`).
- `crates/hpsi-ops/Cargo.toml`: `hpsi-format = { workspace = true }`.
- `apps/hpsi-cli/Cargo.toml`: `hpsi-format`/`hpsi-ops = { workspace = true }`.
- Correção real do P1 de C2e: dependências internas agora têm versão compatível e resolução de workspace, sem duplicar configuração.

### C2f-B — `a948469` `fix: include HPSI package documentation and license files`

- Cópias **byte-idênticas** (hashes conferidos) de `LICENSE-MIT`, `LICENSE-APACHE`, `THIRD-PARTY-NOTICES` para cada crate:
  `crates/hpsi-format/`, `crates/hpsi-ops/`, `apps/hpsi-cli/`.
- `readme = "../../README.md"` nos três manifests (Cargo copia o README da raiz para o pacote).
- Nenhum conteúdo legal alterado; nenhuma titularidade inferida.

### C2f-C — `feat: improve HPSI CLI help and version handling` — **BLOQUEADO**

- `apps/hpsi-cli/src/main.rs` e `apps/hpsi-cli/tests/cli.rs` estão entre os **15 arquivos com 600+ linhas de trabalho preexistente (Fases 4.x)**. O `run()` do CLI foi reescrito por esse trabalho; a adição de `--help`/`--version` cai dentro das regiões já modificadas.
- Staging seletivo produziria um commit inconsistente (HEAD + apenas o novo hunk, sem o código preexistente do qual ele depende) e não pode ser comprovado com segurança.
- Conforme a regra ("se não puder ser comprovado com segurança, não commitar e reportar"), **não alterei esses arquivos** e **não criei o commit**. O CLI continua sem `--help`/`--version` (`exit 1` + `usage`).

### C2f-D — `3fdfa4d` `docs: clarify HPSI v0.1 installation and capabilities`

- `README.md`: novas seções **“Building and installing the CLI”** e **“HPSI Toolkit v0.1 — scope and limitations”** (Rust 1.85, build/install, exit codes, `hpsi: <message>`, RAW-only, codecs 3–6 sem decoder, `inspect`/`verify` não decodificam, Mathematical directory/Shared assets como propostas, toolkit ≠ runtime, ferramentas Python e deps). Registra que `--help`/`--version` **não** existem no v0.1.
- `docs/format-v0.1.md §7`: adicionada a frase explícita de que v0.1 emite/decodifica **somente RAW** e que ZSTD/codecs 2–6 não têm decoder nesta versão.
- **Staging seletivo comprovado:** o commit de `docs/format-v0.1.md` contém apenas o hunk do §7 (blob construído a partir do HEAD + §7); o hunk preexistente §4.1/§4.2 permanece no working tree (`git diff` mostra `Architectural role`).

## 3. Gates executados (C2f-E)

| Gate | Comando | Exit | Duração | Resultado |
|---|---|---:|---:|---|
| Rust fmt | `cargo fmt --all -- --check` | **0** | — | sem divergências |
| Rust clippy | `cargo clippy --workspace --all-targets -- -D warnings` | **0** | 50 s | 0 avisos |
| Rust test | `cargo test --workspace --all-targets` | **0** | 16 s | **278 passed / 0 failed / 0 ignored** |
| Metadados | `cargo metadata --locked --format-version 1` | **0** | — | 3 pacotes |
| Package format | `cargo package -p hpsi-format --locked --allow-dirty --no-verify` | **0** | — | 1 `.crate` |
| Package ops | `… -p hpsi-ops … --config 'patch.crates-io.hpsi-format.path="crates/hpsi-format"'` | **0** | — | 1 `.crate` |
| Package cli | `… -p hpsi-cli …` (patches format+ops) | **0** | — | 1 `.crate` |
| Python compileall | `python -m compileall -q tools tests` | **0** | 0 s | sem saída |
| Python unittest | `python -m unittest discover -s tests -p "test_*.py"` | **0** | 2 s | **Ran 109 tests — OK** |
| Python pytest | `python -m pytest tests -q` | **0** | 6 s | **205 passed** |

**Nota de empacotamento:** sem o auxílio `--config 'patch.crates-io...'`, `cargo package -p hpsi-ops`/`-p hpsi-cli` falha com `no matching package named 'hpsi-format' found; location searched: crates.io index`. O patch é **somente um auxílio local de validação** (não altera manifests); publicar exige publicar `hpsi-format` primeiro (ou registry privado).

## 4. Artefatos `.crate` (tamanho e SHA-256)

| `.crate` | Tamanho | SHA-256 |
|---|---:|---|
| `hpsi-format-0.1.0.crate` | 52.459 B | `6cc14f443d689f9482957e85f5d7246f5e2fcaa27b45b045312e2aaddd6da5c7` |
| `hpsi-ops-0.1.0.crate` | 148.926 B | `0eb1be1772b4e506208db88d12d5bddee3fd325b45d843de5470709e07e4fd9a` |
| `hpsi-cli-0.1.0.crate` | 35.999 B | `14e4a72036e2ef2d69e0295d1e7ae86f3ef09677bab587464df8a37afdb21497` |

Artefatos em `target/package/` (gitignored).

## 5. Inventário físico dos pacotes (extraídos)

Extraídos em `C:/Users/lucas/AppData/Local/Temp/hpsi-c2f-IH5VPL` (fora do repositório) e inspecionados:

- **hpsi-format**: `Cargo.toml`, `README.md`, `LICENSE-MIT`, `LICENSE-APACHE`, `THIRD-PARTY-NOTICES`, `src/*.rs` (11), `tests/{reader,source,writer}.rs`.
- **hpsi-ops**: os mesmos documentos + `src/*.rs` (8), `tests/*.rs` (8), `examples/{audit_reconstruction,vq_benchmark}.rs`.
- **hpsi-cli**: os mesmos documentos + `src/{main,orientation}.rs`, `tests/cli.rs`.

Nenhum pacote inclui `fuzz/`, `.tmp-*`, `prepare_gpt2_test.py`, `.bak`, corpus ou artefatos de fuzzing. **README e os três documentos legais estão fisicamente presentes** em todos os `.crate` (comprovado por `tar -tzf`), não apenas no metadado SPDX.

## 6. Instalação limpa e execução fora do repositório

- `cargo install --path apps/hpsi-cli --root .tmp-phase4/phaseC2f/install-clean --locked` → **exit 0**; `hpsi.exe` 808.448 B.
- Testes do CLI instalado:

| Comando | Exit | Observação |
|---|---:|---|
| `hpsi --help` | **1** | imprime `usage` (`--help` não implementado — C2f-C bloqueado) |
| `hpsi --version` | **1** | idem (`--version` não implementado) |
| `hpsi inspect valid.hpsi` | **0** | HPSI 0.1, 21 tensores |
| `hpsi verify valid.hpsi` | **0** | `verified 22 CRC32C records` |
| `hpsi inspect truncated.hpsi` | **1** | `invalid HPSI header: declared file size differs from actual size` |
| `hpsi bogus` | **1** | `usage` |

- Execução **fora do repositório** em `C:/Users/lucas/AppData/Local/Temp/hpsi-c2f-clean-IMoRCn`: os mesmos resultados; `ldd` contém **0** referências a `lucas`/`HPSI`. Sem dependência do diretório de desenvolvimento.

## 7. Classificação dos gates

| Gate | Classificação |
|---|---|
| Dependências versionadas (A) | **APROVADO** |
| README/licença/notices nos `.crate` (B) | **APROVADO** |
| CLI `--help`/`--version` (C) | **BLOQUEADO** (trabalho preexistente nos arquivos) |
| Docs de instalação/limites (D) | **APROVADO** |
| fmt / clippy / test / metadata | **APROVADO** |
| Empacotamento dos 3 crates (com auxílio local) | **APROVADO** |
| Empacotamento sem auxílio (registry) | **BLOQUEADO** (dep interna não publicada) |
| Instalação limpa + fora do repo | **APROVADO** |
| Python compileall/unittest/pytest | **APROVADO** |
| Publicação em registry | **NÃO EXECUTADO** (proibido) |

## 8. Limitações e itens não executados

1. `cargo package` de dependentes exige `hpsi-format` publicado; validação usou `--config patch` local. Publicar exige ordem de publicação ou registry privado.
2. `--help`/`--version` não implementados (C2f-C bloqueado).
3. Gates executados em um host Windows; Linux/macOS ficam para a CI (sem push).
4. `tools/eval_hpsi_parity.py` não exercitado (torch ausente no Python global; `.venv-capture` tem torch 2.15.0+cpu).
5. Cobertura LLVM e campanhas de 1 h não repetidas (ver C2c/C2d).

## 9. Pendências legais

- **Titularidade:** os textos usam `Copyright (c) 2026 Lucas`, derivado da autoria do repositório (464 commits por `Lucas <bostavacabosta@gmail.com>`). **Isso não comprova a titularidade legal** (apenas um prenome). **Não alterei o texto.** É necessária confirmação do titular legal antes de considerar a release legalmente fechada.
- Nenhum `NOTICE` adicional foi criado além do `THIRD-PARTY-NOTICES` (Unicode-3.0; NCSA do `libfuzzer-sys` é build-only do fuzz, não distribuído).
- Licença dual `MIT OR Apache-2.0` preservada, com os dois textos integrais nos pacotes.

## 10. Preservação e ausência de publicação

- 15 ` M` + 30 `??` preservados; hunk `pytest` de `ci.yml` e hunk §4.1/§4.2 de `docs/format-v0.1.md` intactos; `crates/hpsi-format/` sem diff de código (apenas cópias legais/readme adicionadas e commitadas).
- Nenhuma publicação em crates.io/PyPI/registry; nenhum push, merge, tag, release ou deploy.
- Instalações/extrações em diretórios temporários isolados (`.tmp-phase4/phaseC2f/...` e `%TEMP%`).

---

*Relatório da Etapa C2f. C2f-C bloqueado; demais correções aplicadas e revalidadas.*
