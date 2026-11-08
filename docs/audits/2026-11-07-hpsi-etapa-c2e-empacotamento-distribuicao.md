# HPSI — Etapa C2e: empacotamento, instalação e distribuição v0.1

**Data:** 2026-11-07
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**HEAD-base:** `6a68071e0971bae22dcd2e2c08e5d30d07706a33`
**Escopo:** validar empacotamento Cargo, instalação isolada do CLI, distribuição independente, ferramentas Python e documentação de instalação. Nenhuma publicação, push, tag ou deploy. Único artefato versionado: este relatório (nenhuma correção documental foi aplicada — ver §11).

---

## 1. HEAD e estado Git inicial/final

| Item | Início | Fim |
|---|---|---|
| Branch | `feat/hpsi-core-container` | `feat/hpsi-core-container` |
| HEAD (base C2e) | `6a68071e0971bae22dcd2e2c08e5d30d07706a33` | este relatório (commit próprio) |
| Índice | vazio | vazio após o commit |
| Trabalho pré-existente | 15 ` M` + 30 `??` | 15 ` M` + 30 `??` preservados |
| Hunk `pytest` em `ci.yml` | preservado | preservado (não tocado) |
| `crates/hpsi-format/` | sem diff | sem diff |

Nenhum `reset`, `clean`, `checkout`, `restore`, rebase, push, merge, tag, release ou deploy.

## 2. Inventário dos pacotes

| Pacote | Caminho | Publicável? | Binário | Observação |
|---|---|---|---|---|
| `hpsi-format` | `crates/hpsi-format` | sim (candidato) | lib | `license.workspace`, sem `readme`/`license-file` explícitos |
| `hpsi-ops` | `crates/hpsi-ops` | sim (candidato) | lib | dep path `hpsi-format` **sem `version`** |
| `hpsi-cli` | `apps/hpsi-cli` | sim (candidato) | `[[bin]] hpsi` | deps path `hpsi-format`/`hpsi-ops` **sem `version`** |
| `hpsi-fuzz` | `fuzz/` | **não** (`publish = false`, workspace isolado) | 2 bins de fuzz | harness de verificação |

- **Toolkit v0.1** = `hpsi-format` + `hpsi-ops` + `hpsi-cli` + ferramentas Python + documentação/legals.
- **Fora da release**: `hpsi-fuzz` (dev), `.tmp-*`, `fuzz/corpus`, `fuzz/coverage`, `prepare_gpt2_test.py`, `.bak`.
- **Python**: **não há** `pyproject.toml`, `setup.py`, `setup.cfg` ou `requirements*.txt` → ferramentas distribuídas como **scripts do repositório**.
- Requisitos: Rust `1.85.0` (edition 2024); Python 3.x para as ferramentas; `numpy` (auditor) e `torch`/`transformers` (captura) opcionais.

## 3. Comandos e resultados (gates)

| Gate | Comando | Exit | Duração | Resultado |
|---|---|---:|---:|---|
| Metadados | `cargo metadata --locked --format-version 1` | **0** | — | 3 pacotes HPSI; bin `hpsi` |
| Lista de pacote | `cargo package --workspace --list` | **101** | 10 s | bloqueado por working tree "dirty" (sem `--allow-dirty`) |
| Lista (allow-dirty) | `cargo package --workspace --allow-dirty --list` | **0** | 0 s | 56 linhas; **sem README/LICENSE/NOTICE** |
| Empacotamento | `cargo package --workspace --locked --no-verify --allow-dirty` | **101** | 8 s | `hpsi-format` OK; `hpsi-ops`/`hpsi-cli` falham |
| Instalação isolada | `cargo install --path apps/hpsi-cli --root .tmp-phase4/phaseC2e/install-cli --locked` | **0** | 18 s | `hpsi.exe` instalado |
| CLI (no args/--help/--version) | `hpsi …` | **1** | — | imprime `usage` (não há `--help`/`--version`) |
| CLI inspect válido | `hpsi inspect valid.hpsi` | **0** | — | 21 tensores/21 blocos |
| CLI verify válido | `hpsi verify valid.hpsi` | **0** | — | `verified 22 CRC32C records` |
| CLI inspect truncado | `hpsi inspect truncated.hpsi` | **1** | — | `invalid HPSI header: declared file size differs from actual size` |
| CLI comando inválido | `hpsi bogus` | **1** | — | `usage` |
| Sala limpa | `cd clean && ./bin/hpsi.exe inspect data/valid.hpsi` | **0** | — | idêntico; sem dependência do repo |
| Build de `.crate` | extrair `hpsi-format-0.1.0.crate` + `cargo build --offline` | **0** | 174 s | compila isoladamente |

Logs: `.tmp-phase4/phaseC2e/10-package-list.txt`, `11-package-list-dirty.txt`, `12-package.txt`, `20-install-cli.txt`, `28-cli-validation.txt`, `29-cleanroom.txt`, `30-crate-build.txt`, `40-python-imports.txt`, `41-python-entrypoints.txt`, `13-metadata.json`.

## 4. Resultados de `cargo package`

- **`hpsi-format`**: empacotado com sucesso — **19 arquivos**, 160,5 KiB (29,4 KiB comprimido).
- **`hpsi-ops`**: **FAILED** — `all dependencies must have a version specified when packaging. dependency 'hpsi-format' does not specify a version`.
- **`hpsi-cli`**: não alcançado (mesma causa: deps path sem `version`).
- **Consequência:** não é possível produzir o conjunto completo de `.crate` do toolkit sem adicionar `version` às dependências path — **não foi feito**, por instrução de não alterar manifests para contornar falhas.

**Achado adicional:** a lista e o `.crate` de `hpsi-format` **não incluem** `README`, `LICENSE-MIT`, `LICENSE-APACHE` ou `THIRD-PARTY-NOTICES` (esses arquivos estão na raiz do workspace, e os pacotes têm `readme = false`, sem `license-file`). O `Cargo.toml` normalizado mantém `license = "MIT OR Apache-2.0"`, mas o texto da licença **não viaja** dentro do `.crate`.

## 5. Artefatos gerados (tamanhos e hashes)

| Artefato | Tamanho | SHA-256 |
|---|---:|---|
| `target/package/hpsi-format-0.1.0.crate` | 30.090 B | `af8eecf9f9662114812c9d0d75231b50ac544847c56e5d0f0550f35958a1ad6187` |
| `.tmp-phase4/phaseC2e/install-cli/bin/hpsi.exe` | 808.448 B | `ace104472b22da442c5bf16383a45ca3025b7a582f53661d91c8db9fe21bb484` |

`target/package/` está sob `target/` (gitignored). O build independente do `.crate` ocorreu em diretório temporário isolado: `C:/Users/lucas/AppData/Local/Temp/hpsi-crate-RAcl0x` (fora do repositório).

## 6. Instalação isolada e teste de sala limpa

- `cargo install --path apps/hpsi-cli --root <prefixo> --locked` compilou e instalou `hpsi.exe` **sem registry e sem instalação global**.
- Em `.tmp-phase4/phaseC2e/clean/` (cópia do binário + fixtures descartáveis):
  - `inspect data/valid.hpsi` → exit 0;
  - `verify data/valid.hpsi` → exit 0 (22 CRC32C);
  - `inspect data/truncated.hpsi` → exit 1 com erro tipado.
- `ldd hpsi.exe` mostra **apenas DLLs do sistema** (KERNEL32, ntdll, ucrtbase…); `strings` não contém caminhos do repositório. **Sem dependência do diretório de desenvolvimento.**

## 7. Distribuição independente (`.crate`)

- `hpsi-format-0.1.0.crate` extraído e compilado com `cargo build --offline` em diretório temp **fora do repositório** → **exit 0**. O conteúdo empacotado é autossuficiente (depende só de `serde_json` do registry).
- O mesmo **não** pôde ser demonstrado para `hpsi-ops`/`hpsi-cli`: o empacotamento falha antes. A distribuição correta do conjunto dependeria de versões nas deps path (não aplicado).

## 8. Ferramentas Python

- **Sem configuração de pacote** → distribuídas como **scripts do repositório** (`python tools/<script>.py …`). Nada foi criado (conforme regra).
- Dependências por ferramenta:
  - `audit_hpsi_container.py` → `numpy`;
  - `eval_hpsi_parity.py` → `numpy`, `torch`, e os módulos locais `audit_hpsi_container`/`architecture_oracle`;
  - `capture_transformer_activations.py` → `torch`/`transformers` (uso opcional);
  - `architecture_oracle.py`, `validate_calibration_bundle.py`, `architecture_manifest_from_calibration.py` → stdlib.
- Entry points: `audit --help` 0 · `validate --help` 0 · `capture --help` 0 · `architecture_oracle --help` 0 · **`eval_hpsi_parity --help` 1** (falha por `torch` ausente no Python global; o `.venv-capture` tem `torch 2.15.0+cpu`).
- `python -m py_compile tools/*.py` → OK.
- Distribuição por scripts/documentação é **suficiente para o v0.1**, desde que o usuário instale `numpy`/`torch` conforme o script.

## 9. Documentação de instalação e uso (auditoria)

| Ponto exigido | Situação |
|---|---|
| 1. O que é o Toolkit v0.1 | **parcial** — README descreve o projeto e o estágio, mas não usa o rótulo "Toolkit v0.1" |
| 2. O que faz / não faz | **ok** — "Early-stage", "Explicit non-goals", "Still outstanding" |
| 3. Requisitos de sistema/toolchain | **parcial** — Rust 1.85/edition 2024 citados no status; sem seção dedicada |
| 4. Como compilar e instalar o CLI | **ausente** — exemplos usam `cargo run -p hpsi-cli -- …`; **não há** `cargo build --release` nem `cargo install` |
| 5. `inspect` e `verify` | **ok** — comandos citados; exemplos |
| 6. Ferramentas Python | **ok** — captura, auditor, validador, manifesto, parity documentados |
| 7. Localizar/interpretar erros | **parcial** — o CLI imprime `hpsi: <mensagem>`; não há doc de formato/exit codes |
| 8. Formato e arquivos suportados | **ok** — `docs/format-v0.1.md` |
| 9. Codecs/decoders disponíveis | **parcial** — README diz "RAW-only"/"compressed codecs are not implemented"; `docs/format-v0.1.md §7` descreve `ZSTD_FRAME` com "decoder must enforce output limits" (ambíguo) |

**Limites explícitos verificados:** v0.1 é toolkit, não runtime (README "no model compatibility/inference speedup"); RAW é o único formato emitido/decodificado (README l. 588); `inspect`/`verify` não decodificam codecs (`§7` "The container parser does not decode codecs"); codebook/low-rank/lookup/residual "reserved experimental" (sem decoder); "Mathematical directory"/"Shared assets" constam como **"Proposed"**.

**Inconsistências registradas (não corrigidas nesta etapa):** D1 (redação de `ZSTD_FRAME` em `docs/format-v0.1.md §7`) e D5 (as linhas "Mathematical directory"/"Shared assets" poderiam dizer explicitamente "não presente no v0.1"). Motivo de não corrigir: `docs/format-v0.1.md` é um dos 15 arquivos com alterações pré-existentes (entrelaçaria trabalho anterior) e a correção não é necessária para a instalação. Recomenda-se commit documental dedicado (ver §11).

## 10. Problemas encontrados

| # | Problema | Severidade | Evidência |
|---|---|---|---|
| P1 | `cargo package` falha para `hpsi-ops`/`hpsi-cli`: deps path sem `version` | **Alta** (bloqueia registry) | `12-package.txt` |
| P2 | `.crate` não inclui README/LICENSE/NOTICES | **Alta** (legal/distribuição) | `tar -tzf`; `readme = false` |
| P3 | CLI sem `--help`/`--version` | Média (UX) | `22-help.out`, `23-version.out` |
| P4 | README sem instruções de build/instalação do CLI | Média (doc) | §9 |
| P5 | `docs/format-v0.1.md §7` ambíguo sobre ZSTD | Baixa–Média (doc) | §7 |
| P6 | `eval_hpsi_parity.py` exige `torch` (ausente no Python global) | Baixa | `41-python-entrypoints.txt` |
| P7 | Sem configuração de pacote Python | Baixa (aceitável p/ v0.1) | §8 |

## 11. Dependências de plataforma

- Binário Windows: Rust `1.85.0-msvc`; depende apenas de DLLs do sistema/UCRT (`ldd`). Não testado em Linux/macOS neste host (a CI cobre a matriz).
- O `.crate` de `hpsi-format` depende de `serde_json 1` (registry).
- Ferramentas Python: `numpy` (auditor) e `torch`/`transformers` (captura/parity), não redistribuídos.

## 12. Avaliação de prontidão da distribuição

| Gate | Classificação |
|---|---|
| `cargo metadata` | **APROVADO** |
| `cargo package --list` | **APROVADO** (com `--allow-dirty`, pois o working tree tem o trabalho 4.x) |
| Empacotamento `hpsi-format` | **APROVADO** |
| Empacotamento `hpsi-ops`/`hpsi-cli` | **BLOQUEADO** (P1) |
| `.crate` com licença/README | **REPROVADO** (P2) |
| Instalação isolada do CLI | **APROVADO** |
| Sala limpa (sem repo) | **APROVADO** |
| Build independente do `.crate` | **APROVADO** (apenas `hpsi-format`) |
| Ferramentas Python | **APROVADO como scripts** (sem pacote) |
| Documentação de instalação | **PARCIAL** (P3–P5) |

**Conclusão:** a **instalação a partir do código-fonte** (`cargo install --path` / `cargo build --release`) está **demonstrada e aprovada**. A **distribuição por registry (crates.io)** está **bloqueada** por P1 (deps path sem versão) e P2 (licença/README fora do `.crate`). **Não declaro a distribuição por registry pronta.** Para o v0.1, o caminho recomendado é distribuição via repositório/tarball com `cargo install --path`, enquanto P1/P2 forem resolvidos numa etapa própria.

## 13. Preservação e ausência de publicação

- 15 ` M` + 30 ` ??` preservados; hunk `pytest` de `ci.yml` não incorporado; `crates/hpsi-format/` sem diff; fixture de referência `gpt2-tied.hpsi` com hash inalterado (`56b71fe2…`).
- Nenhuma publicação em crates.io/PyPI/registry; nenhum `push`, `merge`, `tag`, `release` ou `deploy`.
- Instalação de teste isolada em `.tmp-phase4/phaseC2e/install-cli`; build do `.crate` em diretório temp fora do repositório.

---

*Relatório da Etapa C2e. Fonte: logs em `.tmp-phase4/phaseC2e/`. Nenhuma correção de código/manifest/documentação aplicada.*
