# HPSI — C2f-C1: reconciliação segura do CLI (`--help` / `--version`)

**Data:** 2026-11-08
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**HEAD-base:** `27d2bd344c35ef9f5c8383604ae42559cad42123`
**Objetivo:** investigar e, se seguro, implementar `--help`/`--version` sem incorporar, descartar ou reescrever o trabalho preexistente.

---

## 1. Inventário inicial

| Item | Valor |
|---|---|
| Branch | `feat/hpsi-core-container` |
| HEAD | `27d2bd344c35ef9f5c8383604ae42559cad42123` |
| Índice | vazio (`e3b0c442…`) |
| Trabalho preexistente | 15 ` M` + 30 ` ??` |
| Hunk `pytest` em `ci.yml` | presente (não tocado) |
| Commits C2f no histórico | `9726e66`, `a948469`, `3fdfa4d`, `27d2bd3` — preservados |

**Hashes iniciais:**

| Arquivo | SHA-256 |
|---|---|
| `apps/hpsi-cli/src/main.rs` | `866593a73ce6799c00eff520dca5d7927e0c48f35e82d4db481acffdd3928236` |
| `apps/hpsi-cli/tests/cli.rs` | `d027887ee8d5bd4a661414d3697e8d35f8d196fd74bf9f538f6e1917fbc30922` |
| `apps/hpsi-cli/Cargo.toml` | `8e1163baa434616b81f122b3e82a31d7ee148687bbe9945a22edc413cb8eb89f` |

**Observação:** `apps/hpsi-cli/src/cli.rs` **não existe** (o crate tem `src/main.rs` + `src/orientation.rs`). O parser é manual, em `run()`, sem dependência externa.

## 2. Análise de dependências entre hunks (somente leitura)

`git diff HEAD -- apps/hpsi-cli/src/main.rs` = **297 linhas**, 9 hunks, +193/−8. Hunks preexistentes:
imports (`mod orientation`, `ArchitectureFacts`), `run()` (`architecture_path`, `take_architecture_flag`), funções de orientação/auditoria, `import_safetensors_command`, `inspect_safetensors` e **a string de `usage()`** (que ganhou `--architecture`).

Mapa:

| Zona | Situação |
|---|---|
| `let Some(command) = args.first()...` (início de `run()`) | **contexto** (não alterado pelo trabalho preexistente) |
| `if command == "safetensors-import"` | contexto |
| `let tensor_id = match command { ... }` | **hunk preexistente** (`--architecture`) |
| `fn usage()` | **hunk preexistente** (string) |
| `#[cfg(test)] mod tests {` (após `usage()`) | contexto |

**Conclusão:** existe **ponto seguro** para inserir `--help`/`--version` logo após o `let Some(command) = …` (contexto), e `fn help()` logo após o fecho de `usage()`, antes do `#[cfg(test)]`. Nenhuma das inserções referencia símbolos do trabalho preexistente.

## 3. Estratégia escolhida (menor risco)

- **Sem dependência externa de parser** (não há clap; o parser manual é mantido).
- `--help`/`-h` e `--version`/`-V` interceptados no início de `run()`, antes de qualquer outra lógica.
- Versão via `env!("CARGO_PKG_VERSION")` = `0.1.0`.
- `help()` monta o texto a partir de `usage()` (presente em HEAD e no working tree), listando os **comandos reais**: `inspect`, `verify`, `calibrate`, `safetensors-inspect`, `safetensors-import`.
- `inspect`/`verify` não foram tocados; `hpsi` sem argumentos mantém o comportamento documentado (exit 1 + `usage`); argumento inválido continua exit 1.
- `run()` **não** foi reescrito.

## 4. Implementação e arquivos

| Arquivo | Mudança | Status |
|---|---|---|
| `apps/hpsi-cli/src/main.rs` | +8 linhas em `run()` (help/version) + `fn help()` (8 linhas) | modificado |
| `apps/hpsi-cli/tests/cli_meta.rs` | **novo** arquivo de integração, 9 testes | adicionado |

`tests/cli_meta.rs` usa `env!("CARGO_BIN_EXE_hpsi")`, cria um container válido com `ModelWriter::write_path`, e cobre: `--help`, `-h`, `--version`, `-V`, sem argumentos, comando inválido, `inspect`, `verify` e container truncado.

## 5. Staging seguro (comprovado)

Como `main.rs` tem hunks preexistentes, **não** foi feito `git add` do arquivo inteiro. Estratégia:

1. `git show HEAD:apps/hpsi-cli/src/main.rs` → versão base.
2. Aplicadas **as mesmas duas edições** à base e ao working tree.
3. Blob montado a partir da base editada (`git hash-object -w` + `git update-index --cacheinfo`).
4. `git add apps/hpsi-cli/tests/cli_meta.rs` (arquivo novo, sem hunks preexistentes).

`git diff --cached -- apps/hpsi-cli/src/main.rs` mostrou **apenas** as 2 inserções (help/version + `fn help()`), com a string de `usage()` **do HEAD** (sem `--architecture`). `grep` por `ArchitectureFacts|mod orientation|--architecture|summary_lines|audit_container` no staged: **nenhum**. Portanto nenhum hunk preexistente foi incorporado.

**Commit:** `cae9d9e608273596e0e6e2dad2f07685bfb83eec` — `feat: add HPSI CLI help and version options` (2 arquivos, 152 inserções).

## 6. Testes e resultados

| Gate | Comando | Exit | Resultado |
|---|---|---:|---|
| Formatação | `cargo fmt --all -- --check` | **0** | sem divergências |
| CLI | `cargo test -p hpsi-cli` | **0** | 4 unit + 15 `cli` + **9 `cli_meta`** |
| Workspace | `cargo test --workspace --all-targets` | **0** | **287 passed / 0 failed / 0 ignored** |
| Clippy | `cargo clippy --workspace --all-targets -- -D warnings` | **0** | 0 avisos |

Instalação limpa em prefixo novo (`.tmp-phase4/phaseC2fC1/install`, `cargo install --path … --locked`) → exit 0; `hpsi.exe` 809.472 B. Resultados reais:

| Comando | Exit | Saída |
|---|---:|---|
| `hpsi --help` | **0** | `hpsi 0.1.0`, `usage: …`, `Commands:` + 5 comandos, `Options:` |
| `hpsi -h` | **0** | idem |
| `hpsi --version` | **0** | `hpsi 0.1.0` |
| `hpsi -V` | **0** | `hpsi 0.1.0` |
| `hpsi inspect valid.hpsi` | **0** | `HPSI 0.1`, `file_size: 16514 bytes` |
| `hpsi verify valid.hpsi` | **0** | `OK: verified 22 CRC32C records` |
| `hpsi inspect truncated.hpsi` | **1** | `invalid HPSI header: declared file size differs from actual size` |
| `hpsi bogus` | **1** | `usage` |
| `hpsi` (sem args) | **1** | `usage` |

## 7. Estado final do Git

| Item | Valor |
|---|---|
| HEAD | `cae9d9e608273596e0e6e2dad2f07685bfb83eec` (+ commit deste relatório) |
| Índice | vazio |
| Preexistente | 15 ` M` + 30 ` ??` preservados |
| Hunk `pytest` | presente (não tocado) |
| Hunk §4.1/§4.2 de `docs/format-v0.1.md` | presente (não tocado) |
| `crates/hpsi-format/` | sem diff |

**Hashes finais:** `main.rs` `a0bc1f11a870c0590d940f55457a5be55e601052e542147158054e86e5f77b41`; `tests/cli_meta.rs` `04708f9bd95739046c77b6223441c93f6d88cd1032103841b879d1ce3a8291cb`.

## 8. Confirmação de preservação

- Nenhum `reset`, `clean`, `checkout`, `restore`, rebase, `git add -A`, push, merge, tag, release ou deploy.
- `apps/hpsi-cli/src/main.rs`: o trabalho preexistente (193 linhas) **continua no working tree**; apenas as duas inserções foram commitadas.
- `apps/hpsi-cli/tests/cli.rs` **não foi modificado** (testes novos em arquivo separado).
- Hunk `pytest` e hunk §4.1/§4.2 intactos; `crates/hpsi-format` sem diff.

## 9. Conclusão

**APROVADO.** `--help`, `-h`, `--version` e `-V` funcionam (exit 0), `inspect`/`verify` preservados, erro/invalid exit 1, testes passam, e o trabalho preexistente permanece intacto. O commit de implementação é isolado e não contém hunks preexistentes.

---

*Relatório da C2f-C1. Sem publicação, push, merge, tag ou deploy.*
