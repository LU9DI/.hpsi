# HPSI — correções de metadados e validação pré-publicação

**Data:** 2026-11-08
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**HEAD-base:** `6a9beb6603250d9a6be3ed7b32311da1fcd4869a`
**Escopo:** completar `description`/`repository` (e MSRV) dos crates; verificar segurança das dependências; revalidar gates. **Nenhuma publicação** (nem `cargo publish` nem dry-run), tag, release, push, merge ou deploy.

---

## 1. Estado inicial

| Item | Valor |
|---|---|
| Branch | `feat/hpsi-core-container` |
| HEAD | `6a9beb6603250d9a6be3ed7b32311da1fcd4869a` |
| Índice | vazio |
| `git diff HEAD` | vazio |
| Preexistente | **M=0**, **8 itens de scratch** preservados |

## 2. Alterações e justificativas

### 2.1 `description` e `repository`

| Crate | Alteração |
|---|---|
| `hpsi-format` | `+ repository.workspace = true` |
| `hpsi-ops` | `+ rust-version.workspace = true`, `+ repository.workspace = true`, `+ description` |
| `hpsi-cli` | `+ repository.workspace = true`, `+ description` |

- `repository` usa `repository.workspace = true`, herdando o valor já confirmado no workspace: **`https://github.com/LU9DI/.hpsi`**.
- Nome, versão, licença, dependências, features e código funcional **não** foram alterados. `Cargo.lock` permaneceu inalterado (`d3771d3e…`).

### 2.2 Descrições exatas adicionadas

- **`hpsi-ops`**: `Safetensors import, orientation and role audit, and experimental CPU vector-quantized operators for HPSI containers.`
- **`hpsi-cli`**: `Command-line interface to inspect, verify, calibrate and import HPSI v0.1 containers.`

Ambas descrevem funcionalidades **realmente implementadas** no v0.1 (import Safetensors, auditoria de orientação/papel, operadores VQ experimentais em CPU; CLI `inspect`/`verify`/`calibrate`/`import`). Não prometem runtime de inferência, decoders inexistentes ou suporte não implementado.

### 2.3 Decisão fundamentada sobre `rust-version` em `hpsi-ops`

**Evidência (independente do compilador instalado):**
- `[workspace.package] rust-version = "1.85.0"` no `Cargo.toml` raiz;
- `rust-toolchain.toml` fixa `channel = "1.85.0"`;
- CI instala `toolchain: 1.85.0` nos jobs de formatação e Rust;
- `hpsi-ops` usa `edition = "2024"` (exige Rust 1.85+);
- dependência interna `hpsi-format` declara `rust-version = "1.85.0"`; `serde`/`serde_json` 1.x são compatíveis.

**Decisão:** adicionar `rust-version.workspace = true` (→ 1.85.0) a `hpsi-ops`, alinhando-o aos outros dois crates. Não foi inferido apenas da versão do compilador local; a MSRV é declarada pelo projeto e verificada pela CI.

## 3. Segurança das dependências

| Ferramenta | Disponibilidade | Resultado |
|---|---|---|
| `cargo audit` | **não instalado** (`no such command`) | **NÃO VALIDADO** |
| `cargo deny` | **não instalado** | **NÃO VALIDADO** |

- Nenhuma ferramenta foi instalada; nenhum registry/serviço externo foi acessado para auditoria.
- **Permanece sem validação:** advisories de vulnerabilidade (RUSTSEC) e checagem automatizada de licenças/dependências não permitidas.
- O conjunto de dependências está inalterado nesta etapa (`Cargo.lock` idêntico). A auditoria de licenças anterior (C2b) permanece a evidência disponível: `serde`/`serde_core`/`serde_json`/`itoa`/`proc-macro2`/`quote`/`serde_derive`/`syn` = `MIT OR Apache-2.0`; `memchr` = `Unlicense OR MIT`; `unicode-ident` = `(MIT OR Apache-2.0) AND Unicode-3.0`; `zmij` = `MIT`.

## 4. Validações executadas

| Verificação | Comando | Exit | Resultado |
|---|---|---:|---|
| Metadados | `cargo metadata --no-deps --format-version 1` | **0** | ver tabela abaixo |
| Formatação | `cargo fmt --all -- --check` | **0** | sem divergências |
| Testes | `cargo test --workspace --all-targets` | **0** | **287 passed / 0 failed / 0 ignored** |
| Lint | `cargo clippy --workspace --all-targets -- -D warnings` | **0** | 0 avisos |
| Lista de pacote | `cargo package --list -p hpsi-format/ops/cli` | **0/0/0** | README + 3 documentos legais |

> `cargo package --list` foi inicialmente executado com `--allow-dirty` **apenas** porque as correções ainda estavam não commitadas; após o commit, as três listas passam **sem** `--allow-dirty` (exit 0). Não houve mascaramento de alterações alheias.

**Metadados efetivos (todos os três):**

| Crate | description | repository | authors | license | rust-version |
|---|---|---|---|---|---|
| `hpsi-format` | presente | `https://github.com/LU9DI/.hpsi` | `["Lucas Bessa de Araujo"]` | `MIT OR Apache-2.0` | 1.85.0 |
| `hpsi-ops` | presente (nova) | `https://github.com/LU9DI/.hpsi` | `["Lucas Bessa de Araujo"]` | `MIT OR Apache-2.0` | 1.85.0 |
| `hpsi-cli` | presente (nova) | `https://github.com/LU9DI/.hpsi` | `["Lucas Bessa de Araujo"]` | `MIT OR Apache-2.0` | 1.85.0 |

## 5. Estado final

| Item | Valor |
|---|---|
| HEAD | `468ee02f569c36f6629cadde3b7eea4be0fecb21` (+ commit deste relatório) |
| Índice | vazio |
| Preexistente | M=0 · 8 itens de scratch preservados |
| `Cargo.lock` | inalterado |
| Código funcional | inalterado |

**Hashes dos manifests após a correção:** `Cargo.toml` `2ff95220…`; `crates/hpsi-format/Cargo.toml` `d5872ec6…`; `crates/hpsi-ops/Cargo.toml` `63e903a9…`; `apps/hpsi-cli/Cargo.toml` `83ad85c4…`.

## 6. Pendências para publicação

1. **Ordem de publicação:** `hpsi-format` → `hpsi-ops` → `hpsi-cli` (dependências internas exigem as anteriores publicadas).
2. **`cargo audit`/`cargo deny` ausentes** — vulnerabilidades e política de licenças permanecem **não validadas**.
3. **Validação server-side do crates.io** não executada (nenhum dry-run nesta etapa, por instrução).
4. **Campos opcionais ainda ausentes:** `keywords`, `categories`, `documentation`, `homepage`.
5. **Razão social da Filtra.AI** pendente (não bloqueia crates.io; titular do copyright confirmado).

## 7. Prontidão por crate

| Crate | Classificação | Base |
|---|---|---|
| `hpsi-format` | **PRONTO PARA PUBLICAÇÃO** | metadata completa; `--list` OK; dry-run anterior (auditoria prévia) exit 0 |
| `hpsi-ops` | **PRONTO APÓS PUBLICAÇÃO DE `hpsi-format`** | metadata agora completa; bloqueio é apenas de ordem de publicação |
| `hpsi-cli` | **PRONTO APÓS PUBLICAÇÃO DE `hpsi-format` E `hpsi-ops`** | metadata agora completa; bloqueio é apenas de ordem de publicação |
| `fuzz/` | **NÃO APLICÁVEL ao v0.1** | `publish = false` |

**Nenhuma publicação, tag, release, push, merge ou deploy foi executado.**

---

*Relatório da correção de metadados e follow-up de segurança.*
