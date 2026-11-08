# HPSI — auditoria de prontidão para publicação dos crates v0.1

**Data:** 2026-11-08
**Repositório:** `C:\Users\lucas\HPSI`
**Branch auditada:** `feat/hpsi-core-container`
**HEAD auditado:** `d963b9df623fc1609fbc8cddad4698dfe82a3015`
**Escopo:** auditar a prontidão de `hpsi-format`, `hpsi-ops` e `hpsi-cli` para eventual publicação no crates.io. **Nenhuma publicação, tag ou release.** Nenhum arquivo de código/manifest/versão/documentação foi alterado; único artefato novo: este relatório.

---

## 1. SHA e branch auditados

| Item | Valor |
|---|---|
| Branch | `feat/hpsi-core-container` |
| HEAD | `d963b9df623fc1609fbc8cddad4698dfe82a3015` |
| Índice | vazio (`e3b0c442…`) |
| `git diff HEAD` | vazio |
| Preexistente | **M=0**, **8 itens de scratch** preservados |
| Ferramenta | Rust `1.85.0` (pinned), Cargo `1.85.0` |

## 2. Escopo

Três pacotes publicáveis; `fuzz/` (`publish = false`, workspace isolado) está **fora** do escopo v0.1.

## 3. Inventário dos três pacotes

| Crate | Versão | Edition | Licença | Authors | Readme | Description | Repository |
|---|---|---|---|---|---|---|---|
| `hpsi-format` | 0.1.0 | 2024 | MIT OR Apache-2.0 | Lucas Bessa de Araujo | `../../README.md` | presente | **ausente** |
| `hpsi-ops` | 0.1.0 | 2024 | MIT OR Apache-2.0 | Lucas Bessa de Araujo | `../../README.md` | **ausente** | **ausente** |
| `hpsi-cli` | 0.1.0 | 2024 | MIT OR Apache-2.0 | Lucas Bessa de Araujo | `../../README.md` | **ausente** | **ausente** |

Metadados comuns herdados do workspace: `version`, `edition`, `rust-version`, `license`, `authors`. `hpsi-ops` usa campos explícitos e **não** herda `rust-version` nem `repository`.

**Campos ausentes (por crate):**

| Campo | `hpsi-format` | `hpsi-ops` | `hpsi-cli` | Classificação |
|---|---|---|---|---|
| `description` | presente | **ausente** | **ausente** | **Obrigatório pela política do crates.io** (o `cargo publish --dry-run` local apenas avisa; validação server-side não executada) |
| `repository` | ausente | ausente | ausente | **Recomendado** |
| `keywords` / `categories` | ausentes | ausentes | ausentes | Opcional |
| `documentation` / `homepage` | ausentes | ausentes | ausentes | Opcional |
| `rust-version` | presente | **ausente** | presente | Recomendado |
| `readme` / `license` / `authors` | presentes | presentes | presentes | Recomendado/obrigatório — atendidos |

> O workspace define `repository = "https://github.com/LU9DI/.hpsi"`, mas nenhum crate usa `repository.workspace = true`; por isso o campo não aparece no `cargo metadata` nem no `Cargo.toml` empacotado.

**Licença:** `MIT OR Apache-2.0` nos manifests e nos `.crate`; `LICENSE-MIT` (`Copyright (c) 2026 Lucas Bessa de Araujo`), `LICENSE-APACHE` e `THIRD-PARTY-NOTICES` fisicamente incluídos (ver §7).

## 4. Grafo de dependências

```
hpsi-format  ── serde_json ^1 (registry)
hpsi-ops     ── hpsi-format ^0.1.0 (path+version) / serde ^1 / serde_json ^1
hpsi-cli     ── hpsi-format ^0.1.0 / hpsi-ops ^0.1.0 / serde_json ^1
```

- Dependências internas usam `[workspace.dependencies]` com **`path` + `version = "0.1.0"`**.
- Ao empacotar, o Cargo remove o `path` e declara `hpsi-format = "0.1.0"` / `hpsi-ops = "0.1.0"` (registry).
- Não há dependências opcionais nem `[features]`.

## 5. Ordem de publicação justificada

**`hpsi-format` → `hpsi-ops` → `hpsi-cli`**, comprovada por:
1. `hpsi-format` não depende de nenhum crate interno;
2. `hpsi-ops` depende de `hpsi-format`;
3. `hpsi-cli` depende de `hpsi-format` e `hpsi-ops`.

A publicação de `hpsi-ops`/`hpsi-cli` **exige** que as dependências internas já estejam no registry; caso contrário o Cargo falha na resolução.

## 6. Resultado das verificações técnicas

| Verificação | Comando | Exit | Resultado |
|---|---|---:|---|
| Metadados | `cargo metadata --no-deps --format-version 1` | **0** | 3 pacotes; autores/licença corretos |
| Formatação | `cargo fmt --all -- --check` | **0** | sem divergências |
| Testes | `cargo test --workspace --all-targets` | **0** | **287 passed / 0 failed / 0 ignored** |
| Lint | `cargo clippy --workspace --all-targets -- -D warnings` | **0** | 0 avisos |
| Lista de pacote | `cargo package --list -p hpsi-format/ops/cli` | **0/0/0** | ver §7 |

## 7. Conteúdo dos pacotes (`cargo package --list`)

Os três pacotes listam: `README.md`, `LICENSE-MIT`, `LICENSE-APACHE`, `THIRD-PARTY-NOTICES`, fontes necessárias, `Cargo.toml` e testes. **Não** listam `fuzz/`, `.tmp-*`, `.bak`, `prepare_gpt2_test.py`, corpus ou artefatos.

- Referências a caminhos locais: apenas `readme = "../../README.md"` (o Cargo copia para o pacote como `README.md`) e dependências `path` internas (reescritas para registry ao empacotar).
- Nenhum `build.rs`; nenhum script de instalação; sem dependência de arquivos fora do pacote.
- README empacotado contém a seção **“HPSI Toolkit v0.1 — scope and limitations”** (toolkit ≠ runtime; RAW-only; codecs sem decoder; `inspect`/`verify` não decodificam).

`.crate` gerado nesta auditoria (dry-run de `hpsi-format`): `target/package/hpsi-format-0.1.0.crate`, **52.724 B**, SHA-256 `fcb9ad9138eb492330c12c49dafc3571c066da9c3d79385b8df6e99614086d6d`.

> Os `.crate` de `hpsi-ops`/`hpsi-cli` em `target/package` (2026-11-07) são **antigos**, gerados com auxílio de patch na C2f-E; **não** representam o estado atual.

## 8. Resultados dos dry-runs

| Crate | Comando | Exit | Resultado |
|---|---|---:|---|
| `hpsi-format` | `cargo publish --dry-run -p hpsi-format` (árvore limpa, sem `--allow-dirty`) | **0** | empacotou, compilou verificação (7m55s) e **abortou upload** (dry run) |
| `hpsi-ops` | `cargo publish --dry-run -p hpsi-ops` | **101** | `no matching package named 'hpsi-format' found` |
| `hpsi-cli` | `cargo publish --dry-run -p hpsi-cli` | **101** | idem |

**Interpretação:** a falha de `hpsi-ops`/`hpsi-cli` é **bloqueio de ordem de publicação** (dependência interna ainda não publicada), **não** falha funcional. Nenhuma autenticação/credencial foi usada; nenhum upload ocorreu.

## 9. Riscos e bloqueios

| # | Risco | Classificação | Estado |
|---|---|---|---|
| B1 | `hpsi-ops`/`hpsi-cli` exigem `hpsi-format` (e `hpsi-ops`) já publicados | **Bloqueio de ordem de publicação** | Comprovado (dry-run 101) |
| B2 | `description` ausente em `hpsi-ops`/`hpsi-cli` | **Correção obrigatória** (política crates.io; local só avisa) | Pendente |
| B3 | `repository` ausente nos três | Recomendado | Pendente |
| B4 | `rust-version` ausente em `hpsi-ops` | Recomendado | Pendente |
| B5 | `keywords`/`categories`/`documentation`/`homepage` ausentes | Opcional | Pendente |
| B6 | `cargo audit`/`cargo deny` não instalados | **NÃO VALIDADO** (vulnerabilidades não verificadas) | Bloqueio de evidência |
| B7 | Validação server-side do crates.io não executada | **NÃO VALIDADO** (dry-run não envia) | Pendente |
| B8 | Razão social da Filtra.AI não confirmada | Pendência legal (não é bloqueio de crates.io) | Pendente |
| S1 | Sem `build.rs`, sem scripts de instalação, sem `[features]` | — | Comprovado |
| S2 | Nenhum segredo/credencial em arquivos rastreados | — | Comprovado (grep) |
| S3 | Fixtures de teste pequenas e sintéticas (10,9 KB / 15,6 KB) | — | Comprovado |

## 10. Checklist de publicação futura

1. Adicionar `description` a `hpsi-ops` e `hpsi-cli` (e `repository.workspace = true` nos três; `rust-version.workspace = true` em `hpsi-ops`).
2. Opcional: `keywords`, `categories`, `documentation`, `homepage`.
3. Reexecutar `cargo package --list` e `cargo publish --dry-run` em ordem: `hpsi-format` → `hpsi-ops` → `hpsi-cli`.
4. Publicar `hpsi-format`, aguardar o índice do registry, depois publicar `hpsi-ops`, depois `hpsi-cli`.
5. Rodar `cargo audit`/`cargo deny` (ou equivalente) antes da publicação.
6. Confirmar a razão social da Filtra.AI se a empresa for citada em materiais de release.
7. Nunca publicar com `--allow-dirty`; partir de commit limpo e tagueado.

## 11. Conclusão por crate

| Crate | Classificação | Justificativa |
|---|---|---|
| `hpsi-format` | **PRONTO PARA PUBLICAÇÃO** | `publish --dry-run` exit 0; pacote com README/licenças/notices; metadata essencial presente. Pendências apenas recomendadas/opcionais (`repository`, `keywords`, `categories`). |
| `hpsi-ops` | **BLOQUEADO POR DEPENDÊNCIA EXTERNA** + correção de metadata | Depende de `hpsi-format` publicado; `description` ausente (política crates.io); `repository`/`rust-version` recomendados. |
| `hpsi-cli` | **BLOQUEADO POR DEPENDÊNCIA EXTERNA** + correção de metadata | Depende de `hpsi-format` e `hpsi-ops` publicados; `description` ausente; `repository` recomendado. |
| `fuzz/` | **NÃO APLICÁVEL ao escopo v0.1** | `publish = false`, workspace isolado, harness de verificação. |

## 12. Conclusão geral

**APROVADO COM PENDÊNCIAS.**

- **Fatos comprovados:** `hpsi-format` passa no `publish --dry-run` (sem upload); `hpsi-ops`/`hpsi-cli` falham apenas por dependência interna não publicada; todos os gates técnicos passam (287 testes, clippy/fmt 0); os pacotes incluem README e os três documentos legais; não há segredos, `build.rs`, scripts de instalação ou features.
- **Correções necessárias antes de publicar `hpsi-ops`/`hpsi-cli`:** `description` (obrigatório pela política do crates.io) e, recomendadamente, `repository` e `rust-version`.
- **Bloqueios:** ordem de publicação (`format → ops → cli`) e ausência de `cargo audit`/validação server-side.
- **Pendência legal:** razão social da Filtra.AI (não bloqueia crates.io; o titular do copyright — Lucas Bessa de Araujo — está confirmado).

**Nada foi publicado, tagueado, enviado (push), mesclado ou implantado.**

---

*Relatório da auditoria de publicação de crates. Sem publicação.*
