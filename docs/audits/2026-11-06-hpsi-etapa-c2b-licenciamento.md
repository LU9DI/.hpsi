# HPSI — Etapa C2b: licenciamento dual `MIT OR Apache-2.0`

**Data:** 2026-11-06
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**Commit-base:** `4033b7773c146668ee11aa564d630337291d8dc6`
**Escopo:** fechar o licenciamento do HPSI Toolkit v0.1 com licença dual, auditar as dependências e versionar os documentos legais. Alterações restritas a manifests, documentos legais, README e a este relatório.

---

## 1. Estado inicial e final do Git

| Item | Início (C2b) | Fim (C2b) |
|---|---|---|
| Branch | `feat/hpsi-core-container` | `feat/hpsi-core-container` |
| HEAD | `4033b7773c146668ee11aa564d630337291d8dc6` | (novo commit — §8) |
| Índice | vazio (`e3b0c442…`) | vazio após o commit |
| Trabalho pré-existente | 15 ` M` + 29 `??` | preservado (contagens ajustadas — §7) |
| Hunk `pytest` em `.github/workflows/ci.yml` | preservado | preservado (não tocado) |
| `crates/hpsi-format/` | sem diff | sem diff |

Nenhum `reset`, `clean`, `checkout`, `restore`, rebase, push, merge, tag ou deploy.

## 2. Inventário integral de dependências e licenças

### 2.1 Workspace principal (`Cargo.lock` v4 — 14 pacotes)

Evidência: `cargo metadata --locked --format-version 1` (`.tmp-phase4/phaseC2b/main-licenses.txt`), que resolve o lock e lê o `license` de cada pacote.

| Pacote | Versão | Tipo | Licença declarada | Evidência |
|---|---|---|---|---|
| hpsi-format | 0.1.0 | próprio | `MIT` → **`MIT OR Apache-2.0`** | Cargo.toml (alterado) |
| hpsi-ops | 0.1.0 | próprio | `MIT` → **`MIT OR Apache-2.0`** | Cargo.toml (alterado) |
| hpsi-cli | 0.1.0 | próprio | `MIT` → **`MIT OR Apache-2.0`** | via `license.workspace` |
| itoa | 1.0.18 | transitiva (serde_json) | `MIT OR Apache-2.0` | cargo metadata |
| memchr | 2.8.3 | transitiva | `Unlicense OR MIT` | cargo metadata |
| proc-macro2 | 1.0.107 | transitiva (build) | `MIT OR Apache-2.0` | cargo metadata |
| quote | 1.0.47 | transitiva (build) | `MIT OR Apache-2.0` | cargo metadata |
| serde | 1.0.229 | direta (hpsi-ops) | `MIT OR Apache-2.0` | cargo metadata |
| serde_core | 1.0.229 | transitiva | `MIT OR Apache-2.0` | cargo metadata |
| serde_derive | 1.0.229 | transitiva (build) | `MIT OR Apache-2.0` | cargo metadata |
| serde_json | 1.0.154 | direta (todos) | `MIT OR Apache-2.0` | cargo metadata |
| syn | 3.0.9 | transitiva (build) | `MIT OR Apache-2.0` | cargo metadata |
| unicode-ident | 1.0.27 | transitiva (build) | **`(MIT OR Apache-2.0) AND Unicode-3.0`** | cargo metadata |
| zmij | 1.0.23 | transitiva (serde_json) | `MIT` | cargo metadata |

Nenhuma dependência de desenvolvimento no workspace principal (todos os `[dependencies]` batem com o lock; não há `[dev-dependencies]`).

**Achado material:** `unicode-ident` impõe a **Unicode License V3** em adição a MIT/Apache-2.0 → exige reprodução do aviso Unicode (feita em `THIRD-PARTY-NOTICES`).

### 2.2 Workspace de fuzzing (`fuzz/`, isolado, `publish = false`; sem `Cargo.lock`)

Evidência: API do crates.io (`fuzz-licenses.txt`, `fuzz-transitive.txt`). Versões **não fixadas** (o `fuzz/Cargo.lock` não existe; a resolução ocorre no build).

| Pacote | Requisito | Tipo | Licença |
|---|---|---|---|
| libfuzzer-sys | `0.4` (0.4.13 conhecido) | normal | `(MIT OR Apache-2.0) AND NCSA` |
| arbitrary | `^1` (1.5.0) | normal | `MIT OR Apache-2.0` (+ `derive_arbitrary`) |
| cc | `^1.0.83` (1.9.0) | **build** | `MIT OR Apache-2.0` (+ `shlex`, `jobserver`, `libc`, `find-msvc-tools`) |
| flate2, rand | — | **dev** | não aplicáveis ao build do harness |

**Racional:** o workspace `fuzz/` é um harness de desenvolvimento/verificação e **não** integra o binário distribuído do toolkit; nenhum desses crates é redistribuído pelo HPSI. A obrigação NCSA do `libfuzzer-sys` só se aplica a quem compila os alvos de fuzz localmente, e está registrada em `THIRD-PARTY-NOTICES`.

Nenhuma licença permanece não identificada no **workspace principal** (que é o objeto do toolkit). No `fuzz/`, as licenças são conhecidas para `libfuzzer-sys`/`arbitrary`/`cc`; as transitivas do `cc` não foram enumeradas uma a uma por o lock não existir — registrado como pendência informativa (§8).

## 3. Evidências e fontes

| Fonte | Uso |
|---|---|
| `cargo metadata --locked --format-version 1` | licença declarada de cada pacote do workspace principal (a partir do `Cargo.toml` real) |
| `Cargo.lock` (v4) | conjunto exato de versões |
| API `https://crates.io/api/v1/crates/<nome>/<versão>` | licença declarada das dependências do `fuzz/` |
| `static.crates.io` (`libfuzzer-sys-0.4.13.crate`) | arquivos `LICENSE-MIT`/`LICENSE-APACHE` do `libfuzzer-sys` |
| Registry local | textos de MIT/Apache/Unicode para os arquivos legais |
| `git log --format='%an <%ae>'` | autoria (titular) |

Nenhuma licença foi inferida por nome, popularidade ou semelhança.

## 4. Titular e ano usados nos documentos

- **Titular:** `Lucas` — identidade do autor do repositório (464 commits por `Lucas <bostavacabosta@gmail.com>`; único autor humano além do `github-actions[bot]`). Não foi inventado nome.
- **Ano:** `2026` — primeiro commit (2026-11-01) e data corrente (2026-11-06).
- Linha usada: `Copyright (c) 2026 Lucas`.
- **Pendência de confirmação:** o repositório só expõe um prenome (`Lucas`). Se o titular legal for um nome completo/entidade diferente (p.ex. o handle `LU9DI`), o `LICENSE-MIT` deverá ser ajustado. Registrado como decisão do responsável.

## 5. Arquivos legais criados

| Arquivo | Conteúdo | Origem |
|---|---|---|
| `LICENSE-MIT` | texto MIT integral + `Copyright (c) 2026 Lucas` | texto canônico MIT |
| `LICENSE-APACHE` | texto Apache License 2.0 **integral**, incluindo `APPENDIX` (201 linhas) | cópia verbatim de `libfuzzer-sys-0.4.13/LICENSE-APACHE` |
| `THIRD-PARTY-NOTICES` | tabela de dependências; texto integral da Unicode License V3; nota NCSA/`fuzz` | registry + API crates.io |

`NOTICE` separado **não** foi criado: nenhuma dependência distribui arquivo `NOTICE` (`license_file = None` em todas); a atribuição adicional exigida (Unicode-3.0) foi incorporada ao `THIRD-PARTY-NOTICES`.

## 6. Alterações nos manifests e README

```diff
--- Cargo.toml                     license = "MIT" -> "MIT OR Apache-2.0"
--- crates/hpsi-ops/Cargo.toml     license = "MIT" -> "MIT OR Apache-2.0"
--- fuzz/Cargo.toml                + license = "MIT OR Apache-2.0"
--- README.md                      + seção "## License" com links para os 3 documentos
```

`crates/hpsi-format/Cargo.toml` e `apps/hpsi-cli/Cargo.toml` usam `license.workspace = true` e herdam a mudança — **não** precisaram de edição. Nenhuma alteração funcional (sem mudança em código, dependências ou versões).

## 7. Validações executadas

| Validação | Comando | Resultado |
|---|---|---|
| Metadados consistentes pós-mudança | `cargo metadata --locked --format-version 1` | exit 0; `hpsi-{format,ops,cli}` = `MIT OR Apache-2.0` |
| Lock intacto | mtime/`git status` de `Cargo.lock` | inalterado (2026-11-03; segue `??`) |
| Sem referência antiga | `grep 'license = "MIT"' **/*.toml` | nenhuma |
| MIT integral | `grep` | `MIT License` + `Copyright (c) 2026 Lucas` presentes |
| Apache integral | `grep APPENDIX` | presente; 201 linhas |
| Unicode-3.0 | `grep 'UNICODE LICENSE V3'` / copyright | presentes |
| Sem alteração funcional | `git diff` de Cargo.toml/README | apenas linhas de licença/README |

**Contagens de preservação:** início `M=15`, `??=29`. Durante a edição: `M=19` (15 + `Cargo.toml`, `crates/hpsi-ops/Cargo.toml`, `fuzz/Cargo.toml`, `README.md`), `??=32` (29 + 3 arquivos legais). Após o commit, `M` volta a 15 e `??` a 29 (os 8 caminhos do commit deixam de figurar). Toda variação é explicada por arquivos deste escopo.

## 8. Limitações e pendências

1. **Titular** — apenas o prenome `Lucas` está disponível no repositório; confirmar nome legal/entidade.
2. **`fuzz/` sem lock** — versões não fixadas; licenças das transitivas do `cc` não enumeradas individualmente (build-only, não distribuídas).
3. **`cargo-license`/`cargo-deny`** não instalados; a verificação usou `cargo metadata` + API do crates.io, o que cobre as licenças declaradas, mas não uma análise de compatibilidade automatizada.
4. **`serde_core`** aparece como dependência de `serde`; sua licença foi lida diretamente do pacote (não inferida).
5. A CLI/Toolkit não embarca código de terceiros vendorizado (não há diretório `vendor/`).

## 9. Confirmação de preservação

- Os 15 ` M` originais e 29 ` ??` originais permanecem; só mudaram de contagem por causa dos arquivos deste escopo (manifests/README/legais/report).
- O hunk `pytest` pré-existente de `.github/workflows/ci.yml` **permanece** (não tocado).
- `crates/hpsi-format/` sem diff.
- Nenhum arquivo descartado, movido ou sobrescrito.

## 10. Ausência de push, merge, tag e deploy

Confirmado: nenhum `push`, `merge`, `tag` ou `deploy`. O commit é local.

---

*Relatório da Etapa C2b.*
