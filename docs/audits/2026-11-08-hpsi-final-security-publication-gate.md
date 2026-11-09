# HPSI — gate final de segurança e prontidão de publicação

**Data da execução:** 2026-11-09 (UTC), madrugada
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**HEAD auditado:** `a33055fbcc906dbbbca4ed821c2234c5a079e010`
**Natureza:** auditoria operacional somente-leitura; nenhuma publicação, push, merge, tag, release ou deploy. Único artefato novo: este relatório.

---

## 1. Estado inicial

| Item | Valor |
|---|---|
| Branch | `feat/hpsi-core-container` |
| HEAD | `a33055fbcc906dbbbca4ed821c2234c5a079e010` |
| Índice | vazio (`e3b0c442…`) |
| `git diff HEAD` | vazio |
| Preexistente | **M=0**, **8 itens de scratch/backups** preservados |

## 2. Ferramentas

`cargo-audit` e `cargo-deny` **não estavam instalados** (informado antes da instalação). Foram instalados no ambiente do usuário (`~/.cargo/bin`), sem alterar manifests/dependências do projeto, usando a toolchain nightly (a 1.85 do projeto não atende a MSRV das ferramentas atuais).

| Ferramenta | Versão | Instalação |
|---|---|---|
| `cargo-audit` | **0.22.2** | `cargo +nightly install cargo-audit --locked` (52m03s) |
| `cargo-deny` | **0.20.2** | `cargo +nightly install cargo-deny --locked` (48m21s) |
| Advisory DB (RustSec) | commit `7a40350e1b8ba5b656d909f88222a8059a83d88d`, 2026-11-08 | `~/.cargo/advisory-db` |
| Data da consulta | 2026-11-09T03:33Z | — |

## 3. Segurança de dependências

**Comando:** `cargo audit` — **exit 0** (140 s)

```
Fetching advisory database from `https://github.com/RustSec/advisory-db.git`
  Loaded 1422 security advisories
Updating crates.io index
Scanning Cargo.lock for vulnerabilities (14 crate dependencies)
```

**Resultado: 0 vulnerabilidades conhecidas** nas 14 dependências do workspace (`itoa`, `memchr`, `proc-macro2`, `quote`, `serde`, `serde_core`, `serde_derive`, `serde_json`, `syn`, `unicode-ident`, `zmij`, `hpsi-format`, `hpsi-ops`, `hpsi-cli`). Nenhum advisory, nenhuma severidade, nenhuma correção necessária.

## 4. Conformidade de licenças

**Comando:** `cargo deny check` — **exit 4** (256 s)

```
[WARN] unable to find a config path, falling back to default config
advisories ok, bans ok, licenses FAILED, sources ok
```

- **`advisories ok`** — consistente com `cargo audit` (sem vulnerabilidades).
- **`bans ok`** — sem dependências banidas/duplicadas problemáticas.
- **`sources ok`** — todas as fontes são o crates.io.
- **`licenses FAILED`** — **não é uma licença incompatível**: não existe `deny.toml`, então o `cargo-deny` usa a configuração default, cuja allow-list é **vazia** e rejeita **todas** as licenças, inclusive MIT/Apache-2.0/Unlicense. Os 9 pacotes citados (`hpsi-*`, `itoa`, `memchr`, `serde`, `serde_core`, `serde_json`, `zmij`) apresentam apenas licenças permissivas e OSI/FSF-aprovadas: **MIT**, **Apache-2.0**, **Unlicense**.
- **Limitação registrada:** a configuração default **não cobriu** as dependências de build/proc-macro (`serde_derive`, `proc-macro2`, `quote`, `syn`, `unicode-ident` — este último com `(MIT OR Apache-2.0) AND Unicode-3.0`). Portanto a checagem de licenças é **parcial** e exige um `deny.toml` explícito (allow-list) para ser um gate confiável. **Nenhuma política foi inventada** nesta etapa, conforme a regra.

## 5. Empacotamento

| Comando | Exit | Resultado |
|---|---:|---|
| `cargo package --list -p hpsi-format` | **0** | README + LICENSE-MIT + LICENSE-APACHE + THIRD-PARTY-NOTICES |
| `cargo package --list -p hpsi-ops` | **0** | idem |
| `cargo package --list -p hpsi-cli` | **0** | idem |
| `cargo publish --dry-run -p hpsi-format` (**com verificação de build**) | **0** | 23 arquivos, 223,7 KiB; build de verificação **19m28s**; upload **abortado** (dry run) |

- O dry-run **completo** compilou o tarball com sucesso (`Finished dev profile … in 19m 28s`) e abortou o upload. Não houve falha de ambiente nesta execução (a falha `STATUS_DLL_INIT_FAILED` observada anteriormente foi transitória).
- `.crate` atual de `hpsi-format`: `target/package/hpsi-format-0.1.0.crate`, **52.743 B**, SHA-256 `bd5cef851481021a753d5bf3668d580c75913fce00a91d5081cc4ca2be398dbe`.
- `hpsi-ops`/`hpsi-cli` **não** tiveram dry-run: dependem da publicação prévia de `hpsi-format` (e `hpsi-ops`). A ordem **não foi contornada**.
- Os `.crate` de `hpsi-ops`/`hpsi-cli` em `target/package` são antigos (2026-11-07) e foram gerados com patch auxiliar; **não** representam o estado atual.

## 6. Validações complementares (execuções atuais, não históricas)

| Comando | Exit | Duração | Resultado |
|---|---:|---:|---|
| `cargo metadata --no-deps` | **0** | — | 3 pacotes; description/repository/authors/license/rust-version corretos |
| `cargo fmt --all -- --check` | **0** | — | sem divergências |
| `cargo test --workspace --all-targets` | **0** | 409 s | **287 passed / 0 failed / 0 ignored** |
| `cargo clippy --workspace --all-targets -- -D warnings` | **0** | 55 s | 0 avisos |

## 7. Prontidão por crate

| Crate | Classificação | Base |
|---|---|---|
| `hpsi-format` | **APROVADO** | `package --list` OK; `publish --dry-run` **completo** exit 0; metadados completos; sem advisories |
| `hpsi-ops` | **APROVADO COM PENDÊNCIA DE ORDEM** | metadados OK; depende de `hpsi-format` publicado; dry-run não executável antes disso |
| `hpsi-cli` | **APROVADO COM PENDÊNCIA DE ORDEM** | metadados OK; depende de `hpsi-format` e `hpsi-ops` publicados |
| `fuzz/` | **NÃO APLICÁVEL ao v0.1** | `publish = false` |

## 8. Pendências externas e de autorização

1. **Publicação não autorizada** — nenhum `cargo publish` (nem com upload) foi executado; sem token/credencial.
2. **Ordem de publicação** — `hpsi-format` → (índice) → `hpsi-ops` → `hpsi-cli`.
3. **`cargo-deny` sem `deny.toml`** — allow-list de licenças ausente; a checagem de licenças foi parcial (build deps não cobertas).
4. **Legal** — razão social da Filtra.AI pendente de confirmação (não bloqueia crates.io; titular do copyright confirmado: Lucas Bessa de Araujo).
5. **Publicação server-side** — não validada (dry-run não envia).

## 9. Conclusão geral

**`APROVADO COM PENDÊNCIAS`.**

- **Segurança de dependências: APROVADO** — `cargo audit` sem vulnerabilidades (1422 advisories, 14 deps).
- **Conformidade de licenças: APROVADO COM RESSALVAS** — licenças permissivas (MIT/Apache-2.0/Unlicense); o `licenses FAILED` é artefato da allow-list vazia do config default, e a cobertura de build deps é parcial.
- **Empacotamento: APROVADO** — 3 listas OK; `hpsi-format` dry-run completo.
- **Verificação de build: APROVADO** — dry-run compilou o tarball (19m28s) e abortou upload.
- **Prontidão:** `hpsi-format` pronto; `hpsi-ops`/`hpsi-cli` prontos após a ordem de publicação.
- **Bloqueio real restante:** autorização/credencial para publicar e a ordem de publicação; e configurar `deny.toml` para um gate de licenças confiável.

**Nada foi publicado, tagueado, enviado (push), mesclado ou implantado.**

---

*Relatório do gate final de segurança e publicação. Sem publicação.*
