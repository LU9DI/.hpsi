# HPSI — Atualização de autoria e mantenedora

**Data:** 2026-11-08
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**HEAD inicial:** `2b5711b0994ba1112e32b04711bc1a7f2c91d364`
**Escopo:** registrar a identificação pessoal e empresarial aprovada pelo responsável — opções (a) copyright/README + (c) metadados Cargo. Sem `NOTICE`, sem alterar `LICENSE-APACHE`, sem tocar auditorias históricas, código funcional ou scratch.

---

## 1. Identificação confirmada

- **Autor / titular do copyright:** Lucas Bessa de Araujo.
- **GitHub:** `LU9DI`.
- **Mantenedora:** Filtra.AI (CNPJ 69.113.166/0001-10, valor informado pelo responsável).
- **Razão social da empresa:** **pendente de confirmação** — intencionalmente não declarada.

## 2. Alterações aplicadas

### 2.1 `LICENSE-MIT` (4 cópias idênticas)

Somente a linha de copyright mudou; o texto MIT permanece integral.

```diff
-Copyright (c) 2026 Lucas
+Copyright (c) 2026 Lucas Bessa de Araujo
```

Arquivos: `LICENSE-MIT`, `crates/hpsi-format/LICENSE-MIT`, `crates/hpsi-ops/LICENSE-MIT`, `apps/hpsi-cli/LICENSE-MIT`.

| Momento | SHA-256 |
|---|---|
| Antes (4 cópias) | `2907d5e82cf6526f2d6cb64440f88e5db245efa444e5f242feca0ed7ea2d20a2` |
| Depois (4 cópias) | `5c14681d4832939bdba5cee5c575e2d952b5d272a3ce01575ec2d0cb8b8190e7` |

As quatro cópias permanecem **byte-idênticas** entre si.

### 2.2 `README.md` — nova seção `## Project and maintainers`

Inserida antes de `## License`, declarando:

- autor e titular do copyright: Lucas Bessa de Araujo ([LU9DI](https://github.com/LU9DI));
- mantenedora: Filtra.AI (CNPJ 69.113.166/0001-10);
- Filtra.AI **não** é declarada cotitular;
- cessão/cotitularidade depende de instrumento jurídico próprio;
- razão social pendente de confirmação.

SHA-256 do README após a mudança: `e15c40ae5e993cd7b5eb474ee1669b59d7e62f250a39988dca0bf285351ea4ab`.

### 2.3 Metadados Cargo (`authors`)

- `Cargo.toml` → `[workspace.package] authors = ["Lucas Bessa de Araujo"]`.
- `crates/hpsi-format/Cargo.toml` → `authors.workspace = true`.
- `crates/hpsi-ops/Cargo.toml` → `authors.workspace = true` (herança por campo é compatível com os demais campos explícitos).
- `apps/hpsi-cli/Cargo.toml` → `authors.workspace = true`.

Demais metadados preservados.

## 3. Validações executadas

| Validação | Comando | Exit | Resultado |
|---|---|---:|---|
| Metadados | `cargo metadata --no-deps --format-version 1` | **0** | `authors=['Lucas Bessa de Araujo']` nos 3 crates; `license=MIT OR Apache-2.0` |
| Hashes MIT | `sha256sum` nas 4 cópias | — | iguais (`5c14681d…`) |
| Formatação | `cargo fmt --all -- --check` | **0** | sem divergências |
| Lista de pacote | `cargo package --list -p hpsi-format/ops/cli` | **0/0/0** | README + LICENSE-MIT + LICENSE-APACHE + THIRD-PARTY-NOTICES |
| Conteúdo do `.crate` | empacotar `hpsi-format` e extrair | **0** | `README.md` contém “Project and maintainers”; `LICENSE-MIT` contém “Lucas Bessa de Araujo”; `Cargo.toml` normalizado com `authors = ["Lucas Bessa de Araujo"]` |

## 4. Commits

| # | SHA | Mensagem | Conteúdo |
|---|---|---|---|
| 1 | `fc9deee77796952744d0089b9fec5cff24635ecb` | `docs: update HPSI copyright and maintainer attribution` | 4× `LICENSE-MIT` + `README.md` (5 arquivos) |
| 2 | `4f9d1d4cf59067a1fb11b043578e3017db1e16e2` | `chore: add HPSI package author metadata` | 4 manifests |
| 3 | (este relatório) | `docs: record HPSI authorship attribution` | 1 arquivo |

Diff do commit 1 conferido integralmente: apenas a linha de copyright (4×) e a nova seção do README. Diff do commit 2: apenas `+authors` nos 4 manifests.

## 5. Estado Git final

- Branch `feat/hpsi-core-container`.
- HEAD após commits 1–2: `4f9d1d4cf59067a1fb11b043578e3017db1e16e2` (+ commit 3).
- Índice vazio após cada commit.
- Trabalho preexistente preservado: **M=0**; os **8 itens não versionados** (`.tmp-*`, `fuzz/coverage/`, `prepare_gpt2_test.py`, 3 `.bak`) continuam intactos.
- Nenhum `git add -A`, `reset`, `clean`, `checkout`, `restore`, rebase, push, merge, tag, publicação ou deploy.

## 6. Ressalvas legais

1. **Razão social da Filtra.AI não confirmada** — usados apenas a marca e o CNPJ; não foi inventada.
2. **CNPJ** é o valor informado pelo responsável; não verificado independentemente.
3. **Filtra.AI não é cotitular** — apenas mantenedora/apoiadora. Cotitularidade/cessão exige instrumento jurídico escrito.
4. **`LICENSE-APACHE` inalterado** (texto padrão integral preservado).
5. **`NOTICE` não criado** (conforme opção aprovada).
6. **Auditorias históricas** em `docs/audits/` não foram reescritas (registram o estado anterior).
7. Nenhuma alteração de código funcional ou do formato físico v0.1.

---

*Relatório da atualização de autoria. Sem publicação, push, merge, tag ou deploy.*
