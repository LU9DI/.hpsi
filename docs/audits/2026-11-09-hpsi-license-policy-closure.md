# HPSI — fechamento da política de licenças (v0.1)

**Data:** 2026-11-09
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**HEAD (inalterado):** `bffa069a94eaf1345f345a943b1281d937975f6c`
**Natureza:** auditoria e criação de política; **nenhum commit** nesta etapa, nenhuma publicação/push/tag/merge/release/deploy. Nenhum código, manifest, dependência ou versão alterado.
**Revisão:** inclui a verificação de cobertura de dependências opcionais solicitada na revisão (§5.1).

---

## 1. Estado inicial

| Item | Valor |
|---|---|
| Branch | `feat/hpsi-core-container` |
| HEAD | `bffa069a94eaf1345f345a943b1281d937975f6c` |
| Índice | vazio |
| `git diff HEAD` | vazio |
| Preexistente | **M=0**, **8 itens de scratch/backups** preservados |
| Ferramentas | `cargo-deny 0.20.2`, `cargo-audit 0.22.2` |

## 2. Escopo e grafo auditado

### 2.1 Workspace raiz (crates publicáveis)

- **`Cargo.lock`**: **14 pacotes** (3 do workspace + 11 externos).
- **Grafo resolvido por padrão** (`cargo tree -e all`): **9 pacotes** — os 3 HPSI + `itoa`, `memchr`, `serde`, `serde_core`, `serde_json`, `zmij`.
- **5 pacotes no lock não alcançáveis com as features atuais**: `serde_derive`, `proc-macro2`, `quote`, `syn`, `unicode-ident` (dependem da feature opcional `derive` de `serde`, não habilitada; `serde = "1"` com default `std`). **A cobertura de política desses 5 pacotes é o objeto da §5.1.**
- **Build / proc-macro / dev:** não há `build.rs`; `cargo tree -e build`/`-e dev` não adicionam pacotes; não há `[dev-dependencies]`.
- **Fontes:** `Cargo.lock` contém apenas `source = "registry+https://github.com/rust-lang/crates.io-index"` (11 dependências externas). **Zero** dependências git.

### 2.2 Workspace `fuzz/` (fora do escopo)

- Workspace separado (`exclude` no workspace raiz), `publish = false`, 23 pacotes, todas as fontes crates.io.
- Licenças incluem **NCSA** (`libfuzzer-sys`) e as demais permissivas. **Não é coberto pelo `deny.toml` da raiz**; exigiria política própria. Registrado como limitação.

## 3. Licenças encontradas (com evidência)

| Pacote | Versão | Licença (SPDX) | Fonte |
|---|---|---|---|
| hpsi-format / hpsi-ops / hpsi-cli | 0.1.0 | `MIT OR Apache-2.0` | manifest |
| itoa | 1.0.18 | `MIT OR Apache-2.0` | registry `Cargo.toml` |
| memchr | 2.8.3 | `Unlicense OR MIT` | registry |
| serde | 1.0.229 | `MIT OR Apache-2.0` | registry |
| serde_core | 1.0.229 | `MIT OR Apache-2.0` | registry |
| serde_json | 1.0.154 | `MIT OR Apache-2.0` | registry |
| zmij | 1.0.23 | `MIT` | registry |
| serde_derive | 1.0.229 | `MIT OR Apache-2.0` | registry (opcional) |
| proc-macro2 | 1.0.107 | `MIT OR Apache-2.0` | registry (opcional) |
| quote | 1.0.47 | `MIT OR Apache-2.0` | registry (opcional) |
| syn | 3.0.9 | `MIT OR Apache-2.0` | registry (opcional) |
| unicode-ident | 1.0.27 | `(MIT OR Apache-2.0) AND Unicode-3.0` | registry (opcional) |

- **Nenhuma dependência sem licença identificada** no grafo raiz.
- **Licenças múltiplas:** apenas expressões `OR`/`AND` acima; cobertas pelos componentes da allow-list.
- **Exceções:** nenhuma necessária no grafo raiz.

## 4. Política criada (`deny.toml`)

Arquivo novo na raiz (não commitado nesta etapa):

```toml
[licenses]
allow = ["MIT", "Apache-2.0", "Unlicense", "Unicode-3.0"]
confidence-threshold = 0.8
exceptions = []

[sources]
unknown-registry = "deny"
unknown-git = "deny"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
allow-git = []
```

**Justificativa:**
- A allow-list contém **exatamente** as licenças auditadas; nenhum curinga.
- `MIT`, `Apache-2.0`, `Unlicense` cobrem o grafo resolvido por padrão; `Unicode-3.0` cobre `unicode-ident` quando a feature opcional `derive` de `serde` for habilitada (está no lock e foi verificada no registry).
- `exceptions = []` — nenhuma exceção por crate.
- Fontes restritas ao crates.io (`unknown-registry`/`unknown-git = deny`, `allow-git = []`).

**Sobre `unlicensed = "deny"`:** o `cargo-deny 0.20.2` **removeu** essa chave (`error[deprecated]: this key has been removed`, PR #611). A checagem passou a ser **deny-by-default**: qualquer licença (incluindo ausência de licença) que não esteja na allow-list é rejeitada. O efeito equivalente a `unlicensed = "deny"` está preservado **sem** a chave removida — a política não foi enfraquecida. Isso está documentado em comentário no próprio `deny.toml`.

## 5. Resultados dos gates

| Comando | Exit | Duração | Resultado |
|---|---:|---:|---|
| `cargo deny check licenses` | **0** | 20 s | `licenses ok` (warning `license-not-encountered` para `Unicode-3.0`) |
| `cargo deny check` | **0** | 410 s | `advisories ok, bans ok, licenses ok, sources ok` |
| `cargo audit` | **0** | 275 s | 1.430 advisories; **0 vulnerabilidades** (14 deps) |
| `cargo fmt --all -- --check` | **0** | — | sem divergências |
| `cargo test --workspace --all-targets` | **0** | 195 s | **287 passed / 0 failed / 0 ignored** |
| `cargo clippy --workspace --all-targets -- -D warnings` | **0** | 16 s | 0 avisos |

**Sobre a warning `license-not-encountered`:** `Unicode-3.0` não foi encontrada no grafo **resolvido por padrão** porque `unicode-ident` só entra com `serde/derive`; a allowance é intencional e defensiva (está no lock). É warning, não erro; exit 0. Nenhuma política foi relaxada.

Re-execução nesta revisão (cache aquecido): `cargo deny check licenses` — 2,3 s, exit 0,
`licenses ok: 0 errors, 1 warnings`; `cargo deny check` — exit 0,
`advisories ok, bans ok, licenses ok, sources ok`.

## 5.1 Cobertura de dependências opcionais (verificação solicitada)

A revisão apontou corretamente que constar na allow-list **não comprova** que a política
funciona para os 5 pacotes opcionais (`serde_derive`, `proc-macro2`, `quote`, `syn`,
`unicode-ident`). Esta seção fecha essa lacuna com evidência.

### 5.1.1 Os comandos solicitados falham na versão instalada

Executados exatamente como pedido, com `cargo-deny 0.20.2`:

| Comando | Exit | Resultado |
|---|---:|---|
| `cargo deny check licenses --all-features` | **2** | `error: unexpected argument '--all-features' found` |
| `cargo deny check --all-features` | **2** | `error: unexpected argument '--all-features' found` |

O `cargo deny --help` e o `cargo deny check --help` **não expõem** `--all-features`,
`--features` nem `--no-default-features`; a seleção de features só existe via `[graph]`
no arquivo de configuração.

### 5.1.2 Tentativas de habilitar `serde/derive` pela configuração

Todas foram registradas e **não alteraram** o grafo coletado (`gathered 9 crates`):

| Tentativa | Efeito observado |
|---|---|
| `[graph] all-features = true` | 9 crates — inócuo: os 3 membros do workspace **não declaram `[features]`** |
| `[graph] features = ["serde/derive"]` | 9 crates — chave aceita, sem efeito |
| `--metadata-path` com JSON cujo `resolve` contém os 14 pacotes alcançáveis | 9 crates — cargo-deny ainda poda a cadeia |
| `[graph] exclude = ["serde"]` (controle) | **8 crates** — prova que a seção `[graph]` **é** lida; o problema é específico de `features` |

Controle adicional: repetido num crate-sonda isolado, `[graph] features = ["serde/derive"]`
deu **3 crates antes e 3 depois** — a chave é aceita pela configuração mas não habilita
features de dependência. Isto é um comportamento do `cargo-deny 0.20.2`, não da política.

### 5.1.3 Demonstração de que a política funciona para os 5 pacotes

Como a ferramenta não percorre a cadeia a partir deste workspace, a cobertura foi
demonstrada ativando `derive` **no manifest de um crate-sonda** e reutilizando o
**mesmo `deny.toml`** (allow-list, `confidence-threshold`, `exceptions` e fontes
idênticos; nenhuma dependência do HPSI alterada):

```toml
[dependencies]
serde = { version = "1.0.229", features = ["derive"] }
```

Resultado do `cargo deny check licenses` no sonda:

```
INFO  gathered 8 crates
licenses ok: 0 errors, 1 warnings, 8 notes      (exit 0)
```

Crates vistos por `cargo deny list` (todos os 5 opcionais presentes e avaliados):

```
Apache-2.0 (7): proc-macro2@1.0.107, quote@1.0.47, serde@1.0.229, serde_core@1.0.229,
                serde_derive@1.0.229, syn@3.0.10, unicode-ident@1.0.27
MIT (8):        hpsi-optional-license-probe@0.1.0, proc-macro2, quote, serde, serde_core,
                serde_derive, syn, unicode-ident
Unicode-3.0 (1): unicode-ident@1.0.27
```

- `Unicode-3.0` **casou** com `unicode-ident` — a entrada defensiva da allow-list é
  exercitada de fato.
- **0 erros**: a política aceita as licenças dos 5 pacotes.
- O único warning é `license-not-encountered` para `Unlicense`, ausente nesse grafo
  mínimo — esperado e irrelevante para a cobertura em questão.
- Nota de versão: o sonda resolve um lock novo, por isso `syn 3.0.10` em vez de `3.0.9`
  do lock do repositório; ambos declaram `MIT OR Apache-2.0`.

### 5.1.4 Estado do repositório após a verificação

Nenhuma dependência, manifest ou versão foi alterado. A seção `[graph]` experimental foi
**removida** e o `deny.toml` foi restaurado exatamente ao conteúdo em revisão
(SHA-256 `1ABF78BAB7B2B5196064D55EB6FFD54F1F63E84974AFFF630B07AC6AFB150474`). Os
artefatos da investigação ficaram em `.tmp-phase4/` (backup do arquivo, metadata JSON,
log do `check` completo).

## 6. Limitações e pendências

1. **`unlicensed` removido no cargo-deny 0.20** — equivalente deny-by-default documentado;
   se for exigida a chave literal, seria necessário fixar uma versão antiga da ferramenta
   (não feito).

2. **O gate padrão audita 9 dos 14 pacotes do lock — e isso não é corrigível por
   configuração nesta versão.** `cargo-deny 0.20.2` rejeita `--all-features` na CLI (exit 2)
   e a chave `[graph] features` é aceita mas não habilita features de dependência
   (evidência na §5.1). Consequência: a cadeia opcional `serde/derive` **não** é percorrida
   pelo `cargo deny check` padrão, de modo que a cobertura dos 5 pacotes foi demonstrada
   por sonda com features ativas (§5.1.3), **não** por esse gate. O CI atual continua
   auditando apenas o grafo padrão.

3. **`license-not-encountered` (Unicode-3.0)** — esperado no grafo padrão, porque
   `unicode-ident` só entra com `serde/derive`. A allowance **não** é decorativa: foi
   exercitada e casada na sonda da §5.1.3 (`Unicode-3.0 (1): unicode-ident@1.0.27`).
   Mantida por decisão defensiva, já que o pacote está no lock.

4. **Workspace `fuzz/` fora do escopo, com risco concreto.** Verificado: `fuzz/` é
   workspace próprio (`publish = false`), `fuzz/Cargo.lock` tem **23 pacotes** e
   `fuzz/deny.toml` **não existe**, portanto nada o gateia. `libfuzzer-sys 0.4.13` declara
   `(MIT OR Apache-2.0) AND NCSA`; **`NCSA` não está na allow-list da raiz**, logo um
   `deny.toml` de raiz aplicado ao `fuzz/` **falharia**. Exige política própria (incluir
   `NCSA` apenas no escopo do `fuzz/`, nunca na raiz).

5. **Publicação** — não autorizada; pendente de token/ordem de publicação.

6. **Automação da cobertura opcional** — não implementada nesta etapa. Se for desejado
   gate contínuo, o caminho viável é uma etapa de CI que rode um crate-sonda com as
   features ativas (como na §5.1.3), e não o `cargo deny check` do workspace.

## 7. Estado Git final

| Item | Valor |
|---|---|
| Branch | `feat/hpsi-core-container` |
| HEAD | `bffa069a94eaf1345f345a943b1281d937975f6c` (**inalterado**) |
| Índice | vazio |
| `git diff HEAD` | vazio (nenhum arquivo rastreado alterado) |
| **Não rastreados** | os **8 itens de scratch** + `deny.toml` + este relatório (**10 `??`**) |
| Commit | **nenhum** (conforme a etapa) |

Saída observada (`git status --short`), inalterada em relação à etapa anterior:

```
?? .tmp-gpt2-conv1d-test/
?? .tmp-gpt2-final/
?? .tmp-phase4/
?? deny.toml
?? docs/audits/2026-11-09-hpsi-license-policy-closure.md
?? fuzz/coverage/
?? prepare_gpt2_test.py
?? tests/test_capture_transformer_activations.py.20261104-143035.bak
?? tests/test_validate_calibration_bundle.py.bak
?? tools/capture_transformer_activations.py.20261104-143035.bak
```

A verificação da §5.1 gravou artefatos **dentro de `.tmp-phase4/`**, que já era um dos 8
itens de scratch: nada novo apareceu no topo do `git status`. O `deny.toml` voltou ao
conteúdo em revisão (SHA-256 `1ABF78…0474`); nenhum arquivo rastreado foi tocado.

**Confirmação:** nenhum código, manifest, dependência ou versão foi alterado; os 8 itens de scratch/backups foram preservados; nenhum push, tag, merge, release, deploy ou publicação; nenhum commit criado.

---

*Relatório de fechamento da política de licenças. Aguardando revisão antes de qualquer commit.*
