# HPSI — política de licenças do workspace `fuzz`

**Data:** 2026-11-09
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**HEAD (inalterado):** `e86418bbbcb54f8b5f88e2f61ea8b2a60dbb88f8`
**Natureza:** auditoria e criação de política; **nenhum commit** nesta etapa; sem push, tag, merge, release, deploy ou publicação. Nenhum código, manifest, dependência ou versão alterado; `deny.toml` da raiz intocado.
**Revisão 3 (final):** exceção de NCSA fixada por **nome e versão** (`libfuzzer-sys = 0.4.13`), com controle negativo de versão (§6.2.2); gates reexecutados após a restauração (§6); relatório atualizado (§5, §7.6).

---

## 1. Estado inicial

| Item | Valor |
|---|---|
| Branch | `feat/hpsi-core-container` |
| HEAD | `e86418bbbcb54f8b5f88e2f61ea8b2a60dbb88f8` |
| Índice | vazio (0 arquivos staged) |
| `git diff` | vazio |
| Itens de scratch | **8 preservados**: `.tmp-gpt2-conv1d-test/`, `.tmp-gpt2-final/`, `.tmp-phase4/`, `fuzz/coverage/`, `prepare_gpt2_test.py`, `tests/test_capture_transformer_activations.py.20261104-143035.bak`, `tests/test_validate_calibration_bundle.py.bak`, `tools/capture_transformer_activations.py.20261104-143035.bak` |
| `deny.toml` (raiz) | SHA-256 `1abf78bab7b2b5196064d55eb6ffd54f1f63e84974afff630b07ac6afb150474` (inalterado, conferido antes e depois) |
| Ferramentas | `cargo-deny 0.20.2`, `cargo-audit-audit 0.22.2`, `cargo 1.85.0`, `rustc 1.85.0` (toolchain fixada por `rust-toolchain.toml`) |

Nenhuma divergência inesperada. Os artefatos temporários desta auditoria (metadata JSON, saídas de gate, sonda) ficaram em `%TEMP%\hpsi-fuzz-license\`, **fora** do repositório; `.tmp-phase4/` não foi tocado.

## 2. Escopo e grafo auditado

- **Workspace independente:** `fuzz/` é excluído do workspace raiz (`Cargo.toml`, `exclude = ["fuzz"]`) e declara `[workspace] members = ["."]`, `publish = false`.
- **`fuzz/Cargo.lock`:** **23 pacotes** — `hpsi-fuzz` (membro), `hpsi-format` (path `../crates/hpsi-format`) e **21 crates externos**.
- **Grafo padrão no host** (`cargo tree -e normal,build,dev`): **15 pacotes** (2 locais + 13 externos).
- **Grafo avaliado pelo `cargo-deny`** (`cargo deny list`): **17 pacotes** (2 locais + 15 externos) — inclui `libc` e `r-efi`, que são dependências condicionadas por alvo e por isso não aparecem no `cargo tree` do host.
- **Fora do grafo padrão (6 externos):** `serde`, `serde_derive`, `proc-macro2`, `quote`, `syn`, `unicode-ident` — alcançáveis apenas pela feature `derive` de `serde`, não habilitada.
- **Fontes:** 21 pacotes em `registry+https://github.com/rust-lang/crates.io-index` e 1 path local. **Zero** fontes git, **zero** registries alternativos.
- **Publicação:** `hpsi-fuzz` tem `publish = false` (`[]` no metadata). `hpsi-format` não declara `publish` (publicável), mas aqui entra apenas como dependência de path.
- **Tipos de dependência:** há 1 dependência de build (`cc` → build de `libfuzzer-sys`), 1 proc-macro (`serde_derive`, presente no lock, não ativada) e as demais normais/transitivas.

### 2.1 Arestas relevantes (filtros de alvo/feature)

| Aresta | Filtro | Efeito |
|---|---|---|
| `getrandom → r-efi` | `cfg(all(target_os = "uefi", getrandom_backend = "efi_rng"))` | nunca compila em Windows/Linux |
| `cc → libc`, `jobserver → libc`, `getrandom → libc` | `cfg(unix)` | só em alvos Unix |
| `jobserver → getrandom` | `cfg(windows)` | ativo neste host |
| `serde_json → serde` | `cfg(any())` (só por feature) | não ativado |
| `serde_core → serde_derive` | `cfg(any())` (feature `derive`) | não ativado |
| `libfuzzer-sys → cc` | `kind = build` | script de build do runtime |
| `libfuzzer-sys → arbitrary`, `cc → find-msvc-tools`/`shlex`/`jobserver`, `serde_json → itoa`/`memchr`/`serde_core`/`zmij`, `proc-macro2`/`syn → unicode-ident` | normal | sempre ativos quando o pai entra |

## 3. Inventário de dependências e licenças

| Pacote | Versão | Licença (SPDX) | Como entra | Grafo padrão |
|---|---|---|---|---|
| hpsi-fuzz | 0.0.0 | `MIT OR Apache-2.0` | membro do workspace | sim |
| hpsi-format | 0.1.0 | `MIT OR Apache-2.0` | path (`../crates/hpsi-format`) | sim |
| libfuzzer-sys | 0.4.13 | `(MIT OR Apache-2.0) AND NCSA` | normal de `hpsi-fuzz` | sim |
| arbitrary | 1.5.0 | `MIT OR Apache-2.0` | normal de `libfuzzer-sys` | sim |
| cc | 1.11.1 | `MIT OR Apache-2.0` | **build** de `libfuzzer-sys` | sim |
| find-msvc-tools | 0.1.15 | `MIT OR Apache-2.0` | normal de `cc` | sim |
| shlex | 2.0.1 | `MIT OR Apache-2.0` | normal de `cc` | sim |
| jobserver | 0.1.35 | `MIT OR Apache-2.0` | normal de `cc` | sim |
| getrandom | 0.4.3 | `MIT OR Apache-2.0` | normal de `jobserver` (`cfg(windows)`) | sim |
| cfg-if | 1.0.5 | `MIT OR Apache-2.0` | normal de `getrandom` | sim |
| libc | 0.2.191 | `MIT OR Apache-2.0` | `cfg(unix)` | não no host; visto pelo `cargo-deny` |
| r-efi | 6.0.0 | `MIT OR Apache-2.0 OR LGPL-2.1-or-later` | `cfg(all(uefi, efi_rng))` | não; visto pelo `cargo-deny` |
| serde_json | 1.0.154 | `MIT OR Apache-2.0` | normal de `hpsi-format` | sim |
| itoa | 1.0.18 | `MIT OR Apache-2.0` | normal de `serde_json` | sim |
| memchr | 2.8.3 | `Unlicense OR MIT` | normal de `serde_json` | sim |
| zmij | 1.0.23 | `MIT` | normal de `serde_json` | sim |
| serde_core | 1.0.229 | `MIT OR Apache-2.0` | normal de `serde_json` | sim |
| serde | 1.0.229 | `MIT OR Apache-2.0` | `serde_json`, só por feature | não |
| serde_derive | 1.0.229 | `MIT OR Apache-2.0` | `serde_core`, feature `derive` (**proc-macro**) | não |
| proc-macro2 | 1.0.107 | `MIT OR Apache-2.0` | normal de `serde_derive` | não |
| quote | 1.0.47 | `MIT OR Apache-2.0` | normal de `serde_derive` | não |
| syn | 3.0.9 | `MIT OR Apache-2.0` | normal de `serde_derive` | não |
| unicode-ident | 1.0.27 | `(MIT OR Apache-2.0) AND Unicode-3.0` | normal de `proc-macro2`/`syn` | não |

**Resumo das identificações SPDX:** 17 pacotes em `MIT OR Apache-2.0` (16 externos + `hpsi-format`), 1 `MIT`, 1 `Unlicense OR MIT`, 1 `(...) AND NCSA`, 1 `(...) AND Unicode-3.0`, 1 `MIT OR Apache-2.0 OR LGPL-2.1-or-later`.

**Nenhuma licença não identificada; nenhum pacote sem metadado de licença; nenhum `license_file` como fonte única.**

## 4. Evidência das licenças que não estavam na política da raiz

### 4.1 NCSA — `libfuzzer-sys 0.4.13`

- Metadado publicado: `license = "(MIT OR Apache-2.0) AND NCSA"` (idêntico em `Cargo.toml` linha 36 e `Cargo.toml.orig` linha 5).
- A seção *License* do `README.md` do próprio crate declara explicitamente: *"All files in the `libfuzzer` directory are licensed NCSA."* e *"Everything else is dual-licensed Apache 2.0 and MIT."*
- A expressão publicada **corresponde a essa divisão**: a conjunção (`AND`) exige os dois ramos — MIT/Apache-2.0 para o wrapper Rust e NCSA para o runtime vendorizado em `libfuzzer/`.
- O pacote distribui `LICENSE-MIT` e `LICENSE-APACHE`, mas **não inclui um arquivo separado com o texto NCSA**. Isso é **ausência do texto no pacote**, e não ausência da concessão: a concessão está declarada no manifesto e descrita no README do projeto. Para redistribuição, o texto aplicável deve ser obtido do upstream (LLVM/Universidade de Illinois).
- Observação de campo, sem conclusão jurídica: os 53 arquivos de `libfuzzer/` carregam o cabeçalho `SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception` (55 ocorrências do cabeçalho) e nenhum contém o texto "University of Illinois". Essa é a marcação SPDX do próprio upstream LLVM para essas fontes e **coexiste** com a declaração NCSA do crate; qual texto acompanha a redistribuição desses arquivos é uma questão que esta auditoria **registra, mas não decide** (§7.3).
- **Consequência para a política:** o `cargo-deny` avalia a expressão SPDX publicada, que é `(MIT OR Apache-2.0) AND NCSA`. NCSA é concedida por **exceção restrita ao `libfuzzer-sys` na versão `=0.4.13`** (§5), nunca em âmbito global nem para outra versão.

### 4.2 Unicode-3.0 — `unicode-ident 1.0.27`

`(MIT OR Apache-2.0) AND Unicode-3.0`; o pacote distribui `LICENSE-UNICODE` além de MIT/Apache. Já constava da política da raiz pela mesma razão.

### 4.3 LGPL-2.1-or-later — `r-efi 6.0.0`

`MIT OR Apache-2.0 OR LGPL-2.1-or-later`. É a **segunda licença adicional além de NCSA** — não presumida. O pacote **não distribui arquivo de licença**; a expressão no manifesto é a única evidência. O crate só entraria em um alvo UEFI com `getrandom_backend = "efi_rng"`. A política **não libera LGPL**: como a expressão é disjuntiva, o ramo `MIT`/`Apache-2.0` a satisfaz. O `cargo deny check licenses` passa com `r-efi` no grafo (exit 0), o que comprova essa avaliação.

### 4.4 Demais

- `memchr 2.8.3` (`Unlicense OR MIT`) distribui `UNLICENSE`, `COPYING` e `LICENSE-MIT`; `Unlicense` já constava da raiz.
- `zmij 1.0.23` (`MIT`) distribui `LICENSE-MIT`.

## 5. Política criada (`fuzz/deny.toml`)

Arquivo **novo**, exclusivo do workspace `fuzz/`:

```toml
[licenses]
allow = [
    "MIT",
    "Apache-2.0",
    "Unlicense",
    "Unicode-3.0",
]
confidence-threshold = 0.8

[[licenses.exceptions]]
allow = ["NCSA"]
name = "libfuzzer-sys"
version = "=0.4.13"

[sources]
unknown-registry = "deny"
unknown-git = "deny"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
allow-git = []
```

**Justificativa:**

- A allow-list **geral** contém apenas `MIT`, `Apache-2.0`, `Unlicense` e `Unicode-3.0` — exatamente os identificadores SPDX utilizáveis pelas expressões do grafo fora a exceção.
- **`NCSA` não está na allow-list geral.** Ela é concedida por **uma única exceção**, restrita simultaneamente por **nome** e por **versão** (`name = "libfuzzer-sys"`, `version = "=0.4.13"`). Qualquer outro crate que declare NCSA continua rejeitado, e uma versão diferente de `libfuzzer-sys` também não herda a concessão — ambos os casos foram **verificados com controles negativos** (§6.2): nome incorreto e versão incorreta falham com `NCSA … rejected: license is not explicitly allowed` (exit 4).
- **Sem curingas**, `confidence-threshold = 0.8`. A versão exata `=0.4.13` é a versão auditada, presente no lock; qualquer atualização de `libfuzzer-sys` passa a falhar o gate até que a nova expressão seja auditada e a restrição atualizada — comportamento *fail-closed* deliberado (§7.6).
- **LGPL-2.1-or-later deliberadamente fora**: só aparece como alternativa em `r-efi`, cuja expressão é satisfeita por `MIT`/`Apache-2.0`; liberá-la ampliaria a política para uma licença copyleft nunca usada.
- `Unicode-3.0` é mantida porque `unicode-ident` está no lock (entra com `serde/derive`); a warning `license-not-encountered` no grafo padrão é esperada e não é erro — mesma situação documentada na política da raiz.
- **Fontes explícitas:** somente o índice oficial do crates.io; qualquer outro registry ou fonte git é negado; `allow-git = []`.
- **`unlicensed`:** removida no cargo-deny 0.20 (PR #611). A checagem é deny-by-default, então ausência de licença é rejeitada por não constar da allow-list — mesmo efeito, sem a chave removida. Documentado em comentário no arquivo.
- **Relação com a raiz:** `deny.toml` da raiz permanece com escopo restrito a `hpsi-format`/`hpsi-ops`/`hpsi-cli`; **NCSA não foi adicionada à raiz** e o arquivo raiz não foi modificado (hash conferido). `fuzz/deny.toml` é aplicado automaticamente ao rodar `cargo deny` a partir de `fuzz/` (confirmado pela referência ao caminho no diagnóstico da warning).

## 6. Resultados dos gates

Todos executados **dentro de `fuzz/`**, com a toolchain fixada `1.85.0`:

| Comando | Exit | Duração | Resultado |
|---|---:|---:|---|
| `cargo deny check licenses` | **0** | 3,5 s (0,9 s com cache quente) | `licenses ok`; NCSA aceita **apenas** pela exceção `libfuzzer-sys = 0.4.13`; sem warning de exceção; 1 warning `license-not-encountered` (`Unicode-3.0`) |
| `cargo deny check` | **0** | 1 min 33 s (6,2 s após a restauração) | `advisories ok, bans ok, licenses ok, sources ok`; 0 warnings de exceção |
| `cargo audit` | **0** | 58,6 s | 23 dependências do lock; 1.430 advisories; **0 vulnerabilidades** |
| `cargo fmt --all -- --check` | **0** | 1 min 14 s | sem divergências, sem saída |
| `cargo test --workspace --all-targets` | **101** | 6 min 55 s | falha de **link** `LNK1561` nos dois bins — pré-existente, ver §7.2 |
| `cargo test --workspace` | **0** | 5,4 s | `Finished test profile`; **0 testes** (nenhum alvo de teste; bins têm `test = false`) |
| `cargo check --bins` | **0** | 48,2 s | compila os dois alvos (equivalente estável do passo `cargo +nightly check --bins` do CI) |
| `cargo clippy --workspace --all-targets -- -D warnings` | **0** | 52,4 s | 0 avisos |
| `cargo build --bins` | **101** | 11,6 s | mesma falha `LNK1561` de link (não afeta a política; ver §7.2) |
| Sonda (§6.1): `cargo deny --config fuzz/deny.toml check licenses` | **0** | 2 min 56 s | `licenses ok`; 21 externos cobertos com a configuração final |
| Controle negativo (§6.2): exceção com **nome** incorreto | **4** | 1,4 s | `NCSA … rejected: license is not explicitly allowed` — exceção não é global |
| Controle negativo (§6.2): exceção com **versão** incorreta (`=0.4.12`) | **4** | 2,1 s | mesma rejeição + `license-exception-not-encountered` — exceção não vale para outra versão |
| Controle raiz: `cargo deny check licenses` na raiz | **0** | 3,1 s | `licenses ok`; política raiz intacta |

**Sobre a warning `license-not-encountered`:** `Unicode-3.0` não é encontrada no grafo resolvido por padrão porque `unicode-ident` só entra com a feature `derive` de `serde`. A allowance é intencional (o pacote está no lock e foi auditado) e a sonda de §6.1 a exercita de fato. É warning, não erro; **nenhuma política foi relaxada** para obter resultado verde.

### 6.1 Cobertura dos pacotes fora do grafo padrão

O `cargo-deny 0.20.2` poda dependências não ativadas por feature (comportamento já documentado no relatório da raiz, §5.1) e não aceita `--all-features` na CLI. Para provar que a política cobre **os 21 pacotes externos do lock**, foi usada uma sonda isolada em `%TEMP%`, com lock próprio e **a mesma configuração** (`--config fuzz/deny.toml`), habilitando `serde/derive`:

```toml
[dependencies]
serde = { version = "=1.0.229", features = ["derive"] }
serde_json = "=1.0.154"
memchr = "=2.8.3"
zmij = "=1.0.23"
libfuzzer-sys = "=0.4.13"
arbitrary = "=1.5.0"
```

Resultado: `licenses ok` (exit 0, **sem warnings**) e `cargo deny list` com os 21 externos, incluindo `serde`, `serde_derive`, `proc-macro2`, `quote`, `syn`, `unicode-ident`, `libc` e `r-efi`. A sonda resolveu `itoa 1.0.19` e `syn 3.0.10` (patch acima do lock), com as mesmas expressões. A sonda roda fora do repositório e por isso usa a toolchain padrão do host (`stable`, não a fixada por `rust-toolchain.toml`); isso não altera a avaliação de licenças. Nenhuma dependência ou manifest do repositório foi alterado.

A sonda foi **reexecutada após a mudança para exceção** (2 min 56 s, exit 0, `licenses ok` sem warnings): `libfuzzer-sys` passa pela exceção nomeada e os demais 20 externos pela allow-list geral.

### 6.2 Controles negativos da exceção (nome e versão)

Para provar que a exceção **não** equivale a liberar NCSA no grafo, dois controles negativos foram executados contra o workspace real. Em ambos, o resultado esperado é a rejeição de NCSA.

**6.2.1 Nome:** a configuração foi copiada para `%TEMP%` com uma única alteração — `name = "libfuzzer-sys-typo"` — e aplicada com `--config`:

```
$ cargo deny --config .../deny-typo.toml check licenses
error[rejected]: failed to satisfy license requirements
  ┌─ ...\libfuzzer-sys-0.4.13\Cargo.toml:36:36
36 │ license = "(MIT OR Apache-2.0) AND NCSA"
   │                                   │
   │                                   rejected: license is not explicitly allowed
warning[license-exception-not-encountered]: license exception was not encountered
licenses FAILED        (exit 4, 1,4 s)
```

**6.2.2 Versão:** a restrição do próprio repositório foi alterada **temporariamente** para `version = "=0.4.12"` e o gate foi reexecutado:

```
$ cargo deny check licenses
error[rejected]: failed to satisfy license requirements
  ┌─ ...\libfuzzer-sys-0.4.13\Cargo.toml:36:36
36 │ license = "(MIT OR Apache-2.0) AND NCSA"
   │                                   │
   │                                   rejected: license is not explicitly allowed
warning[license-exception-not-encountered]: license exception was not encountered
  ┌─ C:\Users\lucas\HPSI\fuzz\deny.toml:45:9
45 │ name = "libfuzzer-sys"
licenses FAILED        (exit 4, 2,1 s)
```

Em seguida a restrição foi **restaurada** para `version = "=0.4.13"`; a comparação byte a byte com a cópia de referência confirmou que o arquivo voltou idêntico (SHA-256 `9f54c9cc67f90fe00325acf261d9a3cffb9e0b31c6dab44beb862b5a8e978b7d`) e os dois gates foram reexecutados com sucesso (§6).

**Conclusão:** NCSA só é aceita quando o crate é exatamente `libfuzzer-sys` **na versão** `0.4.13`; crate diferente ou versão diferente falham. Nenhuma dependência, manifest ou arquivo rastreado foi alterado por esses testes.

## 7. Limitações e riscos

### 7.1 Cobertura do grafo pelo cargo-deny

O gate padrão avalia 17 dos 23 pacotes do lock (15 dos 21 externos); os 6 restantes são feature-gated e só podem ser avaliados por sonda (§6.1). Não é corrigível por configuração nesta versão (`--all-features` rejeitado, `[graph] features` inócuo — ver relatório da raiz §5.1). Isso é comportamento da ferramenta, não da política.

### 7.2 `cargo test --all-targets` e `cargo build` falham no Windows

Os dois bins são `#![no_main]` e dependem do `main` fornecido pelo runtime libFuzzer; no host Windows/MSVC o link termina com `LNK1561: pontos de entrada devem ser definidos`. É **pré-existente e independente desta política** (`cargo deny` não participa da compilação; `code = 101` ocorre igualmente em `cargo build --bins`). O README do workspace já restringe o libFuzzer a Linux + sanitizer, e o CI compila com `cargo +nightly check --bins` em `ubuntu-latest`, que é o gate equivalente aqui reproduzido com `cargo check --bins` (exit 0). **Nenhum fuzzing real foi executado nesta auditoria**; o workspace não tem testes unitários (`cargo test --workspace` roda 0 testes).

### 7.3 NCSA: concessão declarada × texto não empacotado

- O `libfuzzer-sys` **declara** NCSA no manifesto e explica a divisão no README (arquivos de `libfuzzer/` em NCSA; o restante em MIT/Apache-2.0). O pacote **não traz um arquivo separado com o texto NCSA**.
- Ausência do texto no pacote **não** equivale a ausência da concessão: a concessão consta da declaração SPDX publicada e da documentação do projeto; o texto aplicável pode ser obtido do upstream (LLVM/Universidade de Illinois).
- Os 53 arquivos vendorizados trazem o cabeçalho SPDX `Apache-2.0 WITH LLVM-exception` do próprio LLVM, que **coexiste** com a declaração NCSA do crate.
- **Incerteza remanescente (registrada, não resolvida):** qual texto/versão acompanha a redistribuição do código vendorizado e como interpretar a coexistência entre a declaração do crate e os cabeçalhos dos arquivos. Esta auditoria **não afirma conformidade jurídica absoluta**; se o HPSI distribuir binários que incorporem esse runtime, recomenda-se revisão jurídica específica.
- **Efeito no gate:** o `cargo-deny` valida a expressão publicada. Se o crate alterar a expressão, a exceção precisa ser revista — adicionar ou remover identificadores somente após nova auditoria, nunca por antecipação.

### 7.4 `r-efi` sem texto de licença

Só há a expressão no manifesto. `LGPL-2.1-or-later` não foi aprovada; o uso é pelo ramo MIT/Apache-2.0. Se o crate passar a declarar apenas LGPL, o gate falhará — comportamento desejado.

### 7.5 Dependência da rede e do banco de advisories

`cargo deny check` (8 min 28 s) e `cargo audit` (58,6 s) dependem do banco RustSec remoto e de cache local; primeiras execuções em máquina fria podem ser bem mais lentas e exigem rede. `cargo audit` cobre vulnerabilidades RustSec, não licenças.

### 7.6 Exceção fixada por nome e versão

- A concessão de NCSA é escopada a `libfuzzer-sys` **na versão `=0.4.13`** — a versão auditada, presente no lock.
- Qualquer atualização de versão faz o gate falhar até revisão (comportamento *fail-closed* deliberado), evitando que uma versão futura herde a concessão sem auditoria. O controle negativo de §6.2.2 demonstra exatamente essa falha.
- **Custo operacional registrado:** qualquer bump de `libfuzzer-sys` exige auditar a expressão declarada e atualizar a restrição. Procedimento: auditar a nova versão → ajustar allow-list/exceção → reexecutar os gates → atualizar este relatório.
- Se a expressão futura não exigir mais NCSA, a exceção deve ser **removida**, não apenas mantida por inércia.

## 8. Vereditos

| Requisito | Veredito |
|---|---|
| §1 Pré-verificação (branch, HEAD, índice, árvore, scratch, `deny.toml` raiz) | **PASS** |
| §2 Auditoria integral do workspace `fuzz` (lock, grafo, SPDX, OR/AND, fontes, `publish`, `libfuzzer-sys`/NCSA) | **PASS** |
| §3 `fuzz/deny.toml` com allow-list auditada (sem NCSA global), exceção restrita a `libfuzzer-sys = 0.4.13`, sem curingas, deny-by-default, fontes explícitas, raiz intocada | **PASS** — exceção comprovada por controles negativos de nome e versão (§6.2) |
| §4 Gates dentro de `fuzz/` | **PASS com limitação registrada** — todos os gates aplicáveis verdes; `cargo test --all-targets` falha por limitação de link pré-existente no Windows (§7.2), documentada em vez de mascarada |
| §5 Relatório com inventário, licenças, configuração, resultados e limitações | **PASS** (este documento) |
| §6 Restrições (sem alterar código/deps/manifests/versões, sem tocar raiz/scratch, sem `git add -A`, reset/clean/restore/rebase, sem push/merge/tag/release/deploy, **sem commit**) | **PASS** |

Nenhum requisito foi declarado aprovado apenas porque um comando ficou verde: a cobertura real de pacotes, a natureza da concessão NCSA (texto não empacotado, com incerteza registrada) e a impossibilidade de linkar/testar os alvos no Windows estão explicitadas acima.

## 9. Estado Git final

| Item | Valor |
|---|---|
| Branch | `feat/hpsi-core-container` |
| HEAD | `e86418bbbcb54f8b5f88e2f61ea8b2a60dbb88f8` (inalterado) |
| Índice | vazio |
| Arquivos rastreados alterados | **0** |
| Novos arquivos | `fuzz/deny.toml` (SHA-256 `9f54c9cc67f90fe00325acf261d9a3cffb9e0b31c6dab44beb862b5a8e978b7d`) e este relatório |
| Scratch | os 8 itens preservados, sem alteração |
| `deny.toml` raiz | SHA-256 `1abf78bab7b2b5196064d55eb6ffd54f1f63e84974afff630b07ac6afb150474` (inalterado) |

## 10. Reprodução

```sh
cd fuzz

cargo deny check licenses
cargo deny check
cargo audit
cargo fmt --all -- --check
cargo test --workspace            # 0 testes
cargo test --workspace --all-targets   # falha LNK1561 no Windows (pré-existente)
cargo check --bins
cargo clippy --workspace --all-targets -- -D warnings
```

Sonda de cobertura (fora do repositório):

```sh
cargo deny --config /c/Users/lucas/HPSI/fuzz/deny.toml check licenses
cargo deny --config /c/Users/lucas/HPSI/fuzz/deny.toml list
```

Controles negativos da exceção (devem falhar com exit 4):

```sh
# (a) nome incorreto, em cópia temporária
sed 's/name = "libfuzzer-sys"/name = "libfuzzer-sys-typo"/' fuzz/deny.toml > /tmp/deny-typo.toml
cd fuzz && cargo deny --config /tmp/deny-typo.toml check licenses

# (b) versão incorreta, no próprio arquivo, seguido de restauração
sed -i 's/version = "=0.4.13"/version = "=0.4.12"/' deny.toml
cargo deny check licenses          # deve falhar
sed -i 's/version = "=0.4.12"/version = "=0.4.13"/' deny.toml
cargo deny check licenses && cargo deny check   # deve passar
```
