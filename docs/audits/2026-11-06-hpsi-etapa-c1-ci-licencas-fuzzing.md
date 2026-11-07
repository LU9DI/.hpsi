# HPSI — Etapa C1: auditoria de CI, licenças e fuzzing

**Data:** 2026-11-06
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**HEAD de referência (início):** `68843a93b8cdb51d3d5c905d3d841bfa7d88e34b`
**Escopo:** preparar a Etapa C da release HPSI Toolkit v0.1 auditando três bloqueadores — CI, licenciamento e fuzzing. **Nenhuma mudança de produto**; este relatório é o único artefato versionado.

---

## 1. Estado inicial e final do Git

Evidência: `.tmp-phase4/phaseC1/01-state.txt`.

| Item | Valor |
|---|---|
| Branch | `feat/hpsi-core-container` |
| HEAD | `68843a93b8cdb51d3d5c905d3d841bfa7d88e34b` |
| Índice | vazio (`e3b0c442…`, hash do arquivo vazio) |
| sha256 do diff rastreado | `96a15cc9577a0fb7b1ca4137dc99a467ee6b5bc1bc1910426d9fd4a6ec5cfc0d` |
| Trabalho local preservado | 15 modificados (` M`) + 29 entradas não rastreadas (`??`) |
| `crates/hpsi-format/` | **sem diff** (formato v0.1 intocado) |
| Identidade Git | `lucas <bostavacabosta@gmail.com>` (configurada) |
| `docs/audits/` e este arquivo | **não existiam** antes desta etapa |

**Preservação:** nenhum `reset`, `clean`, `checkout`, `restore` ou rebase; nenhum push/merge/tag/deploy. As 44 entradas da linha de base permanecem. O único commit criado é o deste relatório (§ commit registrado na verificação final).

**Efeito colateral de ambiente (registrado por transparência):** ao verificar a disponibilidade da toolchain, `rustc +nightly --version` disparou o `rustup`, que **instalou** `nightly-x86_64-pc-windows-msvc` (rustc `1.101.0-nightly`). Isso ocorreu fora do repositório e **não alterou nenhum arquivo do projeto** (`git diff` permaneceu `96a15cc9…`). Não foram instalados `cargo-fuzz`, `cargo-license` ou `cargo-deny`.

---

## 2. Auditoria do CI e proposta de correção

Arquivo: `.github/workflows/ci.yml`. O working tree difere do HEAD apenas por **+8 linhas** já existentes (o passo `pytest` e seu comentário), alteração local prévia preservada e **não** parte deste commit.

### 2.1 Estrutura atual

| Job | Runs-on | Condição | Comandos | Git? |
|---|---|---|---|---|
| `python-capture-tools` | ubuntu | sempre | `pip install pytest numpy`; `py_compile`; `unittest discover`; `pytest tests -q` | não |
| `format` | ubuntu | `push` **e** branch `feat/**` | `cargo fmt --all`; **auto-commit** | **sim** |
| `rust` | ubuntu/windows/macos | `needs: format` | `cargo fmt --all -- --check`; `clippy -D warnings`; `cargo test --workspace --all-targets` | não |
| `vq-benchmark` | ubuntu | sempre | `cargo run -p hpsi-ops --example vq_benchmark --release` | não |
| `fuzz-smoke` | ubuntu | sempre | `cargo +nightly fmt -- --check`; `cargo +nightly check --bins` (dir `fuzz`) | não |

- **Permissões:** nível de workflow `permissions: contents: write` (linha 16).
- **Concorrência:** `concurrency: group hpsi-ci-${{ github.ref }}`, `cancel-in-progress: false`.
- **Gatilhos:** `push` em `main` e `feat/**`; `pull_request`; `workflow_dispatch`.

### 2.2 Operações Git do workflow (o problema)

No job `format` (linhas 58–66):

```yaml
- name: Commit formatting changes
  run: |
    if ! git diff --quiet; then
      git config user.name "github-actions[bot]"
      git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
      git add -u
      git commit -m "style: format Rust sources"
      git push
    fi
```

Riscos: (a) a CI **reescreve o branch de feature** com um commit automático; (b) exige `contents: write`; (c) `git add -u` pode capturar mudanças não intencionais presentes no runner; (d) o job `rust` **depende** dele (`needs: format`), então a verificação de testes pode rodar sobre um commit diferente do enviado; (e) um loop de commits pode ocorrer se a formatação gerar diferenças repetidas.

### 2.3 Proposta de correção (não aplicada nesta etapa)

1. **Remover o job `format`** (e sua dependência em `rust`).
2. Manter no job `rust` a verificação estrita já existente: **`cargo fmt --all -- --check`** (linha 90).
3. Reduzir permissões para **`permissions: contents: read`**.
4. Remover o comentário de concorrência sobre o formatter (linhas 9–10) e, se desejado, o bloco `concurrency` deixa de ser necessário para esse fim.
5. Manter `clippy -D warnings`, `cargo test --workspace --all-targets`, `vq-benchmark` e `fuzz-smoke` como estão.

Resultado: nenhum workflow escreve no repositório, o conjunto verificado é exatamente o enviado, e a formatação vira um gate que falha em vez de autocorrigir.

---

## 3. Inventário de licenças e pendências

### 3.1 Metadados do projeto

- `[workspace.package] license = "MIT"`; `hpsi-format`/`hpsi-cli` usam `license.workspace = true`; `hpsi-ops` declara `license = "MIT"`.
- **Não existe `LICENSE` no repositório.**
- `Cargo.lock` (version 4): 14 pacotes; `fuzz/` é workspace isolado (`exclude`) com `libfuzzer-sys`.

### 3.2 Licenças por pacote

| Pacote | Versão | Licença | Fonte |
|---|---|---|---|
| serde | 1.0.229 | `MIT OR Apache-2.0` | **confirmada** (registry src) |
| serde_core | 1.0.229 | `MIT OR Apache-2.0` | **confirmada** |
| serde_json | 1.0.154 | `MIT OR Apache-2.0` | **confirmada** |
| itoa | 1.0.18 | `MIT OR Apache-2.0` | **confirmada** |
| memchr | 2.8.3 | `Unlicense OR MIT` | **confirmada** |
| zmij | 1.0.23 | `MIT` | **confirmada** |
| serde_derive | 1.0.229 | — | **NÃO VERIFICADA** (ausente no registry local) |
| proc-macro2 | 1.0.107 | — | **NÃO VERIFICADA** |
| quote | 1.0.47 | — | **NÃO VERIFICADA** |
| syn | 3.0.9 | — | **NÃO VERIFICADA** |
| unicode-ident | 1.0.27 | — | **NÃO VERIFICADA** |
| libfuzzer-sys | 0.4.x | — | **NÃO VERIFICADA** (dependência só do `fuzz/`) |

Distinção explícita: **confirmadas** = lidas do `Cargo.toml` extraído no registry local; **não verificadas** = crates não presentes localmente e sem rede para consulta (`cargo-license`/`cargo-deny` não instalados).

### 3.3 MIT × MIT OR Apache-2.0

| Aspecto | MIT | MIT OR Apache-2.0 (dual) |
|---|---|---|
| Obrigação básica | incluir texto MIT + copyright | incluir o texto da licença escolhida + avisos |
| `LICENSE` | obrigatório | obrigatório (e `NOTICE` se usar Apache-2.0) |
| Compatibilidade com as deps | ok (deps são `MIT OR Apache-2.0`, `Unlicense OR MIT`, `MIT`) | ok; alinha com a norma do ecossistema Rust |
| Redistribuição binária | simples | exige tratar `NOTICE` ao optar por Apache-2.0 |
| Arquivos necessários | `LICENSE` | `LICENSE-MIT`, `LICENSE-APACHE`, `NOTICE`/`THIRD-PARTY-NOTICES` |

Atribuições de terceiros: ao redistribuir binário, é preciso preservar os avisos das dependências; recomenda-se um arquivo `THIRD-PARTY-NOTICES` gerado por ferramenta. Dependências Python (`torch`, `transformers`, `numpy`, `pytest`) estão apenas em `.venv-capture` (auto-ignorado) e/ou no ambiente do usuário — **não** são redistribuídas com o HPSI.

### 3.4 Ferramentas

`cargo-license` e `cargo-deny`: **não instalados**. Sem eles, a verificação das 6 licenças não confirmadas fica pendente de rede/instalação (decisão do responsável). Nada foi instalado nem criado.

---

## 4. Estado dos alvos de fuzzing

- **Pacote:** `fuzz/` (`hpsi-fuzz`), workspace próprio (`[workspace] members = ["."]`), `publish = false`, edition 2024; depende de `libfuzzer-sys = "0.4"` e do `hpsi-format` local.
- **Alvos:**
  - `fuzz_header` → `Header::parse(data, data.len(), &ReaderLimits::default())`.
  - `fuzz_model_reader` → `ModelReader::open` + `verify_all`, com corte de entrada em 8 MiB.
- **Workflows:** `ci.yml` job `fuzz-smoke` (nightly) compila os alvos; `fuzz.yml` (manual) roda `cargo fuzz run <target> -- -max_total_time=<duration> -print_final_stats=1` e publica artefatos de crash.
- **Ferramentas locais:** `nightly` agora presente (`1.101.0-nightly`); **`cargo-fuzz` ausente**; portanto não há como executar `cargo fuzz` neste host sem instalar `cargo-fuzz`.
- **Corpus:** `fuzz/corpus/` **não localizado**; `fuzz/artifacts/` **não localizado**.
- **Cobertura/relatórios:** não localizados.
- **Classificação correta da evidência:** *não localizada* ≠ *inexistente*. Existem alvos e workflow, e a documentação afirma explicitamente que **nenhuma campanha sustentada foi concluída**: `README.md` linhas 588, 604 e 609 (*"No sustained fuzz run has been completed yet, so no fuzz-coverage claim is made"*). Ou seja: fuzzing **foi implementado como alvo**, mas **não há campanha executada registrada**.
- **Sementes reutilizáveis (locais, não versionadas):** containers válidos `.tmp-phase4/phase45-e2e/gpt2-{tied,untied}.hpsi`, `.tmp-gpt2-final/*.hpsi`, `.tmp-gpt2-conv1d-test/*.hpsi`; e os fixtures inline de `crates/hpsi-format/tests/reader.rs`.

---

## 5. Critérios propostos para a campanha de release

Proposta **mensurável** (a executar na Etapa C, não iniciada):

| Parâmetro | Proposta |
|---|---|
| Duração por alvo | **3600 s** (1 h) para `fuzz_header` e `fuzz_model_reader`; mínimo de 600 s em pré-execução |
| Corpus inicial | semear `fuzz/corpus/fuzz_header/` e `fuzz/corpus/fuzz_model_reader/` com os containers válidos acima **e** truncamentos/variações dos fixtures do reader |
| Nº de execuções | registrar `runs` e `exec/s` de `-print_final_stats=1` (sem meta fixa; a meta é ausência de crash) |
| Crashes/reproduções | crash salvo em `fuzz/artifacts/<target>/`; reproduzir com `cargo fuzz run <target> <arquivo>`; minimizar; adicionar fixture de regressão à suíte normal |
| Cobertura | reportar `cov`/`ft` do `-print_final_stats=1` (e `cargo fuzz coverage` se disponível); anexar ao relatório |
| Versões | `rustc 1.101.0-nightly` (registrar revisão exata), `cargo-fuzz` com `--locked` (registrar versão), SO/runner |
| Critério de aprovação | **0 crashes** nos dois alvos no tempo combinado; se houver crash, o gate só passa após correção + fixture de regressão + re-execução limpa |
| Tratamento de falhas | não silenciar; registrar stack, input minimizado e commit de correção |
| Artefatos | logs `-print_final_stats=1`, lista de crashes, `git rev-parse` do commit testado |

Restrições: `fuzz/corpus` e `fuzz/artifacts` são dados locais (não commitados, exceto fixtures de regressão pequenas e intencionais). A campanha é longa e **não** faz parte desta etapa.

---

## 6. Riscos e recomendações

| # | Risco | Severidade | Recomendação |
|---|---|---|---|
| R1 | CI faz auto-commit/push em `feat/*` com `contents: write` | Alta | Remover o job `format`; usar `cargo fmt --all -- --check`; `contents: read` |
| R2 | `rust` depende do commit automático de formatação | Alta | `rust` sem `needs: format` |
| R3 | Ausência de `LICENSE` + 6 licenças não confirmadas | Alta | Decidir licença; gerar `LICENSE`/`NOTICE`; `cargo-license`/`cargo-deny` |
| R4 | Nenhuma campanha de fuzzing executada/registrada | Alta | Executar a campanha (§5) e arquivar evidência |
| R5 | `cargo-fuzz` ausente no host | Média | Instalar `cargo-fuzz --locked` na Etapa C (com autorização) |
| R6 | Corpus ausente (semente fraca) | Média | Semear a partir dos containers válidos locais |
| R7 | `libfuzzer-sys` sem licença verificada | Baixa | Confirmar junto às demais |
| R8 | Toolchain nightly instalada por efeito colateral do probe | Baixa | Registrar; não instalar mais nada sem autorização |

---

## 7. Decisões que dependem do responsável pelo projeto

1. **CI:** aprovar a remoção do auto-commit e a permissão somente-leitura? (recomendado: sim)
2. **Licença:** manter **MIT** ou adotar dual **`MIT OR Apache-2.0`**? (define os arquivos legais)
3. **Fuzzing:** aprovar duração/escopo da campanha (§5) e a instalação de `cargo-fuzz`.
4. **Ferramentas de licença:** instalar `cargo-license`/`cargo-deny` para fechar as 6 pendências?
5. **Corpus:** incluir sementes derivadas dos fixtures sintéticos versionados ou manter apenas local?

---

## 8. Próxima sequência de implementação (Etapa C — não iniciada)

1. **C2a — CI:** remover o job `format`; ajustar `rust`; `permissions: contents: read`; manter `fmt --check`.
2. **C2b — Licença:** decidir MIT/dual; criar `LICENSE` (+ `NOTICE`/`THIRD-PARTY-NOTICES`); rodar `cargo-license`/`cargo-deny` e registrar o inventário.
3. **C2c — Fuzzing:** instalar `cargo-fuzz`; semear corpus; executar a campanha (§5); arquivar logs e crashes; adicionar fixtures de regressão se necessário.
4. **C2d — Validação:** reexecutar os gates completos (`fmt --check`, `clippy -D warnings`, `test --workspace --all-targets`, `test --doc`, `pytest`, `unittest`, `compileall`, `vq_benchmark --release`) no estado exato do commit de release.
5. **C2e — Empacotamento:** definição do pacote/instalação verificável.

Nada disso é executado nesta etapa.

---

*Relatório da Etapa C1. Único artefato versionado desta etapa; nenhuma alteração de produto.*
