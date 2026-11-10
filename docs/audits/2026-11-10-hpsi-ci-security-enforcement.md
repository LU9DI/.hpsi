# HPSI — enforcement de segurança e licenças no CI

**Data:** 2026-11-10
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**HEAD de referência:** `9180c73ed0965576bfe17f435325792b21492101`
**Natureza:** alteração restrita a workflow e criação deste relatório. Na ocasião da redação, nenhum commit, push, merge, tag, release, deploy ou publicação havia ocorrido; a alteração e este relatório foram posteriormente commitados (`234ba1c` e `5691936`). Nenhum manifest, dependência ou política `deny.toml` alterado.

---

## 1. Estado inicial

| Item | Valor |
|---|---|
| Branch | `feat/hpsi-core-container` |
| HEAD | `9180c73ed0965576bfe17f435325792b21492101` |
| Índice | vazio |
| `git diff HEAD` | vazio |
| Rastreados modificados | 0 |
| Scratch | **8 itens preservados** |
| `.github/workflows/ci.yml` (antes) | SHA-256 `5ebaa0a1d81aa4f0fece49e3a2c41f8351f4284e6a93803dd13c3a8494f66750` |
| `.github/workflows/fuzz.yml` (antes e depois) | SHA-256 `7355a7baa56533df96e68b833468749a5890b3b581c146e9b716738f43a2abed` |
| `deny.toml` | SHA-256 `1abf78bab7b2b5196064d55eb6ffd54f1f63e84974afff630b07ac6afb150474` — **inalterado** |
| `fuzz/deny.toml` | SHA-256 `9f54c9cc67f90fe00325acf261d9a3cffb9e0b31c6dab44beb862b5a8e978b7d` — **inalterado** |

Os oito itens de scratch preservados são os mesmos da auditoria anterior (`.tmp-gpt2-conv1d-test/`, `.tmp-gpt2-final/`, `.tmp-phase4/`, `fuzz/coverage/`, `prepare_gpt2_test.py` e três `.bak`). O relatório `docs/audits/2026-11-10-hpsi-consolidated-history-review.md`, então não rastreado, também foi mantido; ele foi posteriormente versionado no repositório.

---

## 2. Workflows inspecionados (integralmente, antes da alteração)

### 2.1 `.github/workflows/ci.yml`

- Gatilhos: `push` em `main` e `feat/**`, `pull_request` e `workflow_dispatch`.
- `concurrency: hpsi-ci-${{ github.ref }}`, `cancel-in-progress: false`.
- `permissions: contents: read` no nível do workflow.
- Jobs existentes:
  1. `python-capture-tools` — `checkout@v4`, `setup-python@v5` (3.12), `py_compile`, `unittest discover`, `pytest`.
  2. `rust` — matriz `ubuntu-latest`/`windows-latest`/`macos-latest`, `checkout@v4`, `dtolnay/rust-toolchain@master` (1.85.0 + rustfmt/clippy), `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace --all-targets`.
  3. `vq-benchmark` — `checkout@v4`, toolchain 1.85.0, `cargo run -p hpsi-ops --example vq_benchmark --release`.
  4. `fuzz-smoke` — `checkout@v4`, `dtolnay/rust-toolchain@nightly` (rustfmt), `cargo +nightly fmt -- --check` e `cargo +nightly check --bins` em `fuzz/`.

### 2.2 `.github/workflows/fuzz.yml`

- Gatilho apenas `workflow_dispatch` com `duration_seconds` (padrão 600).
- `permissions: contents: read`.
- Job `fuzz`: matriz `fuzz_header`/`fuzz_model_reader`, nightly, `cargo install cargo-fuzz --locked`, execução do alvo e upload de artefatos de crash.

**Nenhum job existente foi alterado.** O `fuzz.yml` permanece intocado.

---

## 3. Alteração aplicada

Um job independente `security` foi **acrescentado ao final** de `ci.yml` (linhas 101–127), sem `needs` e sem alterar os demais jobs, gatilhos, concorrência ou permissões.

```yaml
  # Tools and action commits are pinned; see
  # docs/audits/2026-11-10-hpsi-ci-security-enforcement.md for the audited SHAs.
  security:
    name: Security and license gates
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2

      - name: Install project Rust toolchain
        run: rustup toolchain install 1.85.0 --profile minimal

      - name: Install audit and license tools
        uses: taiki-e/install-action@cfdb7ed4ccc7c7dce37759eae89dc5674b469b39 # v2.90.3
        with:
          tool: cargo-audit@0.22.2,cargo-deny@0.20.2

      - name: Audit workspace dependencies
        run: cargo audit

      - name: Enforce root dependency policy
        run: cargo deny --config deny.toml check

      - name: Enforce fuzz workspace dependency policy
        working-directory: fuzz
        run: cargo deny --config deny.toml check
```

Decisões e garantias:

- **Política correta em cada workspace (restrição 5):** os caminhos são explícitos. Na raiz, `--config deny.toml` aponta para a política raiz; em `working-directory: fuzz`, aponta para `fuzz/deny.toml`. Isso remove qualquer dependência da descoberta automática de configuração.
- **NCSA não é liberado globalmente (restrição 6):** a raiz continua com `exceptions = []`; a exceção `libfuzzer-sys =0.4.13` permanece exclusivamente em `fuzz/deny.toml`, usada apenas pelo passo do workspace `fuzz/`.
- **Permissões (restrição 7):** nenhuma permissão foi adicionada. O job herda `contents: read` do topo do workflow; não usa segredos nem escrita.
- **Jobs preservados (restrição 2):** apenas 27 linhas foram inseridas; nada foi removido ou reordenado.
- **Sem alteração de manifests/dependências (restrição 9):** a árvore de dependências e os lockfiles não foram tocados.

### 3.1 Versões e SHA verificados

| Componente | Versão/tag | Referência usada | Verificação |
|---|---|---|---|
| `actions/checkout` | `v4.2.2` | `11bd71901bbe5b1630ceea73d27597364c9af683` | `git ls-remote` + GitHub API `object.type = "commit"` |
| `taiki-e/install-action` | `v2.90.3` | `cfdb7ed4ccc7c7dce37759eae89dc5674b469b39` | `git ls-remote` + GitHub API `object.type = "commit"` |
| `cargo-audit` | `0.22.2` | binário pré-compilado gerenciado pelo `install-action` | release `rustsec/rustsec` `cargo-audit/v0.22.2`; ativo `cargo-audit-x86_64-unknown-linux-musl-v0.22.2.tgz` responde HTTP 200 |
| `cargo-deny` | `0.20.2` | binário pré-compilado gerenciado pelo `install-action` | release `EmbarkStudios/cargo-deny` `0.20.2`; ativo `cargo-deny-0.20.2-x86_64-unknown-linux-musl.tar.gz` responde HTTP 200 |
| Rust | `1.85.0` | `rustup toolchain install 1.85.0 --profile minimal` | igual ao `rust-toolchain.toml` do repositório |

O `taiki-e/install-action` foi pinado por commit (não por tag). Ele verifica SHA-256 dos binários baixados por padrão e dispensa compilar `cargo-audit`/`cargo-deny` no runner (a compilação local dessas ferramentas levou ~50 min cada). As versões foram **fixadas explicitamente** (`@0.22.2`, `@0.20.2`), sem resolução implícita de "latest".

Nenhuma Action nova foi referenciada por tag ou branch mutável. As referências mutáveis remanescentes são exclusivamente as **pré-existentes** nos jobs antigos, preservadas conforme a restrição 2 e registradas como risco residual na Seção 6.

---

## 4. Validação

### 4.1 Sintaxe YAML

Validado com PyYAML 6.0.3:

- `.github/workflows/ci.yml`: parse OK; jobs `['python-capture-tools', 'rust', 'vq-benchmark', 'fuzz-smoke', 'security']`.
- `.github/workflows/fuzz.yml`: parse OK; job `['fuzz']`.
- Job `security`: `runs-on: ubuntu-latest`; 6 passos na ordem esperada; `working-directory: fuzz` apenas no último passo; permissões de topo `contents: read`.

### 4.2 Comandos equivalentes executados localmente

Executados a partir do HEAD `9180c73`, Windows, com as mesmas versões fixadas (cargo-audit 0.22.2, cargo-deny 0.20.2):

| Comando (igual ao workflow) | Exit | Resultado |
|---|---:|---|
| `cargo audit` | **0** | 1.446 advisories carregados; 14 dependências; 0 vulnerabilidades |
| `cargo deny --config deny.toml check` (raiz) | **0** | `advisories ok, bans ok, licenses ok, sources ok` |
| `cargo deny --config deny.toml check` (`fuzz/`) | **0** | `advisories ok, bans ok, licenses ok, sources ok` (exceção NCSA ativa) |

Observação: ambos os `cargo deny` continuam emitindo o aviso `unmatched license allowance` para `Unicode-3.0` (licença da allow-list ausente do grafo podado por features); é inofensivo e não altera o exit code 0.

### 4.3 Prova de que falhas reprovam o CI

Os cenários foram executados **fora do repositório**, em diretório temporário, sem tocar em manifests ou políticas do projeto.

| Cenário | Comando | Exit | Evidência |
|---|---|---:|---|
| Advisory conhecido | `cargo audit --file Cargo.lock` em projeto temporário com `smallvec =1.6.0` | **1** | `RUSTSEC-2021-0003`, severidade 9.8; `error: 1 vulnerability found!` |
| Licença não permitida | `cargo deny --config <tmp>/deny-fail.toml check licenses` na raiz (allow-list apenas `BSD-3-Clause`) | **4** | múltiplos `error[rejected]: failed to satisfy license requirements` |

Conclusão: qualquer advisory ou violação de licença faz o respectivo passo retornar código não zero e, portanto, **reprova o job `security`** no GitHub Actions (o runner trata exit ≠ 0 como falha de step e de job).

### 4.4 Runner limpo

Não foi possível executar um runner limpo neste ambiente: o daemon do Docker Desktop não está ativo (`npipe:////./pipe/dockerDesktopLinuxEngine` indisponível) e não há runner do GitHub Actions local. Essa verificação **não foi executada** e não é apresentada como feita.

Verificações estáticas realizadas em substituição:

1. Commits das duas Actions existem, são commits (não tags anotadas) e correspondem às tags documentadas (API do GitHub).
2. Os ativos pré-compilados das versões fixadas de `cargo-audit` e `cargo-deny` existem nos releases oficiais (HTTP 200).
3. O `install-action` no commit pinado lista `cargo-audit` e `cargo-deny` como ferramentas suportadas com instalação via GitHub Releases e verificação de SHA-256 habilitada por padrão.
4. Os três comandos executam com sucesso a partir da raiz e de `fuzz/` com as mesmas configurações de caminho do workflow.

---

## 5. Diff

```diff
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 3ae5033..63a3db3 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -98,3 +98,30 @@ jobs:
       - name: Compile libFuzzer targets
         working-directory: fuzz
         run: cargo +nightly check --bins
+
+  # Tools and action commits are pinned; see
+  # docs/audits/2026-11-10-hpsi-ci-security-enforcement.md for the audited SHAs.
+  security:
+    name: Security and license gates
+    runs-on: ubuntu-latest
+    steps:
+      - name: Checkout
+        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
+
+      - name: Install project Rust toolchain
+        run: rustup toolchain install 1.85.0 --profile minimal
+
+      - name: Install audit and license tools
+        uses: taiki-e/install-action@cfdb7ed4ccc7c7dce37759eae89dc5674b469b39 # v2.90.3
+        with:
+          tool: cargo-audit@0.22.2,cargo-deny@0.20.2
+
+      - name: Audit workspace dependencies
+        run: cargo audit
+
+      - name: Enforce root dependency policy
+        run: cargo deny --config deny.toml check
+
+      - name: Enforce fuzz workspace dependency policy
+        working-directory: fuzz
+        run: cargo deny --config deny.toml check
```

Estatística: **1 arquivo alterado, 27 inserções, 0 remoções**.

Hashes após a alteração:

| Arquivo | SHA-256 |
|---|---|
| `.github/workflows/ci.yml` | `cf00602027f19d2911e22786273c1acb101e0d10cbfd5dd7bf884963a095ac35` |
| `.github/workflows/fuzz.yml` | `7355a7baa56533df96e68b833468749a5890b3b581c146e9b716738f43a2abed` (inalterado) |
| `deny.toml` | `1abf78bab7b2b5196064d55eb6ffd54f1f63e84974afff630b07ac6afb150474` (inalterado) |
| `fuzz/deny.toml` | `9f54c9cc67f90fe00325acf261d9a3cffb9e0b31c6dab44beb862b5a8e978b7d` (inalterado) |

---

## 6. Riscos residuais e limitações

1. **Dependência de rede** — `cargo audit` busca a advisory database e `cargo deny` atualiza o índice do crates.io a cada execução. Indisponibilidade de rede reprova o job (fail-closed, comportamento desejado), mas pode gerar falha transitória.
2. **Duração** — a execução local dos três comandos levou alguns minutos, quase todo o tempo em download/varredura do índice e da advisory DB. Não há cache; se o tempo incomodar, um cache de `~/.cargo/advisory-db` pode ser considerado em mudança futura.
3. **Referências mutáveis pré-existentes** — `actions/checkout@v4`, `actions/setup-python@v5`, `dtolnay/rust-toolchain@master`, `dtolnay/rust-toolchain@nightly` e `actions/upload-artifact@v4` continuam por tag/branch nos jobs antigos. Foram preservados por exigência de não alterar os jobs existentes; recomenda-se pinagem por SHA em mudança separada (atenção: pinar `dtolnay/rust-toolchain` exige passar `toolchain:` explicitamente, pois a inferência vem do nome da ref).
4. **Dependência de terceiro** — `taiki-e/install-action` é um terceiro, fixado por commit e com verificação de checksum. Alternativa seria baixar os binários oficiais manualmente; o trade-off foi simplicidade e rastreabilidade das versões.
5. **Cobertura de plataforma** — o job `security` roda apenas em `ubuntu-latest`. `cargo audit` e `cargo deny` operam sobre lockfiles/manifests e não dependem de plataforma; a matriz multiplataforma existente continua cobrindo build/testes.
6. **Aviso `unmatched license allowance`** — permanece para `Unicode-3.0` nos dois `cargo deny`; não é violação e não falha o job. A allow-list não foi alterada.
7. **Validação em runner limpo não executada** — ver 4.4. A primeira execução real no GitHub Actions é a validação definitiva do job.
8. **NCSA** — continua restrito a `libfuzzer-sys =0.4.13` no workspace `fuzz/`; a raiz permanece sem exceções.

---

## 7. Estado Git final

| Item | Valor |
|---|---|
| Branch | `feat/hpsi-core-container` |
| HEAD | `9180c73ed0965576bfe17f435325792b21492101` (inalterado) |
| Índice | vazio (nada adicionado) |
| Modificados | `.github/workflows/ci.yml` (único arquivo rastreado alterado) |
| Novos arquivos | `docs/audits/2026-11-10-hpsi-ci-security-enforcement.md` (então não rastreado; posteriormente versionado e commitado) |
| Scratch | 8 itens preservados |
| `deny.toml` / `fuzz/deny.toml` | inalterados |
| `fuzz.yml` | inalterado |

No momento da redação, nada havia sido commitado, enviado, mesclado, tagueado, publicado ou implantado, e nenhum `git add` havia sido executado.

---

## 8. Conclusão

O job `security` adiciona, em cada `push`/`pull_request`, as três verificações pedidas:

1. `cargo audit` no workspace raiz;
2. `cargo deny check` com a política raiz;
3. `cargo deny check` com `fuzz/deny.toml`.

Os três passam localmente no HEAD atual, e os testes de falha demonstram que um advisory ou uma licença não permitida produzem exit não zero e reprovam o CI. As ferramentas e as Actions novas estão fixadas em versões e commits verificados; nenhuma política foi enfraquecida; nenhuma permissão foi ampliada.

**Pronto para revisão na ocasião.** O diff e os resultados estão acima; nenhum commit havia sido feito à época. A alteração de workflow foi posteriormente commitada em `234ba1c` e este relatório em `5691936`.

---

*Alteração de CI e relatório. No momento da redação, sem commit, push, merge, tag, release, deploy ou publicação; a alteração e este relatório foram posteriormente commitados.*
