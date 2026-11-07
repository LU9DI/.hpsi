# HPSI — Etapa C2a: CI estritamente verificatório (sem auto-commit)

**Data:** 2026-11-06
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**Commit-base:** `f463a1dfe02717aa9fc4c88d85b0749a98fcc835`
**Escopo:** remover a escrita automática do Git Actions no branch e tornar o CI estritamente verificatório. Alterações restritas a `.github/workflows/ci.yml` e a este relatório.

---

## 1. Estado inicial e final do Git

| Item | Início (C2a) | Fim (C2a) |
|---|---|---|
| Branch | `feat/hpsi-core-container` | `feat/hpsi-core-container` |
| HEAD | `f463a1dfe02717aa9fc4c88d85b0749a98fcc835` | (novo commit — §7) |
| Índice | vazio (`e3b0c442…`) | vazio após o commit |
| sha256 do diff rastreado | `96a15cc9…fc0d` | `96a15cc9…fc0d` (inalterado) |
| Trabalho pré-existente | 15 ` M` + 29 `??` | 15 ` M` + 29 `??` (preservado) |
| `crates/hpsi-format/` | sem diff | sem diff |

Nenhum `reset`, `clean`, `checkout`, `restore`, rebase, push, merge, tag ou deploy.

## 2. Diff exato do workflow (o que foi commitado)

O diff **staged** (HEAD → índice) contém apenas a transformação da C2a:

```diff
@@ -6,14 +6,13 @@ on:
-# The formatter may push a commit. Serialize runs per ref so concurrent
-# formatter jobs cannot race while updating the same feature branch.
+# Serialize runs per ref so concurrent runs on the same branch do not overlap.
 concurrency:
   group: hpsi-ci-${{ github.ref }}
   cancel-in-progress: false
 
 permissions:
-  contents: write
+  contents: read
 
@@ -29,38 +28,8 @@ jobs:
       - name: Test capture utility helpers
         run: python -m unittest discover -s tests -p "test_*.py"
-  format:
-    if: github.event_name == 'push' && startsWith(github.ref, 'refs/heads/feat/')
-    runs-on: ubuntu-latest
-    steps:
-      - name: Checkout feature branch
-        uses: actions/checkout@v4
-        with:
-          ref: ${{ github.ref }}
-
-      - name: Install Rust toolchain
-        uses: dtolnay/rust-toolchain@master
-        with:
-          toolchain: 1.85.0
-          components: rustfmt
-
-      - name: Format Rust sources
-        run: cargo fmt --all
-
-      - name: Commit formatting changes
-        run: |
-          if ! git diff --quiet; then
-            git config user.name "github-actions[bot]"
-            git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
-            git add -u
-            git commit -m "style: format Rust sources"
-            git push
-          fi
-
   rust:
     name: Rust checks (${{ matrix.os }})
-    needs: format
-    if: always() && (needs.format.result == 'success' || needs.format.result == 'skipped')
     strategy:
```

### Tratamento das alterações pré-existentes no mesmo arquivo

`.github/workflows/ci.yml` já possuía, **antes** desta etapa, um hunk local pré-existente (+8 linhas: o passo `pytest` e seu comentário, no job `python-capture-tools`). Para respeitar a regra "não incorporar alterações preexistentes", o commit desta etapa foi montado **apenas com a transformação da C2a** (via `git update-index --cacheinfo` sobre o blob `<HEAD + edições C2a>`); o hunk do `pytest` **permanece não commitado** no working tree e não entra neste commit. Consequência: após o commit, `git diff` mostra, em `ci.yml`, **somente** o hunk do `pytest` ainda pendente — nenhuma alteração preexistente foi descartada ou incorporada.

## 3. Permissões e dependências antes/depois

| Item | Antes | Depois |
|---|---|---|
| `permissions` (workflow) | `contents: write` | **`contents: read`** |
| Job `format` | existia (`push`/`feat/**`), `cargo fmt --all` + `git add -u`/`commit`/`push` | **removido** |
| `rust` — dependência | `needs: format` + `if: always() && (needs.format.result …)` | **sem `needs`/`if`** |
| `rust` — verificação de formato | `cargo fmt --all -- --check` | **mantida** |
| Jobs ativos | python-capture-tools, format, rust, vq-benchmark, fuzz-smoke | **python-capture-tools, rust, vq-benchmark, fuzz-smoke** |
| `concurrency` | `group hpsi-ci-${ref}`, `cancel-in-progress: false`, comentário sobre o formatter | mantido; **comentário atualizado** (sem referência ao formatter) |
| Gatilhos (`on:`) | push `main`/`feat/**`, pull_request, workflow_dispatch | **inalterados** |
| Condições dos jobs | — | `rust`/`vq-benchmark`/`fuzz-smoke`/`python` inalterados |

Nenhum outro job dependia de escrita no repositório; portanto `contents: read` é suficiente.

## 4. Validações executadas e resultados

| Validação | Comando | Exit | Resultado |
|---|---|---|---|
| YAML válido (working) | PyYAML `safe_load` | 0 | jobs = `['python-capture-tools','rust','vq-benchmark','fuzz-smoke']`, `permissions = {contents: read}` |
| YAML válido (versão commitada) | PyYAML `safe_load` (blob C2a) | 0 | idem |
| Formatação Rust (gate estrito) | `cargo fmt --all -- --check` | **0** | sem divergências |
| Job `format` removido | `grep '^  format:'` | — | ausente |
| `needs: format` removido | `grep 'needs: format'` | — | ausente |
| `fmt --check` mantido | `grep 'cargo fmt --all -- --check'` | — | presente (linha 59) |
| Sem escrita Git no workflow | `grep -E 'git (add|commit|push|config)|contents: write'` | — | nenhum resultado |
| Índice isolado | `git diff --cached --name-only` | — | exatamente 2 caminhos antes do commit |

Logs: `.tmp-phase4/phaseC2a/fmt-check.txt`, `.tmp-phase4/phaseC2a/yaml-validate.txt`.

## 5. Testes não executados e motivos

| Teste | Motivo |
|---|---|
| `cargo clippy --workspace --all-targets -- -D warnings` | A mudança é apenas YAML de CI; nenhum código Rust foi alterado. Já validado na Fase 4.8; será reexecutado no portão de release (C2d). |
| `cargo test --workspace --all-targets` | idem (mudança de CI, sem impacto em código). |
| `pytest` / `unittest discover` / `compileall` | idem; sem alteração de Python. |
| `vq_benchmark --release` | idem; sem alteração de código. |
| Execução real do workflow no GitHub Actions | Requer push/CI remoto; **push proibido** nesta etapa. Validação limitada a YAML + inspeção estática. |

Nada disso é declarado como executado.

## 6. Confirmação de preservação das alterações anteriores

- Os 15 arquivos modificados e 29 entradas não rastreadas da linha de base continuam presentes; o working tree segue com `M=15` e `??=29` (o `??` de `docs/audits/` desapareceu porque o commit versiona `docs/audits/2026-11-06-hpsi-etapa-c2a-ci.md`).
- O hunk `pytest` pré-existente de `ci.yml` **permanece** no working tree (não commitado).
- `crates/hpsi-format/` sem diff.
- Nenhum arquivo fora do escopo (workflow + relatório) foi alterado, movido ou apagado.

## 7. Commit

- Caminhos adicionados (exclusivamente): `.github/workflows/ci.yml`, `docs/audits/2026-11-06-hpsi-etapa-c2a-ci.md`.
- Método: `git add` do relatório + `git update-index --cacheinfo` do blob C2a para o workflow (evita incorporar o hunk `pytest` pré-existente).
- Mensagem: `ci: make HPSI workflow verification-only`.
- SHA: registrado na verificação final abaixo.

## 8. Ausência de push, merge, tag e deploy

Confirmado: nenhum `push`, `merge`, `tag` ou `deploy` foi executado. O novo commit é **local** e não foi enviado a `origin`.

---

*Relatório da Etapa C2a. Escopo restrito ao workflow e a este documento.*
