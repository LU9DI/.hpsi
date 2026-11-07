# HPSI — Etapa C2c: campanha de fuzzing para release v0.1

**Data:** 2026-11-06/07
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**Commit-base:** `2393ef5c1efc6e1faa06fafe70279ddf53cf3eb9`
**Escopo:** executar campanhas reproduzíveis de 3600 s nos alvos `fuzz_header` e `fuzz_model_reader`. Nenhuma alteração de produto/código. Único artefato versionado: este relatório.

---

## 1. Estado Git inicial e final

| Item | Início | Fim |
|---|---|---|
| Branch | `feat/hpsi-core-container` | `feat/hpsi-core-container` |
| HEAD | `2393ef5c1efc6e1faa06fafe70279ddf53cf3eb9` | (novo commit — §8) |
| Índice | vazio | vazio após o commit |
| Trabalho pré-existente | 15 ` M` + 29 `??` | 15 ` M` + 29 `??` preservados (novos itens de fuzz são scratch — §9) |
| Hunk `pytest` em `ci.yml` | preservado | preservado (não tocado) |
| `crates/hpsi-format/` | sem diff | sem diff |

Nenhum `reset`, `clean`, `checkout`, `restore`, rebase, push, merge, tag ou deploy.

## 2. Ferramentas e versões

| Ferramenta | Versão / caminho |
|---|---|
| Toolchain ativa do repo (`rust-toolchain.toml`) | `1.85.0-x86_64-pc-windows-msvc` |
| Toolchain usada no fuzzing | `nightly-x86_64-pc-windows-msvc` → `rustc 1.101.0-nightly (249c1e4e3 2026-11-05)`; `cargo 1.101.0-nightly` |
| `cargo-fuzz` | **0.13.2** (instalado com `cargo +nightly install cargo-fuzz --locked`; a instalação com o stable 1.85 falhou porque `cargo-platform@0.3.3` exige rustc 1.91) |
| Compilador C++ (libfuzzer-sys) | MSVC `cl.exe` 14.44.35207 (VS 2022 BuildTools) |
| Runtime ASan | `clang_rt.asan_dynamic-x86_64.dll` (mesmo BuildTools) |
| SO | Windows 10 (10.0.19045), shell MINGW64/MSYS |

**Observações de execução:** o binário de fuzz exige `clang_rt.asan_dynamic-x86_64.dll` e a CRT MSVC fora do PATH do shell; foi necessário adicionar duas pastas ao `PATH` (bin `Hostx64\x64` e `Microsoft.VC143.CRT`) para o alvo rodar (sem isso: `STATUS_DLL_NOT_FOUND`, `0xc0000135`).

## 3. Corpus e procedência das sementes

Criado em `fuzz/corpus/<alvo>/` (gitignored), por **cópia** dos originais (nenhum arquivo original movido/alterado; hashes conferidos).

| Alvo | Sementes iniciais | Procedência |
|---|---|---|
| `fuzz_header` | `seed_valid_header.bin` (128 B), `seed_truncated_64.bin`, `seed_magic_only.bin`, `seed_zeros.bin` | primeiros 128/64 B de `gpt2-tied.hpsi`; literais `HPSI` e zeros |
| `fuzz_model_reader` | `seed_tied.hpsi`, `seed_untied.hpsi`, `seed_calibrated.hpsi`, `seed_uncalibrated.hpsi`, `seed_truncated_512.hpsi`, `seed_truncated_200.hpsi` | `.tmp-phase4/phase45-e2e/*.hpsi`, `.tmp-gpt2-conv1d-test/*.hpsi`, truncamentos |

Corpus final após mutação (dados locais, não versionados): `fuzz_header` 42 arquivos; `fuzz_model_reader` 627 arquivos. Conjunto separado por alvo; nenhum resultado de campanha anterior foi misturado.

## 4. Comandos executados

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export PATH=".../VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64:$PATH"
export PATH=".../VC/Redist/MSVC/14.44.35112/x64/Microsoft.VC143.CRT:$PATH"

cd fuzz
cargo +nightly fuzz build   fuzz_header
cargo +nightly fuzz run     fuzz_header        -- -max_total_time=3600 -print_final_stats=1
cargo +nightly fuzz run     fuzz_model_reader  -- -max_total_time=3600 -print_final_stats=1
cargo +nightly fuzz coverage fuzz_header        # falhou: llvm-profdata ausente (§7)
```

Smokes de validação (antes das campanhas): 15–20 s por alvo, bem-sucedidos.

## 5. Estatísticas das campanhas

| Métrica | `fuzz_header` | `fuzz_model_reader` |
|---|---|---|
| Início (UTC) | 2026-11-07T00:50:43Z | 2026-11-07T01:51:52Z |
| Duração efetiva | **3601 s** (`Done … in 3601 second(s)`) | **3601 s** |
| Execuções (`runs`) | **99.874.075** | **1.450.869** |
| `exec/s` médio | 27.735 | 402 |
| Cobertura libFuzzer (`cov`) | **156** | **1.517** |
| Features (`ft`) | **178** | **4.284** |
| `new_units_added` | 0 | 495 |
| Pico RSS | 499 MB | 486 MB |
| Crashes | **0** | **0** |

Logs integrais: `.tmp-phase4/phaseC2c/campaign-fuzz_header.txt`, `.tmp-phase4/phaseC2c/campaign-fuzz_model_reader.txt`; smokes em `smoke-run-*.txt`; build em `smoke-build-header.txt`.

**Resultado:** ambas as campanhas **completaram os 3600 s solicitados**, sem crash. Isto é um gate de robustez (nenhuma entrada derrubou o parser/leitor) e **não** equivale a cobertura completa nem a correção comprovada.

## 6. Crashes, reprodução e regressões

- **Nenhum crash.** `fuzz/artifacts/fuzz_header/` e `fuzz/artifacts/fuzz_model_reader/` vazios (diretórios criados pelo cargo-fuzz, sem arquivos).
- Não houve artefato a reproduzir/minimizar; portanto **nenhum teste de regressão** foi criado nesta etapa.
- Nada foi apagado.

## 7. Cobertura

- **Cobertura LLVM: não disponível.** `cargo +nightly fuzz coverage fuzz_header` executou a instrumentação e o corpus, mas falhou ao mesclar (`llvm-profdata.exe` inexistente: componente `llvm-tools` não instalado). Não foi instalado componente adicional.
- **Cobertura libFuzzer medida (proxy):** `cov` 156 / `ft` 178 (header) e `cov` 1517 / `ft` 4284 (reader), estáveis nas campanhas.
- Distinção explícita: isto é *feature/counter coverage reportada pelo libFuzzer*, **não** um relatório `llvm-cov`. Ausência de crash **não** significa cobertura completa.

## 8. Livro do commit

- Caminho adicionado: `docs/audits/2026-11-06-hpsi-etapa-c2c-fuzzing.md` (exclusivamente).
- Mensagem: `test(fuzz): record HPSI release fuzzing campaign`.
- SHA completo: registrado na resposta final.
- Nenhum corpus mutado, log temporário, `fuzz/Cargo.lock`, `fuzz/coverage/` ou alteração preexistente foi incluído.

## 9. Alterações de ambiente e scratch gerado (não versionado)

| Item | Tipo | Destino |
|---|---|---|
| `cargo-fuzz 0.13.2` | ferramenta instalada | `~/.cargo/bin` |
| `fuzz/Cargo.lock` | scratch de build | **não commitado** (untracked) |
| `fuzz/target/` | build | ignorado (`/fuzz/target/`) |
| `fuzz/corpus/` | corpus mutado | ignorado (`/fuzz/corpus/`) |
| `fuzz/artifacts/` | crashes (vazio) | ignorado (`/fuzz/artifacts/`) |
| `fuzz/coverage/` | scratch da tentativa de cobertura | **não commitado** (untracked) |

O `fuzz/Cargo.lock` e `fuzz/coverage/` aumentam a contagem de `??` de 29 para 31; são scratch desta etapa e **não** entram no commit.

## 10. Limitações

1. **Cobertura LLVM indisponível** (sem `llvm-tools`); relatada como não medida.
2. **Sem sanitizers explícitos**: o build usou ASan (default do cargo-fuzz), mas a campanha é CPU e não substitui análise estática nem testes de limites dedicados.
3. **Corpus de semente pequeno** (6–8 arquivos por alvo) e **sem corpus versionado**; a campanha de 1 h dá confiança de robustez, não prova de exaustão.
4. `fuzz_header` saturou cedo (cov 156 estável; `new_units_added=0`), o que é esperado: o parser de header é pequeno e foi bem coberto pelos seeds.
5. Execução em um único host/OS (Windows); o CI roda o `fuzz-smoke` (compilação) no Ubuntu.
6. As versões de `libfuzzer-sys`/deps são resolvidas no build (criou-se `fuzz/Cargo.lock`, não versionado).

## 11. Confirmação de preservação

- 15 ` M` + 29 `??` da linha de base preservados; os itens novos de fuzz (`fuzz/Cargo.lock`, `fuzz/coverage/`) e o corpus são scratch e não afetam o produto.
- Hunk `pytest` de `.github/workflows/ci.yml` permanece.
- `crates/hpsi-format/` sem diff.
- Nenhum arquivo descartado, movido ou sobrescrito; nenhum crash apagado.

## 12. Ausência de push, merge, tag e deploy

Confirmado: nenhum `push`, `merge`, `tag` ou `deploy`. O commit é local.

---

*Relatório da Etapa C2c. Campanhas completas, 0 crashes, cobertura LLVM não disponível.*
