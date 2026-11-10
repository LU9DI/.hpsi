# HPSI — Relatório de validação da Fase 4.4

> **Nota de proveniência:** registro reconstruído a partir das evidências preservadas no histórico da sessão; não é uma cópia byte a byte do relatório local original. Os números abaixo são os resultados reportados para a execução da Fase 4.4.

## Escopo

Validação e2e de captura, manifesto, importação, container, auditoria Rust/Python e reconstrução, mantendo o formato v0.1 sem alterações.

## Gates reportados

| Gate | Resultado reportado |
|---|---:|
| `cargo fmt --all -- --check` | exit 0 |
| `cargo clippy --workspace --all-targets -- -D warnings` | exit 0 após correção de 10 lints no E2E |
| `cargo test --workspace --all-targets --no-fail-fast` | 257 passed, 0 failed, 0 ignored |
| Doc tests | 0 |
| pytest | 160 passed |
| unittest discover | 64 OK |
| compileall | exit 0 |

## Evidências E2E reportadas

- Cadeia completa executada com fixture GPT-2 sintética: captura → manifest → import → container → auditor.
- Auditorias Rust e Python: `findings: 0`, zero discrepâncias e violações de layout vazias.
- Treze métricas agregadas coincidiram entre os dois auditores.
- RAW foi reconstruído byte a byte; CRC32C foi verificado no container pré-contrato.
- A fixture era sintética, com inicialização aleatória, aproximadamente 3.984 parâmetros, dimensão de embedding 16, vocabulário 10, uma camada e embeddings amarrados. Não comprova qualidade de modelo treinado.

## Bloqueio arquitetural

O capturador inicial não representava adequadamente a identidade compartilhada entre `transformer.wte.weight` (Embedding) e `lm_head.weight` (Linear). O bundle/manifesto derivava apenas um papel de projeção para a chave capturada, sem registrar simultaneamente `embedding_table` e `index_axis`.

## Estado Git reportado

- HEAD: `68843a9`.
- Índice vazio; sem staging, commit, push ou merge.
- `crates/hpsi-format/` permaneceu intocado.
- Relatório original local: `.tmp-phase4/RELATORIO-FASE-4.4.md`.

## Limitações

Os resultados se referem à fixture sintética e à cadeia de contrato/auditoria. Não representam homologação de produção nem avaliação de qualidade de um modelo treinado.
