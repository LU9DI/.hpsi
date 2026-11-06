# HPSI — Relatório da Fase 4.7: extensão tensor_roles

> **Nota de proveniência:** registro reconstruído a partir dos dois resumos finais fornecidos na conversa e do resumo do relatório local de 282 linhas. Não é uma cópia byte a byte do relatório original.

## Resultado

Implementada extensão compatível de metadata para representar múltiplos papéis arquiteturais por tensor dentro da chave `user` do JSON de metadata do container HPSI v0.1, seguindo a recomendação da Fase 4.6.

## Arquivos reportados

Rastreados modificados:
- `crates/hpsi-format/src/metadata.rs`: `TensorRoleState`, `TensorRole` e `Metadata::tensor_roles()`, com validação.
- `crates/hpsi-format/src/lib.rs`: export dos novos tipos.
- `apps/hpsi-cli/src/main.rs`: integração e exibição dos papéis no comando `inspect`.

Novos:
- `tools/validate_tensor_roles.py`: validador Python espelhando a lógica Rust.
- `tests/test_tensor_roles.py`: 22 testes.
- `.tmp-phase4/RELATORIO-FASE-4.7.md`: relatório local de 282 linhas.

## Caso obrigatório reportado

`transformer.wte.weight`:
- `embedding_table` — confirmed, `index_axis=0`;
- `linear_projection` — confirmed.

`lm_head.weight`:
- `linear_projection` — confirmed.

Os nomes/aliases representam a mesma identidade lógica no caso tied; a identidade não é inferida pela igualdade dos valores.

## Compatibilidade declarada

- Física: nenhum byte do layout físico alterado (cabeçalho 128 B, TensorDescriptor 128 B, BlockDescriptor 64 B).
- Schema: whitelist de nível superior mantida; extensão localizada em `user`.
- Semântica: leitores v0.1 antigos devem ignorar conteúdo desconhecido em `user`; leitores novos recuperam os papéis.
- Operacional: CLI `inspect` exibe os papéis sem remover a saída existente; `verify_all()` cobre metadata por CRC32C.

## Testes e bloqueios

- 22 testes Python novos reportados como aprovados.
- Testes existentes de calibração: falhas descritas como preexistentes no resumo fornecido.
- Gates Rust (`cargo fmt`, `cargo clippy`, `cargo test`) não executados porque Cargo não estava instalado no ambiente da execução.
- A revisão Rust foi estática; não equivale a compilação ou execução.

## Estado Git reportado

- Branch `feat/hpsi-core-container`, HEAD `68843a9`.
- Sem commit ou push na execução original.
- Nenhum fixture, log ou relatório anterior alterado.
- A Fase 4.7 não declarou homologação de produção.
