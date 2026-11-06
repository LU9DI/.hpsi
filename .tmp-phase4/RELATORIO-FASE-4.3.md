# HPSI — Relatório de auditoria da Fase 4.3

> **Nota de proveniência:** este arquivo é um registro de arquivo reconstruído a partir das evidências e conclusões preservadas no histórico da sessão. Não é uma cópia byte a byte do relatório local original. Nenhuma evidência ausente foi inventada.

## Escopo

Análise arquitetural, somente leitura, da semântica de orientação, papéis de tensor e organização dos dados no container HPSI v0.1. Nenhuma alteração de produto ou formato foi realizada nesta fase.

## Conclusões registradas

- Não existe campo `arrangement` no schema; o termo havia sido usado informalmente para se referir a `weight_orientation`.
- `HpsiVqLayout` representa dimensões por rows/cols; o decoder não usa orientação arquitetural como informação independente.
- O writer precisa do eixo/orientação para codificar os dados.
- `weight_orientation` é redundante com `source_shape` e `logical_shape` em vários casos e não permite inferência confiável em matrizes quadradas.
- A recomendação foi manter `weight_orientation` para codificação, mas não usá-lo como evidência arquitetural de embeddings.
- `tensor_role` e `index_axis` devem expressar a semântica de tabelas de embedding.
- O manifesto externo já carregava parte da semântica, mas a cadeia de importação E2E ainda precisava de cobertura.
- Foi identificada divergência: Python aceitava metadata sem token em uma situação em que Rust rejeitava.
- O oráculo por módulo tinha uma questão de nomenclatura com pesos tied: o container podia nomear `transformer.wte.weight`, enquanto o oráculo podia emitir `lm_head.weight`.

## Decisão recomendada na fase

Adotar uma evolução compatível sem alteração física do container naquele momento: separar papel arquitetural de orientação; tratar `tensor_role` e `index_axis` como semântica; preservar identidade/aliases de pesos tied; corrigir a divergência Rust/Python e ampliar testes E2E.

## Estado

- Fase de análise; nenhuma alteração de produto.
- Nenhuma mudança de schema/formato.
- Sem commit, push, merge ou deploy decorrente da Fase 4.3.
- Esta cópia de arquivo não substitui o relatório original local.
