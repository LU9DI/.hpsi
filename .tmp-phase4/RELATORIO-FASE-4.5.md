# HPSI — Relatório da Fase 4.5: identidade e papéis múltiplos

> **Nota de proveniência:** registro consolidado/reconstruído a partir dos relatórios e resultados OpenCode/Qoder preservados no histórico da sessão. Não é uma cópia byte a byte do relatório local original. Os resultados de cada execução são identificados separadamente.

## Diagnóstico

A auditoria identificou quatro lacunas no produtor:
1. `nn.Embedding` não era classificado como papel arquitetural.
2. A resolução de nomes por identidade do parâmetro devolvia um único nome e descartava aliases.
3. `module_types[key]` armazenava apenas uma classe por chave.
4. Bundle e gerador de manifesto não representavam múltiplos papéis para a mesma identidade lógica.

## Implementação registrada (trabalho Qoder / sessões anteriores)

- Agrupamento baseado exclusivamente na identidade real do parâmetro, não na igualdade dos valores.
- Preservação de `canonical_name`, aliases, `tensor_roles`, `index_axis`, orientação, shape, dtype, evidências, status e conflitos.
- Bundle v2 aditivo; leitores aceitam versões 1 e 2 e recusam versão 3.
- Manifesto com `aliases` e `role_evidence` opcionais, validados independentemente em Rust e Python.
- Papéis registrados: `linear_projection`, `conv1d_projection`, `embedding_table`, `other` e `unknown`.
- `index_axis=0` para tabela HF no formato [num_embeddings, embedding_dim].
- O executor recusa papéis de embedding/other no caminho de projeção.

## Evidências OpenCode — reexecução independente reportada

| Etapa | GPT-2 tied | GPT-2 untied |
|---|---:|---:|
| Captura | 5 layers, 6 registros | 5 layers, 7 registros |
| Bundle | válido, bundle v2, 2 aliases | válido, bundle v2 |
| Import | 21 tensors, 1 indexed table, 6 papéis confirmados | 22 tensors, 2 indexed tables, 7 papéis confirmados |
| CRC32C | 22 OK | 23 OK |
| Rust/Python | 0 findings, 0 discrepâncias | 0 findings, 0 discrepâncias |
| RAW | byte-identical | byte-identical |

Identidade tied reportada: `m.transformer.wte.weight is m.lm_head.weight == True`, com dois aliases e dois papéis. Identidade untied: IDs distintos, `torch.equal == True`, sem compartilhamento artificial.

Gates OpenCode reportados: fmt exit 0; cargo test 274 passed / 0 failed / 0 ignored; clippy `-D warnings` sem avisos; pytest 205 passed; unittest 109 OK; compileall exit 0.

## Evidências Qoder — implementação

O histórico da execução de implementação reportou 278 testes Rust, 205 pytest e 109 unittest aprovados; bundle v2, aliases e manifesto com papéis múltiplos. O container v0.1 ainda não armazenava normativamente os dois papéis do tensor tied: o papel codificado permanecia no slot único, enquanto a semântica adicional era mantida no manifesto/auditoria.

## Limitação arquitetural remanescente

A representação de múltiplos papéis ainda não era recuperável integralmente apenas do container v0.1. A fase seguinte deveria avaliar uma extensão compatível de metadata.

## Estado

- HEAD reportado: `68843a9`; índice vazio.
- Sem commit, push, merge ou deploy.
- Formato `crates/hpsi-format/` não foi alterado nesta fase.
- Fixtures e logs preservados.
