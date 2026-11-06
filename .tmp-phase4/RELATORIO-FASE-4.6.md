# HPSI — Relatório arquitetural da Fase 4.6

> **Nota de proveniência:** registro reconstruído a partir do resumo estruturado do relatório local de 350 linhas e das conclusões preservadas no histórico. Não é uma cópia byte a byte do relatório original.

## Resultado executivo

**Classificação: EXIGE EXTENSÃO COMPATÍVEL DE METADATA.**

A Fase 4.6 foi somente análise. Nenhuma alternativa foi implementada e nenhum arquivo de produto foi modificado.

## Questão arquitetural

A Fase 4.5 ampliou o produtor e o manifesto para representar um tensor lógico com mais de um papel arquitetural (por exemplo, embedding e projeção linear quando os pesos são tied). O container v0.1, isoladamente, não preservava toda essa semântica normativa para consumidores que não recebem o manifesto externo.

## Escopo de análise registrado

O relatório original documentou:
- Arquitetura do container v0.1: cabeçalho, metadata, diretório de tensores/blocos, integridade, writer, reader, auditores Rust/Python, CLI, executor, manifesto e bundle v2.
- Recuperabilidade sem manifesto: quais informações um consumidor consegue derivar apenas do container.
- Localização da limitação do papel único e ausência do conceito de papéis múltiplos no container.
- Comparação das alternativas A/B/C contra dez requisitos obrigatórios.
- Compatibilidade física, de schema, semântica e operacional.
- Riscos de manifesto externo, associação segura container/manifesto e comportamento de leitores antigos.
- Investigação dos logs `cargo-test-phase45.log` e `cargo-clippy-phase45.log`: arquivos não estavam presentes no repositório no momento da inspeção; nada foi apagado ou modificado.

## Decisão

A recomendação final foi uma extensão compatível de metadata, em vez de alteração física dos descritores binários ou dependência obrigatória de manifesto externo. A extensão deveria permitir ao consumidor recuperar múltiplos papéis por tensor sem invalidar containers v0.1 existentes.

## Estado reportado

- Relatório local: `.tmp-phase4/RELATORIO-FASE-4.6.md` (350 linhas).
- HEAD: `68843a9`; branch `feat/hpsi-core-container`.
- `git diff --stat` vazio para arquivos de produto; apenas o diretório de relatório não rastreado foi criado.
- Nenhuma alternativa implementada; aguardando autorização.
