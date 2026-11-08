# HPSI — auditoria de conformidade arquitetural v0.1

**Data:** 2026-11-08
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**HEAD auditado:** `41c8248f0cbf0f56e9085df8eac95bf45595400e`
**Natureza:** auditoria documental e operacional somente-leitura. Nenhuma funcionalidade implementada; nenhum formato físico alterado; único artefato versionado: este relatório.

---

## 1. Resumo executivo

O **HPSI Toolkit v0.1** é **aderente ao escopo declarado** (contêiner, importação, verificação, auditoria e CLI), com **pendências documentadas** que pertencem a versões futuras ou a validações externas. O formato físico v0.1 está implementado e congelado exatamente como especificado (header 128 B, tensor 128 B, bloco 64 B, integridade 32 B, CRC32C, RAW). As operações VQ e a calibração de ativação existem como camada experimental em CPU, validadas apenas em dados sintéticos. **Não há runtime de inferência**, **não há decoders** para ZSTD/codebooks/low-rank/lookup/residual, e **não há `user.tensor_roles`** (papel único por entrada, por decisão de escopo). O que a arquitetura original previa como visão ampla (runtime, math directory, GPU, MoE, qualidade em modelo real) permanece **futuro**, não implementação atual.

**Classificação geral: `ADERENTE COM PENDÊNCIAS`.**

## 2. Fontes originais e escopo

| Fonte | Conteúdo usado |
|---|---|
| `specs/001-hpsi-core-container/spec.md` | FR-001…FR-016, NFR-001…NFR-007, §7 "Explicitly out of scope" |
| `specs/002-direct-vq-matvec/spec.md` | representação VQ, critérios de aceitação, non-goals |
| `docs/format-v0.1.md` | definição normativa byte-a-byte |
| `docs/orientation-policy.md`, `docs/transformer-activation-capture.md` | papéis/orientação/identidade/calibração |
| `README.md` | visão, milestones M0–M7, POC 4096×4096, restrições |
| `.tmp-phase4/RELATORIO-FASE-4.1…4.8.md`, C1, C2a–C2g | histórico de implementação/auditoria |
| `crates/`, `apps/`, `tools/`, `tests/` | implementação real |

**Distinção aplicada:** *requisito original* (specs/README) ≠ *decisão de escopo v0.1* (toolkit, RAW-only, papel único) ≠ *implementado* (código) ≠ *validado* (testes/campanhas) ≠ *limitação documentada* ≠ *evolução futura*.

## 3. Matriz de conformidade

### 3.1 Contêiner e formato físico

| # | Requisito (fonte) | Estado real | Arquivos/símbolos | Testes/evidência | Classificação |
|---|---|---|---|---|---|
| 1.1 | Header fixo de 128 B (FR-001) | Implementado | `header.rs` (`HEADER_SIZE=128`, `Header::parse`) | unit tests; fuzzing | **IMPLEMENTADO E VALIDADO** |
| 1.2 | Inteiros LE fixos (FR-002) | Implementado | `header.rs`, `tensor.rs`, `block.rs` | unit tests | **IMPLEMENTADO E VALIDADO** |
| 1.3 | Offsets/tamanhos de regiões declarados (FR-003) | Implementado | `header.rs`, `region.rs` | unit tests | **IMPLEMENTADO E VALIDADO** |
| 1.4 | Aritmética verificada e limites (FR-004, FR-010) | Implementado | `region.rs`, `limits.rs` | resource-limit tests | **IMPLEMENTADO E VALIDADO** |
| 1.5 | Metadata JSON limitada (FR-005) | Implementado | `metadata.rs` | unit tests | **IMPLEMENTADO E VALIDADO** |
| 1.6 | Tensor directory 128 B (FR-006) | Implementado | `tensor.rs` | reader/writer tests | **IMPLEMENTADO E VALIDADO** |
| 1.7 | Block directory 64 B (FR-007) | Implementado | `block.rs` | reader/writer tests | **IMPLEMENTADO E VALIDADO** |
| 1.8 | Blocos independentes (FR-008) | Implementado | `reader.rs::read_block` | random-access tests | **IMPLEMENTADO E VALIDADO** |
| 1.9 | Fonte random-access (FR-009) | Implementado | `source.rs` (`FileSource`, `MemorySource`) | source tests | **IMPLEMENTADO E VALIDADO** |
| 1.10 | Validação de ownership/cobertura/aux (FR-011) | Implementado | `reader.rs`, `region.rs` | reader tests | **IMPLEMENTADO E VALIDADO** |
| 1.11 | CRC32C (FR-012) | Implementado | `integrity.rs`, `reader.rs::verify_all` | writer/reader tests | **IMPLEMENTADO E VALIDADO** |
| 1.12 | Writer determinístico (FR-014) | Implementado | `writer.rs` | round-trip tests | **IMPLEMENTADO E VALIDADO** |
| 1.13 | Publicação atômica verificada (FR-015) | Implementado | `writer.rs::write_path` | writer tests | **IMPLEMENTADO E VALIDADO** |
| 1.14 | Taxonomia de erro tipada (FR-016) | Implementado | `error.rs` | unit tests | **IMPLEMENTADO E VALIDADO** |
| 1.15 | Corrupção/truncamento | Implementado | `reader.rs`, `writer.rs` | adversarial tests | **IMPLEMENTADO E VALIDADO** |
| 1.16 | Versionamento/flags (NFR-007) | Implementado | `header.rs` | unit tests | **IMPLEMENTADO E VALIDADO** |
| 1.17 | Assinatura/criptografia | Fora do escopo v0.1 | — | — | **RESERVADO PARA VERSÃO FUTURA** |
| 1.18 | ZSTD/codecs 2–6 | Sem decoder; só identificadores | `docs/format-v0.1.md §7` | — | **RESERVADO PARA VERSÃO FUTURA** |

### 3.2 Organização em blocos

| # | Requisito | Estado real | Arquivos/símbolos | Evidência | Classificação |
|---|---|---|---|---|---|
| 2.1 | Mapeamento tensor→blocos contíguos | Implementado | `reader.rs`, `writer.rs` | reader/writer tests | **IMPLEMENTADO E VALIDADO** |
| 2.2 | Cobertura lógica exata, sem buracos | Implementado | `reader.rs` | validation tests | **IMPLEMENTADO E VALIDADO** |
| 2.3 | Payloads não sobrepostos, dentro da região | Implementado | `region.rs`, `reader.rs` | reader tests | **IMPLEMENTADO E VALIDADO** |
| 2.4 | `codec`/`layout` por bloco | Implementado (parseado/validado) | `block.rs` | reader tests | **IMPLEMENTADO E VALIDADO** |
| 2.5 | `auxiliary_ref` | Implementado | `block.rs`, `reader.rs` | reader tests | **IMPLEMENTADO E VALIDADO** |
| 2.6 | Blocos esparsos/opcionais | Não permitido no v0.1 (regra contígua) | `docs/format-v0.1.md §6` | — | **IMPLEMENTADO E VALIDADO** (decisão de escopo) |
| 2.7 | Mathematical directory / shared assets | Não implementado | — | README "Proposed" | **RESERVADO PARA VERSÃO FUTURA** |

### 3.3 Metadados e tensores

| # | Requisito | Estado real | Arquivos/símbolos | Evidência | Classificação |
|---|---|---|---|---|---|
| 3.1 | dtypes/dimensões/bytes lógicos | Implementado | `tensor.rs` | unit/reader tests | **IMPLEMENTADO E VALIDADO** |
| 3.2 | `model`/`producer`/`user` descritivos | Implementado | `metadata.rs`, `writer.rs` | writer tests | **IMPLEMENTADO E VALIDADO** |
| 3.3 | `tensor_role` (**um por entrada**) | Implementado | `model_import_writer.rs`, `orientation.rs` | tied/untied, 53 casos de paridade | **IMPLEMENTADO E VALIDADO** |
| 3.4 | Orientação/identidade/aliases | Implementado | `orientation.rs`, `model_import.rs` | e2e tied/untied | **IMPLEMENTADO E VALIDADO** |
| 3.5 | Calibração de ativação (proveniência) | Implementado | `activation.rs`, `model_import*.rs` | bundle v1/v2 tests | **IMPLEMENTADO E VALIDADO** |
| 3.6 | `index_axis` | Implementado | `model_import_writer.rs`, `orientation.rs` | parity fixtures | **IMPLEMENTADO E VALIDADO** |
| 3.7 | `user.tensor_roles` (multi-papel por entrada) | **Não implementado** (formato congelado; papel único) | — | — | **RESERVADO PARA VERSÃO FUTURA** |

> O manifesto de arquitetura pode listar múltiplos papéis/aliases externamente; isso **não** comprova suporte operacional a múltiplos papéis por entrada no contêiner.

### 3.4 Importação e operações

| # | Requisito | Estado real | Arquivos/símbolos | Evidência | Classificação |
|---|---|---|---|---|---|
| 4.1 | Importação Safetensors | Implementado | `safetensors.rs`, `model_import*.rs`, CLI | e2e import | **IMPLEMENTADO E VALIDADO** |
| 4.2 | VQ encode + matvec direto (CPU) | Implementado (experimental) | `vq.rs` (`VqMatrix`, `HpsiVqExecutor`) | direct_vq, vq_quality, benchmark | **IMPLEMENTADO, VALIDAÇÃO INCOMPLETA** (só sintético) |
| 4.3 | Calibração de ativação | Implementado | `tools/capture_*`, `activation.rs` | bundle tests | **IMPLEMENTADO, VALIDAÇÃO INCOMPLETA** (sem modelo real) |
| 4.4 | Sem reconstrução densa (kernel VQ) | Implementado no kernel | `vq.rs` | direct matvec tests | **IMPLEMENTADO E VALIDADO** (nível kernel) |
| 4.5 | GPU / CUDA | Non-goal | — | — | **RESERVADO PARA VERSÃO FUTURA** |
| 4.6 | Low-rank / lookup / residual / outliers | Não implementado | — | — | **RESERVADO PARA VERSÃO FUTURA** |
| 4.7 | Runtime de inferência (forward/atenção/tokenizer) | Fora do escopo v0.1 | — | — | **RESERVADO PARA VERSÃO FUTURA** |
| 4.8 | MoE / esparso | Não implementado | — | — | **RESERVADO PARA VERSÃO FUTURA** |
| 4.9 | POC 4096×4096 FP16/INT4/INT2 | Não executado | — | README milestone 3 | **RESERVADO PARA VERSÃO FUTURA** |

### 3.5 CLI

| # | Requisito | Estado real | Arquivos/símbolos | Evidência | Classificação |
|---|---|---|---|---|---|
| 5.1 | `hpsi inspect` | Implementado | `apps/hpsi-cli/src/main.rs` | cli tests | **IMPLEMENTADO E VALIDADO** |
| 5.2 | `hpsi verify` | Implementado | `main.rs` | cli tests | **IMPLEMENTADO E VALIDADO** |
| 5.3 | `safetensors-import` | Implementado | `main.rs` | cli/e2e | **IMPLEMENTADO E VALIDADO** |
| 5.4 | `calibrate` | Implementado | `main.rs` | cli tests | **IMPLEMENTADO E VALIDADO** |
| 5.5 | Erros/códigos de saída previsíveis | Implementado | `main.rs` (`ExitCode`) | cli tests | **IMPLEMENTADO E VALIDADO** |
| 5.6 | `--help`/`--version` | Implementado (C2f-C1) | `main.rs::help` | `cli_meta` tests | **IMPLEMENTADO E VALIDADO** |
| 5.7 | `inspect`/`verify` não decodificam | Implementado | `main.rs`, docs | cli tests | **IMPLEMENTADO E VALIDADO** |

### 3.6 Qualidade e paridade

| # | Requisito | Estado real | Evidência | Classificação |
|---|---|---|---|---|
| 6.1 | Testes Rust | **287 passed / 0 failed / 0 ignored** | `cargo test` | **IMPLEMENTADO E VALIDADO** |
| 6.2 | Testes Python | **205 pytest / 109 unittest OK** | `pytest`, `unittest` | **IMPLEMENTADO E VALIDADO** |
| 6.3 | Fixtures de paridade Rust/Python | 34 + 19 casos | `tests/fixtures/*.json` | **IMPLEMENTADO E VALIDADO** |
| 6.4 | Casos tied/untied | Implementado | `identity_alias_*`, e2e | **IMPLEMENTADO E VALIDADO** |
| 6.5 | Determinismo | Implementado | writer/tests | **IMPLEMENTADO E VALIDADO** |
| 6.6 | Corrupção/truncamento | Implementado | adversarial tests | **IMPLEMENTADO E VALIDADO** |
| 6.7 | Auditores independentes Rust/Python | Implementado | `audit_reconstruction.rs`, `audit_hpsi_container.py` | 0 discrepâncias | **IMPLEMENTADO E VALIDADO** |
| 6.8 | Fuzzing dos alvos | Campanha C2c: 0 crashes (99,8M + 1,45M runs) | `fuzz/` | **IMPLEMENTADO E VALIDADO** |
| 6.9 | Cobertura LLVM | Não disponível (sem `llvm-tools`) | — | **RESERVADO PARA VERSÃO FUTURA** |
| 6.10 | Qualidade em modelo real (perplexidade) | Não executado | — | **NÃO COMPROVADO NO ESCOPO ORIGINAL** (não é requisito do v0.1 toolkit) |

## 4. Diagrama da arquitetura física implementada

Somente o que é comprovado por `docs/format-v0.1.md`, `crates/hpsi-format` e testes.

```
Arquivo .hpsi v0.1 (offsets absolutos, todas as regiões alinhadas a 8 B, padding zero)
┌──────────────────────────────────────────────────────────────────────────────┐
│ HEADER — 128 B                                                               │
│  0   magic "HPSI"           4 B   │  4  format_major u16 = 0                │
│  6   format_minor u16 = 1         │  8  header_size u32 = 128               │
│ 12   flags u32 (reservado)        │ 16  file_size u64                       │
│ 24   metadata_offset u64          │ 32  metadata_length u64                 │
│ 40   tensor_dir_offset u64        │ 48  tensor_count u32                    │
│ 52   tensor_entry_size u32 = 128  │ 56  block_dir_offset u64                │
│ 64   block_count u32              │ 68  block_entry_size u32 = 64           │
│ 72   payload_offset u64           │ 80  payload_length u64                  │
│ 88   integrity_offset u64         │ 96  integrity_count u32                 │
│100   integrity_entry_size u32 = 32│104  artifact_id 16 B (opaco)           │
│120   reserved 8 B = 0                                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│ METADATA — UTF-8 JSON (tamanho limitado)                                     │
│  { format:"hpsi", metadata_version:1, tensor_names:[...],                    │
│    model:{...}, producer:{...}, user:{...} }                                 │
│  user.source_tensors[] → { name, source_dtype, source_shape, disposition,     │
│    tensor_role (ÚNICO por entrada), weight_orientation, orientation_source,   │
│    logical_shape, index_axis?, calibration? }                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ TENSOR DIRECTORY — N entradas de 128 B                                       │
│  tensor_id u32 | rank u8 | dtype u8 | codec u16 | flags u32                  │
│  block_count u32 | first_block_index u64 | logical_nbytes u64                │
│  dimensions u64[8] | name_ref u32 | representation_ref u64                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ BLOCK DIRECTORY — M entradas de 64 B (agrupadas por tensor)                  │
│  tensor_id u32 | block_id u32 | flags u32 | codec u16 | layout u16           │
│  logical_offset u64 | logical_length u64                                     │
│  payload_offset u64 | payload_length u64                                     │
│  auxiliary_ref u64 | checksum_index u32 | reserved u32                       │
├──────────────────────────────────────────────────────────────────────────────┤
│ INTEGRITY DIRECTORY — K entradas de 32 B (opcional)                          │
│  target_kind u32 (1=metadata, 2=block payload) | target_index u32            │
│  algorithm u16 (1=CRC32C) | digest 16 B (CRC32C LE nos 4 primeiros)          │
├──────────────────────────────────────────────────────────────────────────────┤
│ PAYLOAD — bytes opacos; v0.1 emite apenas RAW (codec 1)                      │
│  [ bloco 0 ][ bloco 1 ][ ... ][ bloco M-1 ]                                  │
└──────────────────────────────────────────────────────────────────────────────┘

Mapeamento lógico:
  tensor (tensor_id) ──first_block_index + block_count──▶ faixa contígua de blocos
  bloco ──logical_offset/length──▶ faixa no fluxo lógico do tensor
  bloco ──payload_offset/length──▶ faixa no PAYLOAD (não sobreposta)
  verify_all() ──▶ CRC32C de metadata (target_kind=1) + de cada bloco coberto (target_kind=2)
```

**Fora do diagrama atual (futuro/proposto, não implementado):** Mathematical directory, Shared assets, codecs ZSTD/CODEBOOK/LOW_RANK/LOOKUP/RESIDUAL com decoder, runtime de inferência.

## 5. Testes executados (estado auditado, HEAD `41c8248`)

| Comando | Exit | Duração | Resultado |
|---|---:|---:|---|
| `cargo fmt --all -- --check` | **0** | — | sem divergências |
| `cargo test --workspace --all-targets` | **0** | 673 s | **287 passed / 0 failed / 0 ignored** |
| `cargo clippy --workspace --all-targets -- -D warnings` | **0** | 197 s | 0 avisos |
| `python -m compileall -q tools tests` | **0** | — | sem saída |
| `python -m unittest discover -s tests -p "test_*.py"` | **0** | 2,4 s | **Ran 109 tests — OK** |
| `python -m pytest tests -q` | **0** | 262 s | **205 passed** (61,97 s de execução) |

> Nota de ambiente: a primeira execução de `pytest` ocorreu **em paralelo** com o `cargo test` e estourou o timeout de 600 s por contenção de CPU; foi **reexecutada isolada** e passou (205 passed, exit 0). O `cargo test` (673 s) também reflete a contenção. Nenhum resultado antigo foi reutilizado.

## 6. Limitações e pendências (reais)

1. **Runtime de inferência ausente** — forward, atenção, KV cache e tokenizer não existem; v0.1 é toolkit.
2. **Decoders ausentes** — ZSTD_FRAME (2) e codecs 3–6 são identificadores sem decoder; `inspect`/`verify` não decodificam.
3. **Math directory / shared assets** — propostas, não implementadas.
4. **Papel único por entrada** — `user.tensor_roles`/segundo slot não existem (formato congelado); multi-papel é decisão de schema futuro.
5. **VQ/calibração só sintéticos** — sem avaliação em modelo real treinado; sem perplexidade.
6. **Cobertura LLVM** — não medida (sem `llvm-tools`).
7. **Segurança de dependências** — `cargo audit`/`cargo deny` não instalados; advisories não validados.
8. **Publicação** — não realizada; documentação/legal com pendência de razão social da Filtra.AI.

## 7. Classificação final

**`ADERENTE COM PENDÊNCIAS`.**

- **Aderente:** contêiner/formato físico, organização em blocos, metadados/papéis/orientação, importação, CLI, integridade, paridade e fuzzing correspondem ao especificado e estão validados.
- **Pendências:** tudo o que a arquitetura original previa como visão ampla (runtime, decoders, math directory, GPU, MoE, POC com modelo real) permanece explicitamente **futuro**; e há limitações de validação (cobertura, audit tools, modelo real) registradas.

Não há evidência de desvio do formato v0.1, de `user.tensor_roles`, de decoder inexistente declarado como disponível, ou de runtime de inferência no escopo.

---

*Relatório da auditoria de conformidade arquitetural. Nenhum código, manifest ou formato físico foi alterado.*
