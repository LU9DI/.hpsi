# HPSI — revisão independente da auditoria arquitetural v0.1

**Data:** 2026-11-08
**Repositório:** `C:\Users\lucas\HPSI`
**Branch:** `feat/hpsi-core-container`
**Commit auditado (original):** `c2727323bf971a927a22d47bdbf4bf7b248f3c1c`
**Código efetivamente auditado:** `41c8248f0cbf0f56e9085df8eac95bf45595400e` (pai do commit documental; `c272732` só adiciona este relatório)
**Natureza:** revisão independente, baseada em código, somente-leitura. Nenhum código, manifest, formato ou versão alterado.

---

## 1. Classificação independente

**`ADERENTE COM PENDÊNCIAS`** — a mesma classificação do relatório original, agora confirmada por leitura direta do código. Não encontrei divergência material que mude a classificação. Encontrei, porém, **quatro imprecisões de precisão** (não de escopo) que o relatório original deveria explicitar melhor, além de confirmar que todos os números de teste correspondem ao pai do commit documental.

## 2. Afirmações confirmadas (com evidência de código)

| Afirmação original | Evidência | Veredito |
|---|---|---|
| Header de 128 B com campos nos offsets do diagrama | `crates/hpsi-format/src/header.rs:3` (`HEADER_SIZE=128`); leituras em `4/6/8/12/16/24/32/40/48/52/56/64/68/72/80/88/96/100/104..120/120..128` | **Confirmado** |
| magic `HPSI`, major 0, minor 1; `header_size=128`; `flags` desconhecidos rejeitados | `header.rs:34-52` (`BadMagic`, `UnsupportedVersion`, `InvalidHeader`, `UnsupportedFeature`) | **Confirmado** |
| `file_size` deve coincidir com o tamanho real; reservado final zero | `header.rs:54-59`, `header.rs:120-128` | **Confirmado** |
| Tensor directory 128 B; block 64 B; integrity 32 B | `tensor.rs::ENTRY_SIZE=128`; `block.rs::ENTRY_SIZE=64`; `integrity.rs::ENTRY_SIZE=32`; `header.rs:4-6` | **Confirmado** |
| Alinhamento 8 B, regiões dentro do arquivo, sem sobreposição | `region.rs::validate_regions` (offset≥128, `%8==0`, `end ≤ file_size`, `windows(2)` overlap) | **Confirmado** |
| Payload de bloco contido na região de payload | `reader.rs:490-498` (`"block payload is outside payload region"`) | **Confirmado** |
| Payloads de blocos não sobrepostos | `reader.rs::validate_payload_ranges` (ordenação + `furthest_end`) | **Confirmado** |
| Cobertura lógica contígua e exata | `reader.rs:458` ("tensor block ranges do not cover block directory"), `:519` ("blocks do not cover tensor logical byte range") | **Confirmado** |
| Ownership/ordenação de blocos e tensor IDs crescentes | `reader.rs:471-478`, `:438-446` | **Confirmado** |
| `auxiliary_ref` e `representation_ref` validados (existência e ciclos) | `reader.rs:611-720` (`validate_reference_cycles`, DFS) | **Confirmado** |
| CRC32C, `target_kind` 1=metadata / 2=bloco | `integrity.rs:26-55`; `reader.rs::verify_all` (linhas 225-283) | **Confirmado** |
| `verify_all()` verifica apenas os registros declarados | `reader.rs:226-283` (itera `self.integrity`) | **Confirmado** |
| Writer emite apenas RAW (tensor codec 1; bloco herda 0) | `writer.rs:339` (`put16(entry,6,1)`); bloco com `entry[12..14]` zero | **Confirmado** |
| `tensor_role` único por entrada no contêiner | `crates/hpsi-ops/src/model_import_writer.rs:779` (`"tensor_role": emitted_role.as_str()`); `orientation.rs:1511` lê `as_str()` | **Confirmado** |
| `user.tensor_roles` não existe no Rust | `grep -r "tensor_roles" crates apps --include=*.rs` → nenhum resultado | **Confirmado** |
| Manifesto externo pode listar múltiplos papéis | `orientation.rs:436` (`roles` a partir de `tensor_role` no manifesto) | **Confirmado** (distinção correta) |
| VQ validado só em dados sintéticos | `crates/hpsi-ops/tests/direct_vq.rs:6` (`synthetic_weights`); `vq.rs` (amostras determinísticas) | **Confirmado** |
| Sem runtime de inferência | `grep softmax/attention/kv cache/tokenizer/forward(` em `crates/`/`apps/` → nenhum | **Confirmado** |
| `--help`/`--version` | `apps/hpsi-cli/src/main.rs::help`; `tests/cli_meta.rs` (9 testes) | **Confirmado** |
| Números de teste (287 Rust / 205 pytest / 109 unittest) | executados em `41c8248`; `c272732` só adiciona markdown (`git show --stat c272732` = 1 arquivo) | **Confirmado como pai, não commit posterior** |

## 3. Afirmações parcialmente comprovadas / imprecisões encontradas

### 3.1 Ordem do diagrama × invariante de formato

O diagrama apresenta **header → metadata → tensor dir → block dir → integrity → payload**. Essa é a **ordem canônica do writer** (`writer.rs:392-405`, com `pad_to`), mas **não** é um invariante do formato: `validate_regions` só exige alinhamento, limites e não sobreposição — **não exige ordem**. Um arquivo com regiões em outra ordem, mas offsets consistentes, seria aceito. O diagrama é correto para artefatos do writer de referência; deveria ser rotulado como "layout canônico do writer", não como regra de formato.

**Impacto:** nenhum na classificação; é precisão do diagrama.

### 3.2 Padding zero: regra de formato, verificação opcional

O writer escreve padding zero (`writer.rs:392-405`, `pad_to`), e a regra do formato exige padding zero (`docs/format-v0.1.md §2`). Porém:
- `ModelReader::open` **não** valida padding;
- `verify_all()` **não** valida padding (só CRC);
- o CLI `hpsi verify` chama **apenas** `verify_all()` (`main.rs:129-130`);
- existe `ModelReader::verify_padding(max_bytes)` com testes (`reader.rs:271`, `:864-900`), mas **não é chamado pelo CLI**.

**Impacto:** o diagrama "padding zero" descreve o writer/regra, não a cobertura do `verify` do CLI. O relatório original não afirma que o CLI valida padding, mas deveria explicitar essa limitação.

### 3.3 `layout` de bloco: só 0 é aceito

O diagrama lista `layout u16`. O parser **rejeita** `layout != 0` (`block.rs:47-52`, `"unsupported flags, layout, or reserved field"`). A linha "codec/layout — implementado (parseado/validado)" é correta, mas deveria dizer que **apenas layout 0** é suportado no v0.1.

**Impacto:** nenhum; precisão.

### 3.4 Versionamento: rejeição, não negociação

`header.rs:41-43` fixa `major=0, minor=1` e rejeita qualquer outra versão. A linha "versionamento/flags — IMPLEMENTADO E VALIDADO" é aceitável para o v0.1, mas o que existe é **validação/rejeição** de versão, **não** um mecanismo de compatibilidade forward para minors futuros. Vale registrar para não sugerir capacidade inexistente.

**Impacto:** nenhum; precisão de escopo.

### 3.5 Restrições do header omitidas no diagrama (não exaustivo)

O código impõe, além do diagrama: `tensor_count > 0` e `block_count > 0`; `metadata.length > 0` e `payload.length > 0`; diretório de integridade vazio exige offset 0 e `entry_size` 0; `artifact_id` opaco (o writer grava zero); `reserved` do header zero. O diagrama é um mapa de campos, não a lista completa de invariantes — recomendo uma nota "não exaustivo".

## 4. Revisão do diagrama físico

Conferência campo a campo contra `header.rs`:

| Offset | Campo no diagrama | Código | OK |
|---:|---|---|---|
| 0 | magic "HPSI" | `bytes[0..4]` | ✔ |
| 4/6 | major/minor u16 | `u16_at(4)`, `u16_at(6)` | ✔ |
| 8 | header_size u32 | `u32_at(8)` | ✔ |
| 12 | flags u32 | `u32_at(12)` | ✔ |
| 16 | file_size u64 | `u64_at(16)` | ✔ |
| 24/32 | metadata offset/length | `u64_at(24)`, `u64_at(32)` | ✔ |
| 40/48/52 | tensor dir offset/count/entry | `u64_at(40)`, `u32_at(48/52)` | ✔ |
| 56/64/68 | block dir offset/count/entry | `u64_at(56)`, `u32_at(64/68)` | ✔ |
| 72/80 | payload offset/length | `u64_at(72/80)` | ✔ |
| 88/96/100 | integrity offset/count/entry | `u64_at(88)`, `u32_at(96/100)` | ✔ |
| 104..120 | artifact_id 16 B | `bytes[104..120]` | ✔ |
| 120..128 | reserved 8 B zero | `bytes[120..128]` | ✔ |

Os diretórios (128/64/32 B), a semântica de `checksum_index`, `auxiliary_ref`, cobertura lógica e CRC32C também conferem. **O diagrama está correto em conteúdo**; as ressalvas são de **rotulagem** (§3.1) e de **cobertura do `verify`** (§3.2).

## 5. Testes e correspondência de HEAD

- O relatório original declara `HEAD auditado: 41c8248` e a tabela de testes diz "estado auditado, HEAD 41c8248". O commit documental `c272732` tem pai `41c8248` e altera **somente** o markdown (`1 file changed, 213 insertions`). Portanto os números **correspondem ao pai**, não ao commit posterior. **Confirmado.**
- Números conferem com os logs/execuções: Rust **287**, pytest **205**, unittest **109**, fmt/clippy **0**. O `cargo test`/`pytest` da fase foram lentos por contenção de CPU; o relatório já registra isso honestamente.

## 6. Recomendações mínimas (sem implementar)

1. Rotular o diagrama como **“layout canônico do writer”** e notar que o reader é **offset-driven** (não impõe ordem).
2. Explicitar que `padding zero` é regra do writer/formato e que a verificação é via `verify_padding`, **não** pelo `verify` do CLI.
3. Registrar que **apenas `layout = 0`** é aceito em blocos v0.1.
4. Registrar que o versionamento v0.1 é **rejeição de versão**, não negociação forward.
5. Opcional: marcar o diagrama como **não exaustivo** quanto a `tensor_count/block_count > 0`, metadata/payload não vazios e integridade vazia com offset/entry_size zero.

## 7. Conclusão

Não há **divergência material**: todas as afirmações centrais do relatório original são sustentadas pelo código, e a classificação **`ADERENTE COM PENDÊNCIAS`** se mantém. As diferenças encontradas são **imprecisões de precisão** (ordem do diagrama, cobertura de padding, `layout` só 0, versionamento), que não alteram o escopo nem a prontidão declarada. Nenhum código, manifest ou formato foi alterado nesta revisão.

---

*Revisão independente da auditoria arquitetural. Sem publicação, push, merge, tag ou release.*
