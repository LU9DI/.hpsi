# Weight orientation policy

Scope: how HPSI decides, reports and defends the orientation of a rank-2 weight
matrix. The container fields themselves are specified in
[format-v0.1.md](format-v0.1.md) §4.1; this document states what a consumer may
conclude from them and what it may not.

## 1. Four claims that are not interchangeable

| Claim | Who proves it | Evidence | If it passes, does that mean the weights are oriented correctly? |
|---|---|---|---|
| Container integrity | `hpsi-format` (`verify_all`, `verify_padding`) | CRC32C over metadata, block payloads and padding | No. Metadata is covered by a CRC, so an adulterated document whose CRC was recomputed is perfectly "integrity-valid". |
| Metadata self-consistency | `hpsi_ops::orientation::audit_entry` (`crates/hpsi-ops/src/orientation.rs`) | `weight_orientation` ↔ `orientation_source` ↔ `source_shape` ↔ `logical_shape` ↔ stored selector byte count | No. A matrix transposed at import time leaves a self-consistent document describing the wrong matrix. |
| Architectural compatibility | external model-definition manifest, exact tensor-name match | `ArchitectureFacts::expectation(name) == declared` | This is the only claim about the model, and it requires data from outside the container. |
| Numeric correctness of a reconstruction | auditor comparing against the original weights | `audit_reconstruction` example, `tools/audit_hpsi_container.py` | Proves a specific artifact against a specific source; it is not a property a container can assert about itself. |

`hpsi verify` reports integrity. `hpsi inspect` reports self-consistency and,
only when `--architecture` is supplied, architectural compatibility. No command
presents the first as evidence for the third.

## 2. Assurance states

The single Rust decision point is `OrientationAssurance` in
`crates/hpsi-ops/src/orientation.rs`; the CLI, `resolve_vq_layout` and the audit
example read it from `audit_container` rather than re-deriving it. The Python
auditor (`tools/audit_hpsi_container.py`) implements the same states
independently, so the two can be compared instead of trusted.

| Rust (`hpsi inspect`) | Python auditor | Meaning |
|---|---|---|
| `architecturally_confirmed` | `architecture_confirmed` | Declared orientation equals what the supplied manifest requires for this exact name. The only confirmed state. |
| `declared_not_verified` | `declared_not_verified` | Structurally consistent, but no external information was supplied. |
| `square_not_verifiable` | `square_not_verifiable` | Square matrix and no manifest: nothing in the container discriminates the two orientations. |
| `legacy_undeclared` | `legacy_undeclared` | The container carries no orientation metadata for this tensor. |
| — (a finding, not a status) | `architecture_contradiction` | Declared orientation disagrees with the manifest. Reported as `orientation finding:` by the CLI and as `orientation_disagrees_with_architecture` by the auditors. |
| — (tensor excluded) | `not_a_matrix` | Rank ≠ 2. Only matrices have orientation. |

A manifest that supplies names which match nothing confirms nothing: `inspect`
says so explicitly rather than counting those tensors as confirmed.

## 3. Roles: a second dimension, not a second name for the same thing

`weight_orientation` answers "how is this projection matrix laid out". It cannot
answer "what is this tensor for", and using it for both was the defect that made
an embedding table look like a `nn.Linear` layer. `tensor_role` is the second
dimension, with `index_axis` naming which axis a lookup table indexes:

| role | layout claim | index axis | projection executor |
|---|---|---|---|
| `linear_projection` | requires `linear_out_in` | n/a | allowed |
| `conv1d_projection` | requires `conv1d_in_out` | n/a | allowed |
| `embedding_table` | encoding fact only | required, `0` or `1` | **refused, no flag overrides it** |
| `other` | encoding fact only | forbidden | **refused, no flag overrides it** |
| `unknown` | encoding fact only | forbidden | refused unless the caller accepts unverified |

Three consequences that the tests pin down:

* **Role assurance is reported apart from orientation assurance.** A manifest
  that confirms a role grants no layout guarantee, and a manifest that confirms a
  layout grants no role guarantee. A tied weight is reached as a table *and* as a
  projection, so the model definition may state both roles; the container records
  the one it encoded under and the audit surfaces the full set.
* **`unknown` is not `other`, and neither is `legacy_undeclared`.** Four states
  must stay distinct: a genuinely old artifact (no role key at all), a new import
  whose role was never established (`unknown`), a tensor known to serve some other
  function (`other`), and a role the definition confirms. Collapsing the first two
  would let a new producer hide missing evidence behind legacy semantics.
* **An indexed table never enters the matvec path.** `resolve_vq_layout` refuses
  `embedding_table` and `other` outright, because multiplying by a lookup table
  answers a different question than indexing it, and this contract ships no lookup
  kernel. Producers keep such tensors on the existing RAW path instead of encoding
  them as projections.

Activation statistics are not reused for tables. `diagonal_second_moment` is taken
from a forward pre-hook on a module's *input*; an `nn.Embedding` receives token
ids, so there is no feature-wise second moment of it to record, and attaching a
calibration object to an indexed table is a contract violation, not a soft warning.

### Pending, deliberately not implemented

v0.1 requires `weight_orientation` for **every** rank-2 entry, so a table must
still record the `weight_orientation` its selectors were grouped by. The honest end of
that is an entry that states `tensor_role: embedding_table` and no layout at
all; expressing it means relaxing a documented requirement, which is a
public-schema change. It has not been made. Consequences while it stays pending: an
embedding table's role is confirmed by a model definition, its extra layout
token is treated as an encoding fact and never as architectural evidence, and a
table indexed on axis 1 with no projection provenance cannot be imported at all
(the importer refuses rather than labelling it `conv1d_in_out`, which would be an
invented claim).

## 4. Policy for square `[n, n]` matrices

For a square matrix both orientations produce the identical logical shape, and
for a VQ pair they produce the identical stored selector byte count. Shape,
logical shape and payload size therefore cannot falsify a square declaration.
The policy, in order:

1. **Compare with trustworthy architectural information when it exists.**
   With `--architecture <manifest.json>`, a square matrix is either
   `architecturally_confirmed` or a contradiction. This is the only route to a
   confirmed square matrix.
2. **Otherwise validate against the calibration bound to the tensor.**
   The importer refuses to persist a `calibration` object whose `module_type`
   contradicts the recorded `weight_orientation`, so `orientation_source=calibration`
   means the declaration came from the capture's module class. That is checked
   provenance, not external proof: it travels inside the container and an
   attacker who edits it can recompute the CRC.
3. **Otherwise mark it as not verified.** `square_not_verifiable` — and `inspect`
   states that the declared orientation is a claim the container cannot verify.
4. **Where an operation needs an architectural guarantee, refuse or require
   explicit confirmation.** `resolve_vq_layout(reader, name, facts,
   allow_unverified)` rejects any tensor that is not
   `architecturally_confirmed` unless the caller passes `allow_unverified=true`;
   a contradiction with the manifest is rejected even then. That flag is the
   contract's "explicit confirmation" step, and the error message names the
   manifest as the way to avoid it.

Numeric detection is still possible for a square matrix even when its shape
cannot help, and the auditors use it: the reconstruction is compared with the
source matrix read *in the declared layout* and with its transpose, and a
correctly oriented import must fit the logical matrix strictly better. On the
fixture, the square `transformer.h.0.attn.c_proj.weight` (`[16, 16]`) in the
uncalibrated container scores relative L2 1.3886 against the logical matrix and
0.4186 against its transpose, which is the finding
`reconstruction_matches_transpose_better_than_logical`. Nothing about that
measurement comes from the tensor's name.

## 5. Orientation is never inferred

No consumer, auditor, importer, reader or test helper infers orientation from:
a tensor-name substring (`c_attn`, `c_proj`, `c_fc`, or anything similar), a
module name, a coinciding dimension, or "whichever reference fits better". Roles
are held to the same rule: a role value exists only because a producer that saw
the module class wrote it, and no reader ever assigns `embedding_table` because a
name looks like an embedding or a shape looks square.
Two consequences worth stating plainly:

- An architecture manifest entry must state `weight_orientation` explicitly for
  an exact tensor name; `ArchitectureFacts::from_json` rejects an entry that
  omits it ("orientation is never inferred from a name"), and lookup is exact —
  no prefix, suffix or fuzzy matching.
- Choosing the comparison basis by lowest error would be circular: it would
  turn a transposed import into a passing test. The basis is always the declared
  orientation, and the field `quantization_distortion_basis` records whether
  that declaration was externally confirmed (`architecture_oracle`) or is only
  the container's own claim (`declared_orientation_self_consistency`).

The manifests used as oracles are generated from the model definition, never
from the artifact under audit: `tools/architecture_manifest_from_calibration.py`
maps each captured layer's `module_type` (`linear` → `linear_out_in`, `conv1d`
→ `conv1d_in_out`) and refuses an unknown `module_type` rather than guessing.

## 6. Known limits

- The rectangular collision is real: a VQ `[16, 32]` and a `[32, 16]` store the
  same selector byte count, so a container cannot falsify those declarations
  from its own bytes either. Only a numeric comparison against the original
  weights, or a manifest, distinguishes them.
- A container that declares nothing (`legacy_default`, or no
  `user.source_tensors`) stays readable: readers must surface `legacy_undeclared`
  rather than reinterpreting the layout, and `verify` keeps passing on those
  artifacts because they are not structurally invalid.
- Numeric detection depends on having the original weights. Auditors that lack
  them report the orientation as unverified; they do not emit a verdict.
- The manifest describes module classes, not values. It confirms how a matrix
  is laid out, not that the quantized values are faithful.
- A container records **one** role per tensor: the function it was encoded under.
  A tied embedding/output head is therefore recorded as `linear_projection` while
  the model definition still evidences `embedding_table` too, and the audit
  surfaces the full set as `architecture_roles`. Stating several roles per tensor
  in the artifact would be a schema extension, not a relaxation.
- Role evidence for a lookup table can only come from a model definition. A
  capture bundle states classes it hooked (`linear`, `conv1d`) and never hooks an
  embedding, so a bundle alone cannot produce a table role; the manifest
  generator emits the two projection roles it can see and leaves the rest
  unstated rather than inventing one.
- No gather/row kernel ships yet, so an indexed table that a runtime must index
  has to be reconstructed densely or preserved RAW. Preserving it RAW costs the
  compression the table would otherwise gain.

## 7. Reason codes and the shared case table

Every rejection the layout matrix produces carries a stable code, so the two
auditors can be diffed on *why* they refused, not merely on whether they did:
`orientation_required`, `logical_shape_required`, `role_without_orientation`,
`role_orientation_mismatch`, `axis_without_role`, `axis_for_non_indexed_role`,
`table_without_axis`, `axis_value_unsupported`, `axis_one_unsupported`,
`table_orientation_mismatch`, `logical_shape_conflicts_axis`,
`logical_shape_conflicts_orientation`, `selector_bytes_mismatch`,
`selector_layout_unencodable`, `table_must_not_be_calibrated`,
`role_token_unknown`, `logical_shape_on_non_matrix`, `matrix_role_on_non_matrix`.

The codes are owned by the contract, not by either implementation:
`hpsi_ops::LayoutViolation` and `audit_hpsi_container.entry_layout_violation`
each produce them from their own code, and
`tests/fixtures/embedding_layout_cases.json` is the single table both suites read
(`crates/hpsi-ops/tests/embedding_layout_parity.rs`,
`tests/test_embedding_layout_parity.py`). Neither implementation calls the other.
Renaming a code is a contract change; rewording a reason is not.

**Order is part of the contract.** An entry can violate two rules at once, and
the reported code is the first in this sequence: non-matrix rules, role vocabulary,
`orientation_required`, `logical_shape_required`, projection role rules, index-axis
ownership, then the indexed-table rules (calibration, axis presence, axis value,
axis support, orientation agreement, logical shape, selector payload). A
doubly-broken artifact therefore reports the same single reason in both languages.

`weight_orientation` is the only field that records which axis was grouped. The
words "layout" and "grouping" in prose and in reason strings describe that field
or the physical payload; they are never field names. The container's vocabulary is
exactly `weight_orientation`, `orientation_source`, `tensor_role`, `index_axis`
and `logical_shape`, plus the tensor directory's own shapes. Finding labels such
as `container_arrangement_does_not_match_declared_orientation` are report strings
shared by both auditors and are not part of the container.

## 8. Identity, aliases and several roles at once

A tensor is a **parameter object**, not a name and not a value. The capture groups
what it sees on `id(parameter)` and on nothing else: equal values, equal shapes, a
shared dtype or a name that looks related are all reasons *not* to merge. Two
parameters holding identical arrays stay two logical tensors with independent
roles, metadata and payloads.

One parameter may be reached through several names, and each name may come from a
differently classed module. A tied `transformer.wte.weight` / `lm_head.weight` pair
is one array indexed by an `nn.Embedding` and multiplied by an `nn.Linear`. The
producer records that as one tensor with:

- every alias the model exposes for the object;
- every role those aliases evidence, as a set;
- the **association** — which alias, which module, which class, which role;
- `index_axis` for the indexed role and the layout claim for the projection, each
  stated only by the row that can support it.

Nothing may be recovered from a name. `nn.Embedding` yields
`embedding_table` with `index_axis: 0` because the class indexes rows; `nn.Linear`
yields `linear_projection` with `linear_out_in` because the class declares
`[out_features, in_features]`; Transformers' `Conv1D(nf, nx)` yields
`conv1d_projection` with `conv1d_in_out` because its weight is `[nx, nf]`. Each of
those claims is then checked against the parameter it is made about: a class whose
declared dimensions disagree with its own weight evidences nothing, and the record
says why.

**A disagreement is reported, never resolved by choosing.** If one parameter is
reached as two projections with different layouts, or as a table whose indexing
axis differs from the one another module states, the record becomes
`status: "conflict"` with no role, no layout and no axis, and the manifest
generator omits the entry and lists the name under `unresolved`. An architecture
manifest that pairs a role with an alias or a layout it does not imply — or that
names an alias belonging to another identity — is refused by both auditors, in the
same words, rather than read under whichever half is convenient.

**What the container records for a tied parameter.** v0.1 stores one `tensor_role`
token per source entry, so the container records the role the payload was actually
*encoded* under — for a tied head that is `linear_projection`, because the selector
bytes mean a grouped matvec — and `index_axis` stays absent, since it belongs to the
recorded role. The table role is not discarded: it lives in the model definition and
is reported beside the container's claim as `architecture_roles`,
`architecture_aliases` and `role_evidence` by both auditors. Giving the container a
second role slot would be a public schema change and is not done here; claiming a
`tied` role instead would be a new name for two old claims. A table that was never
measured as a projection is kept out of the VQ path and preserved RAW, which is the
case a positional embedding produces.

The vocabulary of this section is data from **outside** the container. The
container's own fields are unchanged: `weight_orientation`, `orientation_source`,
`tensor_role`, `index_axis`, `logical_shape`. The capture bundle gained an optional
identity section (version 2, `docs/transformer-activation-capture.md`), and the
architecture manifest gained the optional keys `aliases` and `role_evidence`; both
are additive, and a reader that ignores them loses claims, never gains a false one.
