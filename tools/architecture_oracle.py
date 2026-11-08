"""Architecture oracle keyed by tensor IDENTITY, not by module path.

A model definition reaches the same parameter object through several names: with
``tie_word_embeddings`` a GPT-2 exposes one matrix as both
``transformer.wte.weight`` (the ``nn.Embedding`` table) and ``lm_head.weight``
(the output projection). A container records whichever name the checkpoint
stored, so an oracle that keys by module path leaves the stored name unmatched
and a consumer silently loses architectural coverage for exactly the largest
matrix in the model.

This module therefore consumes *facts* — one observation of one parameter object
through one name — and groups them by object identity. Consequences that follow
from that choice, and that the tests pin down:

* two names for one object become one logical entry with two aliases;
* two objects with identical values stay two separate entries, because sharing
  is an identity relation and value equality is not (a checkpoint may legitimately
  store a duplicate);
* the canonical name is chosen deterministically (lexicographically smallest
  alias) so reports are reproducible;
* when the facts contradict each other for one identity — a parameter reached as
  both ``nn.Linear`` and ``transformers.Conv1D``, one name claimed by two
  objects, an indexing axis stated twice with different values — the tensor is
  reported as UNRESOLVED. Nothing here ever fills the gap with a guess from a
  name, a suffix, a rank or a dimension.

Roles and orientations are kept as two separate dimensions: ``tensor_role`` is
the architectural function, ``weight_orientation`` is the layout of a projection
matrix, and an embedding table has an ``index_axis`` instead of an orientation.
One tensor may legitimately carry both a table role and a projection role; the
oracle records both without inventing a precedence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

ORIENTATIONS = ("linear_out_in", "conv1d_in_out")

TENSOR_ROLES = (
    "linear_projection",
    "conv1d_projection",
    "embedding_table",
    "other",
    "unknown",
)

# The orientation a role requires, when the role is a projection. A role that is
# absent from this mapping states no orientation at all.
ROLE_ORIENTATIONS = {
    "linear_projection": "linear_out_in",
    "conv1d_projection": "conv1d_in_out",
}

# Roles that describe a matrix by which axis is indexed rather than by how it
# multiplies.
INDEXED_ROLES = ("embedding_table",)


class LayoutError(ValueError):
    """No layout can be established without inference; nothing is guessed."""


def source_layout(
    role: str | None,
    orientation: str | None,
    index_axis: int | None,
    source_shape: Sequence[int],
) -> tuple[tuple[int, int], bool]:
    """Where the source array sits, given one trusted piece of evidence.

    Returns ``(expected_logical_shape, transpose_to_source_layout)``. Two
    different claims are handled here and never mixed:

    * a **projection** states which axis multiplies, through ``weight_orientation``;
    * an **indexed table** states which axis the ids select, through ``index_axis``.

    A projection orientation is authoritative when present, because the quantizer
    grouped its selectors along exactly that axis. Only without any orientation
    may an indexing role decide the layout, and it is a claim about a different
    thing. Everything else raises: inferring an layout from rank, shape or a
    name pattern is what this contract forbids.
    """
    if len(source_shape) != 2:
        raise LayoutError(f"layout is only defined for rank-2 matrices, got {list(source_shape)}")
    rows, cols = int(source_shape[0]), int(source_shape[1])
    if orientation == "linear_out_in":
        return (rows, cols), False
    if orientation == "conv1d_in_out":
        return (cols, rows), True
    if orientation is None and role == "embedding_table":
        if index_axis == 0:
            return (rows, cols), False
        if index_axis == 1:
            return (cols, rows), True
        raise LayoutError(
            "an embedding table must state index_axis 0 or 1; no indexing layout is derivable"
        )
    raise LayoutError(
        f"no layout evidence for a rank-2 tensor of role {role!r} and orientation "
        f"{orientation!r}; refusing to guess an layout from shape or name"
    )


@dataclass(frozen=True)
class WeightFact:
    """One rank-2 parameter object, seen through one name, classified by class."""

    name: str
    identity: int
    tensor_role: str
    weight_orientation: str | None = None
    index_axis: int | None = None

    def validate(self) -> str | None:
        """Return the reason this fact cannot be trusted, or ``None``."""
        if not self.name or not self.name.strip():
            return "fact has no parameter name"
        if self.tensor_role not in TENSOR_ROLES:
            return f"unknown tensor role {self.tensor_role!r}"
        if self.weight_orientation is not None and self.weight_orientation not in ORIENTATIONS:
            return f"unknown weight orientation {self.weight_orientation!r}"
        required = ROLE_ORIENTATIONS.get(self.tensor_role)
        if required is not None and self.weight_orientation != required:
            return (
                f"role {self.tensor_role!r} requires weight_orientation {required!r}, "
                f"got {self.weight_orientation!r}"
            )
        if self.tensor_role in INDEXED_ROLES:
            if self.weight_orientation is not None:
                return (
                    f"role {self.tensor_role!r} is indexed, not multiplied; it must not "
                    "declare a projection orientation"
                )
            if self.index_axis not in (0, 1):
                return f"role {self.tensor_role!r} requires index_axis 0 or 1"
        elif self.index_axis is not None:
            return f"index_axis is only meaningful for {INDEXED_ROLES}, not {self.tensor_role!r}"
        return None


@dataclass(frozen=True)
class OracleEntry:
    """What the model definition states about one stored tensor name."""

    name: str
    canonical_name: str
    aliases: tuple[str, ...]
    identity: int
    tensor_roles: tuple[str, ...]
    weight_orientation: str | None
    index_axis: int | None
    # (alias, role) for each observation folded into this identity. The role set
    # says what the parameter is reached as; the pairing says which name carries
    # which claim, which is the part a tied table and head cannot recover from a
    # set. Ordered by alias.
    role_evidence: tuple[tuple[str, str], ...] = ()

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "tensor_roles": list(self.tensor_roles),
            "weight_orientation": self.weight_orientation,
            "index_axis": self.index_axis,
            "role_evidence": [
                {"alias": alias, "tensor_role": role} for alias, role in self.role_evidence
            ],
        }


@dataclass(frozen=True)
class UnresolvedTensor:
    """A tensor the model definition does not describe unambiguously."""

    name: str
    reason: str

    def as_record(self) -> dict[str, Any]:
        return {"name": self.name, "reason": self.reason}


class ArchitectureOracle:
    """Identity-resolved view of a model definition's rank-2 weights."""

    def __init__(
        self,
        entries: Iterable[OracleEntry],
        unresolved: Iterable[UnresolvedTensor] = (),
        model_type: str | None = None,
    ) -> None:
        self.entries: dict[str, OracleEntry] = {entry.name: entry for entry in entries}
        self.unresolved: dict[str, UnresolvedTensor] = {
            item.name: item for item in unresolved
        }
        self.model_type = model_type

    def expectation(self, name: str) -> OracleEntry | None:
        """Exact-name lookup. Names are never matched by suffix or pattern; an
        aliased parameter is found under any of its names because identity
        grouping registered every one of them."""
        return self.entries.get(name)

    def aliases_of(self, name: str) -> tuple[str, ...]:
        entry = self.entries.get(name)
        return () if entry is None else entry.aliases

    def role_evidence_of(self, name: str) -> tuple[tuple[str, str], ...]:
        """Which alias of this parameter carries which role, in alias order."""
        entry = self.entries.get(name)
        return () if entry is None else entry.role_evidence

    @property
    def shared_groups(self) -> dict[str, tuple[str, ...]]:
        """Canonical name -> aliases, for identities reached under several names."""
        groups: dict[str, tuple[str, ...]] = {}
        for entry in self.entries.values():
            if len(entry.aliases) > 1:
                groups[entry.canonical_name] = entry.aliases
        return dict(sorted(groups.items()))

    def to_record(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "tensors_described": len(self.entries),
            "distinct_identities": len({entry.identity for entry in self.entries.values()}),
            "shared_groups": {
                name: list(aliases) for name, aliases in self.shared_groups.items()
            },
            "unresolved": [
                item.as_record()
                for item in sorted(self.unresolved.values(), key=lambda item: item.name)
            ],
        }


def _unresolved_group(members: Sequence[WeightFact], reason: str) -> list[UnresolvedTensor]:
    return [UnresolvedTensor(fact.name, reason) for fact in members]


def build_architecture_oracle(
    facts: Sequence[WeightFact], model_type: str | None = None
) -> ArchitectureOracle:
    """Fold per-name observations into per-identity knowledge.

    Every deviation from a single, safe reading of one parameter object is
    reported as unresolved rather than repaired by inference.
    """
    unresolved: list[UnresolvedTensor] = []
    accepted: list[WeightFact] = []
    for fact in facts:
        reason = fact.validate()
        if reason is None:
            accepted.append(fact)
        else:
            unresolved.append(UnresolvedTensor(fact.name, reason))

    # One name claimed by two parameter objects is ambiguous: nothing says which
    # object a container's record of that name describes, so neither is used.
    owners_by_name: dict[str, set[int]] = {}
    for fact in accepted:
        owners_by_name.setdefault(fact.name, set()).add(fact.identity)
    ambiguous = {name for name, owners in owners_by_name.items() if len(owners) > 1}

    groups: dict[int, list[WeightFact]] = {}
    for fact in accepted:
        groups.setdefault(fact.identity, []).append(fact)

    entries: list[OracleEntry] = []
    for identity in sorted(groups):
        members = sorted(groups[identity], key=lambda fact: fact.name)
        names = tuple(sorted({fact.name for fact in members}))
        if ambiguous.intersection(names):
            unresolved.extend(
                _unresolved_group(
                    members, "this parameter name is claimed by more than one tensor object"
                )
            )
            continue
        orientations = sorted(
            {fact.weight_orientation for fact in members if fact.weight_orientation}
        )
        if len(orientations) > 1:
            unresolved.extend(
                _unresolved_group(
                    members,
                    "one tensor object is reached as projections with conflicting layouts "
                    f"{orientations}",
                )
            )
            continue
        axes = sorted({fact.index_axis for fact in members if fact.index_axis is not None})
        if len(axes) > 1:
            unresolved.extend(
                _unresolved_group(
                    members, f"one tensor object states conflicting index axes {axes}"
                )
            )
            continue
        roles = tuple(sorted({fact.tensor_role for fact in members}))
        orientation = orientations[0] if orientations else None
        axis = axes[0] if axes else None
        evidence = tuple(sorted((fact.name, fact.tensor_role) for fact in members))
        for name in names:
            entries.append(
                OracleEntry(
                    name=name,
                    canonical_name=names[0],
                    aliases=names,
                    identity=identity,
                    tensor_roles=roles,
                    weight_orientation=orientation,
                    index_axis=axis,
                    role_evidence=evidence,
                )
            )

    deduped: dict[str, UnresolvedTensor] = {}
    for item in unresolved:
        deduped.setdefault(item.name, item)
    return ArchitectureOracle(
        sorted(entries, key=lambda entry: entry.name),
        sorted(deduped.values(), key=lambda item: item.name),
        model_type,
    )


def classify_module_weight(
    name: str, module: Any, classes: dict[str, Any]
) -> WeightFact | None:
    """Classify one module's ``weight`` into a fact, given the classes to test.

    ``classes`` carries the module classes to check against, so a caller with
    ``torch``/``transformers`` loaded passes the real ones and a test passes
    stand-ins. Only ``nn.Linear``, ``Conv1D`` and ``nn.Embedding`` are
    recognised; every other module returns ``None`` and the caller records the
    tensor as unknown instead of guessing a role from its name or shape.
    """
    weight = getattr(module, "weight", None)
    if weight is None or getattr(weight, "ndim", None) != 2:
        return None
    conv1d = classes.get("conv1d")
    linear = classes.get("linear")
    embedding = classes.get("embedding")
    if conv1d is not None and isinstance(module, conv1d):
        return WeightFact(name, id(weight), "conv1d_projection", "conv1d_in_out")
    if linear is not None and isinstance(module, linear):
        return WeightFact(name, id(weight), "linear_projection", "linear_out_in")
    if embedding is not None and isinstance(module, embedding):
        # HF stores an embedding table as [num_embeddings, embedding_dim], so the
        # indexed axis is axis 0. That is a documented property of the class, not
        # an inference from the matrix's dimensions.
        return WeightFact(name, id(weight), "embedding_table", None, 0)
    return None
