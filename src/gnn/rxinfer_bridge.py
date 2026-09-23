"""Bayes-net <-> RxInfer.jl bridge over the daf-jev GraphSpec interchange.

This module is the GNN-side half of the pinned cross-repo contract with
daf-jev (``dafjev.bayesnet/1`` GraphSpec JSON — see the daf-jev contract
section 3). It consumes and produces GraphSpec JSON, parses a minimal
Bayes-net-oriented ``.gnn`` markdown subset into a GraphSpec, and emits a
standalone deterministic RxInfer.jl script (``@model`` with one
``DiscreteTransition`` factor per CPT) that loads the JSON, conditions on
evidence, and prints marginals.

Jev integration points (daf-jev is the Jev client; GNN is the engine):

* **upstream** — CPTs and network structure are produced by daf-jev
  elicitation (``propose_structure`` + ``elicit_cpts``) and exchanged as
  GraphSpec JSON (format ``dafjev.bayesnet/1``). ``load_graphspec`` /
  ``load_graphspec_file`` consume that interchange; ``parse_gnn_subset``
  and ``render_gnn_subset`` are the GNN-flavored authoring surface.
* **within** — RESERVED. The optional GraphSpec top-level field
  ``jev_factors`` is reserved for Jev-derived factors (zero-shot
  probabilistic factors injected as extra factors at inference time).
  Its semantics are intentionally unpinned: the loader validates only
  the outer shape (a list of string-keyed objects) and preserves the
  entries verbatim through ``to_json``; ``emit_rxinfer_jl`` emits a
  comment noting the reserved field and otherwise ignores it. Any use
  of ``jev_factors`` must be coordinated across both repos in one wave.
* **downstream** — the emitted script prints posterior marginals for the
  non-evidence variables and can write a posteriors sidecar JSON
  (``--out FILE``, format ``dafjev.bayesnet-posteriors/1`` — a plain
  sidecar, deliberately NOT the pinned GraphSpec schema). Re-asking
  daf-jev for follow-up evidence queries then goes through the daf-jev
  CLI (see ``examples/rxinfer/README.md``).
  ``parse_marginals`` parses that printed block back into
  ``{key: {state: probability}}`` and ``write_marginals`` serializes it as
  a ``gnn.marginals/1`` JSON sidecar for daf-jev calibration/re-ask.

Validation here is a deliberate duplicate of the daf-jev ``BayesNet``
rules (this repo must not import ``daf_jev``): unknown edge endpoints,
duplicate edges, self-loops, cycles, missing/duplicate CPTs, CPT parent
set AND order mismatch, non-canonical row order, unknown assignment
labels, wrong row count, non-finite/negative probabilities, and row sums
outside 1e-6 of 1.0 are all rejected with ``ValueError`` messages naming
the context and the offending value (fail-closed).

Canonical CPT row order is the parent-assignment odometer over each
parent's state list, first parent slowest (``itertools.product`` order)
— identical to the daf-jev pinned ordering.

The ``.gnn`` subset accepted by ``parse_gnn_subset`` is documented on
that function; full GNN pipeline files (POMDP tensor blocks and the 25
steps) are out of scope — unknown constructs are rejected with
actionable messages rather than partially parsed.

RxInfer note (validated empirically against RxInfer 5.5.0 and 5.5.2):
GraphPPL requires each ``@model`` argument to be supplied exactly once.
The emitted models keep every latent free and add one observation
interface per variable (``e_<key> ~ DiscreteTransition(<key>,
EYE_<key>)``); evidence is a one-hot through ``data``, unobserved
interfaces go through ``predictvars``/``missing``. Single-parent nets run
end-to-end with correct posteriors; nets containing a multi-parent
``DiscreteTransition`` hit a ReactiveMP structured-rule limitation (see
``examples/rxinfer/README.md`` for the verified gap matrix).
The learning variant replaces fixed CPT arguments with latent
``Dirichlet``/``DirichletCollection`` priors plus the mean-field cut and
marginal initialization RxInfer requires for learning transition tensors.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

__all__ = [
    "GRAPH_SPEC_FORMAT",
    "MARGINALS_FORMAT",
    "GraphVariable",
    "GraphEdge",
    "GraphCPT",
    "GraphSpec",
    "load_graphspec",
    "load_graphspec_file",
    "parse_gnn_subset",
    "render_gnn_subset",
    "emit_rxinfer_jl",
    "parse_marginals",
    "write_marginals",
]

logger = logging.getLogger(__name__)

#: Cross-repo interchange format string; changing it requires both repos
#: (daf-jev and GNN) in the same wave.
GRAPH_SPEC_FORMAT: Final[str] = "dafjev.bayesnet/1"

#: Downstream marginals sidecar format (parse_marginals output serialized
#: by write_marginals; daf-jev consumes it for calibration/re-ask).
#: Changing it requires both repos (daf-jev and GNN) in the same wave.
MARGINALS_FORMAT: Final[str] = "dafjev.bayesnet-posteriors/1"

_ALLOWED_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {"format", "variables", "edges", "cpts", "jev_factors"}
)
_KEY_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_ROW_SUM_TOLERANCE: Final[float] = 1e-6

#: Julia reserved words that cannot be bare identifiers; a variable key
#: colliding with one (after sanitization) is prefixed with ``x_``.
_JULIA_RESERVED: Final[frozenset[str]] = frozenset(
    {
        "abstract",
        "as",
        "baremodule",
        "begin",
        "break",
        "catch",
        "const",
        "continue",
        "do",
        "else",
        "elseif",
        "end",
        "export",
        "false",
        "finally",
        "for",
        "function",
        "global",
        "if",
        "import",
        "in",
        "let",
        "local",
        "macro",
        "module",
        "mutable",
        "new",
        "primitive",
        "quote",
        "return",
        "struct",
        "true",
        "try",
        "type",
        "using",
        "where",
        "while",
    }
)


@dataclass(frozen=True)
class GraphVariable:
    """One discrete variable of a GraphSpec Bayes net."""

    key: str
    description: str
    states: tuple[str, ...]


@dataclass(frozen=True)
class GraphEdge:
    """One directed edge ``parent -> child``."""

    parent: str
    child: str


@dataclass(frozen=True)
class GraphCPT:
    """Conditional probability table for one child variable.

    ``rows`` holds ``(assignment, probabilities)`` pairs where
    ``assignment`` maps each parent key to a state label and
    ``probabilities`` is ordered by the child's state list. Rows are
    stored in the canonical parent-assignment odometer order.
    """

    child: str
    parents: tuple[str, ...]
    rows: tuple[tuple[Mapping[str, str], tuple[float, ...]], ...]


@dataclass(frozen=True)
class GraphSpec:
    """A validated discrete Bayes net in GraphSpec interchange shape.

    ``jev_factors`` is the RESERVED 'within' integration field (see the
    module docstring): Jev-derived factors preserved verbatim, not used
    by the emitter. Construct via :func:`load_graphspec`,
    :func:`parse_gnn_subset`, or directly (direct construction runs
    :meth:`validate`).
    """

    variables: tuple[GraphVariable, ...]
    edges: tuple[GraphEdge, ...]
    cpts: Mapping[str, GraphCPT]
    jev_factors: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    # -- lookups ---------------------------------------------------------

    def variable(self, key: str) -> GraphVariable:
        for var in self.variables:
            if var.key == key:
                return var
        raise ValueError(
            f"GraphSpec variables: unknown variable key {key!r} "
            f"(known keys: {[v.key for v in self.variables]!r})"
        )

    def parents_of(self, key: str) -> tuple[str, ...]:
        return tuple(edge.parent for edge in self.edges if edge.child == key)

    def children_of(self, key: str) -> tuple[str, ...]:
        return tuple(edge.child for edge in self.edges if edge.parent == key)

    def _states_map(self) -> dict[str, tuple[str, ...]]:
        return {var.key: var.states for var in self.variables}

    def topological_order(self) -> tuple[str, ...]:
        """Kahn's algorithm with insertion-order tiebreak (deterministic)."""
        keys = [v.key for v in self.variables]
        parent_map = {
            key: tuple(e.parent for e in self.edges if e.child == key) for key in keys
        }
        return _topological_order(keys, parent_map)

    # -- validation ------------------------------------------------------

    def validate(self) -> None:
        """Re-check internal consistency; raise ``ValueError`` on violation."""
        keys = [var.key for var in self.variables]
        if not keys:
            raise ValueError("GraphSpec variables: at least one variable is required")
        seen: set[str] = set()
        for var in self.variables:
            if not _KEY_RE.match(var.key):
                raise ValueError(
                    f"GraphSpec variables: invalid key {var.key!r} "
                    "(must match [A-Za-z_][A-Za-z0-9_-]*)"
                )
            if var.key in seen:
                raise ValueError(f"GraphSpec variables: duplicate key {var.key!r}")
            seen.add(var.key)
            if len(var.states) < 2:
                raise ValueError(
                    f"GraphSpec variable {var.key}: needs at least 2 states, "
                    f"got {list(var.states)!r}"
                )
            if len(set(var.states)) != len(var.states):
                raise ValueError(
                    f"GraphSpec variable {var.key}: duplicate state labels "
                    f"{list(var.states)!r}"
                )
            for state in var.states:
                if not isinstance(state, str) or not state:
                    raise ValueError(
                        f"GraphSpec variable {var.key}: state labels must be "
                        f"non-empty strings, got {state!r}"
                    )
            if not isinstance(var.description, str):
                raise ValueError(
                    f"GraphSpec variable {var.key}: description must be a "
                    f"string, got {type(var.description).__name__}"
                )

        # Edge structure: unknown endpoints, self-loops, duplicates, cycles.
        _validate_edges_and_order(keys, self.edges)

        states_map = self._states_map()
        parent_map = {v.key: self.parents_of(v.key) for v in self.variables}
        if set(self.cpts) != seen:
            missing = sorted(seen - set(self.cpts))
            extra = sorted(set(self.cpts) - seen)
            raise ValueError(
                f"GraphSpec cpts: CPT keys must equal the variable keys "
                f"(missing: {missing!r}, unexpected: {extra!r})"
            )
        for child in self.variables:  # insertion order = variables order
            cpt = self.cpts[child.key]
            expected_parents = parent_map[child.key]
            if tuple(cpt.parents) != expected_parents:
                raise ValueError(
                    f"GraphSpec cpts[{child.key}]: parents must equal the "
                    f"graph parents in edge order {list(expected_parents)!r}, "
                    f"got {list(cpt.parents)!r}"
                )
            if cpt.child != child.key:
                raise ValueError(
                    f"GraphSpec cpts[{child.key}]: child field "
                    f"{cpt.child!r} does not match the mapping key"
                )
            canonical = _canonical_assignments(
                expected_parents,
                [states_map[p] for p in expected_parents],
            )
            if len(cpt.rows) != len(canonical):
                raise ValueError(
                    f"GraphSpec cpts[{child.key}]: expected {len(canonical)} "
                    f"rows (one per parent assignment), got {len(cpt.rows)}"
                )
            for i, ((assignment, probs), expected) in enumerate(
                zip(cpt.rows, canonical)
            ):
                if dict(assignment) != dict(expected):
                    raise ValueError(
                        f"GraphSpec cpts[{child.key}]: row {i} assignment "
                        f"{dict(assignment)!r} violates the canonical order "
                        f"(expected {dict(expected)!r}; rows must be in "
                        "parent-assignment odometer order, first parent "
                        "slowest)"
                    )
                for parent, label in assignment.items():
                    if label not in states_map[parent]:
                        raise ValueError(
                            f"GraphSpec cpts[{child.key}]: row {i} unknown "
                            f"state {label!r} for parent {parent!r} "
                            f"(states: {list(states_map[parent])!r})"
                        )
                if len(probs) != len(child.states):
                    raise ValueError(
                        f"GraphSpec cpts[{child.key}]: row {i} has "
                        f"{len(probs)} probabilities, expected "
                        f"{len(child.states)} (child states: "
                        f"{list(child.states)!r})"
                    )
                _check_probabilities(child.key, i, probs)

        for factor in self.jev_factors:
            if not isinstance(factor, Mapping) or not all(
                isinstance(k, str) for k in factor
            ):
                raise ValueError(
                    "GraphSpec jev_factors (RESERVED field): entries must be "
                    f"objects with string keys, got {factor!r}"
                )

    # -- interchange -----------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Serialize to the pinned GraphSpec JSON schema (lossless)."""
        doc: dict[str, Any] = {
            "format": GRAPH_SPEC_FORMAT,
            "variables": [
                {
                    "key": var.key,
                    "description": var.description,
                    "states": list(var.states),
                }
                for var in self.variables
            ],
            "edges": [
                {"parent": edge.parent, "child": edge.child} for edge in self.edges
            ],
            "cpts": {
                child.key: {
                    "child": child.key,
                    "parents": list(cpt.parents),
                    "rows": [
                        {
                            "assignment": dict(assignment),
                            "probabilities": list(probs),
                        }
                        for assignment, probs in cpt.rows
                    ],
                }
                for child in self.variables
                for cpt in (self.cpts[child.key],)
            },
        }
        if self.jev_factors:
            doc["jev_factors"] = [dict(factor) for factor in self.jev_factors]
        return doc


def _check_probabilities(child: str, row: int, probs: tuple[float, ...]) -> None:
    """Fail-closed probability checks (finite, >= 0, sums to 1 within 1e-6)."""
    for j, p in enumerate(probs):
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise ValueError(
                f"GraphSpec cpts[{child}]: row {row} probability {j} must be "
                f"a number, got {p!r}"
            )
        if not math.isfinite(float(p)):
            raise ValueError(
                f"GraphSpec cpts[{child}]: row {row} probability {j} is not "
                f"finite: {p!r}"
            )
        if p < 0:
            raise ValueError(
                f"GraphSpec cpts[{child}]: row {row} probability {j} is negative: {p!r}"
            )
    total = float(sum(probs))
    if abs(total - 1.0) > _ROW_SUM_TOLERANCE:
        raise ValueError(
            f"GraphSpec cpts[{child}]: row {row} probabilities sum to "
            f"{total!r}, expected 1.0 within {_ROW_SUM_TOLERANCE}"
        )


def _canonical_assignments(
    parents: tuple[str, ...], states_lists: list[tuple[str, ...]]
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Parent-assignment odometer order; first parent slowest."""
    out: list[tuple[tuple[str, str], ...]] = []
    for combo in itertools.product(*states_lists):
        out.append(tuple(zip(parents, combo)))
    return tuple(out)


def _topological_order(
    keys: list[str], parent_map: Mapping[str, tuple[str, ...]]
) -> tuple[str, ...]:
    """Kahn's algorithm with insertion-order tiebreak (deterministic)."""
    remaining = {key: len(parent_map[key]) for key in keys}
    children: dict[str, list[str]] = {key: [] for key in keys}
    for child, parents in parent_map.items():
        for parent in parents:
            children[parent].append(child)
    order: list[str] = []
    ready = [key for key in keys if remaining[key] == 0]
    while ready:
        key = ready.pop(0)
        order.append(key)
        for child in children[key]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    if len(order) != len(keys):
        cycle = sorted(set(keys) - set(order))
        raise ValueError(f"GraphSpec edges: cycle detected involving {cycle!r}")
    return tuple(order)


def _validate_edges_and_order(
    keys: list[str], edges: tuple[GraphEdge, ...]
) -> dict[str, tuple[str, ...]]:
    """Validate edge structure (endpoints, self-loops, duplicates, acyclic)
    and return the child -> parent-tuple map in edge declaration order."""
    key_set = set(keys)
    edges_seen: set[tuple[str, str]] = set()
    parent_map: dict[str, tuple[str, ...]] = dict.fromkeys(keys, ())
    for edge in edges:
        for endpoint, role in ((edge.parent, "parent"), (edge.child, "child")):
            if endpoint not in key_set:
                raise ValueError(
                    f"GraphSpec edges: unknown {role} {endpoint!r} on edge "
                    f"{edge.parent!r} -> {edge.child!r}"
                )
        if edge.parent == edge.child:
            raise ValueError(f"GraphSpec edges: self-loop on {edge.parent!r}")
        pair = (edge.parent, edge.child)
        if pair in edges_seen:
            raise ValueError(
                f"GraphSpec edges: duplicate edge {edge.parent!r} -> {edge.child!r}"
            )
        edges_seen.add(pair)
        parent_map[edge.child] = parent_map[edge.child] + (edge.parent,)
    _topological_order(keys, parent_map)
    return parent_map


# ---------------------------------------------------------------------------
# GraphSpec JSON interchange (pinned dafjev.bayesnet/1)
# ---------------------------------------------------------------------------


def load_graphspec(data: object) -> GraphSpec:
    """Load and fully validate a GraphSpec JSON document (duplicated rules)."""
    if not isinstance(data, Mapping):
        raise ValueError(
            f"GraphSpec: expected a JSON object, got {type(data).__name__}"
        )
    fmt = data.get("format")
    if fmt != GRAPH_SPEC_FORMAT:
        raise ValueError(
            f"GraphSpec format: expected {GRAPH_SPEC_FORMAT!r}, got {fmt!r}"
        )
    unknown = sorted(set(data) - _ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(
            f"GraphSpec: unknown top-level keys {unknown!r} "
            f"(allowed: {sorted(_ALLOWED_TOP_LEVEL_KEYS)!r})"
        )
    raw_vars = data.get("variables")
    if not isinstance(raw_vars, list) or not raw_vars:
        raise ValueError("GraphSpec variables: must be a non-empty list of objects")
    variables: list[GraphVariable] = []
    for i, raw in enumerate(raw_vars):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"GraphSpec variables[{i}]: expected an object, got {raw!r}"
            )
        key = raw.get("key")
        if not isinstance(key, str) or not _KEY_RE.match(key):
            raise ValueError(
                f"GraphSpec variables[{i}]: invalid key {key!r} "
                "(must match [A-Za-z_][A-Za-z0-9_-]*)"
            )
        if any(v.key == key for v in variables):
            raise ValueError(f"GraphSpec variables[{i}]: duplicate key {key!r}")
        description = raw.get("description")
        if not isinstance(description, str):
            raise ValueError(
                f"GraphSpec variables[{i}] ({key}): description must be a "
                f"string, got {type(description).__name__}"
            )
        states = raw.get("states")
        if (
            not isinstance(states, list)
            or len(states) < 2
            or not all(isinstance(s, str) and s for s in states)
        ):
            raise ValueError(
                f"GraphSpec variables[{i}] ({key}): states must be a list of "
                f"at least 2 non-empty strings, got {states!r}"
            )
        if len(set(states)) != len(states):
            raise ValueError(
                f"GraphSpec variables[{i}] ({key}): duplicate state labels {states!r}"
            )
        variables.append(
            GraphVariable(key=key, description=description, states=tuple(states))
        )

    raw_edges = data.get("edges")
    if not isinstance(raw_edges, list):
        raise ValueError("GraphSpec edges: must be a list of objects")
    edges: list[GraphEdge] = []
    for i, raw in enumerate(raw_edges):
        if not isinstance(raw, Mapping):
            raise ValueError(f"GraphSpec edges[{i}]: expected an object, got {raw!r}")
        parent = raw.get("parent")
        child = raw.get("child")
        if not isinstance(parent, str) or not _KEY_RE.match(parent):
            raise ValueError(
                f"GraphSpec edges[{i}]: invalid parent {parent!r} "
                "(must match [A-Za-z_][A-Za-z0-9_-]*)"
            )
        if not isinstance(child, str) or not _KEY_RE.match(child):
            raise ValueError(
                f"GraphSpec edges[{i}]: invalid child {child!r} "
                "(must match [A-Za-z_][A-Za-z0-9_-]*)"
            )
        edges.append(GraphEdge(parent=parent, child=child))

    keys = [v.key for v in variables]
    parent_map: dict[str, tuple[str, ...]] = _validate_edges_and_order(
        keys, tuple(edges)
    )

    raw_cpts = data.get("cpts")
    if not isinstance(raw_cpts, Mapping):
        raise ValueError("GraphSpec cpts: must be an object keyed by child")
    states_map = {v.key: v.states for v in variables}
    cpts: dict[str, GraphCPT] = {}
    for key in (v.key for v in variables):  # deterministic build order
        raw = raw_cpts.get(key)
        if not isinstance(raw, Mapping):
            raise ValueError(f"GraphSpec cpts[{key}]: missing or non-object CPT entry")
        child = raw.get("child")
        if child != key:
            raise ValueError(
                f"GraphSpec cpts[{key}]: child field {child!r} does not "
                "match the mapping key"
            )
        parents = raw.get("parents")
        if not isinstance(parents, list) or not all(
            isinstance(p, str) for p in parents
        ):
            raise ValueError(
                f"GraphSpec cpts[{key}]: parents must be a list of strings, "
                f"got {parents!r}"
            )
        if tuple(parents) != parent_map[key]:
            raise ValueError(
                f"GraphSpec cpts[{key}]: parents must equal the graph "
                f"parents in edge order {list(parent_map[key])!r}, "
                f"got {parents!r}"
            )
        raw_rows = raw.get("rows")
        if not isinstance(raw_rows, list):
            raise ValueError(f"GraphSpec cpts[{key}]: rows must be a list of objects")
        canonical = _canonical_assignments(
            tuple(parents), [states_map[p] for p in parents]
        )
        if len(raw_rows) != len(canonical):
            raise ValueError(
                f"GraphSpec cpts[{key}]: expected {len(canonical)} rows (one "
                f"per parent assignment), got {len(raw_rows)}"
            )
        rows: list[tuple[dict[str, str], tuple[float, ...]]] = []
        for i, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, Mapping):
                raise ValueError(
                    f"GraphSpec cpts[{key}]: row {i} expected an object, "
                    f"got {raw_row!r}"
                )
            assignment = raw_row.get("assignment")
            probs_raw = raw_row.get("probabilities")
            if not isinstance(assignment, Mapping):
                raise ValueError(
                    f"GraphSpec cpts[{key}]: row {i} assignment must be an "
                    f"object, got {assignment!r}"
                )
            if dict(assignment) != dict(canonical[i]):
                raise ValueError(
                    f"GraphSpec cpts[{key}]: row {i} assignment "
                    f"{dict(assignment)!r} violates the canonical order "
                    f"(expected {dict(canonical[i])!r}; rows must be in "
                    "parent-assignment odometer order, first parent slowest)"
                )
            if not isinstance(probs_raw, list) or not probs_raw:
                raise ValueError(
                    f"GraphSpec cpts[{key}]: row {i} probabilities must be a "
                    f"non-empty list of numbers, got {probs_raw!r}"
                )
            if len(probs_raw) != len(states_map[key]):
                raise ValueError(
                    f"GraphSpec cpts[{key}]: row {i} has {len(probs_raw)} "
                    f"probabilities, expected {len(states_map[key])} "
                    f"(child states: {list(states_map[key])!r})"
                )
            for j, p in enumerate(probs_raw):
                if isinstance(p, bool) or not isinstance(p, (int, float)):
                    raise ValueError(
                        f"GraphSpec cpts[{key}]: row {i} probability {j} is "
                        f"not a number, got {p!r}"
                    )
            probs = tuple(float(p) for p in probs_raw)
            _check_probabilities(key, i, probs)
            rows.append((dict(assignment), probs))
        cpts[key] = GraphCPT(child=key, parents=tuple(parents), rows=tuple(rows))

    jev_factors_raw = data.get("jev_factors", [])
    if not isinstance(jev_factors_raw, list):
        raise ValueError(
            "GraphSpec jev_factors (RESERVED field): must be a list of "
            f"objects, got {jev_factors_raw!r}"
        )
    for factor in jev_factors_raw:
        if not isinstance(factor, Mapping) or not all(
            isinstance(k, str) for k in factor
        ):
            raise ValueError(
                "GraphSpec jev_factors (RESERVED field): entries must be "
                f"objects with string keys, got {factor!r}"
            )
    jev_factors = tuple(dict(f) for f in jev_factors_raw)

    spec = GraphSpec(
        variables=tuple(variables),
        edges=tuple(edges),
        cpts=cpts,
        jev_factors=jev_factors,
    )
    logger.info(
        "loaded GraphSpec: variables=%d edges=%d jev_factors=%d",
        len(spec.variables),
        len(spec.edges),
        len(spec.jev_factors),
    )
    return spec


def load_graphspec_file(path: str | os.PathLike[str]) -> GraphSpec:
    """Load a GraphSpec JSON file from disk."""
    with open(path, encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"GraphSpec file {os.fspath(path)!r}: invalid JSON ({exc})"
            ) from exc
    return load_graphspec(data)


# ---------------------------------------------------------------------------
# Minimal .gnn markdown subset (authoring surface)
# ---------------------------------------------------------------------------

_GNN_SECTION_RE: Final[re.Pattern[str]] = re.compile(r"^##\s+(.+?)\s*$")
_GNN_SUBSECTION_RE: Final[re.Pattern[str]] = re.compile(r"^###\s+(.+?)\s*$")
_GNN_EDGE_RE: Final[re.Pattern[str]] = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_-]*?)\s*(?:->|>)\s*([A-Za-z_][A-Za-z0-9_-]*)\s*$"
)
_GNN_CPT_OPEN_RE: Final[re.Pattern[str]] = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*\{\s*$"
)
_GNN_ROW_RE: Final[re.Pattern[str]] = re.compile(
    r"^\((.*)\)\s*=\s*\(([^)]*)\)\s*,?\s*$"
)
_GNN_STATES_RE: Final[re.Pattern[str]] = re.compile(r"^(?:\[Discrete\]\s*)?(.+)$")


def parse_gnn_subset(text: str) -> GraphSpec:
    """Parse the minimal Bayes-net ``.gnn`` markdown subset into a GraphSpec.

    Accepted subset (anything else is rejected with an actionable message):

    * ``# ...`` comment lines and blank lines anywhere.
    * ``## <key>`` variable blocks with ``### Description`` (free text) and
      ``### States`` (one ``[Discrete] label, label, ...`` line). A
      ``## Variables`` container with ``### <key>`` blocks is also accepted
      (inside a container, any ``### <key>`` that is not ``Description`` or
      ``States`` starts the next variable block).
    * ``## Connections`` with one ``parent>child`` (native GNN arrow) or
      ``parent->child`` line per edge, in declaration order.
    * ``## InitialParameterization`` with one CPT block per variable:
      ``name={`` ... rows ... ``}`` where each row is
      ``(p1=v1, p2=v2) = (0.1, 0.9)`` (assignment in graph parent order,
      probabilities in child state order) or ``() = (0.99, 0.01)`` for a
      prior. Rows may appear in any order; they are canonicalized into
      GraphSpec odometer order.

    Unknown headings, malformed rows, and undeclared endpoints fail
    closed with the line number and offending value.
    """
    variables: dict[str, dict[str, Any]] = {}
    var_order: list[str] = []
    edges: list[tuple[int, str, str]] = []
    cpts: dict[str, dict[str, Any]] = {}

    mode = "none"  # none | variables | varblock | connections | params
    current_key: str | None = None
    description_lines: list[str] = []
    states_line: str | None = None
    cpt_child: str | None = None
    cpt_rows: list[tuple[tuple[tuple[str, str], ...], tuple[float, ...]]] = []
    in_container = False

    def close_variable() -> None:
        nonlocal current_key, description_lines, states_line
        if current_key is None:
            return
        if states_line is None:
            raise ValueError(
                f".gnn variable block '## {current_key}': missing '### States'"
            )
        match = _GNN_STATES_RE.match(states_line)
        if not match:
            raise ValueError(
                f".gnn variable block '## {current_key}': unparsable states "
                f"line {states_line!r} (expected '[Discrete] s1, s2, ...')"
            )
        raw_states = match.group(1)
        states = tuple(s.strip() for s in raw_states.split(",") if s.strip()) or tuple(
            raw_states.split()
        )
        variables[current_key] = {
            "description": " ".join(s.strip() for s in description_lines).strip(),
            "states": states,
        }
        var_order.append(current_key)
        current_key = None
        description_lines = []
        states_line = None

    def close_cpt() -> None:
        nonlocal cpt_child, cpt_rows
        if cpt_child is None:
            return
        cpts[cpt_child] = {
            "assignments": [dict(a) for a, _ in cpt_rows],
            "probabilities": [p for _, p in cpt_rows],
        }
        cpt_child = None
        cpt_rows = []

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#") and not line.startswith("##"):
            # Single-# lines are comments; ## / ### are structural headings.
            continue

        heading = _GNN_SECTION_RE.match(line)
        if heading:
            close_variable()
            close_cpt()
            name = heading.group(1)
            if name in ("Connections", "InitialParameterization"):
                mode = "connections" if name == "Connections" else "params"
                current_key = None
                in_container = False
            elif name == "Variables":
                mode = "variables"
                current_key = None
                in_container = True
            elif mode in ("connections", "params"):
                raise ValueError(
                    f".gnn line {lineno}: unexpected block '## {name}' inside "
                    f"the {mode} section"
                )
            else:
                if not _KEY_RE.match(name):
                    raise ValueError(
                        f".gnn line {lineno}: invalid variable key {name!r} "
                        "(must match [A-Za-z_][A-Za-z0-9_-]*)"
                    )
                if name in variables:
                    raise ValueError(
                        f".gnn line {lineno}: duplicate variable block '## {name}'"
                    )
                mode = "varblock"
                current_key = name
            continue

        subsection = _GNN_SUBSECTION_RE.match(line)
        if subsection:
            name = subsection.group(1)
            if mode == "variables":
                if name in ("Description", "States"):
                    raise ValueError(
                        f".gnn line {lineno}: '### {name}' appears before "
                        "any variable block in the Variables section"
                    )
                if not _KEY_RE.match(name):
                    raise ValueError(
                        f".gnn line {lineno}: invalid variable key {name!r}"
                    )
                if name in variables:
                    raise ValueError(
                        f".gnn line {lineno}: duplicate variable block '### {name}'"
                    )
                mode = "varblock"
                current_key = name
                continue
            if mode == "varblock":
                if name == "Description":
                    description_lines = []
                    continue
                if name == "States":
                    states_line = "__PENDING__"
                    continue
                if in_container:
                    # In a `## Variables` container, `### <key>` starts the
                    # next variable block.
                    close_variable()
                    if name in variables:
                        raise ValueError(
                            f".gnn line {lineno}: duplicate variable block '### {name}'"
                        )
                    current_key = name
                    continue
                raise ValueError(
                    f".gnn line {lineno}: unknown subsection '### {name}' "
                    "(accepted: Description, States)"
                )
            raise ValueError(
                f".gnn line {lineno}: unexpected subsection '### {name}' "
                f"in the {mode} section"
            )

        if mode == "varblock":
            if states_line == "__PENDING__":
                states_line = line
                continue
            description_lines.append(line)
            continue
        if mode == "connections":
            edge_match = _GNN_EDGE_RE.match(line)
            if not edge_match:
                raise ValueError(
                    f".gnn line {lineno}: unparsable edge {line!r} "
                    "(expected 'parent>child' or 'parent->child')"
                )
            edges.append((lineno, edge_match.group(1), edge_match.group(2)))
            continue
        if mode == "params":
            open_match = _GNN_CPT_OPEN_RE.match(line)
            if open_match:
                close_cpt()
                cpt_child = open_match.group(1)
                continue
            if cpt_child is not None:
                if line == "}":
                    close_cpt()
                    continue
                row_match = _GNN_ROW_RE.match(line)
                if not row_match:
                    raise ValueError(
                        f".gnn line {lineno}: unparsable CPT row {line!r} "
                        "(expected '(p1=v1, p2=v2) = (0.1, 0.9)' or "
                        "'() = (0.99, 0.01)')"
                    )
                assignment_part, probs_part = row_match.groups()
                assignment: list[tuple[str, str]] = []
                if assignment_part.strip():
                    for item in assignment_part.split(","):
                        if "=" not in item:
                            raise ValueError(
                                f".gnn line {lineno}: CPT assignment item "
                                f"{item!r} lacks '=' (expected 'parent=state')"
                            )
                        key, label = item.split("=", 1)
                        assignment.append((key.strip(), label.strip()))
                try:
                    probs = tuple(
                        float(p.strip()) for p in probs_part.split(",") if p.strip()
                    )
                except ValueError as exc:
                    raise ValueError(
                        f".gnn line {lineno}: CPT probabilities "
                        f"{probs_part!r} are not all numbers ({exc})"
                    ) from exc
                cpt_rows.append((tuple(assignment), probs))
                continue
            raise ValueError(
                f".gnn line {lineno}: unparsable line {line!r} in the "
                "InitialParameterization section (expected 'name={' to "
                "open a CPT block)"
            )

    close_variable()
    close_cpt()

    if not variables:
        raise ValueError(
            ".gnn: no variable blocks found (expected at least one "
            "'## <key>' block with '### States')"
        )

    variables_out = tuple(
        GraphVariable(
            key=key,
            description=variables[key]["description"],
            states=variables[key]["states"],
        )
        for key in var_order
    )
    keys = set(var_order)
    edges_out = []
    for lineno, parent, child in edges:
        for role, value in (("parent", parent), ("child", child)):
            if value not in keys:
                raise ValueError(
                    f".gnn line {lineno}: edge references undeclared {role} {value!r}"
                )
        edges_out.append(GraphEdge(parent=parent, child=child))
    edges_tuple = tuple(edges_out)

    parent_map: dict[str, tuple[str, ...]] = {
        v.key: tuple(e.parent for e in edges_tuple if e.child == v.key)
        for v in variables_out
    }
    states_map = {v.key: v.states for v in variables_out}
    cpts_out: dict[str, GraphCPT] = {}
    for v in variables_out:
        raw = cpts.get(v.key)
        if raw is None:
            raise ValueError(
                f".gnn: missing CPT block '{v.key}={{...}}' in the "
                "InitialParameterization section"
            )
        parents = parent_map[v.key]
        canonical = _canonical_assignments(parents, [states_map[p] for p in parents])
        rows_raw: list[tuple[dict[str, str], tuple[float, ...]]] = [
            (dict(a), probs)
            for a, probs in zip(raw["assignments"], raw["probabilities"])
        ]
        if len(rows_raw) != len(canonical):
            raise ValueError(
                f".gnn CPT '{v.key}': expected {len(canonical)} rows (one "
                f"per parent assignment), got {len(rows_raw)}"
            )
        lookup: dict[tuple[tuple[str, str], ...], tuple[float, ...]] = {}
        for i, (assign_map, probs) in enumerate(rows_raw):
            if set(assign_map) != set(parents):
                raise ValueError(
                    f".gnn CPT '{v.key}': row {i} assignment "
                    f"{assign_map!r} does not match the declared parents "
                    f"{list(parents)!r}"
                )
            normalized = tuple((p, assign_map[p]) for p in parents)
            if normalized in lookup:
                raise ValueError(
                    f".gnn CPT '{v.key}': duplicate parent assignment {assign_map!r}"
                )
            for parent, label in normalized:
                if label not in states_map[parent]:
                    raise ValueError(
                        f".gnn CPT '{v.key}': row {i} unknown state "
                        f"{label!r} for parent {parent!r} (states: "
                        f"{list(states_map[parent])!r})"
                    )
            if len(probs) != len(v.states):
                raise ValueError(
                    f".gnn CPT '{v.key}': row {i} has {len(probs)} "
                    f"probabilities, expected {len(v.states)} (child "
                    f"states: {list(v.states)!r})"
                )
            _check_probabilities(v.key, i, probs)
            lookup[normalized] = probs
        ordered_rows = []
        for expected in canonical:
            matched_probs = lookup.get(expected)
            if matched_probs is None:
                raise ValueError(
                    f".gnn CPT '{v.key}': missing parent assignment {dict(expected)!r}"
                )
            ordered_rows.append((dict(expected), matched_probs))
        cpts_out[v.key] = GraphCPT(
            child=v.key, parents=parents, rows=tuple(ordered_rows)
        )

    spec = GraphSpec(
        variables=variables_out,
        edges=edges_tuple,
        cpts=cpts_out,
    )
    logger.info(
        "parsed .gnn subset: variables=%d edges=%d",
        len(spec.variables),
        len(spec.edges),
    )
    return spec


def render_gnn_subset(spec: GraphSpec) -> str:
    """Render a GraphSpec back to the minimal ``.gnn`` subset (deterministic).

    ``parse_gnn_subset(render_gnn_subset(spec)) == spec`` holds for every
    validated GraphSpec; rows are emitted in canonical odometer order and
    probabilities with ``repr(float)`` shortest form.
    """
    lines: list[str] = [
        "# Bayes-net .gnn subset for gnn.rxinfer_bridge",
        "# (GraphSpec interchange: dafjev.bayesnet/1; round-trips through",
        "# parse_gnn_subset/render_gnn_subset).",
        "",
    ]
    for var in spec.variables:
        lines.append(f"## {var.key}")
        lines.append("### Description")
        if var.description:
            lines.append(var.description)
        lines.append("### States")
        lines.append(f"[Discrete] {', '.join(var.states)}")
        lines.append("")
    lines.append("## Connections")
    lines.append("")
    for edge in spec.edges:
        lines.append(f"{edge.parent}>{edge.child}")
    lines.append("")
    lines.append("## InitialParameterization")
    lines.append("")
    for var in spec.variables:
        cpt = spec.cpts[var.key]
        lines.append(f"{var.key}={{")
        for assignment, probs in cpt.rows:
            if assignment:
                lhs = ", ".join(
                    f"{parent}={assignment[parent]}" for parent in cpt.parents
                )
            else:
                lhs = ""
            rhs = ", ".join(repr(float(p)) for p in probs)
            lines.append(f"  ({lhs}) = ({rhs})")
        lines.append("}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# RxInfer.jl emitter (deterministic)
# ---------------------------------------------------------------------------


def _julia_ident(key: str, taken: set[str], context: str) -> str:
    """Map a GraphSpec key to a unique Julia identifier (deterministic)."""
    ident = key.replace("-", "_")
    if ident in _JULIA_RESERVED or ident.startswith("_"):
        ident = f"x_{ident}"
    if ident in taken:
        raise ValueError(
            f"emit_rxinfer_jl {context}: keys {key!r} collide on the Julia "
            f"identifier {ident!r} after sanitization"
        )
    taken.add(ident)
    return ident


def emit_rxinfer_jl(spec: GraphSpec, model_name: str = "gnn_bayesnet") -> str:
    """Emit a standalone deterministic RxInfer.jl script for ``spec``.

    The script contains one ``@model`` function with one
    ``Categorical``/``DiscreteTransition`` factor per CPT (Dirichlet
    priors in a learning variant), a fail-closed GraphSpec loader,
    evidence conditioning via per-variable observation interfaces
    (``e_<key> ~ DiscreteTransition(<key>, EYE_<key>)``): every latent stays
    free (its marginal is returned); observed ``e_<key>`` interfaces are
    clamped through ``data`` with a one-hot, unobserved ones are supplied as
    ``predictvars``/``missing`` (GraphPPL requires each ``@model`` argument
    to be supplied exactly once),
    marginal printing in topological order, and an optional posteriors sidecar
    (``--out``). Identical GraphSpec input produces byte-identical output;
    the smoke run in ``examples/rxinfer/README.md`` records the real result.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", model_name):
        raise ValueError(
            f"emit_rxinfer_jl: invalid model_name {model_name!r} "
            "(must be a Julia identifier: [A-Za-z_][A-Za-z0-9_]*)"
        )
    order = spec.topological_order()
    states_map = spec._states_map()
    parent_map = {key: spec.parents_of(key) for key in order}

    idents: dict[str, str] = {}
    taken: set[str] = set()
    for var in spec.variables:
        idents[var.key] = _julia_ident(var.key, taken, "variable keys")
    model_ident_taken: set[str] = set()
    for key in order:
        _julia_ident(key, model_ident_taken, "model variables")

    lines: list[str] = []
    lines.append("#!/usr/bin/env julia")
    lines.append(
        "# RxInfer.jl Bayes-net inference — generated by gnn.rxinfer_bridge (GNN)."
    )
    lines.append(
        "# GraphSpec format: dafjev.bayesnet/1 (pinned cross-repo contract with daf-jev)."
    )
    lines.append("#")
    lines.append(
        "# Jev integration points (upstream / within / downstream; full story in"
    )
    lines.append("# the module docstring of src/gnn/rxinfer_bridge.py and")
    lines.append("# examples/rxinfer/README.md):")
    lines.append("#   upstream   — CPTs and structure produced by daf-jev elicitation")
    lines.append(
        "#                (propose_structure + elicit_cpts), exchanged as this"
    )
    lines.append("#                GraphSpec JSON")
    lines.append(
        '#   within     — RESERVED: the optional GraphSpec "jev_factors" field'
    )
    lines.append(
        "#                carries Jev-derived factors for future use; this script"
    )
    lines.append(
        f"#                ignores it ({'present' if spec.jev_factors else 'absent'} in this input)"
    )
    lines.append("#   downstream — posteriors printed below; --out FILE writes a")
    lines.append("#                posteriors sidecar JSON for re-asking daf-jev")
    lines.append("#                (see examples/rxinfer/README.md)")
    lines.append("#")
    lines.append("# Usage:")
    lines.append(
        "#   julia --project=. <script>.jl <graphspec.json> [--evidence key=state ...] [--out FILE] [--learn]"
    )
    lines.append("#")
    lines.append(
        "# Deterministic: identical GraphSpec input produces identical script text."
    )
    lines.append("")
    lines.append("using RxInfer")
    lines.append("using JSON")
    lines.append("")
    vars_list = ", ".join(f":{idents[key]}" for key in order)
    lines.append(f"const {model_name.upper()}_VARS = [{vars_list}]")
    lines.append(
        f"const {model_name.upper()}_VARS_SET = Set({model_name.upper()}_VARS)"
    )
    lines.append("const CHILD_PARENTS = Dict{Symbol,Vector{Symbol}}(")
    for key in order:
        parents = ", ".join(f":{idents[p]}" for p in parent_map[key])
        lines.append(f"    :{idents[key]} => Symbol[{parents}],")
    lines.append(")")
    for key in order:
        n_states = len(states_map[key])
        lines.append(
            f"const EYE_{idents[key]} = "
            f"[Float64(i == j) for i in 1:{n_states}, j in 1:{n_states}]"
        )
    raw_to_ident = ", ".join(
        f":{raw} => :{ident}" for raw, ident in sorted(idents.items())
    )
    lines.append(f"const RAW_TO_IDENT = Dict{{Symbol,Symbol}}({raw_to_ident})")
    lines.append(
        "const IDENT_TO_RAW = Dict{Symbol,Symbol}(v => k for (k, v) in RAW_TO_IDENT)"
    )
    lines.append("const LEARNING_ITERATIONS = 25")
    lines.append("const LEARNING_PRIOR_SCALE = 10.0")
    lines.append("const LEARNING_ALPHA_EPS = 1e-6")
    lines.append("")

    lines.append('"""')
    lines.append("    load_spec(path) -> (spec, states, tensors)")
    lines.append("")
    lines.append(
        "Load and validate a dafjev.bayesnet/1 GraphSpec JSON; return the parsed"
    )
    lines.append("spec, the states map (Symbol key -> vector of state labels), and the")
    lines.append(
        "CPT tensors (Symbol child -> Array{Float64} with dims (child, p1, p2, ...))."
    )
    lines.append('"""')
    lines.append("function load_spec(path::AbstractString)")
    lines.append('    isfile(path) || error("GraphSpec file not found: $path")')
    lines.append("    spec = JSON.parsefile(path)")
    lines.append('    got = get(spec, "format", nothing)')
    lines.append(
        '    got == "dafjev.bayesnet/1" || error("GraphSpec format: expected \\"dafjev.bayesnet/1\\", got $got")'
    )
    lines.append("    states = Dict{Symbol,Vector{String}}()")
    lines.append('    for var in get(spec, "variables", [])')
    lines.append('        raw_key = Symbol(String(var["key"]))')
    lines.append(
        '        haskey(RAW_TO_IDENT, raw_key) || error("GraphSpec variables: unknown key $raw_key (not in the compiled model)")'
    )
    lines.append("        key = RAW_TO_IDENT[raw_key]")
    lines.append(
        '        haskey(states, key) && error("GraphSpec variables: duplicate key $key")'
    )
    lines.append('        sts = String.(var["states"])')
    lines.append(
        '        length(sts) >= 2 || error("GraphSpec variable $key: needs at least 2 states")'
    )
    lines.append("        states[key] = sts")
    lines.append("    end")
    lines.append("    for key in " + model_name.upper() + "_VARS")
    lines.append(
        '        haskey(states, key) || error("GraphSpec variables: model variable $key missing")'
    )
    lines.append("    end")
    lines.append("    for key in sort!(collect(keys(states)); by=String)")
    lines.append(
        "        key in "
        + model_name.upper()
        + '_VARS_SET || error("GraphSpec variables: unknown key $key not in the compiled model")'
    )
    lines.append("    end")
    lines.append('    cpts = get(spec, "cpts", nothing)')
    lines.append(
        '    cpts isa AbstractDict || error("GraphSpec cpts: must be an object keyed by child")'
    )
    lines.append("    tensors = Dict{Symbol,Any}()")
    lines.append("    for child in " + model_name.upper() + "_VARS")
    lines.append("        parents = CHILD_PARENTS[child]")
    lines.append("        cpt = get(cpts, String(IDENT_TO_RAW[child]), nothing)")
    lines.append(
        '        cpt === nothing && error("GraphSpec cpts: missing CPT for $child")'
    )
    lines.append('        declared = Symbol.(String.(get(cpt, "parents", [])))')
    lines.append(
        '        Tuple(declared) == Tuple(parents) || error("GraphSpec cpts[$child]: parents must be $(parents), got $(declared)")'
    )
    lines.append('        rows = get(cpt, "rows", nothing)')
    lines.append(
        '        rows isa AbstractVector || error("GraphSpec cpts[$child]: rows must be a list")'
    )
    lines.append("        n_child = length(states[child])")
    lines.append("        if isempty(parents)")
    lines.append(
        '            length(rows) == 1 || error("GraphSpec cpts[$child]: expected exactly 1 prior row")'
    )
    lines.append('            probs = Float64.(rows[1]["probabilities"])')
    lines.append(
        '            length(probs) == n_child || error("GraphSpec cpts[$child]: prior row must have $n_child probabilities")'
    )
    lines.append("            _check_row(child, probs)")
    lines.append("            tensors[child] = probs")
    lines.append("        else")
    lines.append("            expected_rows = prod(length(states[p]) for p in parents)")
    lines.append(
        '            length(rows) == expected_rows || error("GraphSpec cpts[$child]: expected $expected_rows rows, got $(length(rows))")'
    )
    lines.append(
        "            dims = vcat([n_child], [length(states[p]) for p in parents])"
    )
    lines.append("            T = zeros(Float64, dims...)")
    lines.append("            for row in rows")
    lines.append('                assignment = get(row, "assignment", nothing)')
    lines.append(
        '                assignment isa AbstractDict || error("GraphSpec cpts[$child]: row assignment must be an object")'
    )
    lines.append("                idx = Int[]")
    lines.append("                for p in parents")
    lines.append("                    label = String(assignment[String(p)])")
    lines.append("                    i = findfirst(==(label), states[p])")
    lines.append(
        '                    i === nothing && error("GraphSpec cpts[$child]: unknown state $label for parent $p")'
    )
    lines.append("                    push!(idx, i)")
    lines.append("                end")
    lines.append('                probs = Float64.(row["probabilities"])')
    lines.append(
        '                length(probs) == n_child || error("GraphSpec cpts[$child]: row must have $n_child probabilities")'
    )
    lines.append("                _check_row(child, probs)")
    lines.append("                T[:, idx...] = probs")
    lines.append("            end")
    lines.append("            tensors[child] = T")
    lines.append("        end")
    lines.append("    end")
    lines.append("    return spec, states, tensors")
    lines.append("end")
    lines.append("")
    lines.append("function _check_row(child::Symbol, probs::AbstractVector{Float64})")
    lines.append("    for p in probs")
    lines.append(
        '        isfinite(p) && p >= 0.0 || error("GraphSpec cpts[$child]: probability $p is not finite and non-negative")'
    )
    lines.append("    end")
    lines.append(
        '    abs(sum(probs) - 1.0) <= 1e-6 || error("GraphSpec cpts[$child]: probabilities sum to $(sum(probs)), expected 1.0 within 1e-6")'
    )
    lines.append("    return nothing")
    lines.append("end")
    lines.append("")

    # --- main @model -----------------------------------------------------
    arg_names = []
    for key in order:
        arg_names.append(("p_" if not parent_map[key] else "A_") + idents[key])
    e_names = [f"e_{idents[key]}" for key in order]
    lines.append(
        "@model function " + model_name + "(" + ", ".join(arg_names + e_names) + ")"
    )
    for key in order:
        if parent_map[key]:
            cpt_parents = parent_map[key]
            t_args = ", ".join(
                [idents[cpt_parents[0]], f"A_{idents[key]}"]
                + [idents[p] for p in cpt_parents[1:]]
            )
            lines.append(f"    {idents[key]} ~ DiscreteTransition({t_args})")
        else:
            lines.append(f"    {idents[key]} ~ Categorical(p_{idents[key]})")
    for key in order:
        lines.append(
            f"    e_{idents[key]} ~ DiscreteTransition({idents[key]}, EYE_{idents[key]})"
        )
    lines.append("end")
    lines.append("")

    # --- learning variant (Dirichlet priors) ------------------------------
    parented = [key for key in order if parent_map[key]]
    learn_args = []
    for key in order:
        learn_args.append(
            ("alpha_p_" if not parent_map[key] else "alpha_A_") + idents[key]
        )
    lines.append("# --- Learning variant (Dirichlet priors for CPT learning) ---")
    lines.append(
        "# Replace fixed data-arg probabilities with latent Dirichlet priors so"
    )
    lines.append(
        "# RxInfer can learn the CPTs from evidence (pass --learn). Priors use a"
    )
    lines.append(
        "# plain Dirichlet; parented CPTs use DirichletCollection (independent"
    )
    lines.append(
        "# Dirichlets along the FIRST dimension, matching A[child, p1, p2, ...])."
    )
    learn_e_args = [f"e_{idents[key]}" for key in order]
    lines.append(
        "@model function "
        + model_name
        + "_learning("
        + ", ".join(learn_args + learn_e_args)
        + ")"
    )
    for key in order:
        if parent_map[key]:
            cpt_parents = parent_map[key]
            t_args = ", ".join(
                [idents[cpt_parents[0]], f"A_{idents[key]}"]
                + [idents[p] for p in cpt_parents[1:]]
            )
            lines.append(
                f"    A_{idents[key]} ~ DirichletCollection(alpha_A_{idents[key]})"
            )
            lines.append(f"    {idents[key]} ~ DiscreteTransition({t_args})")
        else:
            lines.append(f"    p_{idents[key]} ~ Dirichlet(alpha_p_{idents[key]})")
            lines.append(f"    {idents[key]} ~ Categorical(p_{idents[key]})")
    for key in order:
        lines.append(
            f"    e_{idents[key]} ~ DiscreteTransition({idents[key]}, EYE_{idents[key]})"
        )
    lines.append("end")
    lines.append("")
    if parented:
        lines.append("@constraints function " + model_name + "_learning_constraints()")
        for key in parented:
            lines.append(
                f"    q({idents[key]}, A_{idents[key]}) = q({idents[key]})q(A_{idents[key]})"
            )
        lines.append("end")
        lines.append("")
        init_args = []
        for key in parented:
            init_args.append(f"alpha_A_{idents[key]}")
        for key in order:
            init_args.append(f"uniform_{idents[key]}")
        lines.append(
            "@initialization function "
            + model_name
            + "_learning_init("
            + ", ".join(init_args)
            + ")"
        )
        for key in parented:
            lines.append(
                f"    q(A_{idents[key]}) = DirichletCollection(alpha_A_{idents[key]})"
            )
        for key in order:
            lines.append(f"    q({idents[key]}) = Categorical(uniform_{idents[key]})")
        lines.append("end")
        lines.append("")

    lines.append("@initialization function " + model_name + "_init()")
    for key in order:
        n = len(spec._states_map()[key])
        lines.append(f"    q({idents[key]}) = Categorical(fill(1.0 / {n}, {n}))")
        lines.append(f"    q(e_{idents[key]}) = Categorical(fill(1.0 / {n}, {n}))")
    lines.append("end")
    lines.append("")
    # --- args, evidence, main --------------------------------------------
    lines.append("function parse_args(argv::Vector{String})")
    lines.append(
        '    isempty(argv) && error("usage: julia <script>.jl <graphspec.json> [--evidence key=state ...] [--out FILE] [--learn]")'
    )
    lines.append("    spec_path = argv[1]")
    lines.append("    evidence_labels = Dict{Symbol,String}()")
    lines.append("    out_path = nothing")
    lines.append("    learn = false")
    lines.append("    i = 2")
    lines.append("    while i <= length(argv)")
    lines.append("        arg = argv[i]")
    lines.append('        if arg == "--evidence"')
    lines.append(
        '            i + 1 <= length(argv) || error("--evidence requires a key=state value")'
    )
    lines.append('            kv = split(argv[i+1], "="; limit=2)')
    lines.append(
        '            length(kv) == 2 || error("--evidence expects key=state, got $(argv[i+1])")'
    )
    lines.append("            key = Symbol(strip(kv[1]))")
    lines.append(
        '            haskey(evidence_labels, key) && error("duplicate evidence key: $key")'
    )
    lines.append("            evidence_labels[key] = String(strip(kv[2]))")
    lines.append("            i += 2")
    lines.append('        elseif arg == "--out"')
    lines.append('            i + 1 <= length(argv) || error("--out requires a path")')
    lines.append("            out_path = argv[i+1]")
    lines.append("            i += 2")
    lines.append('        elseif arg == "--learn"')
    lines.append("            learn = true")
    lines.append("            i += 1")
    lines.append("        else")
    lines.append('            error("unknown argument: $arg")')
    lines.append("        end")
    lines.append("    end")
    lines.append("    return spec_path, evidence_labels, out_path, learn")
    lines.append("end")
    lines.append("")
    lines.append("function resolve_evidence(evidence_labels, states)")
    lines.append("    evidence = Dict{Symbol,Any}()")
    lines.append("    for ident in " + model_name.upper() + "_VARS")
    lines.append("        evidence[ident] = missing")
    lines.append("    end")
    lines.append(
        "    for (key, label) in sort!(collect(evidence_labels); by=x->String(x[1]))"
    )
    lines.append(
        '        haskey(RAW_TO_IDENT, key) || error("evidence key not in model: $key")'
    )
    lines.append("        ident = RAW_TO_IDENT[key]")
    lines.append("        options = get(states, ident, nothing)")
    lines.append(
        '        options === nothing && error("evidence key not in model: $key")'
    )
    lines.append("        idx = findfirst(==(label), options)")
    lines.append(
        '        idx === nothing && error("evidence state not in states of $key: $label (states: $options)")'
    )
    lines.append("        onehot = zeros(Float64, length(options))")
    lines.append("        onehot[idx] = 1.0")
    lines.append("        evidence[ident] = onehot")
    lines.append("    end")
    lines.append("    return evidence")
    lines.append("end")
    lines.append("")
    lines.append("function main(argv::Vector{String})")
    lines.append("    spec_path, evidence_labels, out_path, learn = parse_args(argv)")
    lines.append("    spec, states, tensors = load_spec(spec_path)")
    lines.append("    evidence = resolve_evidence(evidence_labels, states)")
    if spec.jev_factors:
        lines.append(
            f"    # jev_factors present in this GraphSpec (RESERVED 'within' "
            f"field): {len(spec.jev_factors)} entries; ignored by this script."
        )
    lines.append("    if learn")
    lines.append("        result = _run_learning(states, tensors, evidence)")
    lines.append("    else")
    lines.append("        result = _run_inference(states, tensors, evidence)")
    lines.append("    end")
    lines.append("    if out_path !== nothing")
    lines.append("        _write_sidecar(result, states, evidence, out_path)")
    lines.append("    end")
    lines.append("    return 0")
    lines.append("end")
    lines.append("")

    model_kwargs = ", ".join(
        f"{arg} = tensors[:{idents[key]}]" for arg, key in zip(arg_names, order)
    )
    lines.append("function _run_inference(states, tensors, evidence::Dict{Symbol,Any})")
    lines.append(
        "    predict = Dict{Symbol,Any}("
        "Symbol(:e_, k) => KeepLast() for (k, v) in evidence if v === missing)"
    )
    lines.append(
        f"    result = infer(model = {model_name}({model_kwargs}), data = Dict{{Symbol,Any}}(Symbol(:e_, k) => v for (k, v) in evidence if v !== missing), predictvars = predict, initialization = {model_name}_init())"
    )
    lines.append('    println("Observed evidence:")')
    lines.append("    for key in " + model_name.upper() + "_VARS")
    lines.append("        evidence[key] === missing && continue")
    lines.append(
        '        println("  $(key) = $(states[key][findfirst(==(1.0), evidence[key])])")'
    )
    lines.append("    end")
    lines.append('    println("Posteriors (marginal P(key)):")')
    lines.append("    for key in " + model_name.upper() + "_VARS")
    lines.append("        evidence[key] !== missing && continue")
    lines.append("        probs = result.posteriors[key].p")
    lines.append(
        '        rendered = join(["$(states[key][i])=$(round(probs[i]; digits=6))" for i in 1:length(probs)], "  ")'
    )
    lines.append('        println("  $(key): $(rendered)")')
    lines.append("    end")
    lines.append("    return result")
    lines.append("end")
    lines.append("")

    lines.append("function _run_learning(states, tensors, evidence::Dict{Symbol,Any})")
    lines.append("    alphas = Dict{Symbol,Any}()")
    lines.append("    for key in " + model_name.upper() + "_VARS")
    lines.append(
        "        alphas[key] = tensors[key] .* LEARNING_PRIOR_SCALE .+ LEARNING_ALPHA_EPS"
    )
    lines.append("    end")
    learn_kwargs = ", ".join(
        f"{arg} = alphas[:{idents[key]}]" for arg, key in zip(learn_args, order)
    )
    if parented:
        learn_init_args = ", ".join(
            [f"alphas[:{idents[key]}]" for key in parented]
            + [
                f"fill(1.0 / length(states[:{idents[key]}]), "
                f"length(states[:{idents[key]}]))"
                for key in order
            ]
        )
        lines.append(
            f"    result = infer(model = {model_name}_learning({learn_kwargs}), "
            "data = Dict{Symbol,Any}(Symbol(:e_, k) => v for (k, v) in evidence), "
            f"constraints = {model_name}_learning_constraints(), initialization = "
            f"{model_name}_learning_init({learn_init_args}), iterations = "
            "LEARNING_ITERATIONS, free_energy = true)"
        )
    else:
        lines.append(
            f"    result = infer(model = {model_name}_learning({learn_kwargs}), "
            "data = Dict{Symbol,Any}(Symbol(:e_, k) => v for (k, v) in evidence), "
            "iterations = LEARNING_ITERATIONS, free_energy = true)"
        )
    lines.append(
        '    println("Learning inference completed (iterations = $(LEARNING_ITERATIONS)).")'
    )
    lines.append('    println("Final variational free energy: $(result.free_energy)")')
    lines.append("    return result")
    lines.append("end")
    lines.append("")

    lines.append(
        "function _write_sidecar(result, states, evidence::Dict{Symbol,Any}, out_path::AbstractString)"
    )
    lines.append("    doc = Dict{String,Any}()")
    lines.append('    doc["format"] = "dafjev.bayesnet-posteriors/1"')
    lines.append(
        '    doc["evidence"] = Dict(String(k) => String(states[k][findfirst(==(1.0), v)]) for (k, v) in evidence if v !== missing)'
    )
    lines.append('    doc["posteriors"] = Dict{String,Any}()')
    lines.append("    for key in " + model_name.upper() + "_VARS")
    lines.append("        if evidence[key] !== missing")
    lines.append("            onehot = evidence[key]")
    lines.append(
        '            doc["posteriors"][String(key)] = Dict(String(states[key][i]) => onehot[i] for i in 1:length(onehot))'
    )
    lines.append("        else")
    lines.append("            probs = result.posteriors[key].p")
    lines.append(
        '            doc["posteriors"][String(key)] = Dict(String(states[key][i]) => probs[i] for i in 1:length(probs))'
    )
    lines.append("        end")
    lines.append("    end")
    lines.append('    open(out_path, "w") do io')
    lines.append("        JSON.print(io, doc, 2)")
    lines.append("    end")
    lines.append('    println("Wrote posteriors sidecar: $out_path")')
    lines.append("    return nothing")
    lines.append("end")
    lines.append("")
    lines.append("if abspath(PROGRAM_FILE) == @__FILE__")
    lines.append("    exit(main(ARGS))")
    lines.append("end")
    lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Downstream round-trip: printed marginals -> gnn.marginals/1 JSON
# ---------------------------------------------------------------------------

#: One printed marginal line from the ``_run_inference`` println block of
#: ``emit_rxinfer_jl``: two-space indent, ``<ident>: <state>=<p>  <state>=<p>``
#: (two-space separator between pairs). ``<ident>`` is a sanitized Julia
#: identifier (``MODEL_VARS`` holds ``:idents`` — never raw keys), so the
#: key class admits no hyphens. Evidence lines (``  <ident> = <state>``)
#: and headers never match (no colon right after the key).
_MARGINAL_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^  (?P<key>[A-Za-z_][A-Za-z0-9_]*):\s+(?P<body>.+)$"
)

#: One ``state=probability`` pair of a printed marginal line. The number is
#: Julia's ``round(probs[i]; digits=6)`` text — a non-negative decimal or
#: exponent literal (``0.692308``, ``1.0``, ``0.0``, ``1.0e-7``); anything
#: else fails closed.
_MARGINAL_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<state>.+?)=(?P<num>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)

#: Maximum rounding error of one ``round(p; digits=6)`` value. The
#: ``write_marginals`` row-sum budget is ``_ROW_SUM_TOLERANCE +
#: n_states * _MARGINAL_ROUND_ERROR`` — each independently rounded value
#: may drift this much, so the budget scales with the state count.
_MARGINAL_ROUND_ERROR: Final[float] = 5e-7


def parse_marginals(text: str) -> dict[str, dict[str, float]]:
    """Parse the marginal block an emitted RxInfer.jl script prints to stdout.

    Parses exactly the ``_run_inference`` println block of
    ``emit_rxinfer_jl`` — lines of the shape ``  <key>: <state>=<p>  ...``
    where ``p`` is Julia's ``round(probs[i]; digits=6)`` text (non-negative
    decimal/exponent literals such as ``0.692308``, ``1.0``, ``1.0e-7``).
    Everything else is skipped: the ``Posteriors (marginal P(key)):`` and
    ``Observed evidence:`` headers, evidence lines (``  <key> = <state>``),
    learning-path lines, and surrounding output. A line in the marginal
    shape whose tokens do not parse raises ``ValueError`` naming the
    offending line (fail-closed), as do duplicate keys and duplicate
    states within a line (the emitter prints each variable and state
    once, so a duplicate means mixed output); a literal overflowing to a
    non-finite float (``1e999``) is rejected the same way — the module
    rejects non-finite probabilities everywhere. Text without marginal
    lines parses to ``{}``.

    Printed keys are the sanitized Julia identifiers (``RAW_TO_IDENT``;
    identical to the GraphSpec key unless sanitization renamed it) and the
    pairs are split on the emitter's exact two-space separator, so state
    labels must not contain a run of two spaces.
    """
    marginals: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        match = _MARGINAL_LINE_RE.match(line)
        if match is None:
            continue
        key, body = match.group("key"), match.group("body")
        if key in marginals:
            raise ValueError(
                f"parse_marginals: duplicate marginal key {key!r} on line {line!r}"
            )
        parsed: dict[str, float] = {}
        for token in body.split("  "):
            token_match = _MARGINAL_TOKEN_RE.fullmatch(token)
            if token_match is None:
                raise ValueError(
                    f"parse_marginals: unparsable marginal token {token!r} on "
                    f"line {line!r} (expected '<state>=<probability>')"
                )
            state = token_match.group("state")
            if state in parsed:
                raise ValueError(
                    f"parse_marginals: duplicate state {state!r} on line {line!r}"
                )
            num = float(token_match.group("num"))
            if not math.isfinite(num):
                raise ValueError(
                    f"parse_marginals: marginal {state!r} on line {line!r} "
                    f"is not finite: {token!r}"
                )
            parsed[state] = num
        marginals[key] = parsed
    return marginals


def write_marginals(
    marginals: Mapping[str, Mapping[str, float]],
    path: str | os.PathLike[str],
    *,
    source_model: str | None = None,
) -> Path:
    """Serialize parsed marginals as a ``gnn.marginals/1`` JSON sidecar.

    Document shape (JSON keys and order):

        {"format": "dafjev.bayesnet-posteriors/1",
         "marginals": {"<key>": {"<state>": <p>, ...}, ...},
         "source_model": <name or null>}

    ``source_model`` records the emitting model's name (the
    ``emit_rxinfer_jl`` ``model_name``, e.g. ``asia_model``) when the
    caller knows it; omitted means ``null``. Validation is fail-closed and
    mirrors the GraphSpec probability rules (finite, >= 0, each row
    summing to 1.0) with a rounding-aware row-sum budget
    (``_ROW_SUM_TOLERANCE + n_states * _MARGINAL_ROUND_ERROR``) that
    absorbs the emitter's 6-digit rounding without false-rejecting
    legitimate parsed rows. Insertion order (the printed topological
    order) is preserved in the JSON. Returns the written ``Path``.
    """
    if not isinstance(marginals, Mapping) or not marginals:
        raise ValueError(
            "write_marginals: marginals must be a non-empty mapping of "
            f"variable key -> {{state: probability}}, got {marginals!r}"
        )
    doc_marginals: dict[str, dict[str, float]] = {}
    for key, states in marginals.items():
        if not isinstance(key, str) or not _KEY_RE.match(key):
            raise ValueError(
                f"write_marginals: invalid marginal key {key!r} "
                "(must match [A-Za-z_][A-Za-z0-9_-]*)"
            )
        if not isinstance(states, Mapping) or not states:
            raise ValueError(
                f"write_marginals: marginal {key!r} must be a non-empty "
                f"mapping of state -> probability, got {states!r}"
            )
        probs: dict[str, float] = {}
        for state, p in states.items():
            if not isinstance(state, str) or not state:
                raise ValueError(
                    f"write_marginals: marginal {key!r} state must be a "
                    f"non-empty string, got {state!r}"
                )
            if isinstance(p, bool) or not isinstance(p, (int, float)):
                raise ValueError(
                    f"write_marginals: marginal {key!r} state {state!r} "
                    f"must map to a number, got {p!r}"
                )
            if not math.isfinite(float(p)):
                raise ValueError(
                    f"write_marginals: marginal {key!r} state {state!r} is "
                    f"not finite: {p!r}"
                )
            if p < 0:
                raise ValueError(
                    f"write_marginals: marginal {key!r} state {state!r} is "
                    f"negative: {p!r}"
                )
            probs[state] = float(p)
        total = sum(probs.values())
        tolerance = _ROW_SUM_TOLERANCE + len(probs) * _MARGINAL_ROUND_ERROR
        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"write_marginals: marginal {key!r} probabilities sum to "
                f"{total!r}, expected 1.0 within {tolerance!r}"
            )
        doc_marginals[key] = probs
    if source_model is not None and (
        not isinstance(source_model, str) or not source_model
    ):
        raise ValueError(
            f"write_marginals: source_model must be a non-empty string or "
            f"None, got {source_model!r}"
        )
    doc: dict[str, Any] = {
        "format": MARGINALS_FORMAT,
        "marginals": doc_marginals,
        "source_model": source_model,
    }
    out_path = Path(path)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    return out_path
