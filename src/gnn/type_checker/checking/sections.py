"""Section-scoped GNN content extraction helpers.

The type checker and the resource estimator both need to read canonical
``## Section`` blocks out of a GNN specification without being fooled by
prose, comments, or identically-named tokens appearing outside the
section. Historically each subsystem reimplemented this parsing (the
estimator even used naive whole-content regexes that matched connection
operators inside prose). This module owns the single, section-aware
implementation so the two consumers stay consistent.

Everything here is a pure function over plain strings — no I/O, no
globals — which keeps the type checker and estimator composable and
trivially testable.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from gnn.schemas.section_contract import CANONICAL_GNN_SECTIONS

__all__ = [
    "CANONICAL_GNN_SECTIONS",
    "TimeSpecKind",
    "classify_time_spec",
    "classify_time_spec_kind",
    "connection_group",
    "detect_time_dynamics",
    "extract_markdown_section",
    "parse_resource_connections",
    "section_presence",
]


# Canonical GNN section headers, in declared order — single-sourced from
# ``gnn.schemas.section_contract`` (W2-01) and re-exported here for the
# type-checker/estimator consumers.

# Connection operators recognised in the ``Connections`` section, ordered
# longest/most-specific first so multi-character operators win over their
# single-character prefixes (``<->`` before ``-``).
_CONNECTION_OPERATORS: tuple[str, ...] = ("<->", "->", ">", "|", "-")


def extract_markdown_section(content: str, section_name: str) -> str:
    """Return one canonical Markdown GNN section without adjacent prose.

    Only the body of the ``## <section_name>`` block is returned, stripped
    of surrounding whitespace. Section matching is exact (header text after
    ``## `` must equal ``section_name``). Returns an empty string when the
    section is absent.
    """
    lines: list[str] = []
    in_section = False
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("## "):
            in_section = stripped[3:].strip() == section_name
            continue
        if in_section:
            lines.append(raw_line)
    return "\n".join(lines).strip()


def connection_group(value: str) -> list[str]:
    """Split a connection endpoint group ``"(a, b, pi)"`` into names.

    Parenthesised groups are unwrapped and ``pi`` is normalised to the
    symbolic ``π`` used elsewhere in the type checker. Empty tokens are
    dropped.
    """
    group = value.strip()
    if group.startswith("(") and group.endswith(")"):
        group = group[1:-1]
    return [
        "π" if name.strip().lower() == "pi" else name.strip()
        for name in group.split(",")
        if name.strip()
    ]


def parse_resource_connections(
    content: str, known_variables: set[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse the ``Connections`` section into edges plus diagnostics.

    Only the ``## Connections`` block is scanned, so connection operators
    that appear in prose, equations, or comments are ignored. Each
    discovered edge is ``{"source", "target", "type"}`` where ``type`` is
    ``"undirected"`` for ``-``/``<->`` and ``"directed"`` otherwise.
    Diagnostics flag unparseable lines, empty endpoints, and references to
    variables that were not declared in the state space.
    """
    edges: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    for line_number, raw_line in enumerate(
        extract_markdown_section(content, "Connections").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        operator = next(
            (candidate for candidate in _CONNECTION_OPERATORS if candidate in line),
            None,
        )
        if operator is None:
            diagnostics.append(
                f"Unparseable connection at section line {line_number}: '{line}'"
            )
            continue
        source_text, target_text = line.split(operator, 1)
        target_text = target_text.split(":", 1)[0]
        sources = connection_group(source_text)
        targets = connection_group(target_text)
        if not sources or not targets:
            diagnostics.append(
                f"Connection at section line {line_number} has an empty endpoint: "
                f"'{line}'"
            )
            continue

        edge_type = "undirected" if operator in {"-", "<->"} else "directed"
        for source in sources:
            for target in targets:
                edges.append({"source": source, "target": target, "type": edge_type})
                if source not in known_variables:
                    diagnostics.append(
                        f"Connection at section line {line_number} references "
                        f"undeclared variable '{source}'"
                    )
                if target not in known_variables:
                    diagnostics.append(
                        f"Connection at section line {line_number} references "
                        f"undeclared variable '{target}'"
                    )
    return edges, diagnostics


def section_presence(content: str, sections: tuple[str, ...] = ()) -> dict[str, bool]:
    """Return a ``{section_name: present}`` map for the requested sections.

    Presence means a ``## <section_name>`` header exists in ``content``.
    Defaults to :data:`CANONICAL_GNN_SECTIONS` when ``sections`` is empty.
    """
    wanted = sections or CANONICAL_GNN_SECTIONS
    present = dict.fromkeys(wanted, False)
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("## "):
            name = stripped[3:].strip()
            if name in present:
                present[name] = True
    return present


def detect_time_dynamics(content: str) -> bool:
    """Return True when the ``## Time`` section declares a dynamic model.

    Reads only the canonical ``## Time`` block so a stray ``t`` in prose or
    a variable name cannot flip a static model to dynamic. Recognises
    ``dynamic``, ``continuous-time``, ``time-varying``, and ``regime``
    declarations — a regime-switched schedule selects between finite
    transition tensors per timestep, which is nonstationary and therefore
    dynamic.
    """
    time_text = extract_markdown_section(content, "Time").lower()
    if not time_text:
        return False
    return any(
        marker in time_text
        for marker in (
            "dynamic",
            "continuous-time",
            "continuous_time",
            "time-varying",
            "time_varying",
            "regime",
        )
    )


class TimeSpecKind(Enum):
    """Typed classification of a GNN spec's ``## Time`` section.

    Three kinds, pinned by the nonstationary-semantics contract:

    - ``STATIC`` — no time-variation marker; one transition tensor.
    - ``TIME_VARYING`` — dynamics indexed by time (a ``B_t`` tensor with a
      time axis; parameters evolve over the planning horizon).
    - ``REGIME_SWITCHED`` — dynamics indexed by a discrete regime with an
      explicit schedule (``B_regime`` + ``b_regime_schedule``), selecting
      one of a finite set of transition tensors per timestep.

    A ``Hierarchical`` ``## Time`` declaration is an earlier taxonomy label
    (deep temporal hierarchy), orthogonal to these three kinds: the typed
    API classifies the *time variation of the dynamics*, while the
    :func:`classify_time_spec` string keeps returning ``"Hierarchical"``
    for backward compatibility.
    """

    STATIC = "Static"
    TIME_VARYING = "TimeVarying"
    REGIME_SWITCHED = "RegimeSwitched"


# Earlier string projection consumed by existing callers (the checker's
# ``model_type`` field and the estimator's ``time_spec``): both dynamic
# kinds project to ``"Dynamic"`` so their serialized output is unchanged.
_LEGACY_TIME_SPEC_STRINGS: dict[TimeSpecKind, str] = {
    TimeSpecKind.STATIC: "Static",
    TimeSpecKind.TIME_VARYING: "Dynamic",
    TimeSpecKind.REGIME_SWITCHED: "Dynamic",
}


def classify_time_spec_kind(content: str) -> TimeSpecKind:
    """Classify a GNN spec's ``## Time`` section into the typed kinds.

    Reads only the canonical ``## Time`` block (same scoping rule as
    :func:`detect_time_dynamics`). A regime marker wins over the plain
    dynamic markers — a schedule selecting between finite regimes is the
    more specific contract. Specs with no time-variation marker are
    ``STATIC``; the ``Hierarchical`` label is orthogonal (see
    :class:`TimeSpecKind`).
    """
    time_text = extract_markdown_section(content, "Time").lower()
    if not time_text:
        return TimeSpecKind.STATIC
    if "regime" in time_text:
        return TimeSpecKind.REGIME_SWITCHED
    if detect_time_dynamics(content):
        return TimeSpecKind.TIME_VARYING
    return TimeSpecKind.STATIC


def classify_time_spec(content: str) -> str:
    """Classify a GNN spec's ``## Time`` section into Static/Dynamic/Hierarchical.

    Stable string projection of :func:`classify_time_spec_kind`:
    ``TimeVarying`` and ``RegimeSwitched`` both project to the earlier-name
    ``"Dynamic"`` string so existing callers (the checker's ``model_type``
    field, the estimator's ``time_spec``, and their tests) keep their exact
    contract. Reads only the canonical ``## Time`` block so a stray ``t``
    in prose or a variable name cannot flip a static model to Dynamic.
    Hierarchical wins over Dynamic when both markers appear. The Dynamic
    determination delegates to :func:`detect_time_dynamics` so both share
    one marker set (``dynamic``, ``continuous-time``, ``continuous_time``,
    ``time-varying``, ``regime``) and can never disagree for the same
    content.
    """
    time_text = extract_markdown_section(content, "Time").lower()
    if "hierarchical" in time_text:
        return "Hierarchical"
    return _LEGACY_TIME_SPEC_STRINGS[classify_time_spec_kind(content)]
