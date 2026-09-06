"""Dependency-graph construction and consistency checks for GNN integration.

Builds the system dependency graph from parsed GNN components and connections,
counts informational cycles, detects isolated components, and verifies
``$ref:``/``type:`` cross-references. NetworkX is used when available; a plain
adjacency-dict fallback keeps the module importable (and useful) without it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .parsing import (
    discover_gnn_files,
    parse_connections,
    parse_references,
    parse_state_components,
    parse_type_references,
    parse_yaml_components,
    undefined_type_names,
)

# Cycle metrics are informational: short cycles only, capped in count and time.
MAX_CYCLE_LENGTH = 6
MAX_CYCLE_TIME = 5  # seconds — quick scan, not a critical check
MAX_CYCLE_COUNT = 500


@dataclass
class SystemGraphStats:
    """Structural statistics for the system dependency graph.

    ``cycles``, ``isolated_nodes``, and ``components`` are ``None`` when the
    metric was not computed (the no-NetworkX fallback computes only node and
    edge counts) and are omitted from :meth:`to_dict` accordingly.
    """

    nodes: int = 0
    edges: int = 0
    cycles: int | None = None
    isolated_nodes: int | None = None
    components: int | None = None

    def to_dict(self) -> dict[str, int]:
        data: dict[str, int] = {"nodes": self.nodes, "edges": self.edges}
        for key in ("cycles", "isolated_nodes", "components"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


@dataclass
class SystemAnalysis:
    """Result of a system-consistency analysis over GNN files."""

    stats: SystemGraphStats = field(default_factory=SystemGraphStats)
    #: component name -> file that declares it
    component_locations: dict[str, str] = field(default_factory=dict)
    #: human-readable consistency issues (isolated nodes, undefined refs, …)
    issues: list[str] = field(default_factory=list)
    #: raw dependency graph (NetworkX DiGraph) when NetworkX is available;
    #: plain adjacency dict otherwise. ``None`` only on internal construction.
    graph: Any | None = None


def _count_cycles(graph: Any, logger: logging.Logger) -> int:
    """Count short cycles as informational structure.

    Intra-model cycles from ``## Connections`` are expected mathematical
    relationships (e.g. bidirectional ``s - A`` creates ``s→A→s`` loops), so
    cycles are reported as structure, never as dependency issues.
    """
    import time as _time

    import networkx as nx

    cycle_count = 0
    try:
        deadline = _time.monotonic() + MAX_CYCLE_TIME
        for _ in nx.simple_cycles(graph, length_bound=MAX_CYCLE_LENGTH):
            cycle_count += 1
            if _time.monotonic() > deadline or cycle_count >= MAX_CYCLE_COUNT:
                break
    except TypeError:
        # Older networkx without length_bound
        try:
            deadline = _time.monotonic() + MAX_CYCLE_TIME
            for _ in nx.simple_cycles(graph):
                cycle_count += 1
                if _time.monotonic() > deadline or cycle_count >= MAX_CYCLE_COUNT:
                    break
        except Exception as exc:
            logger.warning("Fallback cycle scan failed: %s", exc)
            cycle_count = 0
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Cycle scan failed: %s", exc)
        cycle_count = 0
    return cycle_count


def build_system_graph(
    gnn_files: Sequence[Path],
    logger: logging.Logger | None = None,
    verbose: bool = False,
) -> SystemAnalysis:
    """Build the system dependency graph from GNN files and analyze it.

    Nodes come from ``## StateSpaceBlock`` declarations and YAML-style
    ``- name:`` entries; edges from ``## Connections`` (``>``, ``-``, ``<``).
    Connection endpoints are added as nodes even when undeclared.
    """
    logger = logger or logging.getLogger(__name__)
    analysis = SystemAnalysis()
    component_locations: dict[str, str] = analysis.component_locations

    try:
        import networkx as nx

        graph: Any = nx.DiGraph()
        has_networkx = True
    except ImportError:
        has_networkx = False
        graph = {}

    for gnn_file in gnn_files:
        try:
            content = gnn_file.read_text()

            # 1. Variables from ## StateSpaceBlock
            for comp in parse_state_components(content):
                component_locations[comp] = gnn_file.name
                if has_networkx:
                    graph.add_node(comp, file=gnn_file.name)
                else:
                    graph.setdefault(comp, [])

            # 2. YAML-style definitions (retained input format)
            for comp in parse_yaml_components(content):
                component_locations[comp] = gnn_file.name
                if has_networkx:
                    graph.add_node(comp, file=gnn_file.name)
                else:
                    graph.setdefault(comp, [])

            # 3. Connections from ## Connections
            for src, op, tgt in parse_connections(content):
                for node in (src, tgt):
                    if has_networkx:
                        if not graph.has_node(node):
                            graph.add_node(node, file=gnn_file.name)
                    else:
                        graph.setdefault(node, [])
                    component_locations.setdefault(node, gnn_file.name)

                if op == ">":
                    if has_networkx:
                        graph.add_edge(src, tgt, type="directional")
                    else:
                        graph.setdefault(src, []).append(tgt)
                elif op == "-":
                    if has_networkx:
                        graph.add_edge(src, tgt, type="bidirectional")
                        graph.add_edge(tgt, src, type="bidirectional")
                    else:
                        graph.setdefault(src, []).append(tgt)
                        graph.setdefault(tgt, []).append(src)
                elif op == "<":
                    if has_networkx:
                        graph.add_edge(tgt, src, type="reverse")
                    else:
                        graph.setdefault(tgt, []).append(src)

            if verbose:
                logger.debug(
                    "Parsed %s: found %d components",
                    gnn_file.name,
                    len(component_locations),
                )

        except Exception as exc:
            logger.warning("Failed to parse %s: %s", gnn_file.name, exc)
            continue

    # NOTE: Cross-file reference edges via content matching have been removed.
    # GNN models share a common mathematical vocabulary (s_prime, beta, alpha,
    # s_tau1, …), so substring matching always generated false-positive edges.
    # Real cross-file dependencies are detected via explicit $ref: syntax in
    # verify_references().

    if has_networkx:
        try:
            import networkx as nx

            cycle_count = _count_cycles(graph, logger)

            isolated = list(nx.isolates(graph))
            if isolated:
                analysis.issues.append(
                    f"Isolated components (no connections): {isolated}"
                )

            analysis.stats = SystemGraphStats(
                nodes=graph.number_of_nodes(),
                edges=graph.number_of_edges(),
                cycles=cycle_count,
                isolated_nodes=len(isolated),
                components=nx.number_weakly_connected_components(graph),
            )

            logger.info(
                "System graph: %d nodes, %d edges, %d intra-model cycles (structural), %d isolated",
                analysis.stats.nodes,
                analysis.stats.edges,
                cycle_count,
                len(isolated),
            )
        except Exception as exc:
            logger.warning("Failed to analyze graph: %s", exc)
    else:
        analysis.stats = SystemGraphStats(
            nodes=len(graph),
            edges=sum(len(edges) for edges in graph.values()),
        )

    analysis.graph = graph
    return analysis


def export_dependency_graph(
    analysis: SystemAnalysis,
    output_path: Path,
) -> Path | None:
    """Export the dependency graph to a machine-readable JSON file.

    With NetworkX available this writes a node-link JSON document (nodes
    carry their ``file`` attribute; edges carry their ``type``). Without
    NetworkX the adjacency mapping is written instead. Returns the written
    path, or ``None`` when the analysis carries no graph data.
    """
    if analysis.graph is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import networkx as nx

        payload: dict[str, Any] = nx.node_link_data(analysis.graph, edges="edges")
    except (ImportError, TypeError):
        # No NetworkX, or an older version without the ``edges`` keyword:
        # fall back to the plain adjacency mapping.
        payload = {"directed": True, "multigraph": False, "adjacency": analysis.graph}

    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path


def verify_references(
    gnn_files: Sequence[Path],
    component_locations: dict[str, str],
    logger: logging.Logger | None = None,
) -> list[str]:
    """Verify ``$ref:`` and capitalized ``type:`` references resolve.

    Returns one issue string per undefined ``$ref:`` and per suspicious
    undefined type. Per-file read failures are logged and skipped.
    """
    logger = logger or logging.getLogger(__name__)
    known: set[str] = set(component_locations)
    issues: list[str] = []

    for gnn_file in gnn_files:
        try:
            content = gnn_file.read_text()
        except Exception as exc:
            logger.warning("Failed to verify references in %s: %s", gnn_file.name, exc)
            continue

        for ref in parse_references(content):
            if ref not in known:
                issues.append(f"Undefined reference '{ref}' in {gnn_file.name}")
        for type_name in undefined_type_names(parse_type_references(content), known):
            issues.append(f"Possible undefined type '{type_name}' in {gnn_file.name}")
    return issues


def analyze_system(
    target_dir: Path,
    logger: logging.Logger | None = None,
    verbose: bool = False,
) -> SystemAnalysis:
    """Discover, parse, and analyze GNN files under ``target_dir`` in one call.

    Convenience composition of :func:`integration.parsing.discover_gnn_files`,
    :func:`build_system_graph`, and :func:`verify_references` for programmatic
    consumers that want the full consistency report without writing any
    output artifacts. Pure: no files are created.
    """
    gnn_files = discover_gnn_files(Path(target_dir))
    analysis = build_system_graph(gnn_files, logger=logger, verbose=verbose)
    analysis.issues.extend(
        verify_references(gnn_files, analysis.component_locations, logger)
    )
    return analysis
