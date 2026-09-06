"""Tests for integration.parsing, integration.graph, and package-level API.

Pins the real behavior of the pure extraction primitives, the system graph
builder, the ``analyze_system()`` one-call API, and the JSON graph export.
Deterministic: temp dirs only, no network, no global state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.integration.graph import (
    SystemAnalysis,
    SystemGraphStats,
    analyze_system,
    build_system_graph,
    export_dependency_graph,
    verify_references,
)
from gnn.integration.parsing import (
    BUILTIN_TYPE_NAMES,
    discover_gnn_files,
    parse_connections,
    parse_references,
    parse_state_components,
    parse_type_references,
    parse_yaml_components,
    undefined_type_names,
)


def _write_gnn(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


SAMPLE = """# Model A

## StateSpaceBlock
A[3,3,type=float]
s[3]
# a comment line

## Connections
s > A
A - s
q < s
"""


class TestParsingPrimitives:
    """Pure text extraction from GNN markdown content."""

    def test_parse_state_components(self) -> None:
        assert parse_state_components(SAMPLE) == ["A", "s"]

    def test_parse_state_components_skip_comments_and_blanks(self) -> None:
        content = "## StateSpaceBlock\n# comment\n\nA[1]\n"
        assert parse_state_components(content) == ["A"]

    def test_parse_state_components_no_section(self) -> None:
        assert parse_state_components("# nothing here\n") == []

    def test_parse_yaml_components(self) -> None:
        content = "components:\n  - name: alpha\n  - name: beta\n"
        assert parse_yaml_components(content) == ["alpha", "beta"]

    def test_parse_connections_operators(self) -> None:
        assert parse_connections(SAMPLE) == [
            ("s", ">", "A"),
            ("A", "-", "s"),
            ("q", "<", "s"),
        ]

    def test_parse_connections_requires_known_operators(self) -> None:
        assert parse_connections("## Connections\nA ~ B\n") == []

    def test_parse_references(self) -> None:
        assert parse_references("$ref: Widget\n$ref: thing") == ["Widget", "thing"]

    def test_parse_type_references(self) -> None:
        content = "type: String\ntype: Missing\n"
        assert parse_type_references(content) == ["String", "Missing"]

    def test_builtin_type_names_frozen(self) -> None:
        assert "String" in BUILTIN_TYPE_NAMES
        with pytest.raises(AttributeError):
            BUILTIN_TYPE_NAMES.add("Nope")  # type: ignore[attr-defined]

    def test_undefined_type_names_ignores_builtins_and_lowercase(self) -> None:
        names = ["Missing", "String", "lowercase", "Known"]
        result = undefined_type_names(names, {"Known"})
        assert result == ["Missing"]

    def test_discover_dedupes_by_filename(self, tmp_path: Path) -> None:
        target = tmp_path / "out"
        target.mkdir()
        _write_gnn(target, "model.md", "# model\n")
        (tmp_path / "input" / "gnn_files").mkdir(parents=True)
        _write_gnn(tmp_path / "input" / "gnn_files", "model.md", "# dup")
        _write_gnn(tmp_path / "input" / "gnn_files", "other.md", "# other")

        files = discover_gnn_files(target)
        names = [f.name for f in files]
        assert names == ["model.md", "other.md"]
        # First occurrence (target dir) wins for the deduplicated name.
        assert files[0].parent == target

    def test_discover_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        assert discover_gnn_files(tmp_path / "missing") == []


class TestSystemGraphStats:
    def test_to_dict_omits_uncomputed_metrics(self) -> None:
        assert SystemGraphStats(nodes=1, edges=2).to_dict() == {"nodes": 1, "edges": 2}

    def test_to_dict_includes_computed_metrics(self) -> None:
        stats = SystemGraphStats(
            nodes=1, edges=2, cycles=3, isolated_nodes=0, components=1
        )
        assert stats.to_dict() == {
            "nodes": 1,
            "edges": 2,
            "cycles": 3,
            "isolated_nodes": 0,
            "components": 1,
        }


class TestBuildSystemGraph:
    def test_nodes_edges_cycles_and_locations(self, tmp_path: Path) -> None:
        path = _write_gnn(tmp_path, "a.md", SAMPLE)
        analysis = build_system_graph([path])

        assert analysis.stats.nodes == 3  # A, s, q

        assert analysis.stats.edges >= 2
        assert analysis.stats.cycles is not None and analysis.stats.cycles >= 1
        assert analysis.component_locations == {"A": "a.md", "s": "a.md", "q": "a.md"}

    def test_isolated_components_reported(self, tmp_path: Path) -> None:
        path = _write_gnn(tmp_path, "lonely.md", "## StateSpaceBlock\nLone[1]\n")
        analysis = build_system_graph([path])
        assert analysis.stats.isolated_nodes == 1
        assert any("Lone" in issue for issue in analysis.issues)

    def test_bidirectional_connection_creates_cycle(self, tmp_path: Path) -> None:
        path = _write_gnn(
            tmp_path, "b.md", "## StateSpaceBlock\nx[1]\ny[1]\n## Connections\nx - y\n"
        )
        analysis = build_system_graph([path])
        assert analysis.stats.cycles is not None and analysis.stats.cycles >= 1

    def test_yaml_components_become_nodes(self, tmp_path: Path) -> None:
        path = _write_gnn(tmp_path, "c.md", "components:\n  - name: YComp\n")
        analysis = build_system_graph([path])
        assert "YComp" in analysis.component_locations
        assert analysis.stats.nodes == 1

    def test_unreadable_file_is_skipped(self, tmp_path: Path) -> None:
        binary = tmp_path / "binary.md"
        binary.write_bytes(b"\x00\x01\x02")
        analysis = build_system_graph([binary])
        assert analysis.stats.nodes == 0
        assert analysis.stats.edges == 0

    def test_analysis_exposes_raw_graph(self, tmp_path: Path) -> None:
        path = _write_gnn(tmp_path, "a.md", SAMPLE)
        analysis = build_system_graph([path])
        assert analysis.graph is not None

    def test_default_analysis_is_empty(self) -> None:
        analysis = SystemAnalysis()
        assert analysis.stats.to_dict() == {"nodes": 0, "edges": 0}
        assert analysis.issues == []
        assert analysis.graph is None


class TestVerifyReferences:
    def test_undefined_ref_and_type(self, tmp_path: Path) -> None:
        path = _write_gnn(
            tmp_path,
            "refs.md",
            "$ref: Ghost\ntype: GhostType\ntype: String\n",
        )
        issues = verify_references([path], {"Real": "other.md"})
        assert "Undefined reference 'Ghost' in refs.md" in issues
        assert "Possible undefined type 'GhostType' in refs.md" in issues
        assert not any("'String'" in issue for issue in issues)

    def test_resolved_refs_produce_no_issues(self, tmp_path: Path) -> None:
        path = _write_gnn(tmp_path, "ok.md", "$ref: Known\n")
        assert verify_references([path], {"Known": "ok.md"}) == []

    def test_unreadable_file_is_skipped(self, tmp_path: Path) -> None:
        binary = tmp_path / "binary.md"
        binary.write_bytes(b"\x00\x01")
        assert verify_references([binary], {}) == []


class TestAnalyzeSystem:
    def test_single_call_reports_graph_and_issues(self, tmp_path: Path) -> None:
        _write_gnn(tmp_path, "a.md", SAMPLE)
        _write_gnn(tmp_path, "b.md", "$ref: Nowhere\n")
        analysis = analyze_system(tmp_path)
        assert analysis.stats.nodes >= 3
        assert any("Nowhere" in issue for issue in analysis.issues)

    def test_pure_no_files_written(self, tmp_path: Path) -> None:
        _write_gnn(tmp_path, "a.md", SAMPLE)
        before = sorted(p.name for p in tmp_path.iterdir())
        analyze_system(tmp_path)
        after = sorted(p.name for p in tmp_path.iterdir())
        assert before == after

    def test_missing_target_dir_is_safe(self, tmp_path: Path) -> None:
        analysis = analyze_system(tmp_path / "missing")
        assert analysis.stats.nodes == 0
        assert analysis.issues == []


class TestExportDependencyGraph:
    def test_node_link_export(self, tmp_path: Path) -> None:
        path = _write_gnn(tmp_path, "a.md", SAMPLE)
        analysis = build_system_graph([path])
        out = tmp_path / "nested" / "graph.json"
        written = export_dependency_graph(analysis, out)
        assert written == out
        data: dict[str, Any] = json.loads(out.read_text())
        assert data["directed"] is True
        node_ids = {n["id"] for n in data["nodes"]}
        assert {"A", "s", "q"} <= node_ids
        assert all("type" in e for e in data["edges"])

    def test_none_graph_returns_none(self, tmp_path: Path) -> None:
        assert export_dependency_graph(SystemAnalysis(), tmp_path / "g.json") is None
        assert not (tmp_path / "g.json").exists()
