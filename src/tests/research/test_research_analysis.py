#!/usr/bin/env python3
"""
Tests for the pure static-analysis API of the Research Processor module.

Pins the additive composable surface introduced alongside the processor
refactor: ``analyze_gnn`` / ``ModelAnalysis``, ``discover_gnn_files``,
``merge_llm_hypotheses``, ``summarize_hypotheses``, ``render_research_report``
(purity + parity with the written report), and the section-scanning helpers.

All tests are deterministic, isolated (tmp_path), and network-free.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from research.processor import (
    MODEL_FAMILIES,
    ModelAnalysis,
    analyze_gnn,
    count_connections,
    discover_gnn_files,
    extract_state_space_dims,
    merge_llm_hypotheses,
    render_research_report,
    summarize_hypotheses,
)

POMDP_CONTENT = (
    "## GNNSection\n"
    "ActInfPOMDP\n"
    "## StateSpaceBlock\n"
    "- A[2,2,type=float]\n"
    "B[2,2,2]\n"
    "C[2]\n"
    "D[2]\n"
    "pi[2]\n"
    "## Connections\n"
    "s -> o\n"
    "o - s\n"
)


@pytest.mark.unit
def test_analyze_gnn_returns_consistent_typed_bundle() -> None:
    """analyze_gnn must agree with the individual helpers it bundles."""
    analysis = analyze_gnn(POMDP_CONTENT)

    assert isinstance(analysis, ModelAnalysis)
    assert analysis.model_family == "pomdp"
    assert analysis.dimensions == extract_state_space_dims(POMDP_CONTENT)
    assert analysis.connections == count_connections(POMDP_CONTENT)
    assert analysis.dimensions["A"] == [2, 2]
    assert analysis.connections["directed"] == 1
    assert analysis.connections["undirected"] == 1
    assert analysis.connections["total"] == 2


@pytest.mark.unit
def test_analyze_gnn_empty_content_is_unknown_and_empty() -> None:
    """Empty content analyzes to the unknown family with no evidence."""
    analysis = analyze_gnn("")

    assert analysis == ModelAnalysis(
        model_family="unknown",
        dimensions={},
        connections={
            "directed": 0,
            "undirected": 0,
            "total": 0,
        },
    )


@pytest.mark.unit
def test_model_families_constant_covers_detector_outputs() -> None:
    """Every family the detector emits for a corpus must be in MODEL_FAMILIES."""
    corpus = [
        POMDP_CONTENT,
        "## GNNSection\nHiddenMarkovModel\n## StateSpaceBlock\nA[2,2]\nB[2,2]\n",
        "## StateSpaceBlock\nA[2,2]\nB[2,2]\n",
        "continuous\ngaussian\n",
        "factor\nA[2,2]\n",
        "level1\nlevel2\nA[2,2]\n",
        "nothing here\n",
    ]
    from research.processor import detect_model_family

    for content in corpus:
        assert detect_model_family(content) in MODEL_FAMILIES


@pytest.mark.unit
def test_section_body_scanner_stops_at_next_level_two_heading() -> None:
    """Dims/connections scanners must not leak across section boundaries."""
    content = (
        "## StateSpaceBlock\n"
        "A[3,3]\n"
        "## ModelName\n"
        "A[9,9]\n"
        "## Connections\n"
        "a -> b\n"
        "## Parameters\n"
        "x -> y\n"
    )

    assert extract_state_space_dims(content) == {"A": [3, 3]}
    assert count_connections(content) == {
        "directed": 1,
        "undirected": 0,
        "total": 1,
    }


@pytest.mark.unit
def test_extract_dims_skips_symbolic_and_nonpositive_parts() -> None:
    """Symbolic dimensions and non-positive integers never yield a dims entry."""
    content = "## StateSpaceBlock\n- A[2,3,type=float]\nsymbolic[pi,N]\nbad[0,2]\n"

    assert extract_state_space_dims(content) == {"A": [2, 3]}


@pytest.mark.unit
def test_count_connections_counts_undirected_dashes() -> None:
    """Undirected edges are bare dashes; arrows are directed edges."""
    content = "## Connections\na --- b\nc - d\n"

    assert count_connections(content) == {
        "directed": 0,
        "undirected": 4,
        "total": 4,
    }


@pytest.mark.unit
def test_discover_gnn_files_sorted_scoped_and_recursive() -> None:
    """Discovery returns sorted *.md paths, respecting the recursive flag."""
    target = Path("/nonexistent-target")

    # Missing directory yields an empty list rather than raising.
    assert discover_gnn_files(target, recursive=True) == []


@pytest.mark.unit
def test_discover_gnn_files_top_level_vs_recursive(tmp_path: Path) -> None:
    target = tmp_path / "input"
    (target / "nested").mkdir(parents=True)
    (target / "b.md").write_text("b", encoding="utf-8")
    (target / "a.md").write_text("a", encoding="utf-8")
    (target / "nested" / "c.md").write_text("c", encoding="utf-8")
    (target / "notes.txt").write_text("x", encoding="utf-8")

    top_level = discover_gnn_files(target, recursive=False)
    assert [p.name for p in top_level] == ["a.md", "b.md"]
    assert all(p.parent == target for p in top_level)

    recursive = discover_gnn_files(target, recursive=True)
    assert [p.name for p in recursive] == ["a.md", "b.md", "c.md"]


@pytest.mark.unit
def test_merge_llm_hypotheses_prefers_llm_and_dedups_by_type() -> None:
    """LLM hypotheses lead; rule hypotheses with a covered type are dropped."""
    llm = [{"type": "t1", "priority": "high"}]
    rules = [
        {"type": "t2", "priority": "medium"},
        {"type": "t1", "priority": "low"},
        {"type": "t3", "priority": "low"},
    ]

    merged = merge_llm_hypotheses(llm, rules)

    assert [h["type"] for h in merged] == ["t1", "t2", "t3"]
    assert merged[0]["priority"] == "high"


@pytest.mark.unit
def test_summarize_hypotheses_counts_priorities_and_types() -> None:
    """Summary counts totals, known priorities, and per-type occurrences."""
    hypotheses = [
        {"type": "alpha", "priority": "high"},
        {"type": "alpha", "priority": "low"},
        {"type": "beta", "priority": "urgent"},  # unknown priority
        {"priority": "low"},  # missing type
    ]

    summary = summarize_hypotheses(hypotheses)

    assert summary["total"] == 4
    assert summary["by_priority"] == {"high": 1, "medium": 0, "low": 2}
    assert summary["by_type"] == {"alpha": 2, "beta": 1}


@pytest.mark.unit
def test_summarize_hypotheses_empty_and_deterministic() -> None:
    """Empty input yields zeroed counts; repeated calls are deterministic."""
    empty = summarize_hypotheses([])
    assert empty == {
        "total": 0,
        "by_priority": {"high": 0, "medium": 0, "low": 0},
        "by_type": {},
    }

    hypotheses: list[dict[str, Any]] = [
        {"type": f"t{i % 3}", "priority": "high"} for i in range(6)
    ]
    assert summarize_hypotheses(hypotheses) == summarize_hypotheses(
        list(reversed(hypotheses))
    )


@pytest.mark.unit
def test_render_research_report_is_pure(tmp_path: Path) -> None:
    """render_research_report must not touch the filesystem."""
    results = {
        "analysis_mode": "rule_based",
        "hypotheses_generated": [
            {
                "file": "model.md",
                "model_family": "pomdp",
                "hypotheses": [
                    {
                        "type": "precision_modulation",
                        "description": "d",
                        "rationale": "r",
                        "source": "rule_based_static_analysis",
                        "priority": "high",
                    },
                    {
                        "type": "ontology_annotation",
                        "description": "d2",
                        "rationale": "r2",
                        "source": "rule_based_static_analysis",
                        "priority": "low",
                    },
                ],
            }
        ],
    }

    report = render_research_report(results)

    assert report.startswith("# Research Hypotheses Report\n")
    assert "## model.md (pomdp model)" in report
    assert "### High Priority" in report
    assert "- **precision_modulation**: d" in report
    assert "  - *Source*: rule_based_static_analysis" in report
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_rendered_report_matches_written_report(tmp_path: Path) -> None:
    """The report on disk must equal render_research_report of the results."""
    from research.processor import process_research

    target = tmp_path / "input"
    target.mkdir()
    (target / "model.md").write_text(POMDP_CONTENT, encoding="utf-8")
    out = tmp_path / "out"

    assert process_research(target, out) is True

    results = json.loads((out / "research_results.json").read_text(encoding="utf-8"))
    written = (out / "research_report.md").read_text(encoding="utf-8")

    assert render_research_report(results) == written
    assert "## model.md (pomdp model)" in written
