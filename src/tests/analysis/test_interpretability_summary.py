from __future__ import annotations

import json
from pathlib import Path

from analysis.interpretability import (
    build_family_interpretability_summary,
    build_model_interpretability_summary,
    render_family_interpretability_markdown,
)


def test_model_interpretability_summary_extracts_variables_edges_and_traces(
    tmp_path: Path,
) -> None:
    model = tmp_path / "demo_model.md"
    model.write_text(
        "\n".join(
            [
                "## ModelName",
                "Demo Model",
                "",
                "## StateSpaceBlock",
                "s[2,type=float]",
                "o[2,type=float]",
                "",
                "## Connections",
                "s>o",
                "",
                "## InitialParameterization",
                "A={(0.9,0.1),(0.1,0.9)}",
                "D={(0.5,0.5)}",
            ]
        ),
        encoding="utf-8",
    )
    execution_dir = tmp_path / "12_execute_output" / "demo_model"
    execution_dir.mkdir(parents=True)
    summary_dir = tmp_path / "00_pipeline_summary"
    summary_dir.mkdir()
    (summary_dir / "pipeline_execution_summary.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"script_name": "11_render.py", "status": "SUCCESS"},
                    {"script_name": "12_execute.py", "status": "SUCCESS"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (execution_dir / "simulation_results.json").write_text(
        json.dumps({"free_energy_trace": [3.0, 2.0, 1.0], "actions": [0, 1]}),
        encoding="utf-8",
    )

    summary = build_model_interpretability_summary(model, tmp_path)

    assert summary["model_name"] == "Demo Model"
    assert summary["variable_count"] == 2
    assert summary["connection_count"] == 1
    assert summary["matrix_shapes"]["A"] == [2, 2]
    assert summary["pipeline_evidence"]["render_status"] == "passed"
    assert summary["pipeline_evidence"]["execution_status"] == "passed"
    assert summary["telemetry_present"] is True
    assert summary["telemetry_preview"]["free_energy_trace"] == [3.0, 2.0, 1.0]
    assert summary["artifact_links"]


def test_family_interpretability_markdown_is_compact(tmp_path: Path) -> None:
    model = tmp_path / "demo.md"
    model.write_text(
        "## ModelName\nDemo\n\n## StateSpaceBlock\ns[2]\n\n## Connections\n",
        encoding="utf-8",
    )

    summary = build_family_interpretability_summary("demo-family", tmp_path, tmp_path)
    markdown = render_family_interpretability_markdown(summary)

    assert summary["family"] == "demo-family"
    assert summary["model_count"] == 1
    assert summary["totals"]["models_with_telemetry"] == 0
    assert "# Model Family Interpretability: demo-family" in markdown
    assert "Models with telemetry" in markdown
    assert "| Demo |" in markdown
