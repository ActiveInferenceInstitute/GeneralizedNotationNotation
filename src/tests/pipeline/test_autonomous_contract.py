from __future__ import annotations

from pathlib import Path

from pipeline.autonomous import (
    collect_observation_streams,
    run_autonomous_proposal_loop,
)


def test_autonomous_loop_writes_proposals_without_source_mutation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "input"
    target.mkdir()
    model = target / "model.md"
    model.write_text("## ModelName\nDemo\n", encoding="utf-8")
    output = tmp_path / "output"
    report = run_autonomous_proposal_loop(target, output)
    assert report["source_mutation_performed"] is False
    assert report["cluster_mutation_performed"] is False
    assert report["container_plan"]["dry_run"] is True
    assert report["container_plan"]["mutation_performed"] is False
    assert (output / "autonomous" / "autonomous_proposals.json").exists()
    assert (output / "autonomous" / "autonomous_evaluation_report.json").exists()
    assert (output / "autonomous" / "candidate-1.gnn.patch").exists()
    assert report["evaluation_report"]["decisions"][0]["status"] == "proposal_only"
    assert report["evaluation_report"]["decisions"][0]["score"]["value"] >= 70
    assert (
        "uv run --extra dev python scripts/check_capability_contracts.py --strict"
        in report["evaluation_report"]["evidence"]["validator_commands"]
    )
    assert model.read_text(encoding="utf-8") == "## ModelName\nDemo\n"


def test_autonomous_observation_streams_include_array_and_manifest_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "observations.npy").write_bytes(b"numpy")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    streams = collect_observation_streams(tmp_path)

    kinds = {Path(stream["path"]).name: stream["kind"] for stream in streams}
    assert kinds["observations.npy"] == "array_file"
    assert kinds["manifest.json"] == "manifest_file"


def test_autonomous_scoring_uses_existing_execution_summary(tmp_path: Path) -> None:
    target = tmp_path / "input"
    target.mkdir()
    (target / "model.md").write_text("## ModelName\nDemo\n", encoding="utf-8")
    output = tmp_path / "output"
    summaries = output / "12_execute_output" / "summaries"
    summaries.mkdir(parents=True)
    (summaries / "execution_summary.json").write_text(
        '{"success_rate": 100.0, "execution_details": []}',
        encoding="utf-8",
    )

    report = run_autonomous_proposal_loop(target, output, max_candidates=1)
    decision = report["evaluation_report"]["decisions"][0]

    assert decision["score"]["recommendation"] == "review_with_validators"
    assert "execution_summary_available" in decision["score"]["reasons"]
    assert report["evaluation_report"]["evidence"]["execution_summary_files"]
