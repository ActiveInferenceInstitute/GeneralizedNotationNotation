from __future__ import annotations

from pathlib import Path

from pipeline.autonomous import (
    collect_evaluation_evidence,
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
    assert report["container_plan"]["security_review_clean"] is True
    assert report["autonomy_policy"]["source_mutation_allowed"] is False
    assert report["autonomy_policy"]["requires_human_approval"] is True
    assert (output / "autonomous" / "autonomous_proposals.json").exists()
    assert (output / "autonomous" / "autonomous_evaluation_report.json").exists()
    assert len(list((output / "autonomous").glob("candidate-1-*.gnn.patch"))) == 1
    decision = report["evaluation_report"]["decisions"][0]
    assert decision["status"] == "proposal_only"
    assert decision["application_allowed"] is False
    assert decision["review_gate"]["required_state"] == "human-reviewed"
    assert decision["rollback_descriptor"]["original_sha256"]
    assert decision["score"]["value"] >= 70
    assert decision["score"]["application_allowed"] is False
    assert (
        "uv run --frozen --extra dev python scripts/check_capability_contracts.py --strict"
        in report["evaluation_report"]["evidence"]["validator_commands"]
    )
    assert report["audit_log"][-1]["automatic_apply_available"] is False
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
    assert "high_execution_success_rate" in decision["score"]["reasons"]
    assert "no_execution_failures_reported" in decision["score"]["reasons"]
    assert report["evaluation_report"]["evidence"]["execution_summary_files"]


def test_autonomous_evidence_collects_release_ledgers(tmp_path: Path) -> None:
    output = tmp_path / "output"
    for relative in (
        "validation/semantic_fidelity_ledger.json",
        "validation/cross_framework_reliability.json",
        "analysis/interpretability_summary.json",
        "report/model_family_report.json",
    ):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    evidence = collect_evaluation_evidence(output)

    assert evidence["semantic_fidelity_ledger_files"]
    assert evidence["cross_framework_ledger_files"]
    assert evidence["interpretability_artifact_files"]
    assert evidence["report_artifact_files"]
