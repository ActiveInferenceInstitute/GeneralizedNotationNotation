"""Bounded autonomous proposal artifacts for the GNN pipeline.

Autonomous mode writes reports and candidate patch files only; it does not edit
repository source files, commit changes, or mutate live infrastructure.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def collect_observation_streams(target_dir: Path) -> List[Dict[str, Any]]:
    """Describe file-backed observation streams without opening live devices."""
    streams: List[Dict[str, Any]] = []
    for path in sorted(target_dir.rglob("*")) if target_dir.exists() else []:
        if path.is_file() and path.suffix.lower() in {
            ".json",
            ".csv",
            ".npy",
            ".npz",
            ".md",
        }:
            suffix = path.suffix.lower()
            if suffix in {".npy", ".npz"}:
                kind = "array_file"
            elif suffix == ".json" and "manifest" in path.name.lower():
                kind = "manifest_file"
            else:
                kind = "file"
            streams.append(
                {
                    "path": str(path),
                    "kind": kind,
                    "suffix": suffix,
                    "size_bytes": path.stat().st_size,
                }
            )
    return streams


def build_container_plan(target_dir: Path) -> Dict[str, Any]:
    """Return a generated container plan, never a live orchestration mutation."""
    return {
        "schema": "gnn_container_plan_v1",
        "target_dir": str(target_dir),
        "services": [
            {
                "name": "gnn-worker",
                "image": "python:3.12-slim",
                "command": "uv run python src/main.py --target-dir /workspace/input/gnn_files",
                "replicas": 1,
            }
        ],
        "dry_run": True,
        "mutation_performed": False,
        "cluster_mutation_performed": False,
    }


def run_autonomous_proposal_loop(
    target_dir: Path, output_dir: Path, *, max_candidates: int = 3
) -> Dict[str, Any]:
    """Write bounded candidate-evaluation artifacts without modifying source files."""
    autonomous_dir = output_dir / "autonomous"
    autonomous_dir.mkdir(parents=True, exist_ok=True)
    gnn_files = sorted(target_dir.rglob("*.md")) if target_dir.exists() else []
    candidates = [
        {
            "candidate_id": f"candidate-{index + 1}",
            "source_file": str(path),
            "proposal": "Evaluate matrix dimensions, execution telemetry, and validation errors before applying any model patch.",
            "patch_artifact": str(autonomous_dir / f"candidate-{index + 1}.gnn.patch"),
            "source_mutation_performed": False,
        }
        for index, path in enumerate(gnn_files[:max_candidates])
    ]
    for candidate in candidates:
        patch_body = _candidate_patch_text(candidate)
        Path(str(candidate["patch_artifact"])).write_text(patch_body, encoding="utf-8")
    evaluation_report = _build_evaluation_report(candidates, target_dir, output_dir)
    report: Dict[str, Any] = {
        "schema": "gnn_autonomous_proposal_loop_v1",
        "created_at": datetime.now().isoformat(),
        "target_dir": str(target_dir),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "observation_streams": collect_observation_streams(target_dir),
        "container_plan": build_container_plan(target_dir),
        "evaluation_report": evaluation_report,
        "source_mutation_performed": False,
        "cluster_mutation_performed": False,
    }
    (autonomous_dir / "autonomous_proposals.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (autonomous_dir / "autonomous_evaluation_report.json").write_text(
        json.dumps(evaluation_report, indent=2), encoding="utf-8"
    )
    (autonomous_dir / "autonomous_evaluation_report.md").write_text(
        _evaluation_report_markdown(evaluation_report), encoding="utf-8"
    )
    (autonomous_dir / "candidate_patch.diff").write_text(
        "\n".join(
            Path(str(candidate["patch_artifact"])).read_text(encoding="utf-8")
            for candidate in candidates
        )
        or "# No candidate patches generated. Autonomous mode only writes proposals.\n",
        encoding="utf-8",
    )
    return report


def _candidate_patch_text(candidate: Dict[str, Any]) -> str:
    """Build a non-applied candidate GNN patch artifact."""
    source_file = candidate["source_file"]
    return (
        f"diff --git a/{source_file} b/{source_file}\n"
        f"--- a/{source_file}\n"
        f"+++ b/{source_file}\n"
        "@@\n"
        "# Proposal only: inspect validation, telemetry, and matrix dimensions before editing this GNN file.\n"
    )


def _build_evaluation_report(
    candidates: List[Dict[str, Any]], target_dir: Path, output_dir: Path
) -> Dict[str, Any]:
    """Build a bounded evaluation report for autonomous candidates."""
    evidence = collect_evaluation_evidence(output_dir)
    return {
        "schema": "gnn_autonomous_evaluation_report_v1",
        "target_dir": str(target_dir),
        "candidate_count": len(candidates),
        "evidence": evidence,
        "decisions": [
            {
                "candidate_id": candidate["candidate_id"],
                "status": "proposal_only",
                "patch_artifact": candidate["patch_artifact"],
                "score": score_candidate_proposal(candidate, evidence),
                "source_mutation_performed": False,
            }
            for candidate in candidates
        ],
        "source_mutation_performed": False,
        "cluster_mutation_performed": False,
    }


def collect_evaluation_evidence(output_dir: Path) -> Dict[str, Any]:
    """Collect existing validator and execution artifacts used for scoring."""
    execution_summaries = sorted(output_dir.rglob("execution_summary.json"))
    validation_artifacts = sorted(output_dir.rglob("*validation*.json"))
    parsed_execution_summaries = [
        _load_json_object(path) for path in execution_summaries
    ]
    success_rates = [
        float(summary["success_rate"])
        for summary in parsed_execution_summaries
        if isinstance(summary.get("success_rate"), (int, float))
    ]
    return {
        "validator_commands": [
            "uv run --extra dev python scripts/check_capability_contracts.py --strict",
            "uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write",
            "uv run --extra dev python -m pytest src/tests/pipeline/test_autonomous_contract.py -q",
        ],
        "execution_summary_files": [str(path) for path in execution_summaries],
        "validation_artifact_files": [str(path) for path in validation_artifacts],
        "execution_success_rate_mean": (
            sum(success_rates) / len(success_rates) if success_rates else None
        ),
    }


def score_candidate_proposal(
    candidate: Dict[str, Any], evidence: Dict[str, Any]
) -> Dict[str, Any]:
    """Score a candidate patch artifact without applying it."""
    score = 40
    reasons: List[str] = ["proposal_only"]
    if Path(str(candidate.get("source_file", ""))).exists():
        score += 20
        reasons.append("source_exists")
    if Path(str(candidate.get("patch_artifact", ""))).exists():
        score += 15
        reasons.append("patch_artifact_written")
    success_rate = evidence.get("execution_success_rate_mean")
    if isinstance(success_rate, (int, float)):
        score += min(15, int(success_rate // 10))
        reasons.append("execution_summary_available")
    if evidence.get("validation_artifact_files"):
        score += 10
        reasons.append("validation_artifacts_available")
    bounded_score = max(0, min(100, score))
    return {
        "value": bounded_score,
        "scale": "0-100",
        "recommendation": (
            "review_with_validators" if bounded_score >= 70 else "needs_more_evidence"
        ),
        "reasons": reasons,
    }


def _load_json_object(path: Path) -> Dict[str, Any]:
    """Load a JSON object, returning an empty dict on invalid content."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _evaluation_report_markdown(report: Dict[str, Any]) -> str:
    """Render a compact Markdown evaluation report."""
    lines = [
        "# Autonomous Evaluation Report",
        "",
        f"- Schema: {report['schema']}",
        f"- Target directory: {report['target_dir']}",
        f"- Candidate count: {report['candidate_count']}",
        "- Source mutation performed: false",
        "- Cluster mutation performed: false",
    ]
    return "\n".join(lines) + "\n"
