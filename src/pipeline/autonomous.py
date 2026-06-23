"""Bounded autonomous proposal artifacts for the GNN pipeline.

Autonomous mode writes reports and candidate patch files only; it does not edit
repository source files, commit changes, or mutate live infrastructure.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from pipeline.container_plan import (
    ResourceLimits,
    generate_container_plan,
    security_review,
)
from pipeline.pipeline_container_plan import PINNED_PIPELINE_IMAGE

VALIDATOR_COMMANDS = [
    "uv run --frozen --extra dev python scripts/check_capability_contracts.py --strict",
    "uv run --frozen --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write",
    "PYTHONPATH=src uv run --frozen python -m pytest src/tests/pipeline/test_autonomous_contract.py -q",
]

AUTONOMY_POLICY = {
    "schema": "gnn_autonomy_policy_v1",
    "mode": "proposal_only",
    "source_mutation_allowed": False,
    "cluster_mutation_allowed": False,
    "requires_human_approval": True,
    "approval_token_required": "human-reviewed",
    "forbidden_actions": [
        "write_source_file",
        "apply_patch_to_repository",
        "git_commit",
        "container_run",
        "cluster_mutation",
    ],
}


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
    plan = generate_container_plan(
        "gnn-autonomous-proposal-review",
        [
            {
                "name": "gnn-autonomous-review",
                "image": PINNED_PIPELINE_IMAGE,
                "command": [
                    "python",
                    "src/main.py",
                    "--autonomous",
                    "--target-dir",
                    str(target_dir),
                    "--output-dir",
                    "output",
                ],
                "mounts": ["gnn-output:/app/output"],
                "resources": ResourceLimits(cpu="1.0", memory="1Gi"),
            }
        ],
    )
    findings = security_review(plan)
    payload = plan.model_dump()
    payload.update(
        {
            "target_dir": str(target_dir),
            "security_review_findings": [finding.model_dump() for finding in findings],
            "security_review_clean": not findings,
        }
    )
    return {
        **payload,
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
        _build_candidate(index, path, autonomous_dir)
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
        "autonomy_policy": dict(AUTONOMY_POLICY),
        "review_workflow": _review_workflow_summary(),
        "audit_log": _audit_log(candidates, evaluation_report),
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


def _build_candidate(index: int, path: Path, autonomous_dir: Path) -> Dict[str, Any]:
    fingerprint = _file_sha256(path)
    candidate_id = f"candidate-{index + 1}-{fingerprint[:8]}"
    return {
        "candidate_id": candidate_id,
        "source_file": str(path),
        "source_sha256": fingerprint,
        "source_summary": _source_summary(path),
        "proposal": (
            "Evaluate matrix dimensions, execution telemetry, validation artifacts, "
            "and interpretability outputs before applying any model patch."
        ),
        "patch_artifact": str(autonomous_dir / f"{candidate_id}.gnn.patch"),
        "review_gate": _review_gate(candidate_id),
        "rollback_descriptor": {
            "schema": "gnn_autonomous_rollback_v1",
            "strategy": "discard_proposal_artifact",
            "source_file": str(path),
            "original_sha256": fingerprint,
            "source_mutation_performed": False,
        },
        "source_mutation_performed": False,
    }


def _candidate_patch_text(candidate: Dict[str, Any]) -> str:
    """Build a non-applied candidate GNN patch artifact."""
    source_file = candidate["source_file"]
    return (
        f"diff --git a/{source_file} b/{source_file}\n"
        f"--- a/{source_file}\n"
        f"+++ b/{source_file}\n"
        "@@\n"
        "# Proposal only: inspect validation, telemetry, matrix dimensions, and human review state before editing this GNN file.\n"
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
                "review_gate": candidate["review_gate"],
                "application_allowed": False,
                "rollback_descriptor": candidate["rollback_descriptor"],
                "source_mutation_performed": False,
            }
            for candidate in candidates
        ],
        "autonomy_policy": dict(AUTONOMY_POLICY),
        "source_mutation_performed": False,
        "cluster_mutation_performed": False,
    }


def collect_evaluation_evidence(output_dir: Path) -> Dict[str, Any]:
    """Collect existing validator and execution artifacts used for scoring."""
    execution_summaries = sorted(output_dir.rglob("execution_summary.json"))
    validation_artifacts = sorted(output_dir.rglob("*validation*.json"))
    semantic_ledgers = sorted(output_dir.rglob("*semantic*fidelity*.json"))
    reliability_ledgers = sorted(output_dir.rglob("*cross*framework*.json"))
    interpretability_artifacts = sorted(output_dir.rglob("*interpretability*.json"))
    report_artifacts = sorted(output_dir.rglob("*report*.json"))
    parsed_execution_summaries = [
        _load_json_object(path) for path in execution_summaries
    ]
    success_rates = [
        float(summary["success_rate"])
        for summary in parsed_execution_summaries
        if isinstance(summary.get("success_rate"), (int, float))
    ]
    failed_counts = [_failed_count(summary) for summary in parsed_execution_summaries]
    return {
        "validator_commands": list(VALIDATOR_COMMANDS),
        "execution_summary_files": [str(path) for path in execution_summaries],
        "validation_artifact_files": [str(path) for path in validation_artifacts],
        "semantic_fidelity_ledger_files": [str(path) for path in semantic_ledgers],
        "cross_framework_ledger_files": [str(path) for path in reliability_ledgers],
        "interpretability_artifact_files": [
            str(path) for path in interpretability_artifacts
        ],
        "report_artifact_files": [str(path) for path in report_artifacts],
        "execution_success_rate_mean": (
            sum(success_rates) / len(success_rates) if success_rates else None
        ),
        "execution_failed_count_total": sum(failed_counts),
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
        if success_rate >= 99:
            score += 5
            reasons.append("high_execution_success_rate")
    if evidence.get("execution_failed_count_total") == 0 and evidence.get(
        "execution_summary_files"
    ):
        score += 5
        reasons.append("no_execution_failures_reported")
    if evidence.get("validation_artifact_files"):
        score += 10
        reasons.append("validation_artifacts_available")
    if evidence.get("semantic_fidelity_ledger_files"):
        score += 5
        reasons.append("semantic_fidelity_ledger_available")
    if evidence.get("cross_framework_ledger_files"):
        score += 5
        reasons.append("cross_framework_ledger_available")
    if evidence.get("interpretability_artifact_files"):
        score += 5
        reasons.append("interpretability_artifacts_available")
    if evidence.get("report_artifact_files"):
        score += 5
        reasons.append("report_artifacts_available")
    bounded_score = max(0, min(100, score))
    return {
        "value": bounded_score,
        "scale": "0-100",
        "recommendation": (
            "review_with_validators" if bounded_score >= 70 else "needs_more_evidence"
        ),
        "reasons": reasons,
        "application_allowed": False,
        "required_review_state": "human-reviewed",
    }


def _load_json_object(path: Path) -> Dict[str, Any]:
    """Load a JSON object, returning an empty dict on invalid content."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _failed_count(summary: Dict[str, Any]) -> int:
    for key in ("failed_count", "failed", "failure_count"):
        value = summary.get(key)
        if isinstance(value, int):
            return max(0, value)
    details = summary.get("execution_details")
    if isinstance(details, list):
        return sum(
            1
            for detail in details
            if isinstance(detail, dict)
            and str(detail.get("status", "")).upper() in {"FAILED", "ERROR"}
        )
    return 0


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_summary(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "line_count": len(text.splitlines()),
        "byte_count": len(text.encode("utf-8")),
        "section_count": sum(1 for line in text.splitlines() if line.startswith("#")),
        "matrix_mentions": sum(
            text.count(token) for token in ("A", "B", "C", "D", "E")
        ),
        "has_model_name": "ModelName" in text,
    }


def _review_gate(candidate_id: str) -> Dict[str, Any]:
    return {
        "schema": "gnn_autonomous_review_gate_v1",
        "candidate_id": candidate_id,
        "required_state": "human-reviewed",
        "current_state": "proposal-only",
        "approval_token": None,
        "application_allowed": False,
        "reviewer_actions": [
            "inspect_patch_artifact",
            "run_validator_commands",
            "record_human_approval_before_any_source_edit",
        ],
    }


def _review_workflow_summary() -> Dict[str, Any]:
    return {
        "schema": "gnn_autonomous_review_workflow_v1",
        "states": ["proposal-only", "human-reviewed", "manually-applied"],
        "initial_state": "proposal-only",
        "automatic_transition_to_manual_apply": False,
        "required_validator_commands": list(VALIDATOR_COMMANDS),
    }


def _audit_log(
    candidates: List[Dict[str, Any]], evaluation_report: Dict[str, Any]
) -> List[Dict[str, Any]]:
    return [
        {
            "event": "autonomous_proposals_emitted",
            "candidate_count": len(candidates),
            "source_mutation_performed": False,
            "cluster_mutation_performed": False,
        },
        {
            "event": "candidate_scores_recorded",
            "decision_count": len(evaluation_report.get("decisions", [])),
            "application_allowed": False,
        },
        {
            "event": "human_review_required",
            "required_state": "human-reviewed",
            "automatic_apply_available": False,
        },
    ]


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
        "- Human review required before any source edit: true",
    ]
    for decision in report.get("decisions", []):
        score = decision.get("score", {})
        lines.append(
            f"- {decision['candidate_id']}: {score.get('value', 'n/a')}/100 "
            f"({score.get('recommendation', 'unknown')})"
        )
    return "\n".join(lines) + "\n"
