"""Report helpers for model-family acceptance ledgers."""

from __future__ import annotations

from typing import Any, Dict


def render_model_family_acceptance_markdown(ledger: Dict[str, Any]) -> str:
    """Render a compact Markdown report from a model-family acceptance ledger."""
    lines = [
        "# GNN Model Family Acceptance Ledger",
        "",
        f"- Schema: {ledger['schema']}",
        f"- Families: {ledger['family_count']}",
        f"- Strict: {str(ledger['strict']).lower()}",
        f"- Only steps: {ledger['only_steps'] or 'full pipeline'}",
        f"- Frameworks: {ledger.get('frameworks') or 'pipeline default'}",
        "",
        "| Family | Status | Models | Passed Steps | Failed Steps | Skipped Steps | Raw Failed Steps | Profiled Unsupported Skips |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in ledger["families"]:
        counts = family["step_status_counts"]
        raw_failed = sum(
            1 for status in family.get("raw_steps", {}).values() if status == "failed"
        )
        profiled_unsupported = sum(
            1
            for evidence in family.get("step_evidence", {}).values()
            if isinstance(evidence, dict)
            and evidence.get("acceptance") == "profiled_unsupported_skip"
        )
        lines.append(
            "| {name} | {status} | {models} | {passed} | {failed} | {skipped} | {raw_failed} | {profiled} |".format(
                name=family["name"],
                status=family["status"],
                models=family["interpretability_summary"]["model_count"],
                passed=counts.get("passed", 0),
                failed=counts.get("failed", 0),
                skipped=counts.get("skipped", 0),
                raw_failed=raw_failed,
                profiled=profiled_unsupported,
            )
        )
    return "\n".join(lines) + "\n"
