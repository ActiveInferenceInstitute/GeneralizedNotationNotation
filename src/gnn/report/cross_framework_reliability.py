"""Markdown rendering for cross-framework reliability ledgers."""

from __future__ import annotations

from typing import Any


def render_cross_framework_reliability_markdown(ledger: dict[str, Any]) -> str:
    """Render a compact Markdown report for v2 cross-framework reliability."""
    lines = [
        "# GNN Cross-Framework Reliability Ledger",
        "",
        f"- Schema: {ledger['schema']}",
        f"- Families: {ledger['family_count']}",
        f"- Strict: {str(ledger['strict']).lower()}",
        f"- Frameworks: {', '.join(ledger.get('frameworks', []))}",
        "",
        "| Family | Status | Comparison | Compared Frameworks | Required Failures |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for family in ledger["families"]:
        comparison = family["comparison"]
        lines.append(
            "| {name} | {status} | {comparison} | {frameworks} | {failures} |".format(
                name=family["name"],
                status=family["status"],
                comparison=comparison["status"],
                frameworks=", ".join(comparison.get("compared_frameworks", []))
                or "none",
                failures=len(family.get("required_framework_failures", [])),
            )
        )
    lines.extend(["", "## Framework Profiles", ""])
    for family in ledger["families"]:
        lines.append(f"### {family['name']}")
        lines.append("")
        lines.append("| Framework | Profile | Status | Reason | Metrics |")
        lines.append("| --- | --- | --- | --- | --- |")
        for framework, result in family["frameworks"].items():
            metrics = result.get("metrics", {})
            metric_status = "available" if metrics.get("available") else "missing"
            lines.append(
                "| {framework} | {profile} | {status} | {reason} | {metrics} |".format(
                    framework=framework,
                    profile=result["profile"],
                    status=result["status"],
                    reason=result.get("reason") or "",
                    metrics=metric_status,
                )
            )
        lines.append("")
    return "\n".join(lines)
