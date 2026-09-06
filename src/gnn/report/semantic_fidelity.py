"""Markdown rendering for semantic fidelity ledgers."""

from __future__ import annotations

from typing import Any


def render_semantic_fidelity_markdown(ledger: dict[str, Any]) -> str:
    """Render a compact Markdown report for the semantic fidelity gate."""
    lines = [
        "# GNN Semantic Fidelity Ledger",
        "",
        f"- Schema: {ledger['schema']}",
        f"- Families: {ledger['family_count']}",
        f"- Strict: {str(ledger['strict']).lower()}",
        f"- Formats: {', '.join(ledger.get('formats', [])) or 'none'}",
        "",
        "| Family | Status | Models | Failed Models |",
        "| --- | --- | ---: | ---: |",
    ]
    for family in ledger["families"]:
        lines.append(
            "| {name} | {status} | {models} | {failed} |".format(
                name=family["name"],
                status=family["status"],
                models=family["model_count"],
                failed=len(family.get("failed_models", [])),
            )
        )
    lines.extend(["", "## Round Trips", ""])
    for family in ledger["families"]:
        for model in family["models"]:
            lines.append(f"### {family['name']} / {model['source_file']}")
            lines.append("")
            lines.append("| Format | Status | Reason | Differences | Artifact |")
            lines.append("| --- | --- | --- | ---: | --- |")
            for round_trip in model["round_trips"]:
                lines.append(
                    "| {fmt} | {status} | {reason} | {diffs} | {artifact} |".format(
                        fmt=round_trip["format"],
                        status=round_trip["status"],
                        reason=round_trip.get("reason") or "",
                        diffs=len(round_trip.get("differences", [])),
                        artifact=round_trip.get("artifact", ""),
                    )
                )
            lines.append("")
    return "\n".join(lines)
