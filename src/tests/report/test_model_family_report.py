from __future__ import annotations

from report.model_family import render_model_family_acceptance_markdown


def test_model_family_acceptance_report_renders_status_table() -> None:
    ledger = {
        "schema": "gnn_model_family_acceptance_ledger_v1",
        "family_count": 1,
        "strict": True,
        "only_steps": "3,5,6",
        "frameworks": "pymdp",
        "families": [
            {
                "name": "basics",
                "status": "passed",
                "step_status_counts": {"passed": 3, "failed": 0, "skipped": 22},
                "raw_steps": {"11": "failed", "12": "passed"},
                "step_evidence": {
                    "11": {"acceptance": "allowed_unsupported"},
                    "12": {"acceptance": "required"},
                },
                "interpretability_summary": {"model_count": 2},
            }
        ],
    }

    markdown = render_model_family_acceptance_markdown(ledger)

    assert "# GNN Model Family Acceptance Ledger" in markdown
    assert "| basics | passed | 2 | 3 | 0 | 22 | 1 | 1 |" in markdown
