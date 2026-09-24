#!/usr/bin/env python3
"""
BC-02c/M-04: degraded validator mode is a visible receipt, never silent.

When ``GNNValidator`` (or the round-trip parser) is unavailable, the MCP
processors fall back to lightweight checks — and the summary plus every
per-file result must say so explicitly via ``validator_mode`` /
``parser_mode`` and ``degraded``, so lightweight output is never
indistinguishable from full validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gnn.schema_validator as schema_validator_module
from gnn.mcp.processors import process_gnn_folder

SAMPLE_GNN = """## ModelName
ReceiptModel
"""


class _BrokenValidator:
    """Stand-in whose construction fails like a missing dependency."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("validator wiring unavailable")


def _make_target(tmp_path: Path) -> tuple[Path, Path]:
    target = tmp_path / "gnn"
    target.mkdir()
    (target / "model.md").write_text(SAMPLE_GNN, encoding="utf-8")
    return target, tmp_path / "out"


def test_full_mode_receipt(tmp_path: Path) -> None:
    """With GNNValidator available the receipt says full and not degraded."""
    target, out = _make_target(tmp_path)

    assert process_gnn_folder(target, out) is True

    summary = json.loads((out / "gnn_processing_summary.json").read_text())
    assert summary["validator_mode"] == "full"
    assert summary["degraded"] is False
    assert "validator_error" not in summary
    assert summary["results"][0]["validator_mode"] == "full"


def test_lightweight_receipt_when_validator_unavailable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Validator init failure must surface as a lightweight receipt."""
    target, out = _make_target(tmp_path)
    monkeypatch.setattr(schema_validator_module, "GNNValidator", _BrokenValidator)

    assert process_gnn_folder(target, out) is True

    summary = json.loads((out / "gnn_processing_summary.json").read_text())
    assert summary["validator_mode"] == "lightweight"
    assert summary["degraded"] is True
    assert "validator wiring unavailable" in summary["validator_error"]
    assert summary["results"][0]["validator_mode"] == "lightweight"
    # Lightweight results must not claim a validation level was applied.
    assert "validation_level" not in summary["results"][0]
