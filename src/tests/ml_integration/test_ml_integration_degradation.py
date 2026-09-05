#!/usr/bin/env python3
"""Degradation-path tests for ml_integration (Step 14).

Pins the contract that when ML training is not possible (scikit-learn
missing, or fewer than two usable GNN samples), ``process_ml_integration``
still succeeds, records structural-analysis fallback entries, and saves the
feature analysis (``feature_statistics`` + ``model_families``) in BOTH
degradation cases.

These tests are deliberately sklearn-aware: assertions that differ between
a sklearn-present and sklearn-absent environment branch on
``importlib.util.find_spec("sklearn")``, so the file passes in both.
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

RESULTS_KEYS = (
    "status",
    "target_dir",
    "output_dir",
    "models_trained",
    "feature_importance",
    "cross_validation",
    "framework_status",
    "extracted_features",
)

GNN_CONTENT = """\
## StateSpaceBlock

A[3, 2]
s[2]
"""


def _load_results(output_dir: Path) -> dict[str, Any]:
    results_file = output_dir / "ml_integration_results.json"
    with open(results_file) as f:
        return cast("dict[str, Any]", json.load(f))


def test_one_file_degradation_records_structural_analysis(tmp_path: Path) -> None:
    from ml_integration import process_ml_integration

    target = tmp_path / "target"
    target.mkdir()
    (target / "model.md").write_text(GNN_CONTENT)
    out = tmp_path / "out"

    success = process_ml_integration(target, out)
    assert success is True

    results = _load_results(out)
    assert results["status"] == "completed"
    models = results["models_trained"]
    assert len(models) == 1
    entry = models[0]
    assert entry["type"] == "structural_analysis"
    assert entry["framework"] == "internal_stats"
    assert entry["validation_status"] == "not_applicable"
    assert entry["source"] == "model.md"
    if HAS_SKLEARN:
        assert entry["note"] == "Need >=2 GNN files for ML classification (have 1)"
    else:
        assert entry["note"] == "sklearn not available"

    # The doc-aligned fix: feature analysis is saved on degradation too,
    # regardless of which degradation branch fired.
    assert results["feature_statistics"]
    assert results["model_families"]


def test_one_file_degradation_framework_status_matches_environment(
    tmp_path: Path,
) -> None:
    from ml_integration import process_ml_integration

    target = tmp_path / "target"
    target.mkdir()
    (target / "model.md").write_text(GNN_CONTENT)
    out = tmp_path / "out"

    assert process_ml_integration(target, out) is True
    results = _load_results(out)
    if HAS_SKLEARN:
        assert results["framework_status"]["sklearn"] == "available"
    else:
        assert results["framework_status"]["sklearn"] == "missing"


def test_zero_files_reports_no_files_and_writes_results(tmp_path: Path) -> None:
    from ml_integration import process_ml_integration

    target = tmp_path / "target"
    target.mkdir()
    out = tmp_path / "out"

    success = process_ml_integration(target, out)
    assert success is True

    assert (out / "ml_integration_results.json").exists()
    results = _load_results(out)
    assert results["status"] == "no_files"
    assert results["models_trained"] == []


def test_results_json_contract_keys(tmp_path: Path) -> None:
    from ml_integration import process_ml_integration

    target = tmp_path / "target"
    target.mkdir()
    (target / "model.md").write_text(GNN_CONTENT)
    out = tmp_path / "out"

    assert process_ml_integration(target, out) is True
    results = _load_results(out)
    for key in RESULTS_KEYS:
        assert key in results
