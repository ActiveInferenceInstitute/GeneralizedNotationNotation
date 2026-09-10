"""Pins for ``type_checker.resource_estimator`` (previously 27%)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import gnn.type_checker.resource_estimator as resource_estimator
from gnn.type_checker.estimation.estimator import GNNResourceEstimator

_SPEC = """## GNNSection
ActInfPOMDP

## ModelName
EstimatorProbe

## StateSpaceBlock
s[3,1,type=float]
o[3,1,type=int]

## Connections
s-s
s-o

## Footer
EstimatorProbe
"""


def test_facade_exposes_the_canonical_estimator() -> None:
    assert resource_estimator.__all__ == ["GNNResourceEstimator"]
    assert resource_estimator.GNNResourceEstimator is GNNResourceEstimator


def test_main_estimates_single_file_and_prints_report(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tmp_path / "probe.gnn"
    spec.write_text(_SPEC, encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["resource_estimator", str(spec), "-o", str(tmp_path)]
    )

    exit_code = resource_estimator.main()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "EstimatorProbe" in captured.out or "probe" in captured.out


def test_main_estimates_directory_recursive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "specs"
    target.mkdir()
    (target / "probe.gnn").write_text(_SPEC, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["resource_estimator", str(target), "--recursive", "-o", str(tmp_path)],
    )

    exit_code = resource_estimator.main()

    assert exit_code == 0
