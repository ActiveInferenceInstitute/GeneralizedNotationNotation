"""Unit tests for ``gnn.testing.round_trip_strategy.RoundTripTestStrategy``.

The strategy module measured 0% in the verification harness (bundled tests
live outside pytest ``testpaths``). The strategy delegates to
``GNNRoundTripTester``; these tests inject a fake tester to pin the
aggregation, summary, error-conversion, and result-persistence behavior
without depending on the real tester's heavy import surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gnn.testing.round_trip_strategy import RoundTripResult, RoundTripTestStrategy


class _FakeReport:
    def __init__(self, results: list[Any]) -> None:
        self.round_trip_results = results


def _make_result(
    *, source_file: str, target_format: str, success: bool
) -> SimpleNamespace:
    return SimpleNamespace(
        source_file=source_file,
        target_format=target_format,
        success=success,
        errors=[] if success else ["boom"],
        warnings=[],
        test_time=0.01,
    )


class TestRoundTripResultDataclass:
    def test_defaults(self) -> None:
        result = RoundTripResult(success=True, target_format="json")

        assert result.errors == []
        assert result.warnings == []
        assert result.test_time == 0.0
        assert result.source_file is None
        assert result.metadata == {}


class TestRoundTripTestStrategy:
    def test_configure_creates_output_dir(self, tmp_path: Path) -> None:
        strategy = RoundTripTestStrategy()
        out = tmp_path / "results"

        strategy.configure(output_dir=out)

        assert out.exists()
        assert strategy.output_dir == out

    def test_test_without_tester_returns_unavailable_envelope(self) -> None:
        strategy = RoundTripTestStrategy()
        strategy.round_trip_tester = None

        result = strategy.test([Path("a.md")])

        assert result == {
            "success": False,
            "error": "Round-trip tester not available",
            "tests_run": 0,
            "files_tested": 0,
        }

    def test_test_aggregates_file_results_and_summary(self) -> None:
        strategy = RoundTripTestStrategy()
        strategy.round_trip_tester = SimpleNamespace(
            run_comprehensive_tests=lambda: _FakeReport(
                [
                    _make_result(
                        source_file="a.md", target_format="json", success=True
                    ),
                    _make_result(
                        source_file="a.md", target_format="pkl", success=False
                    ),
                ]
            )
        )

        result = strategy.test([Path("a.md")])

        assert result["tests_run"] == 1
        assert result["files_tested"] == 1
        assert result["success"] is False  # 50% < 80% threshold
        file_result = result["file_results"]["a.md"]
        assert file_result["success_rate"] == 50.0
        assert file_result["total_formats_tested"] == 2
        summary = result["summary"]
        assert summary["total_files"] == 1
        assert summary["successful_files"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["format_statistics"]["json"] == {"success": 1, "total": 1}
        assert summary["format_statistics"]["pkl"] == {"success": 0, "total": 1}

    def test_test_passes_when_success_rate_meets_threshold(self) -> None:
        strategy = RoundTripTestStrategy()
        strategy.round_trip_tester = SimpleNamespace(
            run_comprehensive_tests=lambda: _FakeReport(
                [
                    _make_result(
                        source_file="a.md", target_format="json", success=True
                    ),
                    _make_result(source_file="a.md", target_format="pkl", success=True),
                ]
            )
        )

        result = strategy.test([Path("a.md")])

        assert result["success"] is True
        assert result["file_results"]["a.md"]["success_rate"] == 100.0

    def test_test_exception_is_converted_to_failed_file_result(self) -> None:
        strategy = RoundTripTestStrategy()

        def boom() -> Any:
            raise RuntimeError("tester exploded")

        strategy.round_trip_tester = SimpleNamespace(run_comprehensive_tests=boom)

        result = strategy.test([Path("a.md")])

        assert result["success"] is False
        assert "tester exploded" in result["file_results"]["a.md"]["error"]

    def test_results_persisted_to_output_dir(self, tmp_path: Path) -> None:
        strategy = RoundTripTestStrategy()
        strategy.round_trip_tester = SimpleNamespace(
            run_comprehensive_tests=lambda: _FakeReport(
                [_make_result(source_file="a.md", target_format="json", success=True)]
            )
        )
        out = tmp_path / "results"
        strategy.configure(output_dir=out)

        strategy.test([Path("a.md")])

        saved = list(out.glob("round_trip_test_results_*.json"))
        assert len(saved) == 1
        payload = json.loads(saved[0].read_text(encoding="utf-8"))
        assert payload["tests_run"] == 1
        assert payload["summary"]["total_files"] == 1

    def test_save_failure_is_logged_not_raised(self, tmp_path: Path) -> None:
        strategy = RoundTripTestStrategy()
        strategy.round_trip_tester = SimpleNamespace(
            run_comprehensive_tests=lambda: _FakeReport(
                [_make_result(source_file="a.md", target_format="json", success=True)]
            )
        )
        # A file path that cannot be opened (directory exists at the name) forces
        # the save's `open(...)` to raise; the strategy must swallow it.
        out = tmp_path / "results"
        strategy.configure(output_dir=out)
        import os

        os.chmod(out, 0o500)  # rx, no write (POSIX)
        try:
            # Should not raise even though the save fails.
            strategy.test([Path("a.md")])
        finally:
            os.chmod(out, 0o700)
