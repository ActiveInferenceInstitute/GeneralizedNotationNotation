"""Unit tests for ``gnn.testing.performance_benchmarks`` and ``gnn.testing.round_trip_config``.

The performance-benchmark helper measured 0% in the verification harness
because it lives under ``src/gnn/testing`` (outside pytest ``testpaths``).
These tests pin its observable behavior with cheap iteration counts against
the shared GNN sample, plus the round-trip config dictionaries.
"""

from __future__ import annotations

from pathlib import Path

from gnn.testing.performance_benchmarks import (
    GNNPerformanceBenchmark,
    PerformanceMetrics,
    benchmark_gnn_files,
)
from gnn.testing.round_trip_config import (
    FORMAT_TEST_CONFIG,
    LOGGING_CONFIG,
    OUTPUT_CONFIG,
    REFERENCE_CONFIG,
    TEST_BEHAVIOR_CONFIG,
)
from tests.helpers.gnn_samples import SAMPLE_GNN_CONTENT, write_sample_gnn_markdown


class TestRoundTripConfig:
    def test_logging_config_shape(self) -> None:
        assert LOGGING_CONFIG["log_level"] == "WARNING"
        assert LOGGING_CONFIG["enable_debug"] is False
        assert isinstance(LOGGING_CONFIG["suppress_parser_warnings"], bool)

    def test_format_config_default_test_formats_include_reference_formats(self) -> None:
        formats = FORMAT_TEST_CONFIG["test_formats"]
        assert "markdown" in formats and "json" in formats
        # PNML is parse-only (SPEC) and deliberately absent from the default list.
        assert "pnml" not in formats
        assert FORMAT_TEST_CONFIG["test_all_formats"] is False

    def test_format_categories_cover_every_kind(self) -> None:
        cats = FORMAT_TEST_CONFIG["test_categories"]
        assert set(cats) == {
            "schema_formats",
            "language_formats",
            "formal_formats",
            "grammar_formats",
            "temporal_formats",
            "binary_formats",
        }
        assert all(cats.values())

    def test_behavior_and_output_and_reference_configs(self) -> None:
        assert TEST_BEHAVIOR_CONFIG["fail_fast"] is False
        assert TEST_BEHAVIOR_CONFIG["compute_checksums"] is True
        assert OUTPUT_CONFIG["export_json_results"] is True
        assert REFERENCE_CONFIG["reference_file"].endswith(".md")
        assert isinstance(REFERENCE_CONFIG["fallback_reference_files"], list)


class TestPerformanceMetricsDataclass:
    def test_default_error_count_is_zero(self) -> None:
        metrics = PerformanceMetrics(
            operation_name="x",
            execution_time=0.001,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            throughput_ops_per_sec=1000.0,
            complexity_score=1.0,
            accuracy_score=1.0,
        )

        assert metrics.error_count == 0
        assert metrics.operation_name == "x"


class TestGNNPerformanceBenchmark:
    def _sample_file(self, tmp_path: Path) -> Path:
        target = tmp_path / "model.md"
        write_sample_gnn_markdown(target)
        return target

    def test_benchmark_parsing_appends_result_and_reports(self, tmp_path: Path) -> None:
        benchmark = GNNPerformanceBenchmark(test_iterations=2)
        metrics = benchmark.benchmark_parsing(self._sample_file(tmp_path))

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.operation_name == "parse_model"
        assert metrics.error_count == 0
        assert metrics.accuracy_score == 1.0
        assert metrics.complexity_score > 0
        assert metrics in benchmark.results

    def test_benchmark_validation_appends_result(self, tmp_path: Path) -> None:
        benchmark = GNNPerformanceBenchmark(test_iterations=2)
        metrics = benchmark.benchmark_validation(self._sample_file(tmp_path))

        # The validator reports per-iteration validation findings (not
        # iteration failures) as error_count; the sample carries findings,
        # so pin the non-negative contract rather than a specific count.
        assert metrics.operation_name == "validate_model"
        assert metrics.error_count >= 0
        assert metrics in benchmark.results

    def test_generate_report_summarizes_recorded_results(self, tmp_path: Path) -> None:
        benchmark = GNNPerformanceBenchmark(test_iterations=1)
        benchmark.benchmark_parsing(self._sample_file(tmp_path))
        benchmark.benchmark_validation(self._sample_file(tmp_path))

        report = benchmark.generate_report()

        assert report["summary"]["total_operations"] == 2
        assert {op["name"] for op in report["operations"]} == {
            "parse_model",
            "validate_model",
        }

    def test_generate_report_returns_error_without_results(self) -> None:
        benchmark = GNNPerformanceBenchmark()

        assert benchmark.generate_report() == {
            "error": "No benchmark results available"
        }


def test_benchmark_gnn_files_runs_both_ops_per_file(tmp_path: Path) -> None:
    target = tmp_path / "model.md"
    write_sample_gnn_markdown(target)

    report = benchmark_gnn_files([target])

    assert "summary" in report
    assert report["summary"]["total_operations"] == 2
    assert {op["name"] for op in report["operations"]} == {
        "parse_model",
        "validate_model",
    }


def test_sample_gnn_content_is_nonempty() -> None:
    assert SAMPLE_GNN_CONTENT.strip()
    assert "## ModelName" in SAMPLE_GNN_CONTENT
