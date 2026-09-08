#!/usr/bin/env python3
"""
Report generation mixin for the GNN round-trip test suite.

Extracted from ``testing.test_round_trip``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from .round_trip_results import ComprehensiveTestReport

logger = logging.getLogger(__name__)


class RoundTripReportMixin:
    """Verbatim methods moved from ``GNNRoundTripTester``."""

    def generate_report(
        self, report: ComprehensiveTestReport, output_file: Optional[Path] = None
    ) -> str:
        """Generate a comprehensive test report."""
        lines: list[Any] = []

        lines.append("# GNN Round-Trip Testing Report")
        lines.append(
            f"**Generated:** {report.test_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append(f"**Reference File:** `{report.reference_file}`")
        lines.append("")

        lines.append("## Summary")
        lines.append(f"- **Total Tests:** {report.total_tests}")
        lines.append(f"- **Successful:** {report.successful_tests}")
        lines.append(f"- **Failed:** {report.failed_tests}")
        lines.append(f"- **Success Rate:** {report.get_success_rate():.1f}%")
        lines.append("")

        # Format summary
        lines.append("## Format Summary")
        format_summary = report.get_format_summary()

        for fmt, stats in format_summary.items():
            success_rate = (
                (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            )
            status = "✅" if success_rate == 100 else "⚠️" if success_rate > 50 else "❌"
            lines.append(
                f"- **{fmt.value}** {status}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)"
            )

        lines.append("")

        # Detailed results
        lines.append("## Detailed Results")

        for result in report.round_trip_results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            lines.append(f"### {result.target_format.value} {status}")

            if result.checksum_original and result.checksum_converted:
                checksum_match = result.checksum_original == result.checksum_converted
                checksum_status = "✅" if checksum_match else "❌"
                lines.append(f"- **Semantic Checksum:** {checksum_status}")

            if result.differences:
                lines.append("- **Differences:**")
                for diff in result.differences:
                    lines.append(f"  - {diff}")

            if result.errors:
                lines.append("- **Errors:**")
                for error in result.errors:
                    lines.append(f"  - {error}")

            if result.warnings:
                lines.append("- **Warnings:**")
                for warning in result.warnings:
                    lines.append(f"  - {warning}")

            lines.append("")

        # Critical issues
        if report.critical_errors:
            lines.append("## Critical Issues")
            for error in report.critical_errors:
                lines.append(f"- ❌ {error}")
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")

        if report.get_success_rate() == 100.0:
            lines.append(
                "🎉 **All tests passed!** The GNN system has 100% confidence in round-trip format conversion."
            )
        else:
            lines.append(
                "⚠️ **Some tests failed.** Review the failed formats and address the differences:"
            )

            failed_formats = [
                result.target_format.value
                for result in report.round_trip_results
                if not result.success
            ]
            for fmt in failed_formats:
                lines.append(f"  - Fix serialization/parsing for {fmt}")

        report_content = "\n".join(lines)

        if output_file:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output_file.parent, delete=False
            ) as tmp_f:
                tmp_f.write(report_content)
            os.replace(tmp_f.name, str(output_file))
            logger.info(f"Report saved to {output_file}")

        return report_content
