#!/usr/bin/env python3
"""
Round-trip result dataclasses for the GNN round-trip test suite.

Extracted from ``testing.test_round_trip``.
"""

from dataclasses import (
    dataclass,
    field,
)
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
)


# Define types locally to avoid circular imports
# Recovery: define simple types to avoid import issues
@dataclass
class RoundTripResult:
    """Record one source-to-target round-trip conversion result."""

    source_format: Any = None
    target_format: Any = None
    success: bool = True
    original_model: Any = None
    converted_content: str = ""
    parsed_back_model: Any = None
    checksum_original: str = ""
    checksum_converted: str = ""
    test_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    differences: List[str] = field(default_factory=list)

    def add_error(self, error: str) -> Any:
        """Record a failing error message and mark the result unsuccessful."""
        self.errors.append(error)
        self.success = False

    def add_warning(self, warning: str) -> Any:
        """Record a non-fatal warning message for the round trip."""
        self.warnings.append(warning)

    def add_difference(self, difference: str) -> Any:
        """Record a detected model/content difference."""
        self.differences.append(difference)


@dataclass
class ComprehensiveTestReport:
    """Aggregate round-trip results and derived report metrics."""

    reference_file: str = ""
    test_timestamp: datetime = field(default_factory=datetime.now)
    round_trip_results: List[RoundTripResult] = field(default_factory=list)
    critical_errors: List[str] = field(default_factory=list)

    def add_result(self, result: RoundTripResult) -> Any:
        """Append one round-trip result to the report."""
        self.round_trip_results.append(result)

    @property
    def total_tests(self) -> int:
        """Return the number of recorded round-trip tests."""
        return len(self.round_trip_results)

    @property
    def successful_tests(self) -> int:
        """Return the count of successful round-trip results."""
        return sum(1 for r in self.round_trip_results if r.success)

    @property
    def failed_tests(self) -> int:
        """Return the count of unsuccessful round-trip results."""
        return self.total_tests - self.successful_tests

    def get_success_rate(self) -> float:
        """Return the success percentage across recorded results."""
        return (
            (self.successful_tests / self.total_tests * 100)
            if self.total_tests > 0
            else 0.0
        )

    def get_format_summary(self) -> Dict[Any, Dict[str, int]]:
        """Return per-target-format success and total counts."""
        summary: dict[Any, Any] = {}
        for result in self.round_trip_results:
            fmt = result.target_format
            if fmt not in summary:
                summary[fmt] = {"success": 0, "total": 0}
            summary[fmt]["total"] += 1
            if result.success:
                summary[fmt]["success"] += 1
        return summary
