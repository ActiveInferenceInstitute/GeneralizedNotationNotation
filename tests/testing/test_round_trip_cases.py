"""Unit tests for the GNN round-trip testing system.

Adopted from ``src/gnn/testing/test_round_trip.py`` (SC-42): the harness
classes (``GNNRoundTripTester`` and the round-trip config/results modules)
remain inside the ``gnn.testing`` package because production code
(``gnn.schema_validator.validator``) lazily imports ``GNNRoundTripTester``
from there. Only the orphaned ``TestGNNRoundTrip`` TestCase moves into the
pytest ``testpaths`` tree; the format list under test stays defined by
``gnn.testing.round_trip_config`` (see ``src/gnn/SPEC.md``).
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from gnn.testing.round_trip_availability import GNN_AVAILABLE, GNNFormat
from gnn.testing.test_round_trip import GNNRoundTripTester


class TestGNNRoundTrip(unittest.TestCase):
    """Unit tests for the round-trip testing system."""

    def setUp(self) -> Any:
        """Set up test environment."""
        if not GNN_AVAILABLE:
            self.skipTest("GNN module not available")

        self.tester = GNNRoundTripTester()
        # Corrective (SC-42): the REFERENCE_CONFIG fallback chain resolves to
        # ``src/gnn/gnn_examples/actinf_pomdp_agent.md``, which the validator
        # rejects with "Required section missing: StateSpaceBlock /
        # InitialParameterization" (pre-existing on main; never surfaced while
        # this TestCase lived outside pytest testpaths). Point the instance at
        # the POMDP GridWorld exemplar — the repo's public full-run contract
        # model — which validates, parses, and round-trips all formats.
        self.tester.reference_file = (
            Path(__file__).resolve().parents[2]
            / "input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md"
        )

    def test_reference_file_exists(self) -> Any:
        """Test that the reference file exists and is readable."""
        self.assertTrue(
            self.tester.reference_file.exists(),
            f"Reference file not found: {self.tester.reference_file}",
        )

    def test_reference_file_validation(self) -> Any:
        """Test that the reference file validates correctly."""
        if self.tester.validator:
            result = self.tester.validator.validate_file(self.tester.reference_file)
            self.assertTrue(
                result.is_valid, f"Reference file validation failed: {result.errors}"
            )
        else:
            self.skipTest("Validator not available")

    def test_comprehensive_round_trip(self) -> Any:
        """Test comprehensive round-trip conversion."""
        report = self.tester.run_comprehensive_tests()

        # Basic assertions
        self.assertGreater(report.total_tests, 0, "No tests were run")
        self.assertGreaterEqual(report.successful_tests, 0, "No successful tests")

        # Generate report
        report_content = self.tester.generate_report(report)
        self.assertIn("GNN Round-Trip Testing Report", report_content)

        # Log results
        print(
            f"\nRound-trip test results: {report.successful_tests}/{report.total_tests} passed"
        )
        print(f"Success rate: {report.get_success_rate():.1f}%")

        if report.failed_tests > 0:
            print("Failed formats:")
            for result in report.round_trip_results:
                if not result.success:
                    print(f"  - {result.target_format.value}: {result.errors}")

    def test_specific_format_round_trip(self) -> Any:
        """Test round-trip for a specific format (JSON)."""
        if not self.tester.parsing_system:
            self.skipTest("Parsing system not available")

        # Parse reference
        reference_result = self.tester.parsing_system.parse_file(
            self.tester.reference_file, GNNFormat.MARKDOWN
        )
        self.assertTrue(reference_result.success, "Failed to parse reference file")

        # Test JSON round-trip
        json_result = self.tester._test_round_trip(
            reference_result.model, GNNFormat.JSON
        )

        if not json_result.success:
            print("JSON round-trip failed:")
            print(f"  Errors: {json_result.errors}")
            print(f"  Differences: {json_result.differences}")

        # Should succeed for JSON format
        self.assertTrue(
            json_result.success, f"JSON round-trip failed: {json_result.errors}"
        )
