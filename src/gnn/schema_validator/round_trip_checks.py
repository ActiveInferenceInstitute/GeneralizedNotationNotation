"""Round-trip and cross-format checks for :mod:`gnn.schema_validator.validator`.

Holds round-trip validation testing, cross-format consistency
validation, and the ParsedGNN-to-markdown conversion used by the
round-trip path, mixed into GNNValidator.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

from gnn.types import GNNFormat, ParsedGNN, RoundTripResult, ValidationResult

logger = logging.getLogger(__name__)


class RoundTripChecksMixin:
    """Mixin holding round-trip and cross-format validation methods."""

    round_trip_tester: Any
    cross_validator: Any

    def _perform_round_trip_validation(
        self, parsed_gnn: ParsedGNN, result: ValidationResult
    ) -> Any:
        """Perform round-trip validation testing."""
        try:
            # Create a temporary markdown file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                # Write parsed content back to markdown
                markdown_content = self._convert_parsed_gnn_to_markdown(parsed_gnn)
                f.write(markdown_content)
                temp_file = Path(f.name)

            try:
                # Run round-trip tests on a subset of formats
                test_formats: list[Any] = [
                    GNNFormat.JSON,
                    GNNFormat.XML,
                    GNNFormat.YAML,
                ]

                for fmt in test_formats:
                    try:
                        parse_result = (
                            self.round_trip_tester._convert_parsed_gnn_to_parse_result(
                                parsed_gnn
                            )
                        )
                        round_trip_model = parse_result.model
                        raw_round_trip = self.round_trip_tester._test_round_trip(
                            round_trip_model, fmt
                        )
                        round_trip_result = RoundTripResult(
                            source_format=raw_round_trip.source_format,
                            target_format=raw_round_trip.target_format,
                            success=raw_round_trip.success,
                            original_model=raw_round_trip.original_model,
                            converted_content=raw_round_trip.converted_content,
                            parsed_back_model=raw_round_trip.parsed_back_model,
                            differences=list(raw_round_trip.differences),
                            warnings=list(raw_round_trip.warnings),
                            errors=list(raw_round_trip.errors),
                            test_time=raw_round_trip.test_time,
                            checksum_original=raw_round_trip.checksum_original,
                            checksum_converted=raw_round_trip.checksum_converted,
                        )
                        result.add_round_trip_result(round_trip_result)

                    except Exception as e:
                        result.warnings.append(
                            f"Round-trip test failed for {fmt.value}: {e}"
                        )

                # Summary
                success_rate = result.get_round_trip_success_rate()
                if success_rate == 100.0:
                    result.suggestions.append(
                        "Perfect round-trip compatibility achieved"
                    )
                elif success_rate >= 80.0:
                    result.warnings.append(
                        f"Good round-trip compatibility: {success_rate:.1f}%"
                    )
                else:
                    result.errors.append(
                        f"Poor round-trip compatibility: {success_rate:.1f}%"
                    )

            finally:
                # Clean up temporary file
                temp_file.unlink(missing_ok=True)

        except Exception as e:
            result.warnings.append(f"Round-trip validation failed: {e}")

    def _validate_cross_format_consistency(
        self, content: str, result: ValidationResult
    ) -> Any:
        """Validate cross-format consistency."""
        try:
            if self.cross_validator is None:
                result.warnings.append("Cross-format validator is not initialized")
                return
            cross_result = self.cross_validator.validate_cross_format_consistency(
                content
            )
            result.cross_format_consistent = cross_result.is_consistent

            if cross_result.is_consistent:
                result.suggestions.append("Cross-format consistency validated")
            else:
                result.warnings.extend(cross_result.inconsistencies)
                result.warnings.extend(cross_result.warnings)

        except Exception as e:
            result.warnings.append(f"Cross-format validation failed: {e}")

    def _convert_parsed_gnn_to_markdown(self, parsed_gnn: ParsedGNN) -> str:
        """Convert ParsedGNN back to markdown format."""
        lines: list[Any] = []

        lines.append("## GNNSection")
        lines.append(parsed_gnn.gnn_section)
        lines.append("")

        lines.append("## GNNVersionAndFlags")
        lines.append(parsed_gnn.version)
        lines.append("")

        lines.append("## ModelName")
        lines.append(parsed_gnn.model_name)
        lines.append("")

        lines.append("## ModelAnnotation")
        lines.append(parsed_gnn.model_annotation)
        lines.append("")

        lines.append("## StateSpaceBlock")
        for var_name, var in parsed_gnn.variables.items():
            dim_str = (
                f"[{','.join(map(str, var.dimensions))}]" if var.dimensions else ""
            )
            type_str = (
                f",type={var.data_type}" if var.data_type != "categorical" else ""
            )
            desc_str = f" # {var.description}" if var.description else ""
            lines.append(f"{var_name}{dim_str}{type_str}{desc_str}")
        lines.append("")

        lines.append("## Connections")
        for conn in parsed_gnn.connections:
            source = (
                ",".join(conn.source) if isinstance(conn.source, list) else conn.source
            )
            target = (
                ",".join(conn.target) if isinstance(conn.target, list) else conn.target
            )
            desc_str = f" # {conn.description}" if conn.description else ""
            lines.append(f"{source}{conn.symbol}{target}{desc_str}")
        lines.append("")

        lines.append("## InitialParameterization")
        for param_name, param_value in parsed_gnn.parameters.items():
            lines.append(f"{param_name}={param_value}")
        lines.append("")

        lines.append("## Time")
        time_type = parsed_gnn.time_config.get("type", "Dynamic")
        lines.append(time_type)
        lines.append("")

        lines.append("## Footer")
        lines.append(parsed_gnn.footer)

        return "\n".join(lines)
