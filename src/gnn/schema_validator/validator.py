"""
GNN schema validation module.

This module provides comprehensive validation, parsing, and analysis capabilities
for GNN (Generalized Notation Notation) model files with enhanced round-trip
testing support and cross-format validation.

The regex-based GNNParser lives in gnn/schema_validator/syntax.py; this module
owns GNNValidator and the validate_gnn_file_comprehensive entry point.

Enhanced Features:
- Complete format ecosystem support (23 formats)
- Binary file validation for pickle/binary formats
- Cross-format consistency validation
- Round-trip semantic preservation testing
- Enhanced error reporting and suggestions
"""

import logging
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any, Optional, Union, cast

from gnn.schemas.section_contract import REQUIRED_SECTIONS
from gnn.types import (
    GNNFormat,
    ParsedGNN,
    RoundTripResult,
    ValidationLevel,
    ValidationResult,
)

# Lark parser removed - too complex and not needed

# Try to import round-trip testing capabilities (owned by syntax.py)
try:
    from gnn.schema_validator.syntax import ROUND_TRIP_AVAILABLE, GNNParser
except ImportError:  # pragma: no cover
    ROUND_TRIP_AVAILABLE = False
    GNNParser = cast(Any, None)  # type: ignore[misc]

# RoundTripResult is already imported from gnn.types above
# No need to import it again from testing module to avoid circular deps

logger = logging.getLogger(__name__)


class GNNValidator:
    """Enhanced validator for GNN files with comprehensive round-trip and cross-format support."""

    def __init__(
        self,
        schema_path: Optional[Path] = None,
        enable_round_trip_testing: bool = False,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        enable_cross_validation: bool = True,
    ) -> None:
        """Initialize the instance.

        ``schema_path`` is retained for cross-format callers that bind a
        validator to a specific schema file; it is not parsed at init.
        """
        if schema_path is None:
            schema_path = Path(__file__).parent.parent / "schemas/json.json"

        self.schema_path = schema_path
        self.enable_round_trip_testing = (
            enable_round_trip_testing and ROUND_TRIP_AVAILABLE
        )
        self.validation_level = validation_level

        # Initialize enhanced parser
        self.parser = GNNParser(enhanced_validation=True)

        # Initialize cross-format validator (optionally, to avoid recursion)
        self.cross_validator = None
        if enable_cross_validation:
            try:
                from gnn.schema_validator.cross_format import CrossFormatValidator

                self.cross_validator = CrossFormatValidator()
            except ImportError:
                self.cross_validator = None

        # Initialize round-trip tester if enabled
        if self.enable_round_trip_testing:
            try:
                from gnn.testing.test_round_trip import GNNRoundTripTester

                self.round_trip_tester = GNNRoundTripTester()
                logger.info("Round-trip testing enabled for comprehensive validation")
            except Exception as e:
                logger.warning(f"Could not initialize round-trip tester: {e}")
                self.enable_round_trip_testing = False

    def _level_rank(self, level: Union[ValidationLevel, str]) -> int:
        """Map validation level to an integer rank for safe comparisons."""
        mapping: dict[Any, Any] = {
            ValidationLevel.BASIC: 10,
            ValidationLevel.STANDARD: 20,
            ValidationLevel.STRICT: 30,
            ValidationLevel.RESEARCH: 40,
            ValidationLevel.ROUND_TRIP: 50,
        }
        if isinstance(level, ValidationLevel):
            return cast("int", mapping.get(level, 0))
        try:
            # allow passing a string
            return cast("int", mapping.get(ValidationLevel(level), 0))
        except (ValueError, TypeError):
            logger.warning("Unknown validation level %r; defaulting rank to 0", level)
            return 0

    def validate_file(
        self,
        file_path: Union[str, Path],
        validation_level: Optional[ValidationLevel] = None,
    ) -> ValidationResult:
        """Enhanced validation with comprehensive testing capabilities."""
        import time

        start_time = time.time()

        validation_level = validation_level or self.validation_level
        file_path = Path(file_path)

        result = ValidationResult(
            is_valid=True,
            validation_level=validation_level,
            format_tested=self._detect_file_format(file_path),
        )

        try:
            # Step 1: Basic file access and format detection
            file_format = self._detect_file_format(file_path)
            result.format_tested = file_format

            # Handle binary formats
            if file_format in ["binary", "pickle"]:
                return self._validate_binary_file(file_path, result)

            # Read text content
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError as e:
                result.errors.append(f"File encoding error: {e}")
                result.is_valid = False
                return result

            # Step 2: Parse the file using enhanced parser
            try:
                parsed_gnn = self.parser.parse_file(file_path)
                result.semantic_checksum = parsed_gnn.semantic_checksum
                result.metadata["parsed_successfully"] = True
                result.metadata["source_format"] = parsed_gnn.source_format
            except Exception as e:
                result.errors.append(f"Parsing failed: {e}")
                result.is_valid = False
                if validation_level == ValidationLevel.BASIC:
                    return result
                # Continue with content-based validation for higher levels
                parsed_gnn = cast(Any, None)

            # Step 3: Validation based on level
            # Use safe rank comparisons to avoid Enum comparison issues
            if self._level_rank(validation_level) >= self._level_rank(
                ValidationLevel.BASIC
            ):
                self._validate_basic_structure(content, result, file_format)

            if (
                self._level_rank(validation_level)
                >= self._level_rank(ValidationLevel.STANDARD)
                and parsed_gnn
            ):
                self._validate_semantics(parsed_gnn, result)

            if self._level_rank(validation_level) >= self._level_rank(
                ValidationLevel.STRICT
            ):
                self._validate_strict_requirements(parsed_gnn, content, result)

            if self._level_rank(validation_level) >= self._level_rank(
                ValidationLevel.RESEARCH
            ):
                self._validate_research_standards(parsed_gnn, content, result)

            # Step 4: Round-trip testing if enabled and requested
            if (
                (
                    isinstance(validation_level, ValidationLevel)
                    and validation_level == ValidationLevel.ROUND_TRIP
                )
                and self.enable_round_trip_testing
                and parsed_gnn
            ):
                self._perform_round_trip_validation(parsed_gnn, result)

            # Step 5: Cross-format consistency if available
            # Use safe comparison via rank to avoid Enum ordering errors
            if (
                self._level_rank(validation_level)
                >= self._level_rank(ValidationLevel.STRICT)
                and self.cross_validator
                and parsed_gnn
            ):
                self._validate_cross_format_consistency(content, result)

            # Final result determination
            result.is_valid = len(result.errors) == 0

            # Performance metrics
            end_time = time.time()
            result.performance_metrics = {
                "validation_time": end_time - start_time,
                "content_length": float(len(content)),
            }
            result.validation_level = validation_level

            return result

        except Exception as e:
            result.errors.append(f"Validation failed with exception: {e}")
            result.is_valid = False
            return result

    def _detect_file_format(self, file_path: Path) -> str:
        """Detect file format from extension."""
        suffix = file_path.suffix.lower()
        format_map: dict[str, Any] = {
            ".md": "markdown",
            ".markdown": "markdown",
            ".json": "json",
            ".xml": "xml",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".pkl": "pkl",
            ".pickle": "pickle",
        }
        return cast("str", format_map.get(suffix, "unknown"))

    def _validate_structured_format(
        self, content: str, result: ValidationResult, file_format: str
    ) -> Any:
        """Validate structured formats like JSON, XML, YAML."""
        try:
            if file_format == "json":
                import json

                data = json.loads(content)
                # Basic validation for JSON structure
                if isinstance(data, dict):
                    if "model_name" in data:
                        result.warnings.append("JSON format validated successfully")
                    else:
                        result.warnings.append(
                            "JSON format valid but missing expected model_name field"
                        )
                else:
                    result.errors.append(
                        "JSON should contain a dictionary/object at root level"
                    )

            elif file_format == "xml":
                import xml.etree.ElementTree as ET  # nosec B405

                try:
                    ET.fromstring(content)  # nosec B314
                    result.warnings.append("XML format validated successfully")
                except ET.ParseError as e:
                    result.errors.append(f"XML parsing error: {e}")

            elif file_format == "yaml":
                try:
                    import yaml

                    data = yaml.safe_load(content)
                    result.warnings.append("YAML format validated successfully")
                except Exception as e:
                    result.errors.append(f"YAML parsing error: {e}")

        except ImportError as e:
            result.warnings.append(
                f"Cannot validate {file_format} format: missing library ({e})"
            )
        except Exception as e:
            result.errors.append(f"Error validating {file_format} format: {e}")

    def _validate_binary_file(
        self, file_path: Path, result: ValidationResult
    ) -> ValidationResult:
        """Validate binary files (pickle format)."""
        try:
            with open(file_path, "rb") as f:
                # Try to read first few bytes to ensure it's accessible
                header = f.read(10)

            # Check for pickle signature
            if header.startswith(b"\x80\x03") or b"pickle" in header:
                result.warnings.append(
                    "Binary pickle format detected - validation limited to accessibility check"
                )
            else:
                result.warnings.append(
                    "Unknown binary format - validation limited to accessibility check"
                )

            result.metadata["binary_format"] = True
            result.metadata["file_size"] = file_path.stat().st_size
            result.is_valid = True

        except Exception as e:
            result.errors.append(f"Binary file access error: {e}")
            result.is_valid = False

        return result

    def _validate_basic_structure(
        self, content: str, result: ValidationResult, file_format: str
    ) -> Any:
        """Enhanced basic validation with format-specific checks."""
        if len(content.strip()) == 0:
            result.errors.append("File is empty")
            return

        # Format-specific basic validation
        if file_format == "markdown":
            self._validate_markdown_structure(content, result)
        elif file_format in ["json", "xml", "yaml"]:
            self._validate_structured_format(content, result, file_format)
        else:
            result.warnings.append(
                f"Unknown file format: {file_format}, using basic validation"
            )
            # Basic text validation
            if len(content) < 10:
                result.warnings.append("File content is very short")
            if "\x00" in content:
                result.warnings.append("File contains null bytes - may be binary")

    def _validate_strict_requirements(
        self, parsed_gnn: Optional[ParsedGNN], content: str, result: ValidationResult
    ) -> Any:
        """Validate strict requirements for research-grade models."""
        if not parsed_gnn:
            result.errors.append("Parsed model required for strict validation")
            return

        # Check for complete documentation
        if (
            not parsed_gnn.model_annotation
            or len(parsed_gnn.model_annotation.strip()) < 50
        ):
            result.warnings.append(
                "Model annotation should be more descriptive for research use"
            )

        # Check for ontology mappings
        if not parsed_gnn.ontology_mappings:
            result.suggestions.append(
                "Consider adding ontology mappings for better interoperability"
            )

        # Check for equations
        if not parsed_gnn.equations:
            result.suggestions.append(
                "Consider adding mathematical equations for clarity"
            )

        # Validate parameter completeness
        if len(parsed_gnn.parameters) < len(parsed_gnn.variables) * 0.5:
            result.warnings.append("Many variables lack parameter specifications")

        # Bridge provenance strictness: a document whose GNNSection
        # identifier carries the `FepLean` prefix (the fep_lean<->GNN
        # bridge convention, bridge contract section 4) MUST carry the
        # bridge provenance keys in its Signature under strict
        # validation. Non-bridge documents are unchanged.
        if str(parsed_gnn.gnn_section or "").startswith("FepLean"):
            missing = [
                key
                for key in (
                    "source_repository",
                    "source_commit",
                    "lean_module",
                    "projection_tool",
                    "target_syntax",
                )
                if not (parsed_gnn.signature or {}).get(key)
            ]
            for key in missing:
                result.errors.append(
                    f"Bridge document (GNNSection {parsed_gnn.gnn_section!r})"
                    f" is missing required provenance key '{key}' in its"
                    " Signature section (bridge contract section 4)"
                )

    def _validate_research_standards(
        self, parsed_gnn: Optional[ParsedGNN], content: str, result: ValidationResult
    ) -> Any:
        """Validate research-grade standards."""
        if not parsed_gnn:
            return

        # Check for signature/provenance
        if not parsed_gnn.signature:
            result.suggestions.append("Add signature section for provenance tracking")

        # Check for time configuration
        if not parsed_gnn.time_config:
            result.suggestions.append("Specify time configuration for reproducibility")

        # Validate model parameters
        if not parsed_gnn.model_parameters:
            result.suggestions.append("Add model parameters for complete specification")

        # Check for research-grade documentation
        research_keywords: list[Any] = [
            "hypothesis",
            "method",
            "experiment",
            "analysis",
            "result",
        ]
        annotation_lower = parsed_gnn.model_annotation.lower()
        found_keywords = [kw for kw in research_keywords if kw in annotation_lower]

        if len(found_keywords) < 2:
            result.suggestions.append(
                "Consider adding research context (hypothesis, methods, etc.)"
            )

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

    def _validate_markdown_structure(
        self, content: str, result: ValidationResult
    ) -> Any:
        """Validate comprehensive GNN markdown file structure and semantics."""
        lines = content.split("\n")

        required_sections: list[str] = list(REQUIRED_SECTIONS)

        found_sections: list[Any] = []
        for line in lines:
            if line.startswith("## "):
                section_name = line[3:].strip()
                found_sections.append(section_name)

        # Check for missing required sections
        missing_sections = set(required_sections) - set(found_sections)
        for section in missing_sections:
            result.errors.append(f"Required section missing: {section}")

        # Also check for required sections that exist but have no substantive content
        try:
            for section in required_sections:
                # Find header and extract following content until next header
                pattern = rf"^##\s+{re.escape(section)}\s*$"
                matches = list(re.finditer(pattern, content, re.MULTILINE))
                if matches:
                    m = matches[0]
                    start = m.end()
                    # Find next header
                    next_header = re.search(r"^##\s+.+$", content[start:], re.MULTILINE)
                    end = start + next_header.start() if next_header else len(content)
                    section_text = content[start:end].strip()
                    # GNN sections routinely open with '#' description
                    # comments (every bundled example does); skip leading
                    # blank/comment lines before judging substance. A
                    # comment-only body still counts as missing.
                    body_lines = [
                        ln
                        for ln in section_text.splitlines()
                        if ln.strip() and not ln.strip().startswith("#")
                    ]
                    if not body_lines or len("\n".join(body_lines)) < 3:
                        result.errors.append(f"Required section missing: {section}")
        except Exception as e:
            # Non-fatal parsing of content; do not stop validation
            logger.debug(f"Non-fatal error during section content validation: {e}")

        # Validate using basic parser
        try:
            parser = GNNParser()
            parsed_gnn = parser.parse_content(content)

            if len(parsed_gnn.variables) == 0:
                result.warnings.append("No variables found in StateSpaceBlock")

            if len(parsed_gnn.connections) == 0:
                result.warnings.append("No connections found in Connections section")

        except Exception as e:
            result.warnings.append(f"Basic parser validation failed: {e}")

    def _validate_structure(self, content: str, result: ValidationResult) -> Any:
        """Previous method - now delegates to markdown validation."""
        self._validate_markdown_structure(content, result)

    def _validate_semantics(self, parsed: ParsedGNN, result: ValidationResult) -> Any:
        """Validate semantic consistency of parsed GNN model."""
        # Validate variable references in connections
        variable_names = set(parsed.variables)

        for connection in parsed.connections:
            # Check source variables
            source_vars = (
                connection.source
                if isinstance(connection.source, list)
                else [connection.source]
            )
            for var in source_vars:
                if var not in variable_names and not self._is_valid_variable_reference(
                    var
                ):
                    result.errors.append(
                        f"Undefined variable in connection source: {var}"
                    )

            # Check target variables
            target_vars = (
                connection.target
                if isinstance(connection.target, list)
                else [connection.target]
            )
            for var in target_vars:
                if var not in variable_names and not self._is_valid_variable_reference(
                    var
                ):
                    result.errors.append(
                        f"Undefined variable in connection target: {var}"
                    )

        # Validate Active Inference conventions
        self._validate_active_inference_conventions(parsed, result)

        # Validate mathematical consistency
        self._validate_mathematical_consistency(parsed, result)

    def _is_valid_variable_reference(self, var: str) -> bool:
        """Check if variable reference follows GNN conventions."""
        # Handle complex variable references like (A,B) or time-indexed variables
        return (var.startswith("(") and var.endswith(")")) or "=" in var

    def _validate_active_inference_conventions(
        self, parsed: ParsedGNN, result: ValidationResult
    ) -> Any:
        """Validate Active Inference naming and structure conventions."""
        # Check for proper A, B, C, D matrix naming
        ai_matrices: dict[str, Any] = {"A": [], "B": [], "C": [], "D": []}

        for var_name, _var in parsed.variables.items():
            if var_name.startswith("A_m"):
                ai_matrices["A"].append(var_name)
            elif var_name.startswith("B_f"):
                ai_matrices["B"].append(var_name)
            elif var_name.startswith("C_m"):
                ai_matrices["C"].append(var_name)
            elif var_name.startswith("D_f"):
                ai_matrices["D"].append(var_name)

        # Validate matrix dimension consistency
        if ai_matrices["A"] and ai_matrices["D"]:
            result.metadata["active_inference_matrices"] = ai_matrices

        # Check for proper state/observation variable naming
        state_vars = [name for name in parsed.variables if name.startswith("s_f")]
        obs_vars = [name for name in parsed.variables if name.startswith("o_m")]

        if state_vars:
            result.metadata["state_variables"] = state_vars
        if obs_vars:
            result.metadata["observation_variables"] = obs_vars

    def _validate_mathematical_consistency(
        self, parsed: ParsedGNN, result: ValidationResult
    ) -> Any:
        """Validate mathematical consistency of parameters and dimensions."""
        # Check for matrix dimension consistency with variable definitions
        for param_name, param_value in parsed.parameters.items():
            if param_name in parsed.variables:
                var = parsed.variables[param_name]

                # Validate matrix dimensions match variable dimensions
                if isinstance(param_value, list) and isinstance(param_value[0], list):
                    # Matrix parameter
                    matrix_rows = len(param_value)
                    matrix_cols = len(param_value[0]) if param_value else 0

                    expected_dims = var.dimensions
                    if len(expected_dims) >= 2:
                        if (
                            matrix_rows != expected_dims[0]
                            or matrix_cols != expected_dims[1]
                        ):
                            result.warnings.append(
                                f"Matrix {param_name} dimensions {matrix_rows}x{matrix_cols} "
                                f"don't match variable definition {expected_dims}"
                            )

                # Check for stochasticity in probability matrices
                if param_name.startswith(("A_", "B_", "D_")) and isinstance(
                    param_value, list
                ):
                    if not self._check_stochasticity(param_value):
                        result.warnings.append(
                            f"Matrix {param_name} may not be properly stochastic (rows should sum to 1)"
                        )

    def _check_stochasticity(self, matrix_data: Any, tolerance: float = 1e-6) -> bool:
        """Check if matrix rows sum to 1 (stochastic constraint)."""
        try:
            if isinstance(matrix_data, list) and matrix_data:
                if isinstance(matrix_data[0], (list, tuple)):
                    # 2D matrix
                    for row in matrix_data:
                        if isinstance(row, (list, tuple)):
                            row_sum = sum(
                                float(x) for x in row if isinstance(x, (int, float))
                            )
                            if abs(row_sum - 1.0) > tolerance:
                                return False
                    return True
                else:
                    # 1D vector
                    total_sum = sum(
                        float(x) for x in matrix_data if isinstance(x, (int, float))
                    )
                    return abs(total_sum - 1.0) <= tolerance
        except (TypeError, ValueError):
            logger.debug(
                "Could not verify stochastic matrix normalization, assuming valid"
            )
        return True  # Conservative: assume valid if can't check


def validate_gnn_file_comprehensive(file_path: Union[str, Path]) -> ValidationResult:
    """Validate a GNN file with the full GNNValidator pipeline."""
    validator = GNNValidator()
    return validator.validate_file(file_path)


def validate_gnn_file(file_path: Union[str, Path]) -> ValidationResult:
    """Old name for :func:`validate_gnn_file_comprehensive`; emits DeprecationWarning."""
    warnings.warn(
        "validate_gnn_file is an old name; use validate_gnn_file_comprehensive instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return validate_gnn_file_comprehensive(file_path)
