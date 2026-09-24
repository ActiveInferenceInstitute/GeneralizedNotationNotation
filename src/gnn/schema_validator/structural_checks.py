"""Structural and format checks for :mod:`gnn.schema_validator.validator`.

Holds file-format detection, binary/structured/markdown structure
validation, and strict/research requirement checks mixed into
GNNValidator.
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional, cast

from gnn.schema_validator.syntax import GNNParser
from gnn.schemas.section_contract import REQUIRED_SECTIONS
from gnn.types import ParsedGNN, ValidationResult

logger = logging.getLogger(__name__)


class StructuralChecksMixin:
    """Mixin holding structural file-format validation methods."""

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
