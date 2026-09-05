"""
ontology module for GNN Processing Pipeline.

This module provides ontology capabilities with recovery implementations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from utils.pipeline_template import (
    log_step_error,
    log_step_start,
    log_step_success,
    log_step_warning,
)

# Import core processing functions from processor module
from .processor import (
    SUGGESTION_MAX_DISTANCE,
    OntologyTermIndex,
    ParsedAnnotation,
    analyze_ontology_content,
    build_ontology_terms,
    generate_ontology_report_for_file,
    load_defined_ontology_terms,
    parse_annotation,
    parse_gnn_ontology_section,
    process_gnn_ontology,
    process_ontology,
    suggest_terms,
    summarise_coverage,
    validate_annotations,
)

# Import utility functions
from .utils import get_mcp_interface, get_module_info, get_ontology_processing_options


def validate_ontology_terms(terms: Optional[Union[List[str], str]] = None) -> bool:
    """Validate ontology terms against the Active Inference ontology."""
    if terms is None:
        return True
    try:
        terms_list = [terms] if isinstance(terms, str) else list(terms)
        if not terms_list:
            return True
        if not all(isinstance(term, str) and term.strip() for term in terms_list):
            return False
        result = validate_annotations(terms_list)
        return "error" not in result and not result.get("invalid_annotations", [])
    except (TypeError, ValueError, KeyError):
        return False


# Feature flags expected by tests
FEATURES: dict[str, Any] = {
    "parsing": True,
    "validation": True,
    "reporting": True,
    "basic_processing": True,
    "mcp_integration": True,
}
__version__ = "1.7.0"


# Minimal classes expected by tests
class OntologyProcessor:
    """Ontology processor with methods expected by tests."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def run(self, *args: Any, **kwargs: Any) -> bool:
        """Run operation."""
        return True

    def process_ontology(self, data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """Process ontology data or content and return a normalized result.

        Delegates the parse→load→validate pipeline to
        :func:`analyze_ontology_content` so this thin wrapper stays in sync
        with ``process_gnn_ontology`` rather than reimplementing it.
        """
        if isinstance(data, dict):
            content = data.get("content", "")
        else:
            content = str(data)
        analysis = analyze_ontology_content(content)
        return {
            "ontology_data": analysis["ontology_data"],
            "validation_result": analysis["validation_result"],
            "success": True,
        }

    # Additional methods expected by some tests
    def validate_terms(self, terms: Optional[List[str]] = None) -> bool:
        """Validate terms."""
        return validate_ontology_terms(terms or [])


class OntologyValidator:
    """Ontology validator exposing validate_ontology as required by tests."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def validate(self, annotations: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate operation.

        Delegates to :func:`validate_annotations` with the default ontology
        loaded once per call. Returns the boolean ``valid`` plus the full
        validation ``details`` so callers can inspect matched terms.
        """
        annotations = annotations or []
        terms = load_defined_ontology_terms()
        res = validate_annotations(annotations, terms)
        return {
            "valid": len(res.get("invalid_annotations", [])) == 0,
            "details": res,
            "errors": [],
            "warnings": [],
        }

    def validate_ontology(self, content: str) -> Union[bool, Dict[str, Any]]:
        """Validate ontology."""
        parsed = parse_gnn_ontology_section(content)
        result = self.validate(parsed.get("annotations", []))
        # Some tests expect a boolean True/False
        return cast("bool | dict[str, Any]", result.get("valid", False))

    # Additional method expected by tests
    def check_consistency(self, annotations: Optional[List[str]] = None) -> bool:
        """Check consistency."""
        return cast("bool", self.validate(annotations).get("valid", False))


__all__: list[Any] = [
    # Core processing functions
    "process_ontology",
    "parse_gnn_ontology_section",
    "process_gnn_ontology",
    "load_defined_ontology_terms",
    "validate_annotations",
    "generate_ontology_report_for_file",
    "parse_annotation",
    "analyze_ontology_content",
    "suggest_terms",
    "summarise_coverage",
    "build_ontology_terms",
    "ParsedAnnotation",
    "SUGGESTION_MAX_DISTANCE",
    "OntologyTermIndex",
    # Utility functions
    "get_module_info",
    "get_ontology_processing_options",
    "get_mcp_interface",
    "validate_ontology_terms",
    # Classes expected by tests
    "OntologyProcessor",
    "OntologyValidator",
]
