"""
MCP integration for the ontology module.

Exposes GNN ontology tools: term validation, ontology mapping,
annotation extraction, and ontology report generation through MCP.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

from . import process_ontology, validate_ontology_terms

import inspect, importlib





def process_ontology_mcp(target_directory: str, output_directory: str,
                         verbose: bool = False) -> Dict[str, Any]:
    """
    Run ontology processing on a directory of GNN files.

    Maps GNN variables to Active Inference Ontology (ActInfO) terms,
    validates annotations, and produces an ontology mapping report.

    Args:
        target_directory: Directory containing GNN files
        output_directory: Directory to save ontology results
        verbose: Enable verbose logging

    Returns:
        Dictionary with success status and mapping summary.
    """
    try:
        success = process_ontology(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose,
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"Ontology processing {'completed successfully' if success else 'completed with issues'}",
        }
    except Exception as e:
        logger.error(f"process_ontology_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def validate_ontology_terms_mcp(terms: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Validate one or more ontology term names against the Active Inference Ontology.

    Args:
        terms: A single term string or list of term strings to validate

    Returns:
        Dictionary with validation results for each term.
    """
    try:
        if isinstance(terms, str):
            terms_list = [t.strip() for t in terms.split(",") if t.strip()]
        else:
            terms_list = list(terms)

        is_valid = validate_ontology_terms(terms_list)
        return {
            "success": True,
            "terms":   terms_list,
            "is_valid": is_valid,
            "message": f"{len(terms_list)} term(s) {'all valid' if is_valid else 'contain invalid entries'}",
        }
    except Exception as e:
        logger.error(f"validate_ontology_terms_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def extract_ontology_annotations_mcp(gnn_content: str) -> Dict[str, Any]:
    """
    Extract ActInfOntologyAnnotation entries from GNN model content.

    Parses the ## ActInfOntologyAnnotation section and returns all
    variable-to-term mappings found.

    Args:
        gnn_content: GNN model content as a string

    Returns:
        Dictionary with extracted annotations and variable-term pairs.
    """
    try:
        lines = gnn_content.splitlines()
        in_annotation = False
        annotations: Dict[str, str] = {}

        for line in lines:
            stripped = line.strip()
            if stripped == "## ActInfOntologyAnnotation":
                in_annotation = True
                continue
            if in_annotation:
                if stripped.startswith("##"):
                    break
                if "=" in stripped and not stripped.startswith("#"):
                    parts = stripped.split("=", 1)
                    var_name = parts[0].strip()
                    term     = parts[1].strip()
                    if var_name:
                        annotations[var_name] = term

        standard_terms = {
            "HiddenState", "Observation", "Action", "PolicyVector",
            "LikelihoodMatrix", "TransitionMatrix", "LogPreferenceVector",
            "PriorOverHiddenStates", "VariationalFreeEnergy", "ExpectedFreeEnergy",
        }
        validated = {k: v for k, v in annotations.items() if v in standard_terms}
        unknown   = {k: v for k, v in annotations.items() if v not in standard_terms}

        return {
            "success":              True,
            "annotations":          annotations,
            "validated_mappings":   validated,
            "unknown_terms":        unknown,
            "total_annotations":    len(annotations),
            "valid_count":          len(validated),
        }
    except Exception as e:
        logger.error(f"extract_ontology_annotations_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_standard_ontology_terms_mcp() -> Dict[str, Any]:
    """
    Return the list of standard Active Inference Ontology (ActInfO) terms.

    Returns:
        Dictionary with term names and their descriptions.
    """
    terms = {
        "HiddenState":            "Latent state variable s",
        "NextHiddenState":         "Next latent state s'",
        "Observation":            "Observable variable o",
        "Action":                 "Action variable u",
        "PolicyVector":           "Policy distribution π",
        "LikelihoodMatrix":       "Observation likelihood A",
        "TransitionMatrix":       "State transition B",
        "LogPreferenceVector":    "Log preference C",
        "PriorOverHiddenStates":  "Prior beliefs D",
        "Habit":                  "Habitual policy E",
        "VariationalFreeEnergy":  "VFE F",
        "ExpectedFreeEnergy":     "EFE G",
        "Time":                   "Discrete time t",
        "Precision":              "Precision parameter γ/β",
    }
    return {"success": True, "terms": terms, "count": len(terms)}


# ── MCP Registration ────────────────────────────────────────────────────────

def register_tools(mcp_instance) -> None:
    """Register ontology tools with the MCP server."""

    mcp_instance.register_tool(
        "process_ontology",
        process_ontology_mcp,
        {"type": "object", "properties": {
            "target_directory": {"type": "string", "description": "Directory with GNN files"},
            "output_directory": {"type": "string", "description": "Directory for ontology results"},
            "verbose":          {"type": "boolean", "default": False},
        }, "required": ["target_directory", "output_directory"]},
        "Map GNN variables to Active Inference Ontology terms and produce an ontology report.",
        module=__package__, category="ontology",
    )

    mcp_instance.register_tool(
        "validate_ontology_terms",
        validate_ontology_terms_mcp,
        {"type": "object", "properties": {
            "terms": {"oneOf": [
                {"type": "string", "description": "Comma-separated term names"},
                {"type": "array", "items": {"type": "string"}},
            ]},
        }, "required": ["terms"]},
        "Validate ontology term names against the Active Inference Ontology.",
        module=__package__, category="ontology",
    )

    mcp_instance.register_tool(
        "extract_ontology_annotations",
        extract_ontology_annotations_mcp,
        {"type": "object", "properties": {
            "gnn_content": {"type": "string", "description": "GNN model content as a string"},
        }, "required": ["gnn_content"]},
        "Extract ActInfOntologyAnnotation variable-to-term mappings from GNN model content.",
        module=__package__, category="ontology",
    )

    mcp_instance.register_tool(
        "list_standard_ontology_terms",
        list_standard_ontology_terms_mcp,
        {},
        "Return the canonical list of Active Inference Ontology (ActInfO) terms and descriptions.",
        module=__package__, category="ontology",
    )

    logger.info("ontology module MCP tools registered (5 tools).")
