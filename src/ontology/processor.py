#!/usr/bin/env python3
"""
Ontology processor module for GNN Processing Pipeline.

This module provides the main ontology processing functionality.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, cast

from utils.pipeline_template import log_step_error, log_step_start, log_step_success

# Import core processing functions from processor module
# Note: Core functions are defined in this module; avoid self-import


def process_ontology(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
) -> bool:
    """
    Process ontology for GNN files.

    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments

    Returns:
        True if processing successful, False otherwise
    """
    logger = logging.getLogger("ontology")
    strict_validation = kwargs.get("strict_validation", False)
    recursive = kwargs.get("recursive", True)
    ontology_terms_file = kwargs.get("ontology_terms_file")

    try:
        log_step_start(logger, "Processing ontology")

        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Any] = {
            "processed_files": 0,
            "reports": [],
            "success": True,
            "errors": [],
            "configuration": {
                "strict_validation": strict_validation,
                "recursive": recursive,
                "ontology_terms_file": str(ontology_terms_file)
                if ontology_terms_file is not None
                else None,
            },
        }

        if not isinstance(strict_validation, bool) or not isinstance(recursive, bool):
            results["success"] = False
            results["errors"].append(
                {
                    "error": "strict_validation and recursive must be booleans",
                    "error_type": "invalid_configuration",
                }
            )

        ontology_terms: dict[str, Any] | None = None
        if results["success"]:
            try:
                ontology_terms = load_defined_ontology_terms(
                    Path(ontology_terms_file)
                    if ontology_terms_file is not None
                    else None
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                results["success"] = False
                results["errors"].append(
                    {
                        "file": str(ontology_terms_file),
                        "error": str(exc),
                        "error_type": "ontology_terms_load_error",
                    }
                )

        # Process each .md file and generate a per-file ontology report.
        discovery = (
            Path(target_dir).rglob("*.md")
            if recursive is True
            else Path(target_dir).glob("*.md")
        )
        gnn_files = sorted(discovery) if results["success"] else []
        results["processed_files"] = len(gnn_files)
        for gnn_file in gnn_files:
            relative_file = gnn_file.relative_to(Path(target_dir))
            report_dir = results_dir / relative_file.parent
            file_report = generate_ontology_report_for_file(
                Path(gnn_file), report_dir, ontology_terms=ontology_terms
            )
            if not file_report.get("success", False):
                results["success"] = False
                results["errors"].append(
                    {
                        "file": str(gnn_file),
                        "error": file_report.get("error", "unknown"),
                    }
                )
            else:
                # Preserve the established report-file API (the path form
                # follows the caller's output_dir) while nested output
                # directories prevent same-stem models from overwriting.
                results["reports"].append(file_report["report_file"])
                invalid = file_report["report"]["validation_result"].get(
                    "invalid_annotations", []
                )
                if strict_validation and invalid:
                    results["success"] = False
                    results["errors"].append(
                        {
                            "file": str(gnn_file),
                            "error": "strict ontology validation failed",
                            "invalid_annotations": invalid,
                        }
                    )
        # Save aggregate results
        results_file = results_dir / "ontology_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        if results["success"]:
            log_step_success(logger, "Ontology processing completed successfully")
        else:
            log_step_error(logger, "Ontology processing failed")

        return cast("bool", results["success"])

    except Exception as e:
        log_step_error(logger, f"Ontology processing failed: {e}")
        return False


def parse_gnn_ontology_section(content: str) -> Dict[str, Any]:
    """
    Parse GNN ontology section from content.

    Args:
        content: GNN file content

    Returns:
        Dictionary with parsed ontology information
    """
    try:
        if not content.strip():
            return {}

        # Basic ontology parsing
        ontology_data: dict[str, Any] = {
            "concepts": [],
            "relations": [],
            "properties": [],
            "annotations": [],
        }

        lines = content.splitlines()
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Markdown section names are case-insensitive for GNN processing,
            # but only an exact level-two ontology heading opens this block.
            if line.startswith("##"):
                heading = line[2:].strip().casefold()
                current_section = (
                    "ontology"
                    if heading in {"ontology", "actinfontologyannotation"}
                    else None
                )
                continue

            if current_section == "ontology":
                if line.startswith("#") or line.startswith("```"):
                    continue
                line = re.sub(r"^[-*+]\s+", "", line)
                # Parse ontology content
                if "=" in line:
                    # Handle A=LikelihoodMatrix style annotations
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    ontology_data["annotations"].append(f"{key}={value}")
                elif ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    if key.lower() in ["concept", "concepts"]:
                        ontology_data["concepts"].append(value)
                    elif key.lower() in ["relation", "relations"]:
                        ontology_data["relations"].append(value)
                    elif key.lower() in ["property", "properties"]:
                        ontology_data["properties"].append(value)
                    elif key.lower() in ["annotation", "annotations"]:
                        ontology_data["annotations"].append(value)

        return ontology_data

    except Exception as e:
        return {
            "error": str(e),
            "concepts": [],
            "relations": [],
            "properties": [],
            "annotations": [],
        }


def process_gnn_ontology(
    gnn_file: str, ontology_terms: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Process ontology for a single GNN file.

    Args:
        gnn_file: Path to the GNN file

    Returns:
        Dictionary with ontology processing results
    """
    try:
        file_path = Path(gnn_file)

        if not file_path.exists():
            return {"success": False, "error": f"File not found: {gnn_file}"}

        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse ontology section
        ontology_data = parse_gnn_ontology_section(content)

        # Load defined ontology terms when the caller did not already resolve a
        # run-specific vocabulary.
        resolved_terms = (
            load_defined_ontology_terms() if ontology_terms is None else ontology_terms
        )

        # Validate annotations
        validation_result = validate_annotations(
            ontology_data.get("annotations", []), resolved_terms
        )

        return {
            "success": True,
            "file_path": str(file_path),
            "ontology_data": ontology_data,
            "validation_result": validation_result,
            "ontology_terms": resolved_terms,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def _normalise_ontology_terms(data: Any) -> Dict[str, Any]:
    """Normalize supported ontology JSON shapes into a term-keyed dictionary."""
    if isinstance(data, dict) and "terms" in data:
        data = data["terms"]

    terms: dict[str, Any] = {}
    casefold_names: dict[str, str] = {}

    def add_term(name: Any, metadata: Any, *, category: str | None = None) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("ontology term names must be non-empty strings")
        normalized_name = name.strip()
        folded_name = normalized_name.casefold()
        previous = casefold_names.get(folded_name)
        if previous is not None and previous != normalized_name:
            raise ValueError(
                f"ontology terms are ambiguous when case-folded: {previous!r} and "
                f"{normalized_name!r}"
            )

        if isinstance(metadata, dict):
            normalized_metadata = dict(metadata)
        elif isinstance(metadata, str):
            normalized_metadata = {"description": metadata}
        elif metadata is None:
            normalized_metadata = {"description": ""}
        else:
            raise ValueError(
                f"ontology metadata for {normalized_name!r} must be an object or string"
            )
        if category is not None:
            normalized_metadata.setdefault("category", category)
        casefold_names[folded_name] = normalized_name
        terms[normalized_name] = normalized_metadata

    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, str):
                add_term(entry, None)
            elif isinstance(entry, dict):
                name = entry.get("name") or entry.get("term") or entry.get("id")
                metadata = {
                    key: value
                    for key, value in entry.items()
                    if key not in {"name", "term", "id"}
                }
                add_term(name, metadata)
            else:
                raise ValueError(
                    "ontology term lists may contain only strings or objects"
                )
    elif isinstance(data, dict):
        category_format = any(isinstance(value, list) for value in data.values())
        if category_format:
            if not all(isinstance(value, list) for value in data.values()):
                raise ValueError("ontology category mappings must contain lists only")
            for category, entries in data.items():
                for entry in entries:
                    if isinstance(entry, str):
                        add_term(entry, None, category=str(category))
                    elif isinstance(entry, dict):
                        name = entry.get("name") or entry.get("term") or entry.get("id")
                        metadata = {
                            key: value
                            for key, value in entry.items()
                            if key not in {"name", "term", "id"}
                        }
                        add_term(name, metadata, category=str(category))
                    else:
                        raise ValueError(
                            "ontology category lists may contain only strings or objects"
                        )
        else:
            for name, metadata in data.items():
                add_term(name, metadata)
    else:
        raise ValueError("ontology JSON must be an object or list")

    if not terms:
        raise ValueError("ontology contains no terms")
    return {
        name: terms[name]
        for name in sorted(terms, key=lambda term_name: term_name.casefold())
    }


def load_defined_ontology_terms(
    ontology_terms_file: Path | None = None,
) -> Dict[str, Any]:
    """
    Load defined ontology terms from the Active Inference ontology terms file.

    Returns:
        Dictionary mapping term names to their definitions (including description and URI)
    """
    logger = logging.getLogger("ontology")

    explicit_file = (
        Path(ontology_terms_file) if ontology_terms_file is not None else None
    )
    # Priority order for ontology files. An explicit file is authoritative and
    # fails closed instead of silently falling back to the built-in vocabulary.
    search_paths: list[Path] = (
        [explicit_file]
        if explicit_file is not None
        else [
            Path(__file__).parent / "act_inf_ontology_terms.json",
            Path("src/ontology/act_inf_ontology_terms.json"),
        ]
    )

    for ontology_file in search_paths:
        if not ontology_file.exists():
            if explicit_file is not None:
                raise FileNotFoundError(
                    f"Ontology terms file not found: {ontology_file}"
                )
            continue
        try:
            with open(ontology_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            terms = _normalise_ontology_terms(data)
            logger.info(
                "Loaded %d Active Inference ontology terms from %s",
                len(terms),
                ontology_file,
            )
            return terms
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            if explicit_file is not None:
                raise ValueError(
                    f"Failed to load ontology from {ontology_file}: {exc}"
                ) from exc
            logger.warning("Failed to load ontology from %s: %s", ontology_file, exc)
            continue

    # Return default Active Inference terms if no file found
    logger.warning("No ontology terms file found, using defaults")
    return {
        "HiddenState": {
            "description": "A state of the environment or agent that is not directly observable.",
            "uri": "obo:ACTO_000001",
        },
        "Observation": {
            "description": "Data received from the environment through sensory input.",
            "uri": "obo:ACTO_000003",
        },
        "Action": {
            "description": "An output of the agent that can affect the environment.",
            "uri": "obo:ACTO_000004",
        },
        "LikelihoodMatrix": {
            "description": "A probabilistic mapping from hidden states to observations.",
            "uri": "obo:TEMP_000061",
        },
        "TransitionMatrix": {
            "description": "A probabilistic mapping defining the dynamics of hidden states.",
            "uri": "obo:ACTO_000009",
        },
        "VariationalFreeEnergy": {
            "description": "A bound on Bayesian model evidence.",
            "uri": "obo:ACTO_000012",
        },
        "ExpectedFreeEnergy": {
            "description": "A quantity minimized by the agent to select policies.",
            "uri": "obo:ACTO_000011",
        },
    }


def parse_annotation(annotation: str) -> tuple:
    """
    Parse a KEY=VALUE annotation into its components.

    Args:
        annotation: Raw annotation string (e.g., "A=LikelihoodMatrix")

    Returns:
        Tuple of (key, value, comment) where any can be None
    """
    comment = None
    if "#" in annotation:
        annotation, comment = annotation.split("#", 1)
        comment = comment.strip()
        annotation = annotation.strip()

    if "=" in annotation:
        key, value = annotation.split("=", 1)
        return key.strip(), value.strip(), comment

    return None, annotation.strip(), comment


def validate_annotations(
    annotations: List[str], ontology_terms: (Dict[str, Any]) | None = None
) -> Dict[str, Any]:
    """
    Validate annotations against ontology terms.

    Supports KEY=VALUE format where VALUE is matched against ontology term names.

    Args:
        annotations: List of annotations to validate (e.g., ["A=LikelihoodMatrix"])
        ontology_terms: Dictionary of ontology terms (loaded if not provided)

    Returns:
        Dictionary with validation results including matched term details
    """
    logger = logging.getLogger("ontology")

    try:
        if ontology_terms is None:
            ontology_terms = load_defined_ontology_terms()

        # Build lookup set of all term names (case-insensitive)
        term_lookup: dict[Any, Any] = {}
        for term_name in sorted(
            ontology_terms, key=lambda candidate: str(candidate).casefold()
        ):
            term_data = ontology_terms[term_name]
            if not isinstance(term_name, str) or not term_name.strip():
                raise ValueError("ontology term names must be non-empty strings")
            term_lookup[term_name.casefold()] = {
                "name": term_name,
                "data": term_data
                if isinstance(term_data, dict)
                else {"description": str(term_data)},
            }

        validation_result: dict[str, Any] = {
            "valid_annotations": [],
            "invalid_annotations": [],
            "matched_terms": {},  # key -> {term_name, description, uri}
            "suggestions": [],
            "coverage_score": 0.0,
            "invalid_details": [],
        }

        for annotation in annotations:
            key, value, comment = parse_annotation(annotation)

            # Mapping annotations are only meaningful when both sides are
            # present.  Previously ``=HiddenState`` was accepted because only
            # the ontology value was checked, losing the variable being
            # annotated.
            if "=" in annotation and (not key or not value):
                validation_result["invalid_annotations"].append(annotation)
                validation_result["invalid_details"].append(
                    {
                        "annotation": annotation,
                        "reason": "mapping annotations require a key and a value",
                    }
                )
                continue

            # Check if value matches any ontology term
            value_lower = value.casefold() if value else ""

            if value_lower in term_lookup:
                matched = term_lookup[value_lower]
                mapping_key = key or value
                existing = validation_result["matched_terms"].get(mapping_key)
                if existing is not None and (
                    str(existing.get("term_name", "")).casefold()
                    != str(matched["name"]).casefold()
                ):
                    validation_result["invalid_annotations"].append(annotation)
                    validation_result["invalid_details"].append(
                        {
                            "annotation": annotation,
                            "reason": "annotation key maps to multiple ontology terms",
                            "previous_term": existing.get("term_name"),
                        }
                    )
                    continue
                validation_result["valid_annotations"].append(annotation)
                validation_result["matched_terms"][mapping_key] = {
                    "annotation": annotation,
                    "term_name": matched["name"],
                    "description": matched["data"].get("description", ""),
                    "uri": matched["data"].get("uri", ""),
                    "key": key,
                    "value": value,
                    "comment": comment,
                }
            else:
                validation_result["invalid_annotations"].append(annotation)
                validation_result["invalid_details"].append(
                    {
                        "annotation": annotation,
                        "reason": "ontology term is not defined",
                    }
                )

                # Find similar terms for suggestions
                for term_name_lower, term_info in term_lookup.items():
                    if (
                        value_lower in term_name_lower
                        or term_name_lower in value_lower
                        or _levenshtein_distance(value_lower, term_name_lower) <= 3
                    ):
                        validation_result["suggestions"].append(
                            {
                                "annotation": annotation,
                                "suggested_term": term_info["name"],
                                "description": term_info["data"].get("description", ""),
                            }
                        )

        # Calculate coverage score
        total_annotations = len(annotations)
        if total_annotations > 0:
            validation_result["coverage_score"] = (
                len(validation_result["valid_annotations"]) / total_annotations
            )

        logger.info(
            f"Validated {len(annotations)} annotations: {len(validation_result['valid_annotations'])} valid, {len(validation_result['invalid_annotations'])} invalid"
        )

        return validation_result

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            "error": str(e),
            "valid_annotations": [],
            "invalid_annotations": annotations,
            "matched_terms": {},
            "suggestions": [],
            "coverage_score": 0.0,
            "invalid_details": [
                {"annotation": annotation, "reason": "validation failed"}
                for annotation in annotations
            ],
        }


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row: list[int] = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row: list[Any] = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def generate_ontology_report_for_file(
    gnn_file: Path,
    output_dir: Path,
    *,
    ontology_terms: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Generate ontology report for a single GNN file.

    Args:
        gnn_file: Path to the GNN file
        output_dir: Output directory for reports

    Returns:
        Dictionary with report generation results
    """
    try:
        # Process ontology for the file
        ontology_result = process_gnn_ontology(str(gnn_file), ontology_terms)

        if not ontology_result["success"]:
            return ontology_result

        # Create report
        report: dict[str, Any] = {
            "file_path": str(gnn_file),
            "file_name": gnn_file.name,
            "ontology_data": ontology_result["ontology_data"],
            "validation_result": ontology_result["validation_result"],
            "summary": {
                "total_concepts": len(ontology_result["ontology_data"]["concepts"]),
                "total_relations": len(ontology_result["ontology_data"]["relations"]),
                "total_properties": len(ontology_result["ontology_data"]["properties"]),
                "total_annotations": len(
                    ontology_result["ontology_data"]["annotations"]
                ),
                "valid_annotations": len(
                    ontology_result["validation_result"]["valid_annotations"]
                ),
                "invalid_annotations": len(
                    ontology_result["validation_result"]["invalid_annotations"]
                ),
                "coverage_score": ontology_result["validation_result"][
                    "coverage_score"
                ],
            },
        }

        # Save report
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / f"{gnn_file.stem}_ontology_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return {"success": True, "report_file": str(report_file), "report": report}

    except Exception as e:
        return {"success": False, "error": str(e)}
