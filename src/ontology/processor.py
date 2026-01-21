#!/usr/bin/env python3
"""
Ontology processor module for GNN Processing Pipeline.

This module provides the main ontology processing functionality.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

# Import core processing functions from processor module
# Note: Core functions are defined in this module; avoid self-import

def process_ontology(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
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
    
    try:
        log_step_start(logger, "Processing ontology")
        
        # Create results directory
        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each .md file and generate a per-file ontology report
        gnn_files = list(Path(target_dir).glob("**/*.md"))
        results = {"processed_files": len(gnn_files), "reports": [], "success": True, "errors": []}
        for gnn_file in gnn_files:
            file_report = generate_ontology_report_for_file(Path(gnn_file), results_dir)
            if not file_report.get("success", False):
                results["success"] = False
                results["errors"].append({"file": str(gnn_file), "error": file_report.get("error", "unknown")})
            else:
                results["reports"].append(file_report["report_file"])
        # Save aggregate results
        results_file = results_dir / "ontology_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if results["success"]:
            log_step_success(logger, "Ontology processing completed successfully")
        else:
            log_step_error(logger, "Ontology processing failed")
        
        return results["success"]
        
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
        ontology_data = {
            "concepts": [],
            "relations": [],
            "properties": [],
            "annotations": []
        }
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for ontology section
            if line.startswith('## Ontology') or line.startswith('## ontology') or line.startswith('## ActInfOntologyAnnotation'):
                current_section = 'ontology'
                continue
            elif line.startswith('##'):
                current_section = None
                continue
                
            if current_section == 'ontology':
                # Parse ontology content
                if '=' in line:
                    # Handle A=LikelihoodMatrix style annotations
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    ontology_data["annotations"].append(f"{key}={value}")
                elif ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key.lower() in ['concept', 'concepts']:
                        ontology_data["concepts"].append(value)
                    elif key.lower() in ['relation', 'relations']:
                        ontology_data["relations"].append(value)
                    elif key.lower() in ['property', 'properties']:
                        ontology_data["properties"].append(value)
                    elif key.lower() in ['annotation', 'annotations']:
                        ontology_data["annotations"].append(value)
        
        return ontology_data
        
    except Exception as e:
        return {
            "error": str(e),
            "concepts": [],
            "relations": [],
            "properties": [],
            "annotations": []
        }

def process_gnn_ontology(gnn_file: str) -> Dict[str, Any]:
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
            return {
                "success": False,
                "error": f"File not found: {gnn_file}"
            }
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse ontology section
        ontology_data = parse_gnn_ontology_section(content)
        
        # Load defined ontology terms
        ontology_terms = load_defined_ontology_terms()
        
        # Validate annotations
        validation_result = validate_annotations(
            ontology_data.get("annotations", []),
            ontology_terms
        )
        
        return {
            "success": True,
            "file_path": str(file_path),
            "ontology_data": ontology_data,
            "validation_result": validation_result,
            "ontology_terms": ontology_terms
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def load_defined_ontology_terms() -> Dict[str, Any]:
    """
    Load defined ontology terms from the Active Inference ontology terms file.
    
    Returns:
        Dictionary mapping term names to their definitions (including description and URI)
    """
    logger = logging.getLogger("ontology")
    
    # Priority order for ontology files
    search_paths = [
        Path(__file__).parent / "act_inf_ontology_terms.json",  # Module dir (canonical)
        Path("src/ontology/act_inf_ontology_terms.json"),        # From project root
    ]
    
    for ontology_file in search_paths:
        if ontology_file.exists():
            try:
                with open(ontology_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Handle both formats:
                # New format: {"TermName": {"description": "...", "uri": "..."}}
                # Old format: {"terms": {...}} or {"category": ["term1", "term2"]}
                if "terms" in data:
                    # Legacy format with "terms" wrapper
                    return data["terms"]
                elif isinstance(data, dict) and data:
                    # Check if it's the ActInf format (term -> {description, uri})
                    first_value = next(iter(data.values()))
                    if isinstance(first_value, dict) and "description" in first_value:
                        logger.info(f"Loaded {len(data)} Active Inference ontology terms from {ontology_file}")
                        return data
                    # Category-based format
                    return data
                    
            except Exception as e:
                logger.warning(f"Failed to load ontology from {ontology_file}: {e}")
                continue
    
    # Return default Active Inference terms if no file found
    logger.warning("No ontology terms file found, using defaults")
    return {
        "HiddenState": {"description": "A state of the environment or agent that is not directly observable.", "uri": "obo:ACTO_000001"},
        "Observation": {"description": "Data received from the environment through sensory input.", "uri": "obo:ACTO_000003"},
        "Action": {"description": "An output of the agent that can affect the environment.", "uri": "obo:ACTO_000004"},
        "LikelihoodMatrix": {"description": "A probabilistic mapping from hidden states to observations.", "uri": "obo:TEMP_000061"},
        "TransitionMatrix": {"description": "A probabilistic mapping defining the dynamics of hidden states.", "uri": "obo:ACTO_000009"},
        "VariationalFreeEnergy": {"description": "A bound on Bayesian model evidence.", "uri": "obo:ACTO_000012"},
        "ExpectedFreeEnergy": {"description": "A quantity minimized by the agent to select policies.", "uri": "obo:ACTO_000011"}
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
    if '#' in annotation:
        annotation, comment = annotation.split('#', 1)
        comment = comment.strip()
        annotation = annotation.strip()
    
    if '=' in annotation:
        key, value = annotation.split('=', 1)
        return key.strip(), value.strip(), comment
    
    return None, annotation.strip(), comment


def validate_annotations(annotations: List[str], ontology_terms: Dict[str, Any] = None) -> Dict[str, Any]:
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
        term_lookup = {}
        for term_name, term_data in ontology_terms.items():
            term_lookup[term_name.lower()] = {
                "name": term_name,
                "data": term_data if isinstance(term_data, dict) else {"description": str(term_data)}
            }
        
        validation_result = {
            "valid_annotations": [],
            "invalid_annotations": [],
            "matched_terms": {},  # key -> {term_name, description, uri}
            "suggestions": [],
            "coverage_score": 0.0
        }
        
        for annotation in annotations:
            key, value, comment = parse_annotation(annotation)
            
            # Check if value matches any ontology term
            value_lower = value.lower() if value else ""
            
            if value_lower in term_lookup:
                matched = term_lookup[value_lower]
                validation_result["valid_annotations"].append(annotation)
                validation_result["matched_terms"][key or value] = {
                    "annotation": annotation,
                    "term_name": matched["name"],
                    "description": matched["data"].get("description", ""),
                    "uri": matched["data"].get("uri", ""),
                    "key": key,
                    "value": value,
                    "comment": comment
                }
            else:
                validation_result["invalid_annotations"].append(annotation)
                
                # Find similar terms for suggestions
                for term_name_lower, term_info in term_lookup.items():
                    if (value_lower in term_name_lower or 
                        term_name_lower in value_lower or
                        _levenshtein_distance(value_lower, term_name_lower) <= 3):
                        validation_result["suggestions"].append({
                            "annotation": annotation,
                            "suggested_term": term_info["name"],
                            "description": term_info["data"].get("description", "")
                        })
        
        # Calculate coverage score
        total_annotations = len(annotations)
        if total_annotations > 0:
            validation_result["coverage_score"] = len(validation_result["valid_annotations"]) / total_annotations
        
        logger.info(f"Validated {len(annotations)} annotations: {len(validation_result['valid_annotations'])} valid, {len(validation_result['invalid_annotations'])} invalid")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            "error": str(e),
            "valid_annotations": [],
            "invalid_annotations": annotations,
            "matched_terms": {},
            "suggestions": [],
            "coverage_score": 0.0
        }


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def generate_ontology_report_for_file(gnn_file: Path, output_dir: Path) -> Dict[str, Any]:
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
        ontology_result = process_gnn_ontology(str(gnn_file))
        
        if not ontology_result["success"]:
            return ontology_result
        
        # Create report
        report = {
            "file_path": str(gnn_file),
            "file_name": gnn_file.name,
            "ontology_data": ontology_result["ontology_data"],
            "validation_result": ontology_result["validation_result"],
            "summary": {
                "total_concepts": len(ontology_result["ontology_data"]["concepts"]),
                "total_relations": len(ontology_result["ontology_data"]["relations"]),
                "total_properties": len(ontology_result["ontology_data"]["properties"]),
                "total_annotations": len(ontology_result["ontology_data"]["annotations"]),
                "valid_annotations": len(ontology_result["validation_result"]["valid_annotations"]),
                "invalid_annotations": len(ontology_result["validation_result"]["invalid_annotations"]),
                "coverage_score": ontology_result["validation_result"]["coverage_score"]
            }
        }
        
        # Save report
        report_file = output_dir / f"{gnn_file.stem}_ontology_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return {
            "success": True,
            "report_file": str(report_file),
            "report": report
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        } 