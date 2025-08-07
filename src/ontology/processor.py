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
        results_dir = output_dir / "ontology_results"
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

def load_defined_ontology_terms() -> Dict[str, List[str]]:
    """
    Load defined ontology terms from the ontology terms file.
    
    Returns:
        Dictionary mapping categories to lists of terms
    """
    try:
        # Try to load from the ontology terms file
        ontology_file = Path("input/ontology_terms.json")
        
        if ontology_file.exists():
            with open(ontology_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Return default ontology terms
            return {
                "cognitive_processes": [
                    "attention", "memory", "learning", "reasoning", "decision_making",
                    "perception", "language", "emotion", "consciousness"
                ],
                "neural_mechanisms": [
                    "synaptic_plasticity", "neurotransmission", "neural_oscillations",
                    "cortical_columns", "neural_networks", "brain_regions"
                ],
                "active_inference": [
                    "free_energy", "variational_inference", "generative_models",
                    "predictive_coding", "belief_propagation", "precision_weighting"
                ],
                "mathematical_concepts": [
                    "probability", "information_theory", "optimization", "dynamics",
                    "geometry", "topology", "category_theory"
                ]
            }
            
    except Exception as e:
        # Return minimal default terms on error
        return {
            "cognitive_processes": ["attention", "memory", "learning"],
            "neural_mechanisms": ["synaptic_plasticity", "neural_networks"],
            "active_inference": ["free_energy", "predictive_coding"],
            "mathematical_concepts": ["probability", "optimization"]
        }

def validate_annotations(annotations: List[str], ontology_terms: Dict[str, List[str]] = None) -> Dict[str, Any]:
    """
    Validate annotations against ontology terms.
    
    Args:
        annotations: List of annotations to validate
        ontology_terms: Dictionary of ontology terms (loaded if not provided)
        
    Returns:
        Dictionary with validation results
    """
    try:
        if ontology_terms is None:
            ontology_terms = load_defined_ontology_terms()
        
        # Flatten all ontology terms
        all_terms = []
        for category, terms in ontology_terms.items():
            all_terms.extend(terms)
        
        validation_result = {
            "valid_annotations": [],
            "invalid_annotations": [],
            "suggestions": [],
            "coverage_score": 0.0
        }
        
        for annotation in annotations:
            annotation_lower = annotation.lower().replace(' ', '_')
            
            if annotation_lower in all_terms:
                validation_result["valid_annotations"].append(annotation)
            else:
                validation_result["invalid_annotations"].append(annotation)
                
                # Find similar terms for suggestions
                for term in all_terms:
                    if annotation_lower in term or term in annotation_lower:
                        validation_result["suggestions"].append({
                            "annotation": annotation,
                            "suggested_term": term
                        })
        
        # Calculate coverage score
        total_annotations = len(annotations)
        if total_annotations > 0:
            validation_result["coverage_score"] = len(validation_result["valid_annotations"]) / total_annotations
        
        return validation_result
        
    except Exception as e:
        return {
            "error": str(e),
            "valid_annotations": [],
            "invalid_annotations": annotations,
            "suggestions": [],
            "coverage_score": 0.0
        }

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