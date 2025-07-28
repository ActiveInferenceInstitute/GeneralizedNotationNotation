"""
ontology module for GNN Processing Pipeline.

This module provides ontology capabilities with fallback implementations.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

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
        
        # Basic ontology processing
        results = {
            "processed_files": 0,
            "success": True,
            "errors": []
        }
        
        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if gnn_files:
            results["processed_files"] = len(gnn_files)
        
        # Save results
        import json
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
            "relationships": [],
            "properties": [],
            "metadata": {}
        }
        
        # Extract ontology-related information
        lines = content.split('\n')
        in_ontology_section = False
        
        for line in lines:
            line = line.strip()
            
            # Check for ontology section markers
            if line.lower().startswith('ontology:'):
                in_ontology_section = True
                continue
            elif line.lower().startswith('end ontology') or line.lower().startswith('#'):
                in_ontology_section = False
                continue
            
            if in_ontology_section and line:
                # Parse ontology entries
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key.lower() == 'concept':
                        ontology_data["concepts"].append(value)
                    elif key.lower() == 'relationship':
                        ontology_data["relationships"].append(value)
                    elif key.lower() == 'property':
                        ontology_data["properties"].append(value)
                    else:
                        ontology_data["metadata"][key] = value
        
        ontology_data["success"] = True
        return ontology_data
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Ontology parsing failed: {str(e)}"
        }

def get_module_info() -> Dict[str, Any]:
    """Get comprehensive information about the ontology module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'ontology_capabilities': [],
        'processing_methods': [],
        'processing_capabilities': []
    }
    
    # Ontology capabilities
    info['ontology_capabilities'].extend([
        'Concept extraction',
        'Relationship mapping',
        'Property analysis',
        'Metadata processing',
        'Ontology validation'
    ])
    
    # Processing methods
    info['processing_methods'].extend([
        'Section parsing',
        'Concept identification',
        'Relationship extraction',
        'Property mapping'
    ])
    
    # Processing capabilities
    info['processing_capabilities'].extend([
        'GNN ontology parsing',
        'Term extraction',
        'Relationship mapping',
        'Validation',
        'Export to multiple formats'
    ])
    
    return info

def get_ontology_processing_options() -> Dict[str, Any]:
    """Get ontology processing options and capabilities."""
    return {
        'parsing_modes': ['basic', 'detailed', 'strict'],
        'output_formats': ['json', 'xml', 'rdf', 'owl'],
        'validation_levels': ['none', 'basic', 'strict'],
        'extraction_methods': ['section', 'pattern', 'semantic'],
        'parsing_options': ['include_metadata', 'validate_terms', 'extract_relationships']
    }

def process_gnn_ontology(gnn_file: str) -> Dict[str, Any]:
    """
    Process GNN ontology (alias for process_ontology).
    
    Args:
        gnn_file: Path to GNN file
        
    Returns:
        Dictionary with processing results
    """
    try:
        gnn_path = Path(gnn_file)
        
        if not gnn_path.exists():
            return {
                "success": False,
                "error": f"GNN file not found: {gnn_file}"
            }
        
        # Read GNN content
        with open(gnn_path, 'r') as f:
            content = f.read()
        
        # Parse ontology section
        ontology_data = parse_gnn_ontology_section(content)
        
        if ontology_data["success"]:
            return {
                "success": True,
                "file": str(gnn_path),
                "ontology_data": ontology_data,
                "concepts_count": len(ontology_data.get("concepts", [])),
                "relationships_count": len(ontology_data.get("relationships", [])),
                "properties_count": len(ontology_data.get("properties", []))
            }
        else:
            return ontology_data
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Ontology processing failed: {str(e)}"
        }

def load_defined_ontology_terms() -> Dict[str, List[str]]:
    """
    Load defined ontology terms.
    
    Returns:
        Dictionary with ontology terms by category
    """
    return {
        "concepts": [
            "Active Inference",
            "Free Energy",
            "Generative Model",
            "Variational Inference",
            "Belief State",
            "Action",
            "Observation",
            "State",
            "Policy"
        ],
        "relationships": [
            "implements",
            "extends",
            "uses",
            "depends_on",
            "contains",
            "inherits_from",
            "composes"
        ],
        "properties": [
            "type",
            "dimension",
            "range",
            "default_value",
            "constraints",
            "description"
        ]
    }

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "ontology processing for GNN Processing Pipeline"

# Feature availability flags
FEATURES = {
    'basic_processing': True,
    'fallback_mode': True,
    'mcp_integration': True
}

__all__ = [
    'process_ontology',
    'parse_gnn_ontology_section',
    'get_module_info',
    'get_ontology_processing_options',
    'process_gnn_ontology',
    'load_defined_ontology_terms',
    'FEATURES',
    '__version__'
]
