"""
Data extraction utilities for advanced visualization of GNN models.

This module provides functionality to extract structured data from GNN files
using the comprehensive GNN parsing system for visualization purposes.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import the GNN parsing system
from gnn.parsers import GNNParsingSystem, GNNFormat


class VisualizationDataExtractor:
    """
    Extracts visualization data from GNN files using the comprehensive parsing system.
    """
    
    def __init__(self, strict_validation: bool = True):
        """
        Initialize the data extractor with the GNN parsing system.
        
        Args:
            strict_validation: Whether to use strict validation during parsing
        """
        self.parsing_system = GNNParsingSystem(strict_validation=strict_validation)
    
    def extract_from_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract visualization data from a GNN file.
        
        Args:
            file_path: Path to the GNN file
            
        Returns:
            Dictionary containing extracted visualization data
        """
        try:
            # Parse the file using the comprehensive parsing system
            parse_result = self.parsing_system.parse_file(file_path)
            
            if not parse_result.success:
                return {
                    "success": False,
                    "errors": parse_result.errors,
                    "warnings": parse_result.warnings,
                    "blocks": [],
                    "connections": [],
                    "total_blocks": 0,
                    "total_connections": 0
                }
            
            # Extract data from the parsed model
            model = parse_result.model
            return self._extract_from_model(model)
            
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "warnings": [],
                "blocks": [],
                "connections": [],
                "total_blocks": 0,
                "total_connections": 0
            }
    
    def extract_from_content(self, content: str, format_hint: Optional[GNNFormat] = None) -> Dict[str, Any]:
        """
        Extract visualization data from GNN content string.
        
        Args:
            content: GNN file content as string
            format_hint: Optional format hint for parsing
            
        Returns:
            Dictionary containing extracted visualization data
        """
        try:
            # Parse the content using the comprehensive parsing system
            if format_hint:
                parse_result = self.parsing_system.parse_string(content, format_hint)
            else:
                # Try to detect format from content
                detected_format = self.parsing_system._detect_format_from_content(content)
                parse_result = self.parsing_system.parse_string(content, detected_format)
            
            if not parse_result.success:
                return {
                    "success": False,
                    "errors": parse_result.errors,
                    "warnings": parse_result.warnings,
                    "blocks": [],
                    "connections": [],
                    "total_blocks": 0,
                    "total_connections": 0
                }
            
            # Extract data from the parsed model
            model = parse_result.model
            return self._extract_from_model(model)
            
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "warnings": [],
                "blocks": [],
                "connections": [],
                "total_blocks": 0,
                "total_connections": 0
            }
    
    def _extract_from_model(self, model) -> Dict[str, Any]:
        """
        Extract visualization data from a parsed GNN model.
        
        Args:
            model: Parsed GNN model (GNNInternalRepresentation)
            
        Returns:
            Dictionary containing extracted visualization data
        """
        # Extract variable blocks
        blocks = []
        for var in model.variables:
            block_data = {
                "name": var.name,
                "type": var.var_type.value if hasattr(var.var_type, 'value') else str(var.var_type),
                "data_type": var.data_type.value if hasattr(var.data_type, 'value') else str(var.data_type),
                "dimensions": var.dimensions,
                "description": var.description or "",
                "constraints": var.constraints
            }
            blocks.append(block_data)
        
        # Extract connections
        connections = []
        for conn in model.connections:
            conn_data = {
                "from": conn.source_variables,
                "to": conn.target_variables,
                "type": conn.connection_type.value if hasattr(conn.connection_type, 'value') else str(conn.connection_type),
                "weight": conn.weight,
                "description": conn.description or ""
            }
            connections.append(conn_data)
        
        # Extract parameters
        parameters = []
        for param in model.parameters:
            param_data = {
                "name": param.name,
                "value": param.value,
                "type_hint": param.type_hint,
                "description": param.description or ""
            }
            parameters.append(param_data)
        
        # Extract equations
        equations = []
        for eq in model.equations:
            eq_data = {
                "label": eq.label,
                "content": eq.content,
                "format": eq.format,
                "description": eq.description or ""
            }
            equations.append(eq_data)
        
        # Extract time specification
        time_spec = None
        if model.time_specification:
            time_spec = {
                "time_type": model.time_specification.time_type,
                "discretization": model.time_specification.discretization,
                "horizon": model.time_specification.horizon,
                "step_size": model.time_specification.step_size
            }
        
        # Extract ontology mappings
        ontology_mappings = []
        for mapping in model.ontology_mappings:
            mapping_data = {
                "variable_name": mapping.variable_name,
                "ontology_term": mapping.ontology_term,
                "description": mapping.description or ""
            }
            ontology_mappings.append(mapping_data)
        
        return {
            "success": True,
            "model_info": {
                "name": model.model_name,
                "version": model.version,
                "annotation": model.annotation,
                "source_format": model.source_format.value if model.source_format else None,
                "created_at": model.created_at.isoformat() if model.created_at else None,
                "modified_at": model.modified_at.isoformat() if model.modified_at else None
            },
            "blocks": blocks,
            "connections": connections,
            "parameters": parameters,
            "equations": equations,
            "time_specification": time_spec,
            "ontology_mappings": ontology_mappings,
            "total_blocks": len(blocks),
            "total_connections": len(connections),
            "total_parameters": len(parameters),
            "total_equations": len(equations),
            "extraction_timestamp": datetime.now().isoformat()
        }
    
    def get_model_statistics(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate statistics from extracted visualization data.
        
        Args:
            extracted_data: Data extracted by extract_from_file or extract_from_content
            
        Returns:
            Dictionary containing model statistics
        """
        if not extracted_data.get("success", False):
            return {"error": "No valid data to analyze"}
        
        blocks = extracted_data.get("blocks", [])
        connections = extracted_data.get("connections", [])
        
        # Variable type statistics
        type_counts = {}
        for block in blocks:
            var_type = block.get("type", "unknown")
            type_counts[var_type] = type_counts.get(var_type, 0) + 1
        
        # Data type statistics
        data_type_counts = {}
        for block in blocks:
            data_type = block.get("data_type", "unknown")
            data_type_counts[data_type] = data_type_counts.get(data_type, 0) + 1
        
        # Connection type statistics
        connection_type_counts = {}
        for conn in connections:
            conn_type = conn.get("type", "unknown")
            connection_type_counts[conn_type] = connection_type_counts.get(conn_type, 0) + 1
        
        # Dimension statistics
        dimension_counts = {}
        for block in blocks:
            dimensions = block.get("dimensions", [])
            dim_key = f"{len(dimensions)}D"
            dimension_counts[dim_key] = dimension_counts.get(dim_key, 0) + 1
        
        return {
            "variable_types": type_counts,
            "data_types": data_type_counts,
            "connection_types": connection_type_counts,
            "dimension_distribution": dimension_counts,
            "total_variables": len(blocks),
            "total_connections": len(connections),
            "total_parameters": extracted_data.get("total_parameters", 0),
            "total_equations": extracted_data.get("total_equations", 0)
        }


def extract_visualization_data(target_dir, output_dir, **kwargs):
    """
    Extract visualization data from GNN files in the target directory.
    
    Args:
        target_dir: Directory containing GNN files
        output_dir: Directory to save extracted data
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with extraction results
    """
    from pathlib import Path
    import json
    
    target_dir = Path(target_dir)
    output_dir = Path(output_dir)
    
    extractor = VisualizationDataExtractor(strict_validation=False)
    
    results = {
        "processed_files": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "extracted_data": {},
        "statistics": {},
        "errors": []
    }
    
    # Find all GNN files
    gnn_extensions = ['.md', '.gnn', '.json', '.yaml', '.yml']
    gnn_files = []
    
    for ext in gnn_extensions:
        gnn_files.extend(target_dir.glob(f"**/*{ext}"))
    
    for gnn_file in gnn_files:
        try:
            extracted_data = extractor.extract_from_file(gnn_file)
            
            results["processed_files"] += 1
            
            if extracted_data.get("success", False):
                results["successful_extractions"] += 1
                
                model_name = gnn_file.stem
                results["extracted_data"][model_name] = extracted_data
                results["statistics"][model_name] = extractor.get_model_statistics(extracted_data)
                
                # Save individual file data
                model_output_dir = output_dir / model_name
                model_output_dir.mkdir(parents=True, exist_ok=True)
                
                with open(model_output_dir / "extracted_data.json", 'w') as f:
                    json.dump(extracted_data, f, indent=2)
                
                with open(model_output_dir / "statistics.json", 'w') as f:
                    json.dump(results["statistics"][model_name], f, indent=2)
            else:
                results["failed_extractions"] += 1
                results["errors"].append(f"Failed to extract from {gnn_file}: {extracted_data.get('errors', [])}")
            
        except Exception as e:
            results["processed_files"] += 1
            results["failed_extractions"] += 1
            results["errors"].append(f"Error processing {gnn_file}: {e}")
    
    # Save overall summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "extraction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results 