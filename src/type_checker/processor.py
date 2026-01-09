#!/usr/bin/env python3
"""
Type checker processor module for GNN pipeline.
"""

from pathlib import Path
from typing import Dict, Any, List, Union
import logging
import json
import re
from datetime import datetime

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

class GNNTypeChecker:
    """Type checker for GNN files."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the GNN type checker."""
        self.validation_rules = self._get_validation_rules()
    
    def validate_gnn_files(
        self,
        target_dir: Path,
        output_dir: Path,
        verbose: bool = False,
        **kwargs
    ) -> bool:
        """
        Validate GNN files for type consistency.
        
        Args:
            target_dir: Directory containing GNN files to validate
            output_dir: Directory to save validation results
            verbose: Enable verbose output
            **kwargs: Additional arguments
            
        Returns:
            True if validation successful, False otherwise
        """
        logger = logging.getLogger("type_checker")
        
        try:
            log_step_start(logger, "Processing type checker")
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create results directory
            results_dir = output_dir
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize results
            results = {
                "timestamp": datetime.now().isoformat(),
                "processed_files": 0,
                "success": True,
                "errors": [],
                "validation_results": [],
                "type_analysis": []
            }
            
            # Find GNN files
            gnn_files = list(target_dir.glob("*.md"))
            if not gnn_files:
                logger.warning("No GNN files found for type checking")
                results["success"] = False
                results["errors"].append("No GNN files found")
            else:
                results["processed_files"] = len(gnn_files)
                
                # Process each GNN file
                for gnn_file in gnn_files:
                    try:
                        # Validate single file
                        validation_result = self.validate_single_gnn_file(gnn_file, verbose)
                        results["validation_results"].append(validation_result)
                        
                        # Analyze types
                        type_analysis = self._analyze_types(gnn_file, verbose)
                        results["type_analysis"].append(type_analysis)
                        
                    except Exception as e:
                        error_info = {
                            "file": str(gnn_file),
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                        results["errors"].append(error_info)
                        logger.error(f"Error processing {gnn_file}: {e}")
            
            # Save detailed results directly in output directory
            results_file = output_dir / "type_check_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            # Generate type check summary
            summary = self._generate_type_check_summary(results)
            summary_file = output_dir / "type_check_summary.md"
            with open(summary_file, 'w') as f:
                f.write(summary)
            
            if results["success"]:
                log_step_success(logger, "Type checking completed successfully")
            else:
                log_step_error(logger, "Type checking failed")
            
            return results["success"]
            
        except Exception as e:
            log_step_error(logger, "Type checking failed", {"error": str(e)})
            return False
    
    def validate_single_gnn_file(self, file_path: Path, verbose: bool = False) -> Dict[str, Any]:
        """Validate a single GNN file for type consistency."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            validation_result = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "valid": True,
                "errors": [],
                "warnings": [],
                "type_issues": [],
                "validation_timestamp": datetime.now().isoformat()
            }
            
            # Check for type definitions
            type_patterns = [
                r'(\w+)\s*:\s*(\w+)',  # name: type
                r'(\w+)\s*\[([^\]]+)\]',  # name[dimensions]
                r'(\w+)\s*=\s*([^;\n]+)',  # name = value
            ]
            
            found_types = []
            for pattern in type_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    var_name = match.group(1)
                    var_type = match.group(2)
                    found_types.append({
                        "name": var_name,
                        "type": var_type,
                        "line": content[:match.start()].count('\n') + 1
                    })
            
            # Validate types
            for type_info in found_types:
                type_validation = self._validate_type(type_info)
                if not type_validation["valid"]:
                    validation_result["type_issues"].append(type_validation)
                    validation_result["warnings"].append(f"Type issue: {type_validation['message']}")
            
            # Check for consistency
            consistency_check = self._check_type_consistency(found_types)
            if not consistency_check["consistent"]:
                validation_result["errors"].append(consistency_check["message"])
                validation_result["valid"] = False
            
            return validation_result
            
        except Exception as e:
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "type_issues": [],
                "validation_timestamp": datetime.now().isoformat()
            }
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for GNN types."""
        return {
            "valid_types": [
                "int", "float", "double", "string", "bool", "array", "matrix",
                "vector", "tensor", "state", "action", "observation", "belief"
            ],
            "type_patterns": {
                "numeric": r"^[0-9]+(\.[0-9]+)?$",
                "identifier": r"^[a-zA-Z_][a-zA-Z0-9_]*$",
                "array": r"^\[.*\]$"
            }
        }
    
    def _validate_type(self, type_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single type definition."""
        var_name = type_info["name"]
        var_type = type_info["type"]
        
        validation = {
            "valid": True,
            "message": "",
            "variable": var_name,
            "type": var_type
        }
        
        # Check if type is in valid types
        if var_type not in self.validation_rules["valid_types"]:
            validation["valid"] = False
            validation["message"] = f"Unknown type '{var_type}' for variable '{var_name}'"
        
        # Check variable name format
        if not re.match(self.validation_rules["type_patterns"]["identifier"], var_name):
            validation["valid"] = False
            validation["message"] = f"Invalid variable name '{var_name}'"
        
        return validation
    
    def _check_type_consistency(self, types: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check consistency of types across the file."""
        consistency = {
            "consistent": True,
            "message": ""
        }
        
        # Check for duplicate variable names
        var_names = [t["name"] for t in types]
        duplicates = [name for name in set(var_names) if var_names.count(name) > 1]
        
        if duplicates:
            consistency["consistent"] = False
            consistency["message"] = f"Duplicate variable names: {', '.join(duplicates)}"
        
        return consistency
    
    def _analyze_types(self, file_path: Path, verbose: bool = False) -> Dict[str, Any]:
        """Analyze types in a GNN file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract type information
            type_patterns = [
                r'(\w+)\s*:\s*(\w+)',  # name: type
                r'(\w+)\s*\[([^\]]+)\]',  # name[dimensions]
            ]
            
            types_found = []
            for pattern in type_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    types_found.append({
                        "name": match.group(1),
                        "type": match.group(2),
                        "line": content[:match.start()].count('\n') + 1
                    })
            
            # Analyze type distribution
            type_counts = {}
            for type_info in types_found:
                var_type = type_info["type"]
                type_counts[var_type] = type_counts.get(var_type, 0) + 1
            
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "types_found": types_found,
                "type_distribution": type_counts,
                "total_variables": len(types_found),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    def _generate_type_check_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary of type checking results."""
        summary = f"""
# Type Check Summary

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Processing Results
- **Files Processed**: {results.get('processed_files', 0)}
- **Success**: {results.get('success', False)}
- **Errors**: {len(results.get('errors', []))}

## Validation Results
- **Files Validated**: {len(results.get('validation_results', []))}
- **Valid Files**: {sum(1 for r in results.get('validation_results', []) if r.get('valid', False))}
- **Invalid Files**: {sum(1 for r in results.get('validation_results', []) if not r.get('valid', False))}

## Type Analysis
- **Type Analyses**: {len(results.get('type_analysis', []))}
- **Total Variables**: {sum(a.get('total_variables', 0) for a in results.get('type_analysis', []))}

## Error Summary
"""
        
        errors = results.get('errors', [])
        if errors:
            for error in errors:
                if isinstance(error, dict):
                    summary += f"- **{error.get('file', 'Unknown')}**: {error.get('error', 'Unknown error')}\n"
                else:
                    summary += f"- {error}\n"
        else:
            summary += "- No errors encountered\n"
        
        return summary

def estimate_file_resources(content: str) -> Dict[str, Any]:
    """Estimate computational resources needed for a GNN file."""
    try:
        # Count variables and connections
        variables = len(re.findall(r'(\w+)\s*[:=]', content))
        connections = len(re.findall(r'(\w+)\s*[->→]\s*(\w+)', content))
        
        # Estimate memory usage
        estimated_memory = variables * 8 + connections * 16  # bytes
        
        # Estimate computation time
        estimated_time = variables * connections * 0.001  # seconds
        
        return {
            "variables": variables,
            "connections": connections,
            "estimated_memory_bytes": estimated_memory,
            "estimated_time_seconds": estimated_time,
            "complexity_score": variables * connections
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "variables": 0,
            "connections": 0,
            "estimated_memory_bytes": 0,
            "estimated_time_seconds": 0,
            "complexity_score": 0
        }
