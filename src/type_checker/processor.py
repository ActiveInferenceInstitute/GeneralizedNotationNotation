#!/usr/bin/env python3
"""
Type checker processor module for GNN pipeline.

This module provides comprehensive type checking and validation for GNN files,
including syntax validation, type consistency checking, and resource estimation.
It follows the thin orchestrator pattern by providing core functionality that
is imported by the numbered pipeline scripts.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
import logging
import json
import re
import math
from datetime import datetime
import tempfile
import shutil

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)
from .pomdp_analyzer import POMDPAnalyzer, load_ontology_terms

class GNNTypeChecker:
    """
    Comprehensive type checker for GNN files.
    
    This class provides robust type checking, validation, and analysis capabilities
    for Generalized Notation Notation (GNN) files. It includes syntax validation,
    type consistency checking, resource estimation, and performance analysis.
    
    Attributes:
        validation_rules: Dictionary containing validation rules and patterns
        logger: Logger instance for this type checker
        performance_metrics: Dictionary tracking performance metrics
    """
    
    def __init__(self, strict_mode: bool = False, verbose: bool = False, 
                 pomdp_mode: bool = False, ontology_file: Optional[Path] = None, *args, **kwargs):
        """
        Initialize the GNN type checker.
        
        Args:
            strict_mode: If True, use strict validation rules
            verbose: If True, enable verbose logging
            pomdp_mode: If True, enable POMDP-specific analysis
            ontology_file: Path to ontology terms JSON file
            **kwargs: Additional configuration options
        """
        self.strict_mode = strict_mode
        self.verbose = verbose
        self.pomdp_mode = pomdp_mode
        self.ontology_file = ontology_file
        self.validation_rules = self._get_validation_rules()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.performance_metrics = {
            "files_processed": 0,
            "total_processing_time": 0,
            "errors_encountered": 0,
            "warnings_generated": 0,
        }
        
        # Initialize POMDP analyzer if in POMDP mode
        if self.pomdp_mode:
            ontology_terms = None
            if self.ontology_file:
                if isinstance(self.ontology_file, str):
                    ontology_file_path = Path(self.ontology_file)
                else:
                    ontology_file_path = self.ontology_file
                if ontology_file_path.exists():
                    ontology_terms = load_ontology_terms(ontology_file_path)
            self.pomdp_analyzer = POMDPAnalyzer(ontology_file=self.ontology_file, ontology_terms=ontology_terms)
        else:
            self.pomdp_analyzer = None
    
    def validate_gnn_files(
        self,
        target_dir: Path,
        output_dir: Path,
        verbose: bool = False,
        **kwargs
    ) -> bool:
        """
        Validate GNN files for type consistency with comprehensive error handling.
        
        Args:
            target_dir: Directory containing GNN files to validate
            output_dir: Directory to save validation results
            verbose: Enable verbose output
            **kwargs: Additional arguments
            
        Returns:
            True if validation successful, False otherwise
        """
        start_time = datetime.now()
        logger = self.logger
        
        try:
            log_step_start(logger, "Processing type checker")
            
            # Validate input parameters
            if not isinstance(target_dir, Path):
                target_dir = Path(target_dir)
            if not isinstance(output_dir, Path):
                output_dir = Path(output_dir)
            
            # Create results directory with proper structure
            results_dir = output_dir / "type_check_results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize comprehensive results structure
            results = {
                "timestamp": start_time.isoformat(),
                "processed_files": 0,
                "success": True,
                "errors": [],
                "warnings": [],
                "validation_results": [],
                "type_analysis": [],
                "performance_metrics": {},
                "summary_statistics": {
                    "total_variables": 0,
                    "total_connections": 0,
                    "valid_files": 0,
                    "invalid_files": 0,
                    "type_errors": 0,
                    "warnings_count": 0,
                }
            }
            
            # Find GNN files with multiple extensions
            gnn_extensions = ["*.md", "*.gnn", "*.txt"]
            gnn_files = []
            for ext in gnn_extensions:
                gnn_files.extend(target_dir.glob(ext))
            
            if not gnn_files:
                warning_msg = "No GNN files found for type checking"
                logger.warning(warning_msg)
                results["warnings"].append(warning_msg)
                results["success"] = False
                results["errors"].append("No GNN files found")
            else:
                results["processed_files"] = len(gnn_files)
                self.performance_metrics["files_processed"] = len(gnn_files)
                
                # Process each GNN file with comprehensive error handling
                for i, gnn_file in enumerate(gnn_files):
                    try:
                        if verbose:
                            logger.info(f"Processing file {i+1}/{len(gnn_files)}: {gnn_file.name}")
                        
                        # Validate single file
                        validation_result = self.validate_single_gnn_file(gnn_file, verbose)
                        results["validation_results"].append(validation_result)
                        
                        # Update summary statistics
                        if validation_result.get("valid", False):
                            results["summary_statistics"]["valid_files"] += 1
                        else:
                            results["summary_statistics"]["invalid_files"] += 1
                        
                        results["summary_statistics"]["type_errors"] += len(validation_result.get("errors", []))
                        results["summary_statistics"]["warnings_count"] += len(validation_result.get("warnings", []))
                        
                        # Analyze types
                        type_analysis = self._analyze_types(gnn_file, verbose)
                        results["type_analysis"].append(type_analysis)
                        
                        # POMDP-specific analysis if enabled
                        if self.pomdp_mode and self.pomdp_analyzer:
                            pomdp_analysis = self.pomdp_analyzer.validate_pomdp_model(gnn_file)
                            if "pomdp_analysis" not in results:
                                results["pomdp_analysis"] = []
                            results["pomdp_analysis"].append(pomdp_analysis)
                            
                            # Add POMDP metrics to summary
                            pomdp_metrics = pomdp_analysis.get("pomdp_specific", {})
                            if "pomdp_metrics" not in results["summary_statistics"]:
                                results["summary_statistics"]["pomdp_metrics"] = {
                                    "total_state_space_size": 0,
                                    "total_observation_space_size": 0,
                                    "total_action_space_size": 0,
                                    "pomdp_models_valid": 0
                                }
                            
                            results["summary_statistics"]["pomdp_metrics"]["total_state_space_size"] += pomdp_metrics.get("state_space_size", 0)
                            results["summary_statistics"]["pomdp_metrics"]["total_observation_space_size"] += pomdp_metrics.get("observation_space_size", 0)
                            results["summary_statistics"]["pomdp_metrics"]["total_action_space_size"] += pomdp_metrics.get("action_space_size", 0)
                            
                            if pomdp_analysis.get("validation_results", {}).get("overall_valid", False):
                                results["summary_statistics"]["pomdp_metrics"]["pomdp_models_valid"] += 1
                        
                        # Update variable and connection counts
                        results["summary_statistics"]["total_variables"] += type_analysis.get("total_variables", 0)
                        
                    except Exception as e:
                        error_info = {
                            "file": str(gnn_file),
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "timestamp": datetime.now().isoformat()
                        }
                        results["errors"].append(error_info)
                        results["summary_statistics"]["type_errors"] += 1
                        self.performance_metrics["errors_encountered"] += 1
                        logger.error(f"Error processing {gnn_file}: {e}")
            
            # Calculate performance metrics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            self.performance_metrics["total_processing_time"] = processing_time
            results["performance_metrics"] = self.performance_metrics.copy()
            
            # Save detailed results with error handling
            try:
                results_file = results_dir / "type_check_results.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                # Generate comprehensive type check summary
                summary = self._generate_type_check_summary(results)
                summary_file = results_dir / "type_check_summary.md"
                with open(summary_file, 'w') as f:
                    f.write(summary)
                
                # Generate performance report
                performance_report = self._generate_performance_report(results)
                performance_file = results_dir / "performance_report.json"
                with open(performance_file, 'w') as f:
                    json.dump(performance_report, f, indent=2, default=str)
                    
            except Exception as e:
                logger.error(f"Error saving results: {e}")
                results["errors"].append(f"Error saving results: {str(e)}")
            
            # Determine overall success
            if results["summary_statistics"]["valid_files"] > 0:
                if results["summary_statistics"]["type_errors"] > 0:
                    log_step_warning(logger, f"Type checking completed with {results['summary_statistics']['type_errors']} errors")
                else:
                    log_step_success(logger, "Type checking completed successfully")
            else:
                log_step_error(logger, "Type checking failed - no valid files found")
                results["success"] = False
            
            return results["success"]
            
        except Exception as e:
            self.performance_metrics["errors_encountered"] += 1
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
        """Generate a comprehensive summary of type checking results."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        stats = results.get('summary_statistics', {})
        perf = results.get('performance_metrics', {})
        
        summary = f"""# Type Check Summary

**Generated**: {timestamp}
**Processing Time**: {perf.get('total_processing_time', 0):.2f} seconds

## Processing Results
- **Files Processed**: {results.get('processed_files', 0)}
- **Success**: {results.get('success', False)}
- **Errors**: {len(results.get('errors', []))}
- **Warnings**: {len(results.get('warnings', []))}

## Validation Results
- **Files Validated**: {len(results.get('validation_results', []))}
- **Valid Files**: {stats.get('valid_files', 0)}
- **Invalid Files**: {stats.get('invalid_files', 0)}
- **Type Errors**: {stats.get('type_errors', 0)}
- **Warnings Count**: {stats.get('warnings_count', 0)}

## Type Analysis
- **Type Analyses**: {len(results.get('type_analysis', []))}
- **Total Variables**: {stats.get('total_variables', 0)}
- **Total Connections**: {stats.get('total_connections', 0)}

## Performance Metrics
- **Files Processed**: {perf.get('files_processed', 0)}
- **Total Processing Time**: {perf.get('total_processing_time', 0):.2f}s
- **Errors Encountered**: {perf.get('errors_encountered', 0)}
- **Warnings Generated**: {perf.get('warnings_generated', 0)}

## Error Summary
"""
        
        errors = results.get('errors', [])
        if errors:
            for i, error in enumerate(errors, 1):
                if isinstance(error, dict):
                    summary += f"{i}. **{error.get('file', 'Unknown')}**: {error.get('error', 'Unknown error')}\n"
                    if 'error_type' in error:
                        summary += f"   - Error Type: {error['error_type']}\n"
                    if 'timestamp' in error:
                        summary += f"   - Timestamp: {error['timestamp']}\n"
                else:
                    summary += f"{i}. {error}\n"
        else:
            summary += "- No errors encountered\n"
        
        # Add warnings section
        warnings = results.get('warnings', [])
        if warnings:
            summary += "\n## Warnings\n"
            for i, warning in enumerate(warnings, 1):
                summary += f"{i}. {warning}\n"
        
        return summary
    
    def _generate_performance_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a detailed performance report."""
        perf = results.get('performance_metrics', {})
        stats = results.get('summary_statistics', {})
        
        # Calculate additional metrics
        files_processed = perf.get('files_processed', 0)
        processing_time = perf.get('total_processing_time', 0)
        
        avg_time_per_file = processing_time / files_processed if files_processed > 0 else 0
        error_rate = (perf.get('errors_encountered', 0) / files_processed * 100) if files_processed > 0 else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": perf,
            "summary_statistics": stats,
            "calculated_metrics": {
                "avg_time_per_file_seconds": avg_time_per_file,
                "error_rate_percent": error_rate,
                "files_per_second": files_processed / processing_time if processing_time > 0 else 0,
                "efficiency_score": max(0, 100 - error_rate),
            },
            "recommendations": self._generate_performance_recommendations(perf, stats)
        }
    
    def _generate_performance_recommendations(self, perf: Dict[str, Any], stats: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []
        
        error_rate = (perf.get('errors_encountered', 0) / perf.get('files_processed', 1)) * 100
        processing_time = perf.get('total_processing_time', 0)
        files_processed = perf.get('files_processed', 0)
        
        if error_rate > 20:
            recommendations.append("High error rate detected - consider reviewing input files for consistency")
        
        if processing_time > 60 and files_processed < 10:
            recommendations.append("Slow processing detected - consider optimizing file content or using parallel processing")
        
        if stats.get('type_errors', 0) > stats.get('valid_files', 1):
            recommendations.append("More type errors than valid files - consider improving GNN syntax")
        
        if not recommendations:
            recommendations.append("Performance metrics look good - no immediate optimizations needed")
        
        return recommendations
    
    def validate_pomdp_file(self, file_path: Path, verbose: bool = False) -> Dict[str, Any]:
        """
        Validate a single POMDP GNN file with specialized analysis.
        
        Args:
            file_path: Path to the GNN file
            verbose: Enable verbose output
            
        Returns:
            Dictionary containing POMDP validation results
        """
        if not self.pomdp_mode or not self.pomdp_analyzer:
            return {
                "error": "POMDP mode not enabled",
                "file_path": str(file_path),
                "valid": False
            }
        
        try:
            pomdp_result = self.pomdp_analyzer.validate_pomdp_model(file_path)
            
            # Add complexity estimation
            if pomdp_result.get("validation_results", {}).get("overall_valid", False):
                complexity = self.pomdp_analyzer.estimate_pomdp_complexity(pomdp_result)
                pomdp_result["complexity_estimation"] = complexity
            
            return {
                "pomdp_analysis": pomdp_result,
                "file_path": str(file_path),
                "valid": pomdp_result.get("overall_valid", False)
            }
            
        except Exception as e:
            self.logger.error(f"Error validating POMDP file {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "validation_results": {
                    "overall_valid": False,
                    "structure_valid": False,
                    "dimension_consistency": False,
                    "ontology_compliance": False
                },
                "errors": [f"POMDP validation error: {str(e)}"],
                "timestamp": datetime.now().isoformat()
            }
    
    def get_pomdp_analysis_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate a summary of POMDP analysis results.
        
        Args:
            results: Results dictionary containing POMDP analysis
            
        Returns:
            Formatted summary string
        """
        if not self.pomdp_mode or "pomdp_analysis" not in results:
            return "POMDP analysis not available"
        
        pomdp_analyses = results["pomdp_analysis"]
        pomdp_metrics = results.get("summary_statistics", {}).get("pomdp_metrics", {})
        
        summary = f"""
# POMDP Analysis Summary

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## POMDP Models Processed
- **Total POMDP Models**: {len(pomdp_analyses)}
- **Valid POMDP Models**: {pomdp_metrics.get('pomdp_models_valid', 0)}
- **Invalid POMDP Models**: {len(pomdp_analyses) - pomdp_metrics.get('pomdp_models_valid', 0)}

## Space Dimensions
- **Total State Space Size**: {pomdp_metrics.get('total_state_space_size', 0)}
- **Total Observation Space Size**: {pomdp_metrics.get('total_observation_space_size', 0)}
- **Total Action Space Size**: {pomdp_metrics.get('total_action_space_size', 0)}

## Model Details
"""
        
        for i, analysis in enumerate(pomdp_analyses, 1):
            file_name = analysis.get("file_name", f"Model {i}")
            # Handle nested pomdp_analysis structure
            if "pomdp_analysis" in analysis:
                inner_analysis = analysis["pomdp_analysis"]
                pomdp_specific = inner_analysis.get("pomdp_specific", {})
                validation = inner_analysis.get("validation_results", {})
                overall_valid = inner_analysis.get("overall_valid", False)
            else:
                pomdp_specific = analysis.get("pomdp_specific", {})
                validation = analysis.get("validation_results", {})
                overall_valid = analysis.get("overall_valid", False)
            
            summary += f"""
### {file_name}
- **State Space**: {pomdp_specific.get('state_space_size', 0)}
- **Observation Space**: {pomdp_specific.get('observation_space_size', 0)}
- **Action Space**: {pomdp_specific.get('action_space_size', 0)}
- **Model Complexity**: {pomdp_specific.get('model_complexity', 'unknown')}
- **Valid**: {overall_valid}
- **Structure Valid**: {validation.get('structure_valid', False)}
- **Dimension Consistent**: {validation.get('dimension_consistency', False)}
- **Ontology Compliant**: {validation.get('ontology_compliance', False)}
"""
            
            if analysis.get("errors"):
                summary += f"- **Errors**: {len(analysis['errors'])}\n"
            if analysis.get("warnings"):
                summary += f"- **Warnings**: {len(analysis['warnings'])}\n"
        
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
