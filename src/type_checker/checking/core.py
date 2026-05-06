"""
Core Type Checker Processor.

Provides the ``GNNTypeChecker`` class which orchestrates type checking
of GNN files, validating syntax, dimensions, and type consistency.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from utils.pipeline_template import log_step_error, log_step_start, log_step_success
from .dimensions import extract_gnn_dimensions, validate_dimension_compatibility
from .rules import (
    check_type_consistency,
    extract_types_from_content,
    get_validation_rules,
    validate_type,
)

# We need to import the estimator lazily or through the standard pipeline
# to avoid circular dependencies if we refactor heavily, but for now we'll 
# import it directly from the estimation package.
from type_checker.estimation.strategies import (
    calculate_complexity,
    estimate_memory,
)

_module_logger = logging.getLogger(__name__)


def estimate_file_resources(content: str) -> Dict[str, Any]:
    """Estimate computational resources needed for a GNN file using core framework logic.
    
    This function bridges the type checker to the estimation subsystem
    for generation of Baseball Cards during the standard validation pass.
    """
    import math
    from type_checker.estimation.estimator import GNNResourceEstimator

    try:
        # Extract structured dimensions
        variables_with_dims = extract_gnn_dimensions(content)

        # Formulate variables map compatible with rigorous estimators
        vars_map = {}
        for k, v in variables_with_dims.items():
            vars_map[k] = {"dimensions": v, "type": "float"}

        # Connections math
        directed = re.findall(r'(\w+)\s*>\s*(\w+)', content)
        undirected = re.findall(r'(\w+)\s*-\s*(\w+)', content)
        edges = [{"source": u, "target": v, "type": "directed"} for u, v in directed]
        edges.extend([{"source": u, "target": v, "type": "undirected"} for u, v in undirected])

        equations = "\n".join([f"{u}={v}" for u, v in re.findall(r'(\w+)\s*=\s*(.+)', content)])

        memory_bytes = estimate_memory(vars_map, GNNResourceEstimator.MEMORY_FACTORS) * 1024 # convert kb to bytes loosely
        complexity_metrics = calculate_complexity(vars_map, edges, equations)

        total_parameters = sum([math.prod(v) if isinstance(v, list) else 1 for v in variables_with_dims.values()])
        if total_parameters == 0:
            total_parameters = len(re.findall(r'(\w+)\s*[:=]', content)) * 9

        complexity_tier = "minimal"
        score = complexity_metrics.get("overall_complexity", 0)
        if score > 2.0: complexity_tier = "small"
        if score > 5.0: complexity_tier = "medium"
        if score > 8.0: complexity_tier = "large"

        return {
            "complexity_tier": complexity_tier,
            "estimated_memory_bytes": int(memory_bytes),
            "total_parameters": total_parameters,
            "variables": len(variables_with_dims),
            "connections": len(edges),
            "flops_estimate": score * 500.0, # Rough FLOPS parameter correlation
            "complexity_score": score
        }
    except Exception as e:
        return {
            "complexity_tier": "unknown",
            "estimated_memory_bytes": 0,
            "total_parameters": 0,
            "variables": 0,
            "connections": 0,
            "flops_estimate": 0,
            "complexity_score": 0,
            "error": str(e)
        }


class GNNTypeChecker:
    """Type checker for GNN files."""

    def __init__(self, *args, **kwargs):
        """Initialize the GNN type checker."""
        self.validation_rules = get_validation_rules()

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

            output_dir.mkdir(parents=True, exist_ok=True)

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
            gnn_files = list(target_dir.rglob("*.md"))
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

            # Generate visualizations natively from results matrix
            from type_checker.visualizer import generate_all_visualizations
            visual_embeddings = generate_all_visualizations(results, output_dir)
            if visual_embeddings:
                results["visual_embeddings"] = visual_embeddings

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
            found_types = extract_types_from_content(content)

            # Validate types
            for type_info in found_types:
                type_validation = validate_type(type_info)
                if not type_validation["valid"]:
                    validation_result["type_issues"].append(type_validation)
                    validation_result["warnings"].append(f"Type issue: {type_validation['message']}")

            # Check for consistency
            consistency_check = check_type_consistency(found_types)
            if not consistency_check["consistent"]:
                validation_result["errors"].append(consistency_check["message"])
                validation_result["valid"] = False

            # Validate dimension compatibility
            gnn_dims = extract_gnn_dimensions(content)
            if gnn_dims:
                dim_check = validate_dimension_compatibility(gnn_dims)
                validation_result["dimension_compatibility"] = dim_check
                if not dim_check["compatible"]:
                    for issue in dim_check["issues"]:
                        validation_result["errors"].append(issue)
                    validation_result["valid"] = False
                for warning in dim_check["warnings"]:
                    validation_result["warnings"].append(warning)

            # Assign resource estimation metadata for baseball cards
            resources = estimate_file_resources(content)
            validation_result["resource_estimation"] = resources

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

    def _analyze_types(self, file_path: Path, verbose: bool = False) -> Dict[str, Any]:
        """Analyze types in a GNN file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Extract type information
            types_found = extract_types_from_content(content)

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
        summary = f"""# Type Check Summary

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

## Graphical Abstracts
"""
        visual_embeddings = results.get("visual_embeddings", [])
        if visual_embeddings:
            for embedding in visual_embeddings:
                summary += f"\\n{embedding}\\n"
        else:
            summary += "\\n*No visual summaries could be generated.*\\n"

        summary += """
## Error Summary
"""

        errors = results.get('errors', [])
        if errors:
            for error in errors:
                if isinstance(error, dict):
                    summary += f"- **{error.get('file', 'Unknown')}**: {error.get('error', 'Unknown error')}\\n"
                else:
                    summary += f"- {error}\\n"
        else:
            summary += "- No errors encountered\\n"

        return summary
