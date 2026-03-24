#!/usr/bin/env python3
"""
Type checker processor module for GNN pipeline.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_module_logger = logging.getLogger(__name__)

from utils.pipeline_template import log_step_error, log_step_start, log_step_success


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
    """Estimate computational resources needed for a GNN file.

    Uses framework-aware complexity tiers based on variable dimensions
    rather than naive variable*connection multiplication.
    """
    try:
        # Extract structured dimensions
        variables_with_dims = extract_gnn_dimensions(content)

        # Count connections (directed and undirected)
        directed_connections = len(re.findall(r'(\w+)\s*>\s*(\w+)', content))
        undirected_connections = len(re.findall(r'(\w+)\s*-\s*(\w+)', content))
        total_connections = directed_connections + undirected_connections

        # Compute total parameter count from actual dimensions
        total_parameters = 0
        max_single_var = 0
        for _, dims in variables_with_dims.items():
            elements = 1
            for d in dims:
                elements *= d
            total_parameters += elements
            max_single_var = max(max_single_var, elements)

        # Recovery: use regex variable count if no structured dims found
        if total_parameters == 0:
            var_count = len(re.findall(r'(\w+)\s*[:=]', content))
            total_parameters = var_count * 9  # assume 3x3 average

        # Framework-aware complexity tiers
        if total_parameters < 100:
            complexity_tier = "minimal"
            estimated_time = 0.1
            estimated_memory = total_parameters * 8
        elif total_parameters < 1000:
            complexity_tier = "small"
            estimated_time = 0.5
            estimated_memory = total_parameters * 8 + 1024 * 1024  # 1MB overhead
        elif total_parameters < 10000:
            complexity_tier = "medium"
            estimated_time = 5.0
            estimated_memory = total_parameters * 8 + 10 * 1024 * 1024  # 10MB overhead
        elif total_parameters < 100000:
            complexity_tier = "large"
            estimated_time = 30.0
            estimated_memory = total_parameters * 8 + 100 * 1024 * 1024  # 100MB overhead
        else:
            complexity_tier = "very_large"
            estimated_time = 120.0
            estimated_memory = total_parameters * 8 + 500 * 1024 * 1024  # 500MB overhead

        return {
            "variables": len(variables_with_dims),
            "connections": total_connections,
            "total_parameters": total_parameters,
            "max_single_variable_elements": max_single_var,
            "estimated_memory_bytes": int(estimated_memory),
            "estimated_time_seconds": estimated_time,
            "complexity_score": total_parameters * (1 + total_connections * 0.1),
            "complexity_tier": complexity_tier,
            "dimension_map": variables_with_dims
        }

    except Exception as e:
        return {
            "error": str(e),
            "variables": 0,
            "connections": 0,
            "total_parameters": 0,
            "estimated_memory_bytes": 0,
            "estimated_time_seconds": 0,
            "complexity_score": 0,
            "complexity_tier": "unknown"
        }


def validate_dimension_compatibility(variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that matrix/tensor dimensions are compatible in a GNN model.

    Checks Active Inference POMDP constraints:
    - Likelihood matrix A[obs, states]: columns must match hidden state count
    - Transition tensor B[states, states, actions]: first two dims must match
    - Preference C[obs]: length must match A's first dimension
    - Prior D[states]: length must match A's second dimension

    Args:
        variables: Dict mapping variable names to their dimension specs
                   e.g. {"A": [3,3], "B": [3,3,3], "s": [3,1]}

    Returns:
        Dict with keys: compatible (bool), issues (list of str), warnings (list of str)
    """
    issues = []
    warnings = []

    # Parse dimension specs: extract variables with numeric dimensions
    dims = {}
    for name, spec in variables.items():
        if isinstance(spec, (list, tuple)) and all(isinstance(d, int) for d in spec):
            dims[name] = list(spec)

    # Check A-s compatibility: A[obs, states], s[states, 1]
    if "A" in dims and "s" in dims:
        a_dims = dims["A"]
        s_dims = dims["s"]
        if len(a_dims) >= 2 and len(s_dims) >= 1:
            if a_dims[1] != s_dims[0]:
                issues.append(
                    f"Dimension mismatch: A[{a_dims[0]},{a_dims[1]}] column count ({a_dims[1]}) "
                    f"!= s[{s_dims[0]},...] row count ({s_dims[0]}). "
                    f"A's columns must equal the number of hidden states."
                )

    # Check B symmetry: B[states, states, actions] -- first two dims must match
    if "B" in dims:
        b_dims = dims["B"]
        if len(b_dims) >= 2 and b_dims[0] != b_dims[1]:
            issues.append(
                f"Transition matrix B[{','.join(str(d) for d in b_dims)}]: "
                f"first two dimensions must match (got {b_dims[0]} != {b_dims[1]}). "
                f"B[next_states, prev_states, actions] requires next_states == prev_states."
            )

    # Check A-B state dimension consistency
    if "A" in dims and "B" in dims:
        a_dims = dims["A"]
        b_dims = dims["B"]
        if len(a_dims) >= 2 and len(b_dims) >= 1:
            if a_dims[1] != b_dims[0]:
                issues.append(
                    f"State dimension mismatch between A and B: "
                    f"A has {a_dims[1]} hidden states, B has {b_dims[0]} states. "
                    f"Must be equal."
                )

    # Check C-A observation compatibility
    if "C" in dims and "A" in dims:
        c_dims = dims["C"]
        a_dims = dims["A"]
        if len(c_dims) >= 1 and len(a_dims) >= 1:
            c_obs = c_dims[0]
            a_obs = a_dims[0]
            if c_obs != a_obs:
                issues.append(
                    f"Preference vector C[{c_obs}] length != A observation dimension A[{a_obs},...]. "
                    f"C must have one entry per observation outcome."
                )

    # Check D-s prior compatibility
    if "D" in dims and "s" in dims:
        d_dims = dims["D"]
        s_dims = dims["s"]
        if len(d_dims) >= 1 and len(s_dims) >= 1:
            if d_dims[0] != s_dims[0]:
                issues.append(
                    f"Prior D[{d_dims[0]}] length != hidden state s[{s_dims[0]},...] count. "
                    f"D must have one entry per hidden state."
                )

    # Warn about very large dimensions (tractability)
    for name, d in dims.items():
        total_elements = 1
        for dim in d:
            total_elements *= dim
        if total_elements > 10000:
            warnings.append(
                f"Variable {name} with dimensions {d} has {total_elements} total elements. "
                f"Consider dimensionality reduction for tractable inference."
            )

    return {
        "compatible": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "variables_checked": list(dims.keys()),
        "dimension_map": dims
    }


def extract_gnn_dimensions(content: str) -> Dict[str, Any]:
    """
    Extract variable dimensions from GNN StateSpaceBlock content.

    Parses patterns like: A[3,3,type=float], s[3,1,type=float]

    Args:
        content: Full GNN file content as string

    Returns:
        Dict mapping variable names to their dimension lists
    """
    variables = {}

    # Match: varname[dim1,dim2,...,type=xxx] or varname[dim1,dim2,...]
    pattern = r'^([A-Za-z_][A-Za-z0-9_\']*)\s*\[([^\]]+)\]'

    in_state_space = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("## StateSpaceBlock"):
            in_state_space = True
            continue
        elif stripped.startswith("##") and in_state_space:
            in_state_space = False
            continue

        if in_state_space:
            match = re.match(pattern, stripped)
            if match:
                var_name = match.group(1)
                dim_str = match.group(2)
                # Parse dimensions (skip type=xxx entries)
                dims = []
                for part in dim_str.split(","):
                    part = part.strip()
                    if part.startswith("type=") or part.startswith("π") or not part:
                        continue
                    try:
                        dims.append(int(part))
                    except ValueError:
                        _module_logger.debug("Skipping non-integer dimension token: %s", part)
                if dims:
                    variables[var_name] = dims

    return variables
