"""
Core Type Checker Processor.

Provides the ``GNNTypeChecker`` class which orchestrates type checking
of GNN files, validating syntax, dimensions, and type consistency.
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TypedDict

from utils.pipeline_template import (
    log_step_error,
    log_step_start,
    log_step_success,
    log_step_warning,
)

# We need to import the estimator lazily or through the standard pipeline
# to avoid circular dependencies if we refactor heavily, but for now we'll
# import it directly from the estimation package.
from ..estimation.strategies import VariableMap, calculate_complexity
from .dimensions import (
    extract_b_matrix_evidence,
    extract_gnn_dimensions_with_diagnostics,
    validate_dimension_compatibility,
)
from .rules import (
    check_type_consistency,
    extract_types_from_content,
    get_validation_rules,
    validate_type,
)

_module_logger = logging.getLogger(__name__)


class ResourceEstimate(TypedDict):
    """Stable resource-estimation result emitted by the type checker."""

    complexity_tier: str
    estimated_memory_bytes: int
    total_parameters: int
    variables: int
    connections: int
    flops_estimate: float
    complexity_score: float
    diagnostics: list[str]


def _extract_markdown_section(content: str, section_name: str) -> str:
    """Return one canonical Markdown GNN section without adjacent prose."""
    lines: list[str] = []
    in_section = False
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("## "):
            in_section = stripped[3:].strip() == section_name
            continue
        if in_section:
            lines.append(raw_line)
    return "\n".join(lines).strip()


def _connection_group(value: str) -> list[str]:
    group = value.strip()
    if group.startswith("(") and group.endswith(")"):
        group = group[1:-1]
    return [
        "π" if name.strip().lower() == "pi" else name.strip()
        for name in group.split(",")
        if name.strip()
    ]


def _parse_resource_connections(
    content: str, known_variables: set[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse GNN connection groups without matching prose outside the section."""
    edges: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    for line_number, raw_line in enumerate(
        _extract_markdown_section(content, "Connections").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        operator = next(
            (
                candidate
                for candidate in ("<->", "->", ">", "|", "-")
                if candidate in line
            ),
            None,
        )
        if operator is None:
            diagnostics.append(
                f"Unparseable connection at section line {line_number}: '{line}'"
            )
            continue
        source_text, target_text = line.split(operator, 1)
        target_text = target_text.split(":", 1)[0]
        sources = _connection_group(source_text)
        targets = _connection_group(target_text)
        if not sources or not targets:
            diagnostics.append(
                f"Connection at section line {line_number} has an empty endpoint: '{line}'"
            )
            continue

        edge_type = "undirected" if operator in {"-", "<->"} else "directed"
        for source in sources:
            for target in targets:
                edges.append({"source": source, "target": target, "type": edge_type})
                if source not in known_variables:
                    diagnostics.append(
                        f"Connection at section line {line_number} references undeclared variable '{source}'"
                    )
                if target not in known_variables:
                    diagnostics.append(
                        f"Connection at section line {line_number} references undeclared variable '{target}'"
                    )
    return edges, diagnostics


def estimate_file_resources(content: str) -> ResourceEstimate:
    """Estimate computational resources needed for a GNN file using core framework logic.

    This function bridges the type checker to the estimation subsystem
    for generation of Baseball Cards during the standard validation pass.
    """
    from ..estimation.estimator import GNNResourceEstimator

    variables_with_dims, diagnostics = extract_gnn_dimensions_with_diagnostics(content)
    variable_types = {
        str(type_info["name"]): str(type_info["type"])
        for type_info in extract_types_from_content(content)
    }

    variables: VariableMap = {
        name: {
            "dimensions": dimensions,
            "type": variable_types.get(name, "float"),
        }
        for name, dimensions in variables_with_dims.items()
    }

    edges, connection_diagnostics = _parse_resource_connections(content, set(variables))
    diagnostics.extend(connection_diagnostics)
    equations = _extract_markdown_section(content, "Equations")

    total_parameters = 0
    memory_bytes = 0
    for name, dimensions in variables_with_dims.items():
        parameter_count = math.prod(dimensions)
        total_parameters += parameter_count
        variable_type = variable_types.get(name, "float")
        bytes_per_element = GNNResourceEstimator.MEMORY_FACTORS.get(
            variable_type,
            GNNResourceEstimator.MEMORY_FACTORS["float"],
        )
        memory_bytes += parameter_count * bytes_per_element

    complexity_metrics = calculate_complexity(variables, edges, equations)
    score = complexity_metrics["overall_complexity"]
    if score > 8.0:
        complexity_tier = "large"
    elif score > 5.0:
        complexity_tier = "medium"
    elif score > 2.0:
        complexity_tier = "small"
    else:
        complexity_tier = "minimal"

    return {
        "complexity_tier": complexity_tier,
        "estimated_memory_bytes": memory_bytes,
        "total_parameters": total_parameters,
        "variables": len(variables),
        "connections": len(edges),
        "flops_estimate": score * 500.0,
        "complexity_score": score,
        "diagnostics": diagnostics,
    }


class GNNTypeChecker:
    """Type checker for GNN files."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the GNN type checker."""
        self.validation_rules = get_validation_rules()

    def check_file(
        self, file_path: str | Path
    ) -> tuple[bool, list[str], list[str], Dict[str, Any]]:
        """Validate one GNN file and return the CLI result tuple."""
        result = self.validate_single_gnn_file(Path(file_path))
        return (
            bool(result.get("valid", False)),
            list(result.get("errors", [])),
            list(result.get("warnings", [])),
            result,
        )

    def generate_report(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        report_md_filename: str = "TYPE_CHECK_REPORT.md",
        project_root_path: str | None = None,
    ) -> str:
        """Generate the Markdown report consumed by the type-checker CLI."""
        output_dir.mkdir(parents=True, exist_ok=True)
        validation_results = [
            {"file_path": path, **data} for path, data in results.items()
        ]
        summary_data: Dict[str, Any] = {
            "processed_files": len(results),
            "success": all(
                item.get("valid", item.get("is_valid", False))
                for item in results.values()
            ),
            "errors": [
                error for item in results.values() for error in item.get("errors", [])
            ],
            "validation_results": validation_results,
            "type_analysis": [],
            "project_root": project_root_path,
        }
        report_text = self._generate_type_check_summary(summary_data)
        (output_dir / report_md_filename).write_text(report_text, encoding="utf-8")
        return report_text

    def generate_json_data(self, results: Dict[str, Any], output_path: Path) -> None:
        """Write type-check results as JSON for downstream analysis."""
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    def _discover_gnn_files(self, target_dir: Path) -> list[Path]:
        """Discover GNN specs across every registered non-binary extension.

        The parser stack registers more than just ``.md`` (notably ``.gnn``);
        a directory holding only a non-markdown spec must still be found.
        Binary pickle specs are excluded — they are not type-checked. Sorted
        for deterministic, reproducible discovery.
        """
        from gnn.discovery import is_model_source_path
        from gnn.parsers.common import get_supported_gnn_extensions

        return sorted(
            path
            for ext in get_supported_gnn_extensions(include_binary_pickle=False)
            for path in target_dir.rglob(f"*{ext}")
            if is_model_source_path(path)
        )

    def validate_gnn_files(
        self, target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
    ) -> bool | int:
        """
        Validate GNN files for type consistency.

        Args:
            target_dir: Directory containing GNN files to validate
            output_dir: Directory to save validation results
            verbose: Enable verbose output
            **kwargs: Additional arguments

        Returns:
            ``True`` (coerces to pipeline exit 0) when every file validates;
            ``2`` (coerces to pipeline exit ``SUCCESS_WITH_WARNINGS``) when the
            run completed but one or more files had type/consistency problems
            (recoverable); ``False`` (coerces to exit 1) when a hard failure
            occurred (exception / nothing to process).
        """
        logger = logging.getLogger("type_checker")

        try:
            log_step_start(logger, "Processing type checker")

            output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize results
            results: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "processed_files": 0,
                "success": True,
                "errors": [],
                "validation_results": [],
                "type_analysis": [],
            }
            hard_failure = False  # any exception / nothing-to-process (vs recoverable invalid files)
            b_orientation_failed = False  # any [GNN-E002] B orientation/contradiction error
            strict = bool(kwargs.get("strict", False))

            # Find GNN files across every registered spec extension (the
            # parser stack supports more than markdown — discovering only
            # *.md silently ignored e.g. *.gnn files and reported
            # "No GNN files found" for a directory that held a valid spec).
            gnn_files = self._discover_gnn_files(target_dir)
            if not gnn_files:
                logger.warning("No GNN files found for type checking")
                results["success"] = False
                hard_failure = True
                results["errors"].append("No GNN files found")
            else:
                results["processed_files"] = len(gnn_files)

                # Process each GNN file
                for gnn_file in gnn_files:
                    try:
                        # Validate single file
                        validation_result = self.validate_single_gnn_file(
                            gnn_file, verbose, strict=strict
                        )
                        results["validation_results"].append(validation_result)
                        if not validation_result.get("valid", False):
                            results["success"] = False
                        if any(
                            "[GNN-E002]" in str(error)
                            for error in validation_result.get("errors", [])
                        ):
                            b_orientation_failed = True

                        # Analyze types
                        type_analysis = self._analyze_types(gnn_file, verbose)
                        results["type_analysis"].append(type_analysis)

                    except Exception as e:
                        results["success"] = False
                        hard_failure = True
                        error_info: dict[str, Any] = {
                            "file": str(gnn_file),
                            "error": str(e),
                            "error_type": type(e).__name__,
                        }
                        results["errors"].append(error_info)
                        logger.error(f"Error processing {gnn_file}: {e}")

            # Save detailed results directly in output directory
            results_file = output_dir / "type_check_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            # Generate visualizations natively from results matrix
            from ..visualizer import generate_all_visualizations

            visual_embeddings = generate_all_visualizations(results, output_dir)
            if visual_embeddings:
                results["visual_embeddings"] = visual_embeddings

            # Generate type check summary
            summary = self._generate_type_check_summary(results)
            summary_file = output_dir / "type_check_summary.md"
            with open(summary_file, "w") as f:
                f.write(summary)

            if results["success"]:
                log_step_success(logger, "Type checking completed successfully")
                return True
            if hard_failure:
                log_step_error(logger, "Type checking failed")
                return False
            if strict and b_orientation_failed:
                log_step_error(
                    logger,
                    "Type checking failed (strict mode: B orientation "
                    "contradictions [GNN-E002] are errors)",
                )
                return False
            log_step_warning(
                logger, "Type checking completed with warnings (some files invalid)"
            )
            return 2

        except Exception as e:
            log_step_error(logger, "Type checking failed", error=str(e))
            return False

    def validate_single_gnn_file(
        self, file_path: Path, verbose: bool = False, strict: bool = False
    ) -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            validation_result: dict[str, Any] = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "valid": True,
                "errors": [],
                "warnings": [],
                "type_issues": [],
                "validation_timestamp": datetime.now().isoformat(),
            }

            # Check for type definitions
            found_types = extract_types_from_content(content)

            # Validate types
            for type_info in found_types:
                type_validation = validate_type(type_info)
                if not type_validation["valid"]:
                    validation_result["type_issues"].append(type_validation)
                    validation_result["errors"].append(
                        f"Type issue: {type_validation['message']}"
                    )
                    validation_result["valid"] = False

            # Check for consistency
            consistency_check = check_type_consistency(found_types)
            if not consistency_check["consistent"]:
                validation_result["errors"].append(consistency_check["message"])
                validation_result["valid"] = False

            # Validate dimension compatibility
            gnn_dims, dimension_diagnostics = extract_gnn_dimensions_with_diagnostics(
                content
            )
            if dimension_diagnostics:
                validation_result["errors"].extend(dimension_diagnostics)
                validation_result["valid"] = False
            if gnn_dims:
                b_evidence = extract_b_matrix_evidence(content)
                dim_check = validate_dimension_compatibility(
                    gnn_dims, b_evidence=b_evidence, strict=strict
                )
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
                "validation_timestamp": datetime.now().isoformat(),
            }

    def _analyze_types(self, file_path: Path, verbose: bool = False) -> Dict[str, Any]:
        """Analyze types in a GNN file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract type information
            types_found = extract_types_from_content(content)

            # Analyze type distribution
            type_counts: dict[Any, Any] = {}
            for type_info in types_found:
                var_type = type_info["type"]
                type_counts[var_type] = type_counts.get(var_type, 0) + 1

            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "types_found": types_found,
                "type_distribution": type_counts,
                "total_variables": len(types_found),
                "analysis_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
            }

    def _generate_type_check_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary of type checking results."""
        summary = f"""# Type Check Summary

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Processing Results
- **Files Processed**: {results.get("processed_files", 0)}
- **Success**: {results.get("success", False)}
- **Errors**: {len(results.get("errors", []))}

## Validation Results
- **Files Validated**: {len(results.get("validation_results", []))}
- **Valid Files**: {sum(1 for r in results.get("validation_results", []) if r.get("valid", False))}
- **Invalid Files**: {sum(1 for r in results.get("validation_results", []) if not r.get("valid", False))}

## Type Analysis
- **Type Analyses**: {len(results.get("type_analysis", []))}
- **Total Variables**: {sum(a.get("total_variables", 0) for a in results.get("type_analysis", []))}

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

        errors = results.get("errors", [])
        if errors:
            for error in errors:
                if isinstance(error, dict):
                    summary += f"- **{error.get('file', 'Unknown')}**: {error.get('error', 'Unknown error')}\\n"
                else:
                    summary += f"- {error}\\n"
        else:
            summary += "- No errors encountered\\n"

        return summary
