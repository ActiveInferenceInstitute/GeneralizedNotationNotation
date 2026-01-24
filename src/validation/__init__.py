"""
Validation Module

This module provides comprehensive validation capabilities for GNN models,
including semantic validation, performance profiling, and consistency checking.
"""

__version__ = "1.1.3"
FEATURES = {
    "semantic_validation": True,
    "performance_profiling": True,
    "consistency_checking": True,
    "multi_model_validation": True,
    "mcp_integration": True
}

from pathlib import Path
from typing import Dict, Any, Optional, List

# Import core validation functionality
from .semantic_validator import SemanticValidator, process_semantic_validation
from .performance_profiler import PerformanceProfiler, profile_performance
from .consistency_checker import ConsistencyChecker, check_consistency

def process_validation(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool:
    """
    Main validation processing function for GNN models.

    This function orchestrates the complete validation workflow including:
    - Semantic validation
    - Performance profiling
    - Consistency checking

    Args:
        target_dir: Directory containing GNN files to validate
        output_dir: Output directory for validation results
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options

    Returns:
        True if validation succeeded, False otherwise
    """
    import json
    import datetime
    import logging
    from pathlib import Path

    # Setup logging
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load parsed GNN data from previous step (step 3)
        from pipeline.config import get_output_dir_for_script
        # Look in the base output directory, not the step-specific directory
        base_output_dir = Path(output_dir).parent if Path(output_dir).name.startswith(('6_validation', '7_export', '8_visualization')) else output_dir
        gnn_output_dir = get_output_dir_for_script("3_gnn.py", base_output_dir)
        gnn_results_file = gnn_output_dir / "gnn_processing_results.json"

        if not gnn_results_file.exists():
            logger.error(f"GNN processing results not found at {gnn_results_file}. Run step 3 first.")
            logger.error(f"Expected file location: {gnn_results_file}")
            logger.error(f"GNN output directory: {gnn_output_dir}")
            logger.error(f"GNN output directory exists: {gnn_output_dir.exists()}")
            if gnn_output_dir.exists():
                logger.error(f"Contents: {list(gnn_output_dir.iterdir())}")
            return False

        with open(gnn_results_file, 'r') as f:
            gnn_results = json.load(f)

        logger.info(f"Loaded {len(gnn_results['processed_files'])} parsed GNN files")

        # Validation results
        validation_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source_directory": str(target_dir),
            "output_directory": str(output_dir),
            "files_validated": [],
            "summary": {
                "total_files": 0,
                "successful_validations": 0,
                "failed_validations": 0,
                "validation_scores": {
                    "semantic": [],
                    "performance": [],
                    "consistency": []
                }
            }
        }

        # Process each file
        for file_result in gnn_results["processed_files"]:
            if not file_result["parse_success"]:
                continue

            file_name = file_result["file_name"]
            logger.info(f"Validating: {file_name}")

            # Load the actual parsed GNN specification
            parsed_model_file = file_result.get("parsed_model_file")
            if parsed_model_file and Path(parsed_model_file).exists():
                try:
                    with open(parsed_model_file, 'r') as f:
                        actual_gnn_spec = json.load(f)
                    logger.info(f"Loaded parsed GNN specification from {parsed_model_file}")
                    model_data = actual_gnn_spec
                except Exception as e:
                    logger.error(f"Failed to load parsed GNN spec from {parsed_model_file}: {e}")
                    model_data = file_result
            else:
                logger.warning(f"Parsed model file not found for {file_name}, using summary data")
                model_data = file_result

            file_validation_result = {
                "file_name": file_name,
                "file_path": file_result["file_path"],
                "validations": {},
                "success": True
            }

            # Perform semantic validation
            try:
                semantic_result = process_semantic_validation(model_data)
                file_validation_result["validations"]["semantic"] = semantic_result
                validation_results["summary"]["validation_scores"]["semantic"].append(
                    semantic_result.get("semantic_score", 0.0)
                )
                logger.info(f"Semantic validation completed for {file_name}")
            except Exception as e:
                logger.error(f"Semantic validation failed for {file_name}: {e}")
                file_validation_result["success"] = False

            # Perform performance profiling
            try:
                performance_result = profile_performance(model_data)
                file_validation_result["validations"]["performance"] = performance_result
                validation_results["summary"]["validation_scores"]["performance"].append(
                    performance_result.get("performance_score", 0.0)
                )
                logger.info(f"Performance profiling completed for {file_name}")
            except Exception as e:
                logger.error(f"Performance profiling failed for {file_name}: {e}")
                file_validation_result["success"] = False

            # Perform consistency checking
            try:
                consistency_result = check_consistency(model_data)
                file_validation_result["validations"]["consistency"] = consistency_result
                validation_results["summary"]["validation_scores"]["consistency"].append(
                    consistency_result.get("consistency_score", 0.0)
                )
                logger.info(f"Consistency checking completed for {file_name}")
            except Exception as e:
                logger.error(f"Consistency checking failed for {file_name}: {e}")
                file_validation_result["success"] = False

            validation_results["files_validated"].append(file_validation_result)
            validation_results["summary"]["total_files"] += 1

            if file_validation_result["success"]:
                validation_results["summary"]["successful_validations"] += 1
            else:
                validation_results["summary"]["failed_validations"] += 1

        # Calculate average scores
        for score_type in ["semantic", "performance", "consistency"]:
            scores = validation_results["summary"]["validation_scores"][score_type]
            if scores:
                avg_score = sum(scores) / len(scores)
                validation_results["summary"]["validation_scores"][f"avg_{score_type}_score"] = avg_score

        # Save validation results
        validation_results_file = output_dir / "validation_results.json"
        with open(validation_results_file, 'w') as f:
            json.dump(validation_results, f, indent=2)

        # Save validation summary
        validation_summary_file = output_dir / "validation_summary.json"
        with open(validation_summary_file, 'w') as f:
            json.dump(validation_results["summary"], f, indent=2)

        logger.info(f"Validation processing completed:")
        logger.info(f"  Total files: {validation_results['summary']['total_files']}")
        logger.info(f"  Successful validations: {validation_results['summary']['successful_validations']}")
        logger.info(f"  Failed validations: {validation_results['summary']['failed_validations']}")

        if validation_results["summary"]["validation_scores"]["semantic"]:
            avg_semantic = validation_results["summary"]["validation_scores"]["avg_semantic_score"]
            logger.info(f"  Average semantic score: {avg_semantic:.2f}")

        if validation_results["summary"]["validation_scores"]["performance"]:
            avg_performance = validation_results["summary"]["validation_scores"]["avg_performance_score"]
            logger.info(f"  Average performance score: {avg_performance:.2f}")

        if validation_results["summary"]["validation_scores"]["consistency"]:
            avg_consistency = validation_results["summary"]["validation_scores"]["avg_consistency_score"]
            logger.info(f"  Average consistency score: {avg_consistency:.2f}")

        success = validation_results["summary"]["successful_validations"] > 0
        return success

    except Exception as e:
        logger.error(f"Validation processing failed: {e}")
        return False


# Re-export main classes and functions
__all__ = [
    '__version__',
    'FEATURES',
    'SemanticValidator',
    'PerformanceProfiler',
    'ConsistencyChecker',
    'process_semantic_validation',
    'profile_performance',
    'check_consistency',
    'process_validation'
]
