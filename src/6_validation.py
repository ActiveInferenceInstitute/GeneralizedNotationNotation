#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 6: Validation (Thin Orchestrator)

This step performs validation and quality assurance on GNN models,
including semantic validation, performance profiling, and consistency checking.

How to run:
  python src/6_validation.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Validation results in the specified output directory
  - Semantic validation reports and scores
  - Performance profiling and resource estimates
  - Consistency checking and quality metrics
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that validation dependencies are installed
  - Check that src/validation/ contains validation modules
  - Check that the output directory is writable
  - Verify validation configuration and requirements
"""

import sys
import logging
from pathlib import Path
import json
import datetime
from typing import Dict, Any, List, Optional, Union

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import centralized utilities and configuration
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    performance_tracker
)

from pipeline import (
    get_output_dir_for_script,
    get_pipeline_config
)

from utils.pipeline_template import create_standardized_pipeline_script

# Import core validation functions from validation module
try:
    from validation import (
        process_semantic_validation,
        profile_performance,
        check_consistency,
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

def process_validation_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized validation processing function.
    
    Args:
        target_dir: Directory containing GNN files to validate
        output_dir: Output directory for validation results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Require validation module
        if not VALIDATION_AVAILABLE:
            log_step_error(logger, "Validation module not available")
            return False
        
        # Start performance tracking
        with performance_tracker.track_operation("validation_processing", {"verbose": verbose, "recursive": recursive}):
            # Update logger verbosity if needed
            if verbose:
                logger.setLevel(logging.DEBUG)
            
            # Get configuration
            config = get_pipeline_config()
            step_config = config.get_step_config("6_validation.py")
            
            # Set up output directory
            step_output_dir = get_output_dir_for_script("6_validation.py", output_dir)
            step_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Log processing parameters
            logger.info(f"Processing GNN files from: {target_dir}")
            logger.info(f"Output directory: {step_output_dir}")
            logger.info(f"Recursive processing: {recursive}")
            
            # Load parsed GNN data from previous step
            gnn_output_dir = get_output_dir_for_script("3_gnn.py", output_dir)
            gnn_results_file = gnn_output_dir / "gnn_processing_results.json"
            
            if not gnn_results_file.exists():
                log_step_error(logger, "GNN processing results not found. Run step 3 first.")
                return False
            
            with open(gnn_results_file, 'r') as f:
                gnn_results = json.load(f)
            
            logger.info(f"Loaded {len(gnn_results['processed_files'])} parsed GNN files")
            
            # Validation results
            validation_results = {
                "timestamp": datetime.datetime.now().isoformat(),
                "source_directory": str(target_dir),
                "output_directory": str(step_output_dir),
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
            validation_results_file = step_output_dir / "validation_results.json"
            with open(validation_results_file, 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            # Save validation summary
            validation_summary_file = step_output_dir / "validation_summary.json"
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
            
            log_step_success(logger, "Validation processing completed successfully")
            return True
        
    except Exception as e:
        log_step_error(logger, f"Validation processing failed: {e}")
        return False

def main():
    """Main validation processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("6_validation.py")
    
    # Setup logging
    logger = setup_step_logging("validation", args)
    
    # Check if validation module is available
    if not VALIDATION_AVAILABLE:
        log_step_warning(logger, "Validation module not available, using fallback functions")
    
    # Process validation
    success = process_validation_standardized(
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        logger=logger,
        recursive=args.recursive,
        verbose=args.verbose
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 