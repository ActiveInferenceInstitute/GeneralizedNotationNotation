#!/usr/bin/env python3
"""
Step 6: Validation Processing (Thin Orchestrator)

This step performs validation and quality assurance on GNN models,
including semantic validation, performance profiling, and consistency checking.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/validation/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the validation module.

Pipeline Flow:
    main.py → 6_validation.py (this script) → validation/ (modular implementation)

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

Exit Codes:
  0: SUCCESS - Validation completed successfully
  1: CRITICAL_ERROR - Critical error preventing validation (missing dependencies, invalid configuration)
  2: SUCCESS_WITH_WARNINGS - Validation completed with warnings (non-critical issues found)
"""

import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import standardized utilities with error handling and logging
from utils import (
    ArgumentParser,
    PipelineErrorHandler,
    StructuredLogger,
    ExitCode,
    ErrorCategory,
    generate_correlation_id,
    set_correlation_context,
    log_step_start,
    log_step_success,
    log_step_warning,
    log_step_error,
    log_pipeline_complete
)

# Import module function
try:
    from validation import process_validation
except ImportError as e:
    process_validation = None


def validate_dependencies(logger: StructuredLogger, error_handler: PipelineErrorHandler) -> bool:
    """Validate that all required dependencies are available."""
    if process_validation is None:
        error = error_handler.create_error(
            "validation",
            ImportError("validation module not available"),
            ErrorCategory.DEPENDENCY,
            context={"module": "validation", "function": "process_validation"}
        )
        error_handler.handle_error(error)
        return False

    logger.info("All validation dependencies are available")
    return True


def setup_validation_environment(args, logger: StructuredLogger, error_handler: PipelineErrorHandler) -> Optional[Path]:
    """Setup the validation environment and validate inputs."""
    try:
        # Validate output directory
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        elif not output_dir.is_dir():
            error = error_handler.create_error(
                "validation",
                ValueError(f"Output path is not a directory: {output_dir}"),
                ErrorCategory.FILE_SYSTEM,
                context={"path": str(output_dir)}
            )
            error_handler.handle_error(error)
            return None

        # Validate target directory
        target_dir = Path(args.target_dir)
        if not target_dir.exists():
            error = error_handler.create_error(
                "validation",
                FileNotFoundError(f"Target directory does not exist: {target_dir}"),
                ErrorCategory.FILE_SYSTEM,
                context={"path": str(target_dir)}
            )
            error_handler.handle_error(error)
            return None

        # Check for GNN processing results from step 3
        gnn_output_dir = Path(args.output_dir) / "3_gnn_output"
        gnn_nested_dir = gnn_output_dir / "3_gnn_output"
        gnn_results_file = gnn_nested_dir / "gnn_processing_results.json"

        if not gnn_results_file.exists():
            error = error_handler.create_error(
                "validation",
                FileNotFoundError("GNN processing results not found. Run step 3 first."),
                ErrorCategory.VALIDATION,
                context={"expected_file": str(gnn_results_file)}
            )
            error_handler.handle_error(error)
            return None

        logger.info("Validation environment setup complete")
        return output_dir

    except Exception as e:
        error = error_handler.create_error(
            "validation",
            e,
            ErrorCategory.EXECUTION,
            context={"operation": "environment_setup"}
        )
        error_handler.handle_error(error)
        return None


def execute_validation(output_dir: Path, args, logger: StructuredLogger, error_handler: PipelineErrorHandler) -> int:
    """Execute the validation processing."""
    start_time = time.time()

    try:
        logger.set_context(operation="validation_processing")

        # Call the validation module
        result = process_validation(
            target_dir=args.target_dir,
            output_dir=str(output_dir),
            verbose=args.verbose,
            recursive=getattr(args, 'recursive', False)
        )

        duration = time.time() - start_time
        logger.info(".2f")

        if result:
            log_step_success(logger, "Validation processing completed successfully", duration=duration)
            return ExitCode.SUCCESS
        else:
            log_step_warning(logger, "Validation processing completed with warnings", duration=duration)
            return ExitCode.SUCCESS_WITH_WARNINGS

    except Exception as e:
        duration = time.time() - start_time
        error = error_handler.create_error(
            "validation",
            e,
            ErrorCategory.EXECUTION,
            context={"operation": "validation_processing", "duration_seconds": duration}
        )
        exit_code = error_handler.handle_error(error)

        log_step_error(logger, f"Validation processing failed: {e}", duration=duration)
        return exit_code


def main() -> int:
    """Main entry point for the validation step with enhanced error handling."""
    # Generate correlation ID for this execution
    correlation_id = generate_correlation_id("validation")
    set_correlation_context(correlation_id)

    # Initialize logger and error handler
    logger = StructuredLogger("validation", correlation_id)
    error_handler = PipelineErrorHandler(logger.base_logger, correlation_id)

    # Log step start
    log_step_start(logger, "validation")

    try:
        # Parse arguments
        parser = ArgumentParser()
        parser.add_argument('--target-dir', required=True, help='Target directory containing GNN files')
        parser.add_argument('--output-dir', required=True, help='Output directory for results')
        parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
        parser.add_argument('--recursive', action='store_true', help='Process directories recursively')

        args = parser.parse_args()

        # Validate dependencies
        if not validate_dependencies(logger, error_handler):
            return ExitCode.CRITICAL_ERROR

        # Setup environment
        output_dir = setup_validation_environment(args, logger, error_handler)
        if output_dir is None:
            return ExitCode.CRITICAL_ERROR

        # Execute validation
        exit_code = execute_validation(output_dir, args, logger, error_handler)

        # Generate error report if there were errors
        if error_handler.errors:
            error_report_path = error_handler.generate_error_report(output_dir)
            logger.info(f"Error report generated: {error_report_path}")

        # Log final status
        if exit_code == ExitCode.SUCCESS:
            log_pipeline_complete(logger, "validation", "SUCCESS")
        elif exit_code == ExitCode.SUCCESS_WITH_WARNINGS:
            log_pipeline_complete(logger, "validation", "SUCCESS_WITH_WARNINGS")
        else:
            log_pipeline_complete(logger, "validation", "FAILED")

        return exit_code

    except Exception as e:
        # Handle any unexpected errors in main
        error = error_handler.create_error(
            "validation",
            e,
            ErrorCategory.EXECUTION,
            context={"operation": "main_execution"}
        )
        exit_code = error_handler.handle_error(error)

        log_step_error(logger, f"Critical error in validation step: {e}")
        return exit_code


if __name__ == "__main__":
    sys.exit(main()) 