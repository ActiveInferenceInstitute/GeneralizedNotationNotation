#!/usr/bin/env python3
"""
Report Generation Pipeline Step

This step generates comprehensive analysis reports from all previous pipeline outputs,
providing a unified view of the GNN processing results.

Usage:
    python 14_report.py [options]
    (Typically called by main.py)
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List, Any, Dict

# Standard imports for all pipeline steps
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    performance_tracker,
    EnhancedArgumentParser,
    UTILS_AVAILABLE
)

from pipeline import (
    STEP_METADATA,
    get_output_dir_for_script,
    get_pipeline_config
)

from utils.pipeline_template import create_standardized_pipeline_script

# Import the report module functions
from report.generator import generate_comprehensive_report

# Initialize logger for this step
logger = setup_step_logging("14_report", verbose=False)

def process_report_generation(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized report generation processing function.
    
    Args:
        target_dir: Directory containing pipeline outputs to analyze
        output_dir: Output directory for generated reports
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        log_step_start(logger, "Starting comprehensive report generation from pipeline outputs")
        
        # Update logger verbosity if needed
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Get the report output directory for this step
        report_output_dir = get_output_dir_for_script("14_report.py", output_dir)
        report_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline output directory: {output_dir}")
        logger.info(f"Report output directory: {report_output_dir}")
        
        # Validate that the output directory exists and has content
        if not output_dir.exists():
            log_step_error(logger, f"Pipeline output directory does not exist: {output_dir}")
            return False
        
        # Generate comprehensive report using the modular report generator
        success = generate_comprehensive_report(output_dir, report_output_dir, logger)
        
        if success:
            report_file = report_output_dir / "comprehensive_analysis_report.html"
            if report_file.exists():
                file_size_mb = report_file.stat().st_size / (1024 * 1024)
                log_step_success(logger, f"Comprehensive report generated successfully: {report_file} ({file_size_mb:.2f} MB)")
                return True
            else:
                log_step_error(logger, f"Report generation reported success but report file not found at {report_file}")
                return False
        else:
            log_step_error(logger, "Report generation failed")
            return False
        
    except Exception as e:
        log_step_error(logger, f"Report generation failed with exception: {e}")
        if verbose:
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

# Create the standardized pipeline script using the template
run_script = create_standardized_pipeline_script(
    "14_report.py",
    process_report_generation,
    "Comprehensive analysis report generation",
    additional_arguments={
        "report_format": {"type": str, "default": "html", "help": "Report format (html, markdown, json)"}
    }
)

if __name__ == '__main__':
    sys.exit(run_script())
