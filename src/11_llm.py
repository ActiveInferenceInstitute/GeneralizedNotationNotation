#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 11: LLM

This script performs LLM-enhanced analysis and processing.

Usage:
    python 11_llm.py [options]
    (Typically called by main.py)
"""

import sys
import json
import asyncio
import logging
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional

# Import centralized utilities and configuration
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    performance_tracker,
    UTILS_AVAILABLE
)

from pipeline import (
    STEP_METADATA,
    get_output_dir_for_script
)

from llm.analyzer import analyze_gnn_files

# Initialize logger for this step
logger = setup_step_logging("11_llm", verbose=False)

def parse_task_list(tasks_str: str) -> List[str]:
    """
    Parse the tasks string into a list of task names.
    
    Args:
        tasks_str: Comma-separated list of task names or 'all'
        
    Returns:
        List of task names to execute
    """
    if tasks_str.lower() == 'all':
        return ['summarize_content', 'explain_model']  # Placeholder tasks
    
    # Map string names to task names
    task_map = {
        'explain': 'explain_model',
        'explain_model': 'explain_model',
        'structure': 'analyze_structure',
        'analyze_structure': 'analyze_structure',
        'summary': 'summarize_content',
        'summarize': 'summarize_content',
        'summarize_content': 'summarize_content',
        'components': 'identify_components',
        'identify_components': 'identify_components',
        'math': 'mathematical_analysis',
        'mathematical_analysis': 'mathematical_analysis',
        'applications': 'practical_applications',
        'practical_applications': 'practical_applications',
        'parameters': 'extract_parameters',
        'extract_parameters': 'extract_parameters',
        'improvements': 'suggest_improvements',
        'suggest_improvements': 'suggest_improvements',
    }
    
    tasks = []
    for task_name in tasks_str.split(','):
        task_name = task_name.strip().lower()
        if task_name in task_map:
            tasks.append(task_map[task_name])
        else:
            logger.warning(f"Unknown task name: {task_name}")
    
    return tasks

def process_llm_analysis(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized LLM analysis processing function.
    
    Args:
        target_dir: Directory containing GNN files to analyze
        output_dir: Output directory for analysis results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Use centralized output directory configuration
        llm_output_dir = get_output_dir_for_script("11_llm.py", output_dir)
        llm_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Call the existing analyze_gnn_files function
        overall_results = analyze_gnn_files(
            target_dir=target_dir,
            output_dir=llm_output_dir,
            logger=logger,
            recursive=recursive,
            verbose=verbose
        )
        
        # Log results summary
        if overall_results.get('files_processed', 0) > 0:
            success_rate = (overall_results.get('successful_analyses', 0) / overall_results.get('total_analyses', 1)) * 100
            log_step_success(logger, 
                f"LLM processing completed: {overall_results.get('files_processed', 0)} files, "
                f"{overall_results.get('successful_analyses', 0)}/{overall_results.get('total_analyses', 0)} analyses successful ({success_rate:.1f}%)")
        else:
            log_step_warning(logger, "No files were successfully processed")
            return False
        
        return overall_results.get('success', False)
        
    except Exception as e:
        log_step_error(logger, f"LLM processing failed: {e}")
        return False

def main(parsed_args):
    """Main function for LLM processing."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("11_llm.py", {})
    log_step_start(logger, f"{step_info.get('description', 'LLM-enhanced analysis and processing')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Parse tasks
    tasks_str = getattr(parsed_args, 'llm_tasks', 'all')
    tasks = parse_task_list(tasks_str)
    
    logger.info(f"Will execute {len(tasks)} LLM analysis tasks: {[t for t in tasks]}")
    
    # Process LLM analysis
    try:
        success = process_llm_analysis(
            target_dir=Path(parsed_args.target_dir),
            output_dir=Path(parsed_args.output_dir),
            logger=logger,
            recursive=getattr(parsed_args, 'recursive', False),
            verbose=getattr(parsed_args, 'verbose', False),
            timeout=getattr(parsed_args, 'llm_timeout', 360)
        )
        
        if success:
            log_step_success(logger, "LLM processing completed successfully")
            return 0
        else:
            log_step_error(logger, "LLM processing failed")
            return 1
    
    except Exception as e:
        log_step_error(logger, f"Unexpected error in LLM processing: {e}")
        return 1

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("11_llm")
    else:
        # Fallback argument parsing
        import argparse
        parser = argparse.ArgumentParser(description="LLM-enhanced analysis and processing")
        parser.add_argument("--target-dir", type=Path, required=True,
                          help="Target directory containing GNN files")
        parser.add_argument("--output-dir", type=Path, required=True,
                          help="Output directory for generated artifacts")
        parser.add_argument("--recursive", action="store_true",
                          help="Search recursively in subdirectories")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose output")
        parser.add_argument("--llm-tasks", type=str, default="all",
                          help="Comma-separated list of LLM tasks or 'all'")
        parser.add_argument("--llm-timeout", type=int, default=360,
                          help="Timeout for LLM operations in seconds")
        parsed_args = parser.parse_args()
    
    exit_code = main(parsed_args)
    sys.exit(exit_code) 