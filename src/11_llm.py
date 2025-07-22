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
    get_output_dir_for_script
)

from llm.analyzer import analyze_gnn_files
from utils.pipeline_template import create_standardized_pipeline_script

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
    Enhanced LLM analysis processing function with improved error handling.
    
    Args:
        target_dir: Directory containing GNN files to analyze
        output_dir: Output directory for analysis results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options (llm_tasks, llm_timeout)
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        log_step_start(logger, "Starting enhanced LLM analysis with comprehensive reporting")
        
        # Update logger verbosity if needed
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Use centralized output directory configuration
        llm_output_dir = get_output_dir_for_script("11_llm.py", output_dir)
        llm_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LLM analysis source directory: {target_dir}")
        logger.info(f"LLM analysis output directory: {llm_output_dir}")
        
        # Validate that the target directory exists
        if not target_dir.exists():
            log_step_error(logger, f"Target directory does not exist: {target_dir}")
            return False
        
        # Pass through additional kwargs like llm_tasks and llm_timeout
        analysis_kwargs = {
            'llm_tasks': kwargs.get('llm_tasks', 'all'),
            'llm_timeout': kwargs.get('llm_timeout', 360)
        }
        
        logger.debug(f"Analysis configuration: {analysis_kwargs}")
        
        # Call the enhanced analyze_gnn_files function
        with performance_tracker.track_operation("llm_analysis_complete"):
            overall_results = analyze_gnn_files(
                target_dir=target_dir,
                output_dir=llm_output_dir,
                logger=logger,
                recursive=recursive,
                verbose=verbose,
                **analysis_kwargs
            )
        
        # Enhanced result processing
        if not overall_results:
            log_step_error(logger, "No results returned from LLM analysis")
            return False
        
        # Extract key metrics
        status = overall_results.get('status', 'unknown')
        files_processed = overall_results.get('files_processed', 0)
        successful_analyses = overall_results.get('successful_analyses', 0)
        total_analyses = overall_results.get('total_analyses', 0)
        success_rate = overall_results.get('success_rate', 0)
        
        # Log detailed results
        logger.info(f"=== LLM Analysis Results ===")
        logger.info(f"Status: {status}")
        logger.info(f"Files processed: {files_processed}")
        logger.info(f"Successful analyses: {successful_analyses}")
        logger.info(f"Total analyses: {total_analyses}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        # Determine success based on different criteria
        if status == 'no_providers_available':
            log_step_warning(logger, "No LLM providers available - analysis skipped")
            return True  # Don't fail pipeline for missing providers
        elif status == 'no_files_found':
            log_step_warning(logger, "No GNN files found for analysis")
            return True  # Not an error condition
        elif status == 'error':
            error_msg = overall_results.get('error', 'Unknown error')
            log_step_error(logger, f"LLM analysis error: {error_msg}")
            return False
        elif status == 'completed':
            if successful_analyses > 0:
                log_step_success(logger, 
                    f"LLM analysis completed successfully: {successful_analyses}/{total_analyses} analyses successful ({success_rate:.1f}%)")
                
                # Additional logging for analysis tasks
                analysis_tasks = overall_results.get('analysis_tasks', [])
                available_providers = overall_results.get('available_providers', [])
                logger.info(f"Analysis tasks performed: {', '.join(analysis_tasks)}")
                logger.info(f"Providers used: {', '.join(available_providers)}")
                
                return True
            else:
                log_step_warning(logger, "LLM analysis completed but no analyses were successful")
                return True  # Don't fail pipeline for failed analyses
        else:
            log_step_warning(logger, f"Unknown LLM analysis status: {status}")
            return overall_results.get('success', False)
        
    except Exception as e:
        log_step_error(logger, f"LLM failed: {e}")
        if verbose:
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

run_script = create_standardized_pipeline_script(
    "11_llm.py",
    process_llm_analysis,
    "LLM-enhanced analysis and processing",
    additional_arguments={
        "llm_tasks": {"type": str, "default": "all", "help": "Comma-separated list of LLM tasks"},
        "llm_timeout": {"type": int, "default": 360, "help": "Timeout for LLM operations in seconds"}
    }
)

if __name__ == '__main__':
    sys.exit(run_script()) 