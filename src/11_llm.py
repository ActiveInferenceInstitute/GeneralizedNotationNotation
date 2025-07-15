#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 11: LLM

This script performs LLM-enhanced analysis and processing of GNN models using
structured prompts to generate comprehensive analysis reports.

Usage:
    python 11_llm.py [options]
    (Typically called by main.py)
"""

import sys
import json
import asyncio
from pathlib import Path
import argparse
import logging
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

def parse_task_list(tasks_str: str) -> List[str]: # Changed PromptType to str
    """
    Parse the tasks string into a list of PromptType enums.
    
    Args:
        tasks_str: Comma-separated list of task names or 'all'
        
    Returns:
        List of PromptType enums to execute
    """
    # Removed LLM_AVAILABLE check as LLM modules are no longer imported directly
    
    if tasks_str.lower() == 'all':
        # Assuming get_default_prompt_sequence is no longer available or not needed
        # For now, return a placeholder or raise an error if not available
        return ['summarize_content', 'explain_model'] # Placeholder tasks
    
    # Map string names to PromptType enums
    task_map = {
        'explain': 'explain_model', # Changed PromptType.EXPLAIN_MODEL to 'explain_model'
        'explain_model': 'explain_model',
        'structure': 'analyze_structure', # Changed PromptType.ANALYZE_STRUCTURE to 'analyze_structure'
        'analyze_structure': 'analyze_structure',
        'summary': 'summarize_content', # Changed PromptType.SUMMARIZE_CONTENT to 'summarize_content'
        'summarize': 'summarize_content',
        'summarize_content': 'summarize_content',
        'components': 'identify_components', # Changed PromptType.IDENTIFY_COMPONENTS to 'identify_components'
        'identify_components': 'identify_components',
        'math': 'mathematical_analysis', # Changed PromptType.MATHEMATICAL_ANALYSIS to 'mathematical_analysis'
        'mathematical_analysis': 'mathematical_analysis',
        'applications': 'practical_applications', # Changed PromptType.PRACTICAL_APPLICATIONS to 'practical_applications'
        'practical_applications': 'practical_applications',
        'parameters': 'extract_parameters', # Changed PromptType.EXTRACT_PARAMETERS to 'extract_parameters'
        'extract_parameters': 'extract_parameters',
        'improvements': 'suggest_improvements', # Changed PromptType.SUGGEST_IMPROVEMENTS to 'suggest_improvements'
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

# Removed analyze_gnn_file_with_llm function

async def process_llm_analysis(
    target_dir: Path, 
    output_dir: Path, 
    tasks: List[str], # Changed PromptType to str
    recursive: bool = False,
    timeout: int = 360
) -> Dict[str, Any]:
    """Process GNN files with LLM-enhanced analysis."""
    log_step_start(logger, "Processing GNN files with LLM-enhanced analysis")
    
    # Removed LLM_AVAILABLE check
    
    # Use centralized output directory configuration
    llm_output_dir = get_output_dir_for_script("11_llm.py", output_dir)
    llm_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find GNN files
    if recursive:
        gnn_files = list(target_dir.rglob("*.md"))
    else:
        gnn_files = list(target_dir.glob("*.md"))
    
    if not gnn_files:
        log_step_warning(logger, f"No GNN files found in {target_dir}")
        return {'success': False, 'error': 'No GNN files found'}
    
    try:
        # Call the new analyze_gnn_files function
        overall_results = await analyze_gnn_files(
            gnn_files=gnn_files,
            output_dir=llm_output_dir,
            tasks=tasks,
            logger=logger,
            timeout=timeout
        )
        
        # Log results summary
        if overall_results['files_processed'] > 0:
            success_rate = (overall_results['successful_analyses'] / overall_results['total_analyses']) * 100
            log_step_success(logger, 
                f"LLM processing completed: {overall_results['files_processed']} files, "
                f"{overall_results['successful_analyses']}/{overall_results['total_analyses']} analyses successful ({success_rate:.1f}%)")
        else:
            log_step_warning(logger, "No files were successfully processed")
            overall_results['success'] = False
        
        return overall_results
        
    except Exception as e:
        log_step_error(logger, f"LLM processing failed: {e}")
        return {'success': False, 'error': str(e)}

# Removed create_placeholder_analysis function

def main(parsed_args: argparse.Namespace):
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
        results = asyncio.run(process_llm_analysis(
            target_dir=Path(parsed_args.target_dir),
            output_dir=Path(parsed_args.output_dir),
            tasks=tasks,
            recursive=getattr(parsed_args, 'recursive', False),
            timeout=getattr(parsed_args, 'llm_timeout', 360)
        ))
        
        if results.get('success', False):
            # Removed setup_required check as it's no longer handled locally
            log_step_success(logger, "LLM processing completed successfully")
            return 0
        else:
            log_step_error(logger, f"LLM processing failed: {results.get('error', 'Unknown error')}")
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