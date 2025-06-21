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

# Initialize logger for this step
logger = setup_step_logging("11_llm", verbose=False)

# Import LLM functionality
try:
    from llm.prompts import (
        PromptType, 
        get_prompt, 
        get_default_prompt_sequence,
        get_prompt_title,
        get_all_prompt_types
    )
    from llm.llm_processor import LLMProcessor, AnalysisType, load_api_keys_from_env, get_default_provider_configs
    from llm.llm_operations import LLMOperations
    logger.debug("Successfully imported LLM modules")
    LLM_AVAILABLE = True
except ImportError as e:
    log_step_error(logger, f"Could not import LLM modules: {e}")
    LLM_AVAILABLE = False
    PromptType = None
    get_prompt = None
    get_default_prompt_sequence = None
    LLMProcessor = None
    LLMOperations = None

def parse_task_list(tasks_str: str) -> List[PromptType]:
    """
    Parse the tasks string into a list of PromptType enums.
    
    Args:
        tasks_str: Comma-separated list of task names or 'all'
        
    Returns:
        List of PromptType enums to execute
    """
    if not LLM_AVAILABLE:
        return []
    
    if tasks_str.lower() == 'all':
        return get_default_prompt_sequence()
    
    # Map string names to PromptType enums
    task_map = {
        'explain': PromptType.EXPLAIN_MODEL,
        'explain_model': PromptType.EXPLAIN_MODEL,
        'structure': PromptType.ANALYZE_STRUCTURE,
        'analyze_structure': PromptType.ANALYZE_STRUCTURE,
        'summary': PromptType.SUMMARIZE_CONTENT,
        'summarize': PromptType.SUMMARIZE_CONTENT,
        'summarize_content': PromptType.SUMMARIZE_CONTENT,
        'components': PromptType.IDENTIFY_COMPONENTS,
        'identify_components': PromptType.IDENTIFY_COMPONENTS,
        'math': PromptType.MATHEMATICAL_ANALYSIS,
        'mathematical_analysis': PromptType.MATHEMATICAL_ANALYSIS,
        'applications': PromptType.PRACTICAL_APPLICATIONS,
        'practical_applications': PromptType.PRACTICAL_APPLICATIONS,
        'parameters': PromptType.EXTRACT_PARAMETERS,
        'extract_parameters': PromptType.EXTRACT_PARAMETERS,
        'improvements': PromptType.SUGGEST_IMPROVEMENTS,
        'suggest_improvements': PromptType.SUGGEST_IMPROVEMENTS,
    }
    
    tasks = []
    for task_name in tasks_str.split(','):
        task_name = task_name.strip().lower()
        if task_name in task_map:
            tasks.append(task_map[task_name])
        else:
            logger.warning(f"Unknown task name: {task_name}")
    
    return tasks

async def analyze_gnn_file_with_llm(
    gnn_file: Path, 
    gnn_content: str,
    llm_processor: LLMProcessor,
    tasks: List[PromptType],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Analyze a single GNN file using LLM with multiple prompts.
    
    Args:
        gnn_file: Path to the GNN file
        gnn_content: Content of the GNN file
        llm_processor: Initialized LLM processor
        tasks: List of analysis tasks to perform
        output_dir: Directory to save results
        
    Returns:
        Dictionary with analysis results and metadata
    """
    results = {
        'file': str(gnn_file),
        'file_name': gnn_file.stem,
        'analyses': {},
        'errors': [],
        'success_count': 0,
        'total_tasks': len(tasks)
    }
    
    # Create subdirectory for this file's analyses
    file_output_dir = output_dir / gnn_file.stem
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Analyzing {gnn_file.name} with {len(tasks)} LLM tasks")
    
    for task_type in tasks:
        try:
            logger.debug(f"Running {task_type.value} analysis on {gnn_file.name}")
            
            # Get the formatted prompt for this task
            prompt_config = get_prompt(task_type, gnn_content)
            
            # Make the LLM call - convert to LLMMessage objects
            from llm.providers.base_provider import LLMMessage
            messages = [
                LLMMessage(role="system", content=prompt_config["system_message"]),
                LLMMessage(role="user", content=prompt_config["user_prompt"])
            ]
            
            response = await llm_processor.get_response(
                messages=messages,
                max_tokens=prompt_config.get("max_tokens", 1600),
                temperature=0.3
            )
            
            if response and response.content:
                # Save the analysis result
                task_title = get_prompt_title(task_type)
                analysis_file = file_output_dir / f"{task_type.value}_analysis.md"
                
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {task_title}\n\n")
                    f.write(f"**File:** {gnn_file.name}\n\n")
                    f.write(f"**Analysis Type:** {task_type.value}\n\n")
                    f.write(f"**Generated:** {performance_tracker.get_timestamp()}\n\n")
                    f.write("---\n\n")
                    f.write(response.content)
                    f.write("\n\n---\n\n")
                    f.write(f"*Analysis generated using LLM provider: {response.provider}*\n")
                
                results['analyses'][task_type.value] = {
                    'success': True,
                    'file': str(analysis_file),
                    'provider': response.provider,
                    'content_length': len(response.content),
                    'tokens_used': getattr(response, 'tokens_used', None)
                }
                results['success_count'] += 1
                
                log_step_success(logger, f"Completed {task_type.value} analysis for {gnn_file.name}")
                
            else:
                error_msg = f"Empty response for {task_type.value} analysis"
                results['errors'].append(error_msg)
                results['analyses'][task_type.value] = {
                    'success': False,
                    'error': error_msg
                }
                log_step_warning(logger, f"{error_msg} on {gnn_file.name}")
                
        except Exception as e:
            error_msg = f"Failed {task_type.value} analysis: {str(e)}"
            results['errors'].append(error_msg)
            results['analyses'][task_type.value] = {
                'success': False,
                'error': error_msg
            }
            log_step_warning(logger, f"{error_msg} on {gnn_file.name}")
    
    # Save summary results
    summary_file = file_output_dir / "analysis_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    return results

async def process_llm_analysis(
    target_dir: Path, 
    output_dir: Path, 
    tasks: List[PromptType],
    recursive: bool = False,
    timeout: int = 360
) -> Dict[str, Any]:
    """Process GNN files with LLM-enhanced analysis."""
    log_step_start(logger, "Processing GNN files with LLM-enhanced analysis")
    
    if not LLM_AVAILABLE:
        log_step_error(logger, "LLM modules not available")
        return {'success': False, 'error': 'LLM modules not available'}
    
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
        # Initialize LLM processor
        with performance_tracker.track_operation("llm_processor_initialization"):
            logger.info("Initializing LLM processor...")
            api_keys = load_api_keys_from_env()
            
            if not api_keys:
                log_step_warning(logger, "No API keys found in environment. LLM analysis will not be available.")
                # Create placeholder analysis files
                return await create_placeholder_analysis(gnn_files, llm_output_dir, tasks)
            
            llm_processor = LLMProcessor(
                api_keys=api_keys,
                provider_configs=get_default_provider_configs()
            )
            
            initialized = await llm_processor.initialize()
            if not initialized:
                log_step_warning(logger, "LLM processor failed to initialize. Creating placeholder analysis.")
                return await create_placeholder_analysis(gnn_files, llm_output_dir, tasks)
        
        # Process each GNN file
        overall_results = {
            'success': True,
            'files_processed': 0,
            'total_files': len(gnn_files),
            'total_analyses': 0,
            'successful_analyses': 0,
            'files': {}
        }
        
        with performance_tracker.track_operation("llm_analysis_batch"):
            for gnn_file in gnn_files:
                try:
                    # Read GNN file content
                    gnn_content = gnn_file.read_text(encoding='utf-8')
                    
                    # Analyze with LLM
                    file_results = await analyze_gnn_file_with_llm(
                        gnn_file, gnn_content, llm_processor, tasks, llm_output_dir
                    )
                    
                    overall_results['files'][gnn_file.stem] = file_results
                    overall_results['files_processed'] += 1
                    overall_results['total_analyses'] += file_results['total_tasks']
                    overall_results['successful_analyses'] += file_results['success_count']
                    
                    log_step_success(logger, f"Processed {gnn_file.name}: {file_results['success_count']}/{file_results['total_tasks']} analyses successful")
                    
                except Exception as e:
                    log_step_warning(logger, f"Failed to process {gnn_file.name}: {e}")
                    overall_results['files'][gnn_file.stem] = {
                        'file': str(gnn_file),
                        'error': str(e),
                        'success': False
                    }
        
        # Save overall results
        results_file = llm_output_dir / "llm_analysis_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(overall_results, f, indent=2)
        
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

async def create_placeholder_analysis(
    gnn_files: List[Path], 
    output_dir: Path, 
    tasks: List[PromptType]
) -> Dict[str, Any]:
    """Create placeholder analysis files when LLM is not available."""
    log_step_warning(logger, "Creating placeholder analysis files (LLM not available)")
    
    results = {
        'success': True,
        'files_processed': len(gnn_files),
        'total_files': len(gnn_files),
        'placeholder_mode': True,
        'files': {}
    }
    
    for gnn_file in gnn_files:
        file_output_dir = output_dir / gnn_file.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        file_results = {
            'file': str(gnn_file),
            'file_name': gnn_file.stem,
            'analyses': {},
            'placeholder': True
        }
        
        for task_type in tasks:
            task_title = get_prompt_title(task_type)
            analysis_file = file_output_dir / f"{task_type.value}_analysis.md"
            
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(f"# {task_title} (Placeholder)\n\n")
                f.write(f"**File:** {gnn_file.name}\n\n")
                f.write(f"**Analysis Type:** {task_type.value}\n\n")
                f.write(f"**Generated:** {performance_tracker.get_timestamp()}\n\n")
                f.write("---\n\n")
                f.write("**Note:** This is a placeholder analysis file. ")
                f.write("LLM analysis was not available during pipeline execution.\n\n")
                f.write("To generate actual LLM analysis:\n")
                f.write("1. Set up API keys in environment variables (OPENAI_API_KEY, etc.)\n")
                f.write("2. Install required dependencies (aiohttp, openai, etc.)\n")
                f.write("3. Re-run the LLM processing step\n\n")
                f.write(f"**Intended Analysis:** {task_title}\n\n")
                f.write("This analysis would have provided:\n")
                
                # Add task-specific placeholder content
                if task_type == PromptType.EXPLAIN_MODEL:
                    f.write("- Comprehensive explanation of what the generative model does\n")
                    f.write("- Analysis of model purpose and real-world applications\n")
                    f.write("- Breakdown of core components and their relationships\n")
                elif task_type == PromptType.ANALYZE_STRUCTURE:
                    f.write("- Detailed structural analysis of the model graph\n")
                    f.write("- Variable analysis and dependencies\n")
                    f.write("- Complexity assessment and design patterns\n")
                elif task_type == PromptType.SUMMARIZE_CONTENT:
                    f.write("- Concise summary of the GNN specification\n")
                    f.write("- Key variables and parameters overview\n")
                    f.write("- Notable features and use cases\n")
                # Add more task-specific descriptions as needed
            
            file_results['analyses'][task_type.value] = {
                'success': True,
                'file': str(analysis_file),
                'placeholder': True
            }
        
        results['files'][gnn_file.stem] = file_results
    
    return results

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
    if LLM_AVAILABLE:
        tasks = parse_task_list(tasks_str)
    else:
        # Use default tasks for placeholder mode
        tasks = [
            PromptType.SUMMARIZE_CONTENT if PromptType else 'summarize_content',
            PromptType.EXPLAIN_MODEL if PromptType else 'explain_model'
        ]
    
    logger.info(f"Will execute {len(tasks)} LLM analysis tasks: {[t.value if hasattr(t, 'value') else t for t in tasks]}")
    
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
            if results.get('placeholder_mode', False):
                log_step_warning(logger, "LLM processing completed in placeholder mode (no LLM available)")
                return 2  # Success with warnings
            else:
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