#!/usr/bin/env python3
"""
Pipeline Module Template

This template shows the recommended structure for all GNN pipeline modules.
Copy this structure for consistent argument handling, logging, and error management.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Any, Callable, List, Dict
from pipeline.config import get_output_dir_for_script

# Standard import pattern for all pipeline modules
try:
    from utils import (
        setup_step_logging,
        log_step_start,
        log_step_success, 
        log_step_warning,
        log_step_error,
        validate_output_directory,
        get_pipeline_utilities,
        UTILS_AVAILABLE
    )
    
    # Enhanced imports for more complex modules
    from utils import (
        ArgumentParser,
        PipelineArguments,
        PerformanceTracker,
        performance_tracker
    )
    
except ImportError as e:
    # Fallback logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create minimal compatibility functions
    def setup_step_logging(name: str, verbose: bool = False):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        return logger
    
    def log_step_start(logger, message): logger.info(f"🚀 {message}")
    def log_step_success(logger, message): logger.info(f"✅ {message}")
    def log_step_warning(logger, message): logger.warning(f"⚠️ {message}")
    def log_step_error(logger, message): logger.error(f"❌ {message}")
    def validate_output_directory(output_dir, step_name): 
        try:
            (output_dir / f"{step_name}_step").mkdir(parents=True, exist_ok=True)
            return True
        except:
            return False
    
    UTILS_AVAILABLE = False

def create_standard_pipeline_script(
    step_name: str,
    module_function: Callable,
    fallback_parser_description: str,
    additional_arguments: Optional[Dict[str, Any]] = None,
    step_specific_imports: Optional[List[str]] = None
) -> Callable:
    """
    Create a standardized pipeline script with consistent argument parsing and error handling.
    
    Args:
        step_name: Name of the step (e.g., "1_gnn", "5_export")
        module_function: The main function to call for processing
        fallback_parser_description: Description for fallback argument parser
        additional_arguments: Additional arguments to add to the parser
        step_specific_imports: Additional imports needed for the step
        
    Returns:
        Function that can be called to run the standardized script
    """
    def run_standardized_script():
        """Standardized script execution function."""
        try:
            # Import step-specific modules if provided
            if step_specific_imports:
                for import_name in step_specific_imports:
                    try:
                        __import__(import_name)
                    except ImportError as e:
                        logging.warning(f"Failed to import {import_name}: {e}")
            
            # Parse arguments
            if UTILS_AVAILABLE:
                try:
                    parsed_args = ArgumentParser.parse_step_arguments(step_name)
                except Exception as e:
                    logging.warning(f"Failed to use enhanced argument parser: {e}")
                    parsed_args = _create_fallback_parser(fallback_parser_description, additional_arguments).parse_args()
            else:
                parsed_args = _create_fallback_parser(fallback_parser_description, additional_arguments).parse_args()
            
            # Set up logging
            logger = setup_step_logging(step_name, getattr(parsed_args, 'verbose', False))
            
            # Convert paths with proper handling for None values
            target_dir_raw = getattr(parsed_args, 'target_dir', None)
            if target_dir_raw is None:
                # Set default based on step type
                if step_name == "11_render.py":
                    # Step 9 should process exports from step 5 by default
                    target_dir = Path('output/gnn_exports/gnn_exports')
                elif step_name == "12_execute.py":
                    # Step 10 should process rendered simulators from step 9 by default
                    target_dir = Path('output/gnn_rendered_simulators')
                else:
                    # Default for other steps
                    target_dir = Path('input/gnn_files')
            else:
                target_dir = Path(target_dir_raw)
            
            output_dir_raw = getattr(parsed_args, 'output_dir', 'output')
            output_dir = Path(output_dir_raw) if output_dir_raw is not None else Path('output')
            
            # Set recursive flag default based on step type
            if step_name in ["11_render.py", "12_execute.py"]:
                # Steps 9 and 10 need recursive processing by default
                recursive_default = True
            else:
                recursive_default = False
            
            # Normalize the output directory to the standardized numbered step folder
            try:
                step_output_dir = get_output_dir_for_script(step_name if not step_name.endswith('.py') else step_name[:-3], output_dir)
            except Exception:
                step_output_dir = output_dir

            # Call the module function with the standardized step output dir
            success = module_function(
                target_dir=target_dir,
                output_dir=step_output_dir,
                logger=logger,
                recursive=getattr(parsed_args, 'recursive', recursive_default),
                verbose=getattr(parsed_args, 'verbose', False),
                **{k: v for k, v in vars(parsed_args).items() 
                   if k not in ['target_dir', 'output_dir', 'recursive', 'verbose']}
            )
            
            return 0 if success else 1
            
        except Exception as e:
            logging.error(f"Script execution failed: {e}")
            return 1
    
    return run_standardized_script

def _create_fallback_parser(description: str, additional_arguments: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    """Create a fallback argument parser with standard arguments."""
    parser = argparse.ArgumentParser(description=description)
    
    # Standard arguments - use None for defaults that get set by pipeline template
    parser.add_argument("--target-dir", type=Path, default=None,
                       help="Target directory containing files to process")
    parser.add_argument("--output-dir", type=Path, default=Path("output"),
                       help="Output directory for generated artifacts")
    parser.add_argument("--recursive", action="store_true",
                       help="Process files recursively")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    # Add additional arguments if provided
    if additional_arguments:
        for arg_name, arg_config in additional_arguments.items():
            if isinstance(arg_config, dict):
                # Use custom flag if provided, otherwise default to --{arg_name}
                config_copy = arg_config.copy()
                flag = config_copy.pop("flag", f"--{arg_name}")
                parser.add_argument(flag, **config_copy)
            else:
                parser.add_argument(f"--{arg_name}", default=arg_config)
    
    return parser

"""
Standardized Pipeline Module Function Template

This module provides templates and utilities for creating consistent module functions
across all pipeline steps.
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import Callable, Optional, Dict, Any, List, Union
# Import utilities - these are already imported above
# from . import (
#     setup_step_logging, log_step_start, log_step_success,
#     log_step_warning, log_step_error, ArgumentParser, UTILS_AVAILABLE
# )
# from ..pipeline import STEP_METADATA, get_output_dir_for_script

def standard_module_function(
    target_dir: Path,
    output_dir: Path, 
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standard template for module functions used by pipeline scripts.
    
    Args:
        target_dir: Directory containing input files to process
        output_dir: Base output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional step-specific arguments
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Validate inputs
        if not target_dir.exists():
            log_step_error(logger, f"Target directory does not exist: {target_dir}")
            return False
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find files to process
        from .shared_functions import find_gnn_files
        gnn_files = find_gnn_files(target_dir, recursive)
        
        if not gnn_files:
            log_step_warning(logger, f"No GNN files found in {target_dir}")
            return True  # Not an error, just no files to process
        
        # Process files
        processed_files = []
        errors = []
        warnings = []
        
        for file_path in gnn_files:
            try:
                # Process individual file (to be implemented by specific modules)
                result = process_single_file(file_path, output_dir, **kwargs)
                if result:
                    processed_files.append(file_path)
                else:
                    errors.append(f"Failed to process {file_path}")
            except Exception as e:
                errors.append(f"Error processing {file_path}: {e}")
        
        # Create and save report
        from .shared_functions import create_processing_report, save_processing_report, log_processing_summary
        report = create_processing_report(
            step_name="standard_module",
            target_dir=target_dir,
            output_dir=output_dir,
            processed_files=processed_files,
            errors=errors,
            warnings=warnings
        )
        save_processing_report(report, output_dir)
        
        # Log summary
        log_processing_summary(
            logger, "standard_module", len(gnn_files), 
            len(processed_files), len(errors), len(warnings)
        )
        
        return len(errors) == 0
        
    except Exception as e:
        log_step_error(logger, f"Standard module processing failed: {e}")
        return False

def process_single_file(file_path: Path, output_dir: Path, **kwargs) -> bool:
    """
    Template for processing a single file. Override this in specific modules.
    
    Args:
        file_path: Path to the file to process
        output_dir: Output directory for results
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    # This is a template - specific modules should override this
    return True

def create_standard_module_function(
    step_name: str,
    process_function: Callable,
    additional_params: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Create a standardized module function with the given process function.
    
    Args:
        step_name: Name of the step (e.g., "gnn", "export")
        process_function: Function to process individual files
        additional_params: Additional parameters to pass to the process function
        
    Returns:
        Standardized module function
    """
    def standard_function(
        target_dir: Path,
        output_dir: Path,
        logger: logging.Logger,
        recursive: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> bool:
        """Standardized module function for {step_name}."""
        try:
            # Validate inputs
            if not target_dir.exists():
                log_step_error(logger, f"Target directory does not exist: {target_dir}")
                return False
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find files to process
            from .shared_functions import find_gnn_files
            gnn_files = find_gnn_files(target_dir, recursive)
            
            if not gnn_files:
                log_step_warning(logger, f"No GNN files found in {target_dir}")
                return True  # Not an error, just no files to process
            
            # Process files
            processed_files = []
            errors = []
            warnings = []
            
            # Merge additional parameters with kwargs
            all_params = additional_params or {}
            all_params.update(kwargs)
            
            for file_path in gnn_files:
                try:
                    # Call the specific process function
                    result = process_function(file_path, output_dir, **all_params)
                    if result:
                        processed_files.append(file_path)
                    else:
                        errors.append(f"Failed to process {file_path}")
                except Exception as e:
                    errors.append(f"Error processing {file_path}: {e}")
            
            # Create and save report
            from .shared_functions import create_processing_report, save_processing_report, log_processing_summary
            report = create_processing_report(
                step_name=step_name,
                target_dir=target_dir,
                output_dir=output_dir,
                processed_files=processed_files,
                errors=errors,
                warnings=warnings
            )
            save_processing_report(report, output_dir)
            
            # Log summary
            log_processing_summary(
                logger, step_name, len(gnn_files), 
                len(processed_files), len(errors), len(warnings)
            )
            
            return len(errors) == 0
            
        except Exception as e:
            log_step_error(logger, f"{step_name} processing failed: {e}")
            return False
    
    return standard_function

def create_standardized_pipeline_script(
    step_name: str,
    module_function: Callable,
    fallback_parser_description: str,
    additional_arguments: Optional[Dict[str, Any]] = None,
    step_specific_imports: Optional[List[str]] = None
) -> Callable:
    """
    Create a standardized pipeline script with consistent argument parsing and error handling.
    
    Args:
        step_name: Name of the step (e.g., "1_gnn", "5_export")
        module_function: The main function to call for processing
        fallback_parser_description: Description for fallback argument parser
        additional_arguments: Additional arguments to add to the parser
        step_specific_imports: Additional imports needed for the step
        
    Returns:
        Function that can be called to run the standardized script
    """
    def run_standardized_script():
        """Standardized script execution function."""
        try:
            # Import step-specific modules if provided
            if step_specific_imports:
                for import_name in step_specific_imports:
                    try:
                        __import__(import_name)
                    except ImportError as e:
                        logging.warning(f"Failed to import {import_name}: {e}")
            
            # Parse arguments - try enhanced first, fall back gracefully
            try:
                # Import enhanced parser locally to avoid import issues
                from utils import ArgumentParser
                parsed_args = ArgumentParser.parse_step_arguments(step_name)
            except Exception as e:
                # Create fallback parser with step-specific arguments
                fallback_additional_args = additional_arguments or {}
                
                # Add step-specific arguments from pipeline template configuration
                if step_name == "10_ontology.py" and "ontology_terms_file" not in fallback_additional_args:
                    fallback_additional_args["ontology_terms_file"] = {
                        "type": Path, 
                        "help": "Path to ontology terms JSON file",
                        "flag": "--ontology-terms-file"
                    }
                
                logging.warning(f"Enhanced parser failed for {step_name}, using fallback: {e}")
                parsed_args = _create_fallback_parser(fallback_parser_description, fallback_additional_args).parse_args()
            
            # Set up logging
            logger = setup_step_logging(step_name, getattr(parsed_args, 'verbose', False))
            
            # Convert paths with proper handling for None values
            target_dir_raw = getattr(parsed_args, 'target_dir', None)
            if target_dir_raw is None:
                # Set default based on step type
                if step_name == "11_render.py":
                    # Step 9 should process exports from step 5 by default
                    target_dir = Path('output/gnn_exports/gnn_exports')
                elif step_name == "12_execute.py":
                    # Step 10 should process rendered simulators from step 9 by default
                    target_dir = Path('output/gnn_rendered_simulators')
                else:
                    # Default for other steps
                    target_dir = Path('input/gnn_files')
            else:
                target_dir = Path(target_dir_raw)
            
            output_dir_raw = getattr(parsed_args, 'output_dir', 'output')
            output_dir = Path(output_dir_raw) if output_dir_raw is not None else Path('output')
            
            # Set recursive flag default based on step type
            if step_name in ["11_render.py", "12_execute.py"]:
                # Steps 9 and 10 need recursive processing by default
                recursive_default = True
            else:
                recursive_default = False
            
            # Normalize the output directory to the standardized numbered step folder
            try:
                normalized_step = step_name if not step_name.endswith('.py') else step_name[:-3]
                step_output_dir = get_output_dir_for_script(normalized_step, output_dir)
            except Exception:
                step_output_dir = output_dir

            # Call the module function with the standardized step output dir
            success = module_function(
                target_dir=target_dir,
                output_dir=step_output_dir,
                logger=logger,
                recursive=getattr(parsed_args, 'recursive', recursive_default),
                verbose=getattr(parsed_args, 'verbose', False),
                **{k: v for k, v in vars(parsed_args).items() 
                   if k not in ['target_dir', 'output_dir', 'recursive', 'verbose']}
            )
            
            return 0 if success else 1
            
        except Exception as e:
            logging.error(f"Script execution failed: {e}")
            return 1
    
    return run_standardized_script

def get_standard_function_name(step_name: str) -> str:
    """Get the standard function name for a step."""
    return f"process_{step_name.replace('_', '_')}_files"

def validate_module_function_signature(func: Callable) -> bool:
    """Validate that a module function has the correct signature."""
    import inspect
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    
    # Check for required parameters
    required_params = ['target_dir', 'output_dir', 'logger']
    for param in required_params:
        if param not in params:
            return False
    
    return True

# Standard parameter types for module functions
ModuleFunctionParams = {
    "target_dir": Path,
    "output_dir": Path, 
    "logger": logging.Logger,
    "recursive": bool,
    "verbose": bool
}

# Common additional parameters
CommonModuleParams = {
    "strict": bool,  # For type checking
    "estimate_resources": bool,  # For type checking
    "duration": float,  # For SAPF
    "llm_tasks": str,  # For LLM
    "llm_timeout": int,  # For LLM
    "ontology_terms_file": Path,  # For ontology
    "recreate_venv": bool,  # For setup
    "dev": bool,  # For setup
}

# Standardized naming conventions for module functions
STANDARD_MODULE_FUNCTION_NAMES = {
    "1_gnn": "process_gnn_files",
    "2_setup": "perform_setup", 
    "3_tests": "run_tests",
    "4_type_checker": "process_type_checking",
    "5_export": "process_export",
    "6_visualization": "process_visualization", 
    "7_mcp": "process_mcp_operations",
    "8_ontology": "process_ontology_operations",
    "9_render": "process_rendering",
    "10_execute": "process_execution",
    "11_llm": "process_llm_analysis",
    "12_website": "process_website_generation",
    "13_website": "process_website_generation",
"14_report": "process_report_generation"
}

# Standard additional arguments for each step
STEP_ADDITIONAL_ARGUMENTS = {
    "4_type_checker": {
        "strict": {"type": bool, "help": "Enable strict validation mode"},
        "estimate_resources": {"type": bool, "help": "Estimate computational resources"}
    },
    "11_llm": {
        "llm_tasks": {"type": str, "default": "all", "help": "Comma-separated list of LLM tasks"},
        "llm_timeout": {"type": int, "default": 360, "help": "Timeout for LLM operations in seconds"}
    },
    "13_website": {
        "website_html_filename": {"type": str, "default": "gnn_pipeline_summary_website.html", "help": "Filename for generated HTML website"}
    },
    "14_report": {
        # No additional arguments for report generation
    },
    "8_ontology": {
        "ontology_terms_file": {"type": Path, "help": "Path to ontology terms JSON file", "flag": "--ontology-terms-file"}
    },
    "2_setup": {
        "recreate_venv": {"type": bool, "help": "Recreate virtual environment"},
        "dev": {"type": bool, "help": "Install development dependencies"}
    },
    "21_mcp.py": {
        "performance-mode": {
            "type": str,
            "default": "low",
            "choices": ["low", "high"],
            "help": "Performance mode for MCP",
            "flag": "--performance-mode"
        }
    }
} 