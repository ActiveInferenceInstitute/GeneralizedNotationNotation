#!/usr/bin/env python3
"""
Utils Pipeline module for GNN Processing Pipeline.

This module provides pipeline utility functions.
"""

import logging
from pathlib import Path
from typing import Optional, Any, Tuple, Dict, List

logger = logging.getLogger(__name__)

def setup_step_logging(step_name: str, verbose: bool = False):
    """
    Setup logging for a pipeline step.
    
    Args:
        step_name: Name of the pipeline step
        verbose: Enable verbose logging
    """
    try:
        from .logging_utils import setup_step_logging as _setup_step_logging
        return _setup_step_logging(step_name, verbose)
    except ImportError:
        # Fallback logging setup
        logger = logging.getLogger(step_name)
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        return logger

class MockArgumentParser:
    """
    Mock argument parser for fallback scenarios.
    """
    
    @staticmethod
    def parse_step_arguments(step_name): 
        """
        Parse step arguments (mock implementation).
        
        Args:
            step_name: Name of the step
            
        Returns:
            Mock arguments object
        """
        class MockArgs:
            def __init__(self):
                self.verbose = False
                self.output_dir = Path("output")
                self.step_name = step_name
        
        return MockArgs()

def get_pipeline_utilities(step_name: str, verbose: bool = False) -> Tuple[Any, ...]:
    """
    Get pipeline utilities for a step.
    
    Args:
        step_name: Name of the pipeline step
        verbose: Enable verbose output
        
    Returns:
        Tuple of pipeline utilities
    """
    try:
        # Try to import actual utilities
        from .logging_utils import setup_step_logging
        from .argument_utils import EnhancedArgumentParser
        
        logger = setup_step_logging(step_name, verbose)
        parser = EnhancedArgumentParser()
        
        return logger, parser
        
    except ImportError:
        # Fallback to mock utilities
        logger = setup_step_logging(step_name, verbose)
        parser = MockArgumentParser()
        
        return logger, parser

def validate_output_directory(output_dir: Path, step_name: str) -> bool:
    """
    Validate output directory for a pipeline step.
    
    Args:
        output_dir: Output directory path
        step_name: Name of the pipeline step
        
    Returns:
        True if directory is valid, False otherwise
    """
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if directory is writable
        test_file = output_dir / f"{step_name}_test.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            logger.error(f"Output directory {output_dir} is not writable: {e}")
            return False
        
        logger.info(f"Output directory validated: {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate output directory {output_dir}: {e}")
        return False

def execute_pipeline_step_template(
    step_name: str,
    step_description: str,
    main_function,
    import_dependencies: Optional[List[str]] = None
):
    """
    Execute a pipeline step using the standard template.
    
    Args:
        step_name: Name of the pipeline step
        step_description: Description of the step
        main_function: Main function to execute
        import_dependencies: Optional list of dependencies to import
    """
    try:
        # Setup logging
        logger = setup_step_logging(step_name, verbose=True)
        
        # Log step start
        from .logging_utils import log_step_start
        log_step_start(logger, step_description)
        
        # Import dependencies if specified
        if import_dependencies:
            for dep in import_dependencies:
                try:
                    __import__(dep)
                    logger.debug(f"Imported dependency: {dep}")
                except ImportError as e:
                    logger.warning(f"Failed to import dependency {dep}: {e}")
        
        # Execute main function
        result = main_function()
        
        # Log step completion
        from .logging_utils import log_step_success
        log_step_success(logger, f"{step_description} completed successfully")
        
        return result
        
    except Exception as e:
        # Log step error
        from .logging_utils import log_step_error
        log_step_error(logger, f"{step_description} failed: {e}")
        raise
