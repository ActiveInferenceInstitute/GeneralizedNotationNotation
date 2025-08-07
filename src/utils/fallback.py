#!/usr/bin/env python3
"""
Utils Fallback module for GNN Processing Pipeline.

This module provides fallback implementations when core modules are not available.
"""

import logging
from pathlib import Path

class MockArgumentParser:
    """
    Fallback argument parser for when argument_utils is not available.
    """
    @staticmethod
    def parse_step_arguments(step_name): 
        """
        Parse step arguments with fallback implementation.
        
        Args:
            step_name: Name of the step
            
        Returns:
            MockArgs object with default values
        """
        class MockArgs:
            def __init__(self):
                self.verbose = False
                self.output_dir = Path("output")
                self.step_name = step_name
        return MockArgs()

def setup_step_logging(step_name: str, verbose: bool = False):
    """
    Fallback step logging setup.
    
    Args:
        step_name: Name of the step
        verbose: Enable verbose logging
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(step_name)
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger
