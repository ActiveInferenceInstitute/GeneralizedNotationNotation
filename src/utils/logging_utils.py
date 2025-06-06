"""
Centralized logging utilities for the GNN project.

This module provides consistent logging configuration across all
pipeline scripts, ensuring logs are properly formatted, colored, and 
directed to both console and files as appropriate.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, Union, List

def setup_standalone_logging(
    level: int = logging.INFO,
    logger_name: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    log_filename: Optional[str] = None,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None
) -> logging.Logger:
    """
    Sets up logging with a consistent format for both console and file output.
    
    Args:
        level: Base logging level (applies to logger, not handlers)
        logger_name: Name of the logger to configure (None for root logger)
        output_dir: Directory to save log files (None for no file logging)
        log_filename: Name of the log file (default: based on logger name)
        console_level: Specific level for console output (defaults to level)
        file_level: Specific level for file output (defaults to DEBUG)
        
    Returns:
        The configured logger
    """
    # Get the logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates if called multiple times
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # Set default handler levels
    if console_level is None:
        console_level = level
    if file_level is None:
        file_level = logging.DEBUG  # Default to detailed logs in file
    
    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    console_handler.flush = lambda: sys.stdout.flush()  # Force flush to ensure immediate visibility
    logger.addHandler(console_handler)
    
    # File handler (if output_dir is provided)
    if output_dir:
        output_path = Path(output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename based on logger name if not provided
            if log_filename is None:
                safe_name = logger_name.replace(".", "_") if logger_name else "root"
                log_filename = f"{safe_name}.log"
                
            log_file_path = output_path / log_filename
            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_handler.setLevel(file_level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            logger.debug(f"File logging configured to: {log_file_path}")
        except Exception as e:
            console_handler.setLevel(min(console_level, logging.WARNING))
            logger.warning(f"Failed to set up file logging to {output_dir}: {e}. Continuing with console logging only.")
    
    # Turn off propagation if this is not the root logger to avoid duplicate logs
    if logger_name:
        logger.propagate = False
        
    return logger

def silence_noisy_modules_in_console(modules: List[str] = None):
    """
    Increases the log level for known noisy modules, but only for console output.
    Keeps detailed logs in the log file.
    
    Args:
        modules: List of module names to silence. If None, uses a default list.
    """
    if modules is None:
        modules = [
            'PIL',
            'matplotlib',
            'discopy',
            'jax',
            'tensorflow',
            'urllib3',
            'asyncio',
        ]
    
    for module_name in modules:
        module_logger = logging.getLogger(module_name)
        
        # Only adjust console handlers, leave file handlers at their current level
        for handler in module_logger.handlers + logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
                handler.setLevel(logging.WARNING)

def set_verbose_mode(enable: bool = True, logger_names: List[str] = None):
    """
    Sets verbose mode (DEBUG level) for specified loggers.
    
    Args:
        enable: Whether to enable (True) or disable (False) verbose mode
        logger_names: List of logger names to modify. If None, modifies root logger.
    """
    level = logging.DEBUG if enable else logging.INFO
    
    if not logger_names:
        logger_names = ['']  # Root logger
        
    for name in logger_names:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Also update existing handlers
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
                handler.setLevel(level)

def log_section_header(logger: logging.Logger, title: str, char: str = '=', length: int = 80):
    """
    Logs a section header with the given title.
    
    Args:
        logger: The logger to use
        title: The title of the section
        char: The character to use for the border
        length: The total length of the header
    """
    border = char * length
    centered_title = f" {title} ".center(length, char)
    
    logger.info("")
    logger.info(border)
    logger.info(centered_title)
    logger.info(border)
    logger.info("")
    sys.stdout.flush()

# Add a custom log level for pipeline steps
STEP = 25  # Between INFO and WARNING
logging.addLevelName(STEP, "STEP")

# Add a method to Logger instances
def step(self, message, *args, **kwargs):
    """
    Logs a message with level STEP.
    This is used for marking the start/end of pipeline steps.
    """
    if self.isEnabledFor(STEP):
        self._log(STEP, message, args, **kwargs)
        sys.stdout.flush()

# Add the step method to the Logger class
logging.Logger.step = step 