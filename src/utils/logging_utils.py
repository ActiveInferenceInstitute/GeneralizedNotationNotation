"""
Logging utilities for the GNN Pipeline.

This module provides consistent logging configuration across the GNN pipeline,
including functionality to separate console and file logging with different
verbosity levels.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List, Union, Set

# Module-level constants for common log levels
SILENT = 100  # Custom level above CRITICAL to effectively silence loggers
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# Dictionary of modules that are too verbose even at INFO level and should be silenced in console
# (but still logged to file at appropriate levels)
NOISY_MODULES = {
    # Module loggers to silence completely from console but still log to file
    'matplotlib': logging.WARNING,
    'PIL': logging.WARNING,
    'discopy_translator_module.translator': logging.INFO,
    'execute.pymdp_runner': logging.INFO,
    'execute.rxinfer_runner': logging.INFO,
    'mcp': logging.INFO,
    'src.mcp': logging.INFO,
    
    # Modules to show only warnings and errors (e.g., DEBUG in files, WARNING in console)
    'visualization': {'console': logging.WARNING, 'file': logging.DEBUG},
}

# Track the console silenced loggers to ensure we don't configure them multiple times
_console_silenced_loggers: Set[str] = set()

def setup_standalone_logging(
    level: int = logging.INFO,
    logger_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
    log_filename: Optional[str] = None,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None
) -> logging.Logger:
    """
    Set up logging for standalone script execution with optional file logging.
    
    Args:
        level: The default logging level for both console and file (if not overridden)
        logger_name: The name of the logger to configure (None for root logger)
        output_dir: Directory to save log files (None for console-only logging)
        log_filename: Name of the log file (default: based on logger_name)
        console_level: Override console logging level (default: level parameter)
        file_level: Override file logging level (default: level parameter)
        
    Returns:
        The configured logger
    """
    # Get the target logger
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    
    # Set the logger's overall level to the most verbose of console/file
    effective_console_level = console_level if console_level is not None else level
    effective_file_level = file_level if file_level is not None else level
    logger.setLevel(min(effective_console_level, effective_file_level))
    
    # Clear existing handlers if requested to reconfigure
    # (typically only needed in special scenarios)
    # for handler in logger.handlers[:]:
    #    logger.removeHandler(handler)
    
    # Only create handlers if there aren't any yet
    if not logger.handlers and not logging.getLogger().handlers:
        # Create console handler
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(effective_console_level)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Create file handler if output_dir is provided
        if output_dir is not None:
            # Ensure the output directory exists
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine the log filename
            if log_filename is None:
                log_name = logger_name or "root"
                log_filename = f"{log_name.replace('.', '_')}.log"
            
            # Create the file handler
            log_file_path = output_dir / log_filename
            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_handler.setLevel(effective_file_level)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"File logging configured to: {log_file_path}")
    
    # Set up silenced modules for console output
    silence_noisy_modules_in_console()
    
    return logger

def silence_noisy_modules_in_console() -> None:
    """
    Configure noisy modules to be silenced in console output but still logged to file.
    This function sets up a filter on console handlers to avoid showing low-level
    messages from certain modules.
    """
    global _console_silenced_loggers
    
    # Define a filter class that allows controlling log level by handler
    class ModuleLevelFilter(logging.Filter):
        def __init__(self, module_name: str, min_level: int):
            super().__init__()
            self.module_name = module_name
            self.min_level = min_level
            
        def filter(self, record: logging.LogRecord) -> bool:
            # Allow records from this module if they're at or above min_level
            return not record.name.startswith(self.module_name) or record.levelno >= self.min_level
    
    # Apply silencing to all console handlers
    for module_name, level_config in NOISY_MODULES.items():
        # Skip if already silenced
        if module_name in _console_silenced_loggers:
            continue
            
        if isinstance(level_config, dict):
            console_level = level_config.get('console', logging.WARNING)
        else:
            console_level = level_config
            
        # Get all console handlers from root and other loggers
        all_loggers = [logging.getLogger()]  # Start with root logger
        
        # Add other loggers that might have console handlers
        for name in logging.root.manager.loggerDict:
            all_loggers.append(logging.getLogger(name))
            
        # Apply filter to console handlers
        for logger in all_loggers:
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
                    handler.addFilter(ModuleLevelFilter(module_name, console_level))
        
        _console_silenced_loggers.add(module_name)

def get_dual_logger(
    name: str, 
    console_level: int = logging.INFO, 
    file_level: int = logging.DEBUG,
    output_dir: Optional[Path] = None,
    log_filename: Optional[str] = None
) -> logging.Logger:
    """
    Get a logger that logs at different levels to console and file.
    
    Args:
        name: The name of the logger
        console_level: The logging level for console output
        file_level: The logging level for file output
        output_dir: Directory to save log files (None for console-only)
        log_filename: Name of the log file (default: based on name)
        
    Returns:
        The configured logger
    """
    return setup_standalone_logging(
        level=min(console_level, file_level),
        logger_name=name,
        output_dir=output_dir,
        log_filename=log_filename,
        console_level=console_level,
        file_level=file_level
    )

def set_verbose_mode(verbose: bool = False) -> None:
    """
    Set verbose mode for the entire pipeline. This affects console logging level
    but keeps detailed logs in files.
    
    Args:
        verbose: Whether to show verbose output on console
    """
    root_logger = logging.getLogger()
    pipeline_logger = logging.getLogger("GNN_Pipeline")
    
    # Update console handlers
    for logger in [root_logger, pipeline_logger]:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
                handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Always silence noisy modules regardless of verbose mode
    silence_noisy_modules_in_console() 