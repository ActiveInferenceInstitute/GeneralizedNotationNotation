"""
Logging utilities for the GNN Processing Pipeline.

This module provides consistent logging configuration and utilities
for all pipeline steps and modules.
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
import json

class GNNLogger:
    """Centralized logger for GNN pipeline with consistent error handling."""
    
    def __init__(self, name: str, log_file: Optional[Path] = None, level: int = logging.INFO):
        """
        Initialize a GNN logger with consistent formatting.
        
        Args:
            name: Logger name (usually module name)
            log_file: Optional log file path
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(log_file)
    
    def _setup_handlers(self, log_file: Optional[Path]):
        """Setup console and file handlers with consistent formatting."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler - only show WARNING and above
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler - show all levels
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message with optional exception info."""
        self.logger.error(message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs):
        """Log critical message with exception info."""  
        self.logger.critical(message, exc_info=exc_info, **kwargs)
    
    def log_exception(self, message: str, exception: Exception):
        """Log an exception with full traceback."""
        self.logger.error(f"{message}: {str(exception)}")
        self.logger.debug("Full traceback:", exc_info=True)
    
    def log_step_start(self, step_name: str, params: Optional[Dict[str, Any]] = None):
        """Log the start of a pipeline step."""
        msg = f"Starting step: {step_name}"
        if params:
            msg += f" with params: {params}"
        self.info(msg)
    
    def log_step_success(self, step_name: str, result_summary: Optional[str] = None):
        """Log successful completion of a pipeline step."""
        msg = f"Successfully completed step: {step_name}"
        if result_summary:
            msg += f" - {result_summary}"
        self.info(msg)
    
    def log_step_failure(self, step_name: str, error: Exception, critical: bool = False):
        """Log failure of a pipeline step."""
        msg = f"Failed step: {step_name} - {str(error)}"
        if critical:
            self.critical(msg, exc_info=True)
        else:
            self.error(msg, exc_info=True)


class PipelineErrorHandler:
    """Centralized error handling for pipeline steps."""
    
    def __init__(self, logger: GNNLogger):
        self.logger = logger
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
    
    def handle_error(self, step_name: str, error: Exception, critical: bool = False, 
                    context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Handle an error in a pipeline step.
        
        Args:
            step_name: Name of the step where error occurred
            error: The exception that occurred
            critical: Whether this is a critical error that should stop pipeline
            context: Additional context information
            
        Returns:
            bool: True if pipeline should continue, False if it should stop
        """
        error_info = {
            'step_name': step_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'critical': critical,
            'context': context or {}
        }
        
        if critical:
            self.errors.append(error_info)
            self.logger.log_step_failure(step_name, error, critical=True)
            return False
        else:
            self.warnings.append(error_info)
            self.logger.log_step_failure(step_name, error, critical=False)
            return True
    
    def handle_validation_error(self, step_name: str, validation_errors: List[str], 
                              critical: bool = True) -> bool:
        """Handle validation errors from a pipeline step."""
        error_msg = f"Validation failed: {'; '.join(validation_errors)}"
        validation_error = ValueError(error_msg)
        return self.handle_error(step_name, validation_error, critical=critical)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors and warnings."""
        return {
            'critical_errors': len(self.errors),
            'warnings': len(self.warnings),
            'errors': self.errors,
            'warning_details': self.warnings
        }
    
    def save_error_report(self, output_path: Path):
        """Save error report to a JSON file."""
        error_report = self.get_error_summary()
        error_report['generated_at'] = datetime.now().isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(error_report, f, indent=2)


def setup_pipeline_logging(log_dir: Path, log_level: int = logging.INFO) -> GNNLogger:
    """
    Setup logging for the entire pipeline.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        
    Returns:
        GNNLogger: Configured logger instance
    """
    log_file = log_dir / "pipeline.log"
    return GNNLogger("pipeline", log_file, log_level)


def get_module_logger(module_name: str, log_dir: Optional[Path] = None) -> GNNLogger:
    """
    Get a logger for a specific module.
    
    Args:
        module_name: Name of the module
        log_dir: Optional directory for log files
        
    Returns:
        GNNLogger: Configured logger instance
    """
    log_file = None
    if log_dir:
        log_file = log_dir / f"{module_name}.log"
    
    return GNNLogger(module_name, log_file)


def setup_standalone_logging(
    level: int = logging.INFO,
    logger_name: str = "GNN_Pipeline",
    output_dir: Optional[Path] = None,
    log_filename: str = "pipeline.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Set up standalone logging for pipeline steps.
    
    Args:
        level: Base logging level
        logger_name: Name of the logger
        output_dir: Directory for log files
        log_filename: Name of the log file
        console_level: Console logging level
        file_level: File logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(min(console_level, file_level))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if output directory is provided
    if output_dir:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(output_dir / log_filename, mode='a')
            file_handler.setLevel(file_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to set up file logging: {e}")
    
    return logger


# Legacy function for backward compatibility
def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """Legacy logging setup function - use GNNLogger instead."""
    if log_file:
        log_path = Path(log_file)
        return GNNLogger("legacy", log_path, level)
    else:
        return GNNLogger("legacy", None, level)

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