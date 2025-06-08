"""
Streamlined Logging Utilities for GNN Processing Pipeline.

Provides coherent, correlation-based logging across all pipeline steps
with consistent formatting and centralized configuration.
"""

import logging
import sys
import uuid
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

# Thread-local storage for correlation context
_correlation_context = threading.local()

class CorrelationFormatter(logging.Formatter):
    """Formatter that includes correlation IDs for tracing across pipeline steps."""
    
    def format(self, record):
        # Add correlation ID to log record
        correlation_id = getattr(_correlation_context, 'correlation_id', 'MAIN')
        step_name = getattr(_correlation_context, 'step_name', 'pipeline')
        
        # Create enhanced record with correlation info
        record.correlation_id = correlation_id
        record.step_name = step_name
        
        return super().format(record)

class PipelineLogger:
    """Centralized logger for the GNN pipeline with correlation support."""
    
    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False
    _log_file_handler: Optional[logging.FileHandler] = None
    
    @classmethod
    def initialize(cls, log_dir: Optional[Path] = None, console_level: int = logging.INFO, 
                  file_level: int = logging.DEBUG) -> None:
        """Initialize the centralized logging system."""
        if cls._initialized:
            return
            
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        # Create correlation-aware formatter
        console_formatter = CorrelationFormatter(
            '%(asctime)s [%(correlation_id)s:%(step_name)s] %(name)s - %(levelname)s - %(message)s'
        )
        file_formatter = CorrelationFormatter(
            '%(asctime)s [%(correlation_id)s:%(step_name)s] %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (if log directory provided)
        if log_dir:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "pipeline.log"
                cls._log_file_handler = logging.FileHandler(log_file, mode='w')
                cls._log_file_handler.setLevel(file_level)
                cls._log_file_handler.setFormatter(file_formatter)
                root_logger.addHandler(cls._log_file_handler)
            except Exception as e:
                console_handler.emit(logging.LogRecord(
                    name="PipelineLogger", level=logging.ERROR, pathname="", lineno=0,
                    msg=f"Failed to setup file logging: {e}", args=(), exc_info=None
                ))
        
        # Silence noisy third-party libraries
        for noisy_lib in ['PIL', 'matplotlib', 'urllib3', 'requests']:
            logging.getLogger(noisy_lib).setLevel(logging.WARNING)
            
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger with the given name, ensuring it's properly configured."""
        if not cls._initialized:
            cls.initialize()
            
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            # Ensure it inherits from root configuration
            logger.propagate = True
            cls._loggers[name] = logger
            
        return cls._loggers[name]
    
    @classmethod
    def set_correlation_context(cls, step_name: str, correlation_id: Optional[str] = None) -> str:
        """Set correlation context for current thread."""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())[:8]
            
        _correlation_context.correlation_id = correlation_id
        _correlation_context.step_name = step_name
        
        return correlation_id
    
    @classmethod
    def clear_correlation_context(cls):
        """Clear correlation context for current thread."""
        if hasattr(_correlation_context, 'correlation_id'):
            delattr(_correlation_context, 'correlation_id')
        if hasattr(_correlation_context, 'step_name'):
            delattr(_correlation_context, 'step_name')
    
    @classmethod
    def set_verbosity(cls, verbose: bool):
        """Update console log level based on verbosity."""
        level = logging.DEBUG if verbose else logging.INFO
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(level)
                break

def setup_step_logging(step_name: str, verbose: bool = False) -> logging.Logger:
    """
    Setup logging for a pipeline step with correlation support.
    
    Args:
        step_name: Name of the pipeline step (e.g., "1_gnn", "setup")
        verbose: Whether to enable verbose logging
        
    Returns:
        Configured logger for the step
    """
    # Ensure pipeline logging is initialized
    PipelineLogger.initialize()
    
    # Set correlation context for this step
    correlation_id = PipelineLogger.set_correlation_context(step_name)
    
    # Get step-specific logger
    logger = PipelineLogger.get_logger(step_name)
    
    # Update verbosity if needed
    if verbose:
        PipelineLogger.set_verbosity(True)
    
    logger.info(f"Initialized logging for step: {step_name} [correlation_id: {correlation_id}]")
    
    return logger

def setup_main_logging(log_dir: Optional[Path] = None, verbose: bool = False) -> logging.Logger:
    """
    Setup logging for the main pipeline orchestrator.
    
    Args:
        log_dir: Directory for log files
        verbose: Whether to enable verbose logging
        
    Returns:
        Configured main logger
    """
    console_level = logging.DEBUG if verbose else logging.INFO
    PipelineLogger.initialize(log_dir, console_level)
    
    # Set main pipeline context
    correlation_id = PipelineLogger.set_correlation_context("main")
    
    logger = PipelineLogger.get_logger("GNN_Pipeline")
    logger.info(f"GNN Pipeline logging initialized [correlation_id: {correlation_id}]")
    
    return logger

# Legacy function for backward compatibility
def setup_standalone_logging(
    level: int = logging.INFO,
    logger_name: str = "GNN_Pipeline", 
    output_dir: Optional[Path] = None,
    log_filename: str = "pipeline.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Legacy function for backward compatibility.
    Now routes to the new streamlined system.
    """
    # Extract step name from logger name
    step_name = logger_name.split('.')[-1] if '.' in logger_name else logger_name
    step_name = step_name.replace('src.', '').replace('_', '-')
    
    # Initialize if needed  
    if not PipelineLogger._initialized:
        PipelineLogger.initialize(output_dir, console_level, file_level)
    
    return setup_step_logging(step_name, verbose=(console_level <= logging.DEBUG))

# Convenience functions
def silence_noisy_modules_in_console():
    """Silence noisy third-party modules (handled automatically now)."""
    pass  # This is now handled in PipelineLogger.initialize()

def set_verbose_mode(verbose: bool):
    """Set verbose mode for the entire pipeline."""
    PipelineLogger.set_verbosity(verbose)

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