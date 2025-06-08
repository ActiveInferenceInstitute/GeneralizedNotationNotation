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
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import json
import time
from contextlib import contextmanager
import os

# Thread-local storage for correlation context
_correlation_context = threading.local()

# Add performance tracking capabilities
class PerformanceTracker:
    """Track performance metrics across pipeline steps."""
    
    def __init__(self):
        self._metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
    
    def record_timing(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Record timing information for an operation."""
        with self._lock:
            if operation not in self._metrics:
                self._metrics[operation] = []
            
            self._metrics[operation].append({
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all tracked operations."""
        with self._lock:
            summary = {}
            for operation, measurements in self._metrics.items():
                durations = [m['duration'] for m in measurements]
                summary[operation] = {
                    'count': len(durations),
                    'total_duration': sum(durations),
                    'avg_duration': sum(durations) / len(durations) if durations else 0,
                    'min_duration': min(durations) if durations else 0,
                    'max_duration': max(durations) if durations else 0
                }
            return summary

# Global performance tracker instance
performance_tracker = PerformanceTracker()

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
    Setup streamlined logging for a pipeline step.
    
    This is the main entry point for pipeline step logging.
    All modules should use this function.
    
    Args:
        step_name: Name of the pipeline step
        verbose: Whether to enable verbose logging
        
    Returns:
        Configured logger for the step
    """
    # Initialize enhanced logging if not already done
    if not EnhancedPipelineLogger._initialized:
        EnhancedPipelineLogger.initialize(enable_structured=True)
    
    # Set correlation context
    correlation_id = EnhancedPipelineLogger.set_correlation_context(step_name)
    
    # Get step-specific logger
    logger = EnhancedPipelineLogger.get_logger(step_name)
    
    # Update verbosity
    if verbose:
        EnhancedPipelineLogger.set_verbosity(True)
    
    # Log initialization
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

# Enhanced formatter with structured data support
class StructuredFormatter(CorrelationFormatter):
    """Formatter that supports structured logging with JSON metadata."""
    
    def format(self, record):
        # Extract structured data if present
        structured_data = getattr(record, 'structured_data', None)
        
        # Format basic record
        formatted = super().format(record)
        
        # Append structured data as JSON if present
        if structured_data:
            try:
                json_data = json.dumps(structured_data, default=str)
                formatted += f" | STRUCTURED_DATA: {json_data}"
            except (TypeError, ValueError):
                # Fallback if JSON serialization fails
                formatted += f" | STRUCTURED_DATA: {str(structured_data)}"
        
        return formatted

class EnhancedPipelineLogger(PipelineLogger):
    """Enhanced pipeline logger with structured logging and performance tracking."""
    
    @classmethod
    def initialize(cls, log_dir: Optional[Path] = None, console_level: int = logging.INFO, 
                  file_level: int = logging.DEBUG, enable_structured: bool = True) -> None:
        """Initialize enhanced logging system."""
        if cls._initialized:
            return
            
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Choose formatter based on structured logging preference
        if enable_structured:
            console_formatter = StructuredFormatter(
                '%(asctime)s [%(correlation_id)s:%(step_name)s] %(name)s - %(levelname)s - %(message)s'
            )
            file_formatter = StructuredFormatter(
                '%(asctime)s [%(correlation_id)s:%(step_name)s] %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        else:
            console_formatter = CorrelationFormatter(
                '%(asctime)s [%(correlation_id)s:%(step_name)s] %(name)s - %(levelname)s - %(message)s'
            )
            file_formatter = CorrelationFormatter(
                '%(asctime)s [%(correlation_id)s:%(step_name)s] %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        
        # Console handler with enhanced formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with rotation support
        if log_dir:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                
                # Main pipeline log
                log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                cls._log_file_handler = logging.FileHandler(log_file, mode='w')
                cls._log_file_handler.setLevel(file_level)
                cls._log_file_handler.setFormatter(file_formatter)
                root_logger.addHandler(cls._log_file_handler)
                
                # Performance metrics log
                perf_log_file = log_dir / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                perf_handler = logging.FileHandler(perf_log_file, mode='w')
                perf_handler.setLevel(logging.INFO)
                perf_handler.setFormatter(file_formatter)
                
                # Create performance logger
                perf_logger = logging.getLogger("PERFORMANCE")
                perf_logger.addHandler(perf_handler)
                perf_logger.setLevel(logging.INFO)
                
            except Exception as e:
                console_handler.emit(logging.LogRecord(
                    name="EnhancedPipelineLogger", level=logging.ERROR, pathname="", lineno=0,
                    msg=f"Failed to setup file logging: {e}", args=(), exc_info=None
                ))
        
        # Enhanced third-party library control
        noisy_libraries = [
            'PIL', 'matplotlib', 'urllib3', 'requests', 'httpx', 'openai',
            'jax', 'jaxlib', 'discopy', 'networkx', 'graphviz'
        ]
        for lib in noisy_libraries:
            logging.getLogger(lib).setLevel(logging.WARNING)
            
        cls._initialized = True
    
    @classmethod
    def log_structured(cls, logger: logging.Logger, level: int, message: str, **structured_data):
        """Log a message with structured data."""
        record = logging.LogRecord(
            name=logger.name, level=level, pathname="", lineno=0,
            msg=message, args=(), exc_info=None
        )
        record.structured_data = structured_data
        logger.handle(record)
    
    @classmethod
    @contextmanager
    def timed_operation(cls, operation_name: str, logger: Optional[logging.Logger] = None, 
                       metadata: Optional[Dict[str, Any]] = None):
        """Context manager for timing operations with automatic logging."""
        start_time = time.time()
        
        if logger:
            logger.info(f"🚀 Starting operation: {operation_name}")
        
        try:
            yield
            duration = time.time() - start_time
            
            # Record performance metrics
            performance_tracker.record_timing(operation_name, duration, metadata)
            
            if logger:
                cls.log_structured(
                    logger, logging.INFO,
                    f"✅ Completed operation: {operation_name}",
                    operation=operation_name,
                    duration_seconds=duration,
                    status="success",
                    metadata=metadata
                )
        except Exception as e:
            duration = time.time() - start_time
            
            # Record failed operation
            error_metadata = metadata.copy() if metadata else {}
            error_metadata.update({'error': str(e), 'error_type': type(e).__name__})
            performance_tracker.record_timing(f"{operation_name}_failed", duration, error_metadata)
            
            if logger:
                cls.log_structured(
                    logger, logging.ERROR,
                    f"❌ Failed operation: {operation_name}",
                    operation=operation_name,
                    duration_seconds=duration,
                    status="failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    metadata=metadata
                )
            raise

def setup_enhanced_step_logging(step_name: str, verbose: bool = False, 
                               enable_structured: bool = True) -> logging.Logger:
    """
    Setup enhanced logging for a pipeline step with structured logging support.
    
    Args:
        step_name: Name of the pipeline step
        verbose: Whether to enable verbose logging
        enable_structured: Whether to enable structured logging
        
    Returns:
        Configured logger for the step
    """
    # Initialize enhanced logging
    EnhancedPipelineLogger.initialize(enable_structured=enable_structured)
    
    # Set correlation context
    correlation_id = EnhancedPipelineLogger.set_correlation_context(step_name)
    
    # Get step-specific logger
    logger = EnhancedPipelineLogger.get_logger(step_name)
    
    # Update verbosity
    if verbose:
        EnhancedPipelineLogger.set_verbosity(True)
    
    # Log initialization with structured data
    EnhancedPipelineLogger.log_structured(
        logger, logging.INFO,
        f"Initialized enhanced logging for step: {step_name}",
        step_name=step_name,
        correlation_id=correlation_id,
        verbose=verbose,
        structured_logging=enable_structured
    )
    
    return logger

# Simplified convenience logging functions with flexible signatures
def log_step_start(logger_or_step_name, message: str = None, **metadata):
    """
    Log the start of a pipeline step with flexible signature support.
    
    Args:
        logger_or_step_name: Either a logger instance or step name string
        message: Log message (if logger_or_step_name is a logger) or None
        **metadata: Additional structured metadata
    """
    if isinstance(logger_or_step_name, str):
        # Called with step name only - get logger and use as message
        step_name = logger_or_step_name
        message = message or f"Starting {step_name}"
        logger = EnhancedPipelineLogger.get_logger(step_name)
    else:
        # Called with logger instance
        logger = logger_or_step_name
        message = message or "Starting step"
    
    EnhancedPipelineLogger.log_structured(
        logger, logging.INFO,
        f"🚀 {message}",
        event_type="step_start",
        **metadata
    )

def log_step_success(logger_or_step_name, message: str = None, **metadata):
    """
    Log successful completion of a pipeline step.
    
    Args:
        logger_or_step_name: Either a logger instance or step name string  
        message: Log message (if logger_or_step_name is a logger) or None
        **metadata: Additional structured metadata
    """
    if isinstance(logger_or_step_name, str):
        # Called with step name only
        step_name = logger_or_step_name
        message = message or f"{step_name} completed successfully"
        logger = EnhancedPipelineLogger.get_logger(step_name)
    else:
        # Called with logger instance
        logger = logger_or_step_name
        message = message or "Step completed successfully"
    
    EnhancedPipelineLogger.log_structured(
        logger, logging.INFO,
        f"✅ {message}",
        event_type="step_success", 
        **metadata
    )

def log_step_warning(logger_or_step_name, message: str = None, **metadata):
    """
    Log a warning during a pipeline step.
    
    Args:
        logger_or_step_name: Either a logger instance or step name string
        message: Log message (if logger_or_step_name is a logger) or None  
        **metadata: Additional structured metadata
    """
    if isinstance(logger_or_step_name, str):
        # Called with step name only
        step_name = logger_or_step_name
        message = message or f"Warning in {step_name}"
        logger = EnhancedPipelineLogger.get_logger(step_name)
    else:
        # Called with logger instance
        logger = logger_or_step_name
        message = message or "Step warning"
    
    EnhancedPipelineLogger.log_structured(
        logger, logging.WARNING,
        f"⚠️ {message}",
        event_type="step_warning",
        **metadata
    )

def log_step_error(logger_or_step_name, message: str = None, **metadata):
    """
    Log an error during a pipeline step.
    
    Args:
        logger_or_step_name: Either a logger instance or step name string
        message: Log message (if logger_or_step_name is a logger) or None
        **metadata: Additional structured metadata  
    """
    if isinstance(logger_or_step_name, str):
        # Called with step name only
        step_name = logger_or_step_name
        message = message or f"Error in {step_name}"
        logger = EnhancedPipelineLogger.get_logger(step_name)
    else:
        # Called with logger instance
        logger = logger_or_step_name
        message = message or "Step error"
    
    EnhancedPipelineLogger.log_structured(
        logger, logging.ERROR,
        f"❌ {message}",
        event_type="step_error",
        **metadata
    )

def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary for the current pipeline run."""
    return performance_tracker.get_summary() 