#!/usr/bin/env python3
"""
Streamlined Logging Utilities for GNN Processing Pipeline.

Provides coherent, correlation-based logging across all pipeline steps
with enhanced visual formatting and centralized configuration.
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
import shutil

# Thread-local storage for correlation context
_correlation_context = threading.local()

# Import performance tracking from dedicated module
from .performance_tracker import PerformanceTracker, performance_tracker

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
    # Initialize pipeline logger
    PipelineLogger.initialize()
    
    # Set correlation context
    correlation_id = PipelineLogger.set_correlation_context(step_name.replace('.py', ''))
    
    # Configure verbosity
    PipelineLogger.set_verbosity(verbose)
    
    # Get logger instance
    logger = PipelineLogger.get_logger(step_name)
    
    # Add step-specific attributes
    logger.step_name = step_name
    logger.correlation_id = correlation_id
    
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
    PipelineLogger.initialize(log_dir=log_dir)
    PipelineLogger.set_verbosity(verbose)
    
    correlation_id = PipelineLogger.set_correlation_context("main")
    logger = PipelineLogger.get_logger("GNN_Pipeline")
    
    logger.info(f"GNN Pipeline logging initialized [correlation_id: {correlation_id}]")
    
    return logger

# Compatibility functions for legacy code
def setup_standalone_logging(
    level: int = logging.INFO,
    logger_name: str = "GNN_Pipeline", 
    output_dir: Optional[Path] = None,
    log_filename: str = "pipeline.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """Legacy compatibility function."""
    log_dir = output_dir / "logs" if output_dir else None
    PipelineLogger.initialize(log_dir=log_dir, console_level=console_level, file_level=file_level)
    return PipelineLogger.get_logger(logger_name)

def silence_noisy_modules_in_console():
    """Reduce console output from verbose third-party modules."""
    for module in ['PIL', 'matplotlib', 'urllib3', 'requests']:
        logging.getLogger(module).setLevel(logging.WARNING)

def set_verbose_mode(verbose: bool):
    """Set verbose mode for console output."""
    PipelineLogger.set_verbosity(verbose)

def log_section_header(logger: logging.Logger, title: str, char: str = '=', length: int = 80):
    """Log a formatted section header."""
    border = char * length
    padded_title = f" {title} ".center(length, char)
    
    logger.info("")
    logger.info(border)
    logger.info(padded_title)
    logger.info(border)
    logger.info("")

# Enhanced logging functionality
logging.STEP = 25  # Custom log level between INFO and WARNING
logging.addLevelName(logging.STEP, "STEP")

def step(self, message, *args, **kwargs):
    """Log a step-level message."""
    if self.isEnabledFor(logging.STEP):
        self._log(logging.STEP, message, args, **kwargs)

logging.Logger.step = step

class StructuredFormatter(CorrelationFormatter):
    """Formatter that handles structured logging data."""
    
    def format(self, record):
        # Extract structured data if present
        if hasattr(record, 'structured_data'):
            structured_data = record.structured_data
            
            # Add structured data to the log message
            if isinstance(structured_data, dict) and structured_data:
                structured_str = " | ".join([f"{k}={v}" for k, v in structured_data.items() if k != 'event_type'])
                if structured_str:
                    record.msg = f"{record.msg} [{structured_str}]"
        
        # Continue with correlation formatting
        return super().format(record)

class EnhancedPipelineLogger(PipelineLogger):
    """Enhanced pipeline logger with structured logging support."""
    
    @classmethod
    def initialize(cls, log_dir: Optional[Path] = None, console_level: int = logging.INFO, 
                  file_level: int = logging.DEBUG, enable_structured: bool = True) -> None:
        """Initialize enhanced logging with structured data support."""
        if cls._initialized:
            return
            
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers.clear()
        
        # Choose formatter based on structured logging preference
        if enable_structured:
            console_formatter = EnhancedVisualFormatter(
                '%(asctime)s [%(correlation_id)s:%(step_name)s] %(name)s - %(levelname)s - %(message)s',
                include_performance=True
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
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
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
                    name="EnhancedPipelineLogger", level=logging.ERROR, pathname="", lineno=0,
                    msg=f"Failed to setup file logging: {e}", args=(), exc_info=None
                ))
        
        # Silence noisy libraries
        for noisy_lib in ['PIL', 'matplotlib', 'urllib3', 'requests', 'werkzeug']:
            logging.getLogger(noisy_lib).setLevel(logging.WARNING)
            
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
        """Context manager for timing operations with structured logging."""
        if logger is None:
            logger = cls.get_logger("performance")
        
        start_time = time.time()
        cls.log_structured(
            logger, logging.INFO, 
            f"🚀 Starting {operation_name}",
            event_type="operation_start",
            operation=operation_name,
            **(metadata or {})
        )
        
        try:
            with performance_tracker.track_operation(operation_name, metadata):
                yield
                
            duration = time.time() - start_time
            cls.log_structured(
                logger, logging.INFO,
                f"✅ Completed {operation_name}",
                event_type="operation_complete", 
                operation=operation_name,
                duration_seconds=round(duration, 3),
                **(metadata or {})
            )
            
        except Exception as e:
            duration = time.time() - start_time
            cls.log_structured(
                logger, logging.ERROR,
                f"❌ Failed {operation_name}: {e}",
                event_type="operation_error",
                operation=operation_name,
                duration_seconds=round(duration, 3),
                error=str(e),
                **(metadata or {})
            )
            raise

def setup_enhanced_step_logging(step_name: str, verbose: bool = False, 
                               enable_structured: bool = True) -> logging.Logger:
    """
    Setup enhanced logging for a pipeline step with structured data support.
    
    Args:
        step_name: Name of the pipeline step
        verbose: Whether to enable verbose logging
        enable_structured: Whether to enable structured logging
        
    Returns:
        Configured logger for the step
    """
    # Initialize enhanced pipeline logger
    EnhancedPipelineLogger.initialize(enable_structured=enable_structured)
    
    # Set correlation context
    correlation_id = EnhancedPipelineLogger.set_correlation_context(step_name.replace('.py', ''))
    
    # Configure verbosity
    EnhancedPipelineLogger.set_verbosity(verbose)
    
    # Get logger instance
    logger = EnhancedPipelineLogger.get_logger(step_name)
    
    # Add step-specific attributes
    logger.step_name = step_name
    logger.correlation_id = correlation_id
    
    return logger

class VisualLoggingEnhancer:
    """Enhanced visual formatting for pipeline logging with progress tracking."""
    
    # Color codes for terminal output
    COLORS = {
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'DIM': '\033[2m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m', 
        'RED': '\033[91m',
        'BLUE': '\033[94m',
        'MAGENTA': '\033[95m',
        'CYAN': '\033[96m',
        'WHITE': '\033[97m',
        'BG_GREEN': '\033[102m',
        'BG_YELLOW': '\033[103m',
        'BG_RED': '\033[101m'
    }
    
    # Progress indicators
    PROGRESS_CHARS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    
    @classmethod
    def supports_color(cls) -> bool:
        """Check if terminal supports color output."""
        if not sys.stdout.isatty():
            return False
        
        # Check environment variables
        if os.getenv('NO_COLOR'):
            return False
        if os.getenv('FORCE_COLOR'):
            return True
            
        # Check terminal type
        term = os.getenv('TERM', '').lower()
        if 'color' in term or term in ['xterm', 'xterm-256color', 'screen', 'tmux']:
            return True
            
        # Check if we have a known terminal with color support
        return shutil.which('tput') is not None
    
    @classmethod
    def colorize(cls, text: str, color: str, bold: bool = False) -> str:
        """Apply color formatting to text if terminal supports it."""
        if not cls.supports_color():
            return text
            
        color_code = cls.COLORS.get(color.upper(), '')
        bold_code = cls.COLORS['BOLD'] if bold else ''
        reset_code = cls.COLORS['RESET']
        
        return f"{color_code}{bold_code}{text}{reset_code}"
    
    @classmethod
    def format_step_header(cls, step_num: int, total_steps: int, step_name: str, 
                          status: str = "RUNNING") -> str:
        """Create a formatted step header with progress bar."""
        progress = step_num / total_steps
        bar_length = 30
        filled_length = int(bar_length * progress)
        
        # Create progress bar
        bar = '█' * filled_length + '▒' * (bar_length - filled_length)
        percentage = int(progress * 100)
        
        # Color code based on status
        if status == "SUCCESS":
            status_colored = cls.colorize(status, "GREEN", True)
            bar_colored = cls.colorize(bar, "GREEN")
        elif status == "FAILED":
            status_colored = cls.colorize(status, "RED", True)
            bar_colored = cls.colorize(bar, "RED")
        elif status == "WARNING":
            status_colored = cls.colorize(status, "YELLOW", True)
            bar_colored = cls.colorize(bar, "YELLOW")
        else:  # RUNNING
            status_colored = cls.colorize(status, "CYAN", True)
            bar_colored = cls.colorize(bar, "CYAN")
        
        step_info = cls.colorize(f"Step {step_num}/{total_steps}", "WHITE", True)
        step_name_colored = cls.colorize(step_name, "MAGENTA")
        percentage_colored = cls.colorize(f"({percentage}%)", "DIM")
        
        return f"┌─ {step_info}: {step_name_colored} {status_colored}\n└─ Progress: [{bar_colored}] {percentage_colored}"
    
    @classmethod
    def format_duration(cls, duration_seconds: float) -> str:
        """Format duration with appropriate units and color coding."""
        if duration_seconds < 1:
            duration_str = f"{duration_seconds*1000:.0f}ms"
            color = "GREEN"
        elif duration_seconds < 10:
            duration_str = f"{duration_seconds:.2f}s"
            color = "GREEN"
        elif duration_seconds < 60:
            duration_str = f"{duration_seconds:.1f}s"
            color = "YELLOW"
        elif duration_seconds < 300:
            minutes = int(duration_seconds // 60)
            seconds = duration_seconds % 60
            duration_str = f"{minutes}m{seconds:.0f}s"
            color = "YELLOW"
        else:
            minutes = int(duration_seconds // 60)
            hours = minutes // 60
            minutes = minutes % 60
            if hours > 0:
                duration_str = f"{hours}h{minutes}m"
            else:
                duration_str = f"{minutes}m"
            color = "RED"
        
        return cls.colorize(duration_str, color)
    
    @classmethod
    def format_memory_usage(cls, memory_mb: float) -> str:
        """Format memory usage with appropriate units and color coding."""
        if memory_mb < 100:
            memory_str = f"{memory_mb:.1f}MB"
            color = "GREEN"
        elif memory_mb < 1000:
            memory_str = f"{memory_mb:.0f}MB"
            color = "YELLOW"
        else:
            memory_gb = memory_mb / 1024
            memory_str = f"{memory_gb:.1f}GB"
            color = "RED"
        
        return cls.colorize(memory_str, color)

class PipelineProgressTracker:
    """Track pipeline progress across steps with visual indicators."""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_status: Dict[int, str] = {}
        self.step_durations: Dict[int, float] = {}
        self.start_time = time.time()
    
    def start_step(self, step_num: int, step_name: str) -> str:
        """Mark step as started and return formatted header."""
        self.current_step = step_num
        self.step_status[step_num] = "RUNNING"
        
        return VisualLoggingEnhancer.format_step_header(
            step_num, self.total_steps, step_name, "RUNNING"
        )
    
    def complete_step(self, step_num: int, status: str, duration: float = None) -> str:
        """Mark step as completed and return summary."""
        self.step_status[step_num] = status
        if duration:
            self.step_durations[step_num] = duration
        
        # Calculate completion stats
        completed = len([s for s in self.step_status.values() if s != "RUNNING"])
        success_count = len([s for s in self.step_status.values() if "SUCCESS" in s])
        
        duration_str = f" in {VisualLoggingEnhancer.format_duration(duration)}" if duration else ""
        
        if status == "SUCCESS":
            icon = "✅"
            color = "GREEN"
        elif "WARNING" in status:
            icon = "⚠️"
            color = "YELLOW"
        else:
            icon = "❌"
            color = "RED"
        
        completion_text = VisualLoggingEnhancer.colorize(
            f"{icon} Step {step_num} completed with {status}{duration_str}",
            color, True
        )
        
        progress_text = VisualLoggingEnhancer.colorize(
            f"Progress: {completed}/{self.total_steps} steps ({success_count} successful)",
            "CYAN"
        )
        
        return f"{completion_text}\n{progress_text}"
    
    def get_overall_progress(self) -> str:
        """Get overall pipeline progress summary."""
        completed = len([s for s in self.step_status.values() if s != "RUNNING"])
        success_count = len([s for s in self.step_status.values() if "SUCCESS" in s])
        warning_count = len([s for s in self.step_status.values() if "WARNING" in s])
        failed_count = len([s for s in self.step_status.values() if "FAILED" in s or "ERROR" in s])
        
        elapsed = time.time() - self.start_time
        
        progress_bar = VisualLoggingEnhancer.format_step_header(
            completed, self.total_steps, "Overall Progress", 
            "SUCCESS" if completed == self.total_steps and failed_count == 0 else "RUNNING"
        )
        
        stats = [
            f"✅ Success: {success_count}",
            f"⚠️ Warnings: {warning_count}", 
            f"❌ Failed: {failed_count}",
            f"⏱️ Elapsed: {VisualLoggingEnhancer.format_duration(elapsed)}"
        ]
        
        return f"{progress_bar}\nStats: {' | '.join(stats)}"

class EnhancedVisualFormatter(StructuredFormatter):
    """Enhanced formatter with visual improvements and performance context."""
    
    def __init__(self, format_string, include_performance=True, use_colors=True):
        super().__init__(format_string)
        self.include_performance = include_performance
        self.use_colors = use_colors and VisualLoggingEnhancer.supports_color()
    
    def format(self, record):
        # Get base formatted message
        formatted = super().format(record)
        
        # Add color coding for log levels and emojis
        if self.use_colors:
            if "🚀" in formatted:
                formatted = formatted.replace("🚀", VisualLoggingEnhancer.colorize("🚀", "BLUE", True))
            elif "✅" in formatted:
                formatted = formatted.replace("✅", VisualLoggingEnhancer.colorize("✅", "GREEN", True))
            elif "⚠️" in formatted:
                formatted = formatted.replace("⚠️", VisualLoggingEnhancer.colorize("⚠️", "YELLOW", True))
            elif "❌" in formatted:
                formatted = formatted.replace("❌", VisualLoggingEnhancer.colorize("❌", "RED", True))
        
        # Add performance context if available
        if self.include_performance and hasattr(record, 'performance_context'):
            perf_data = record.performance_context
            perf_parts = []
            
            if 'duration' in perf_data:
                duration_str = VisualLoggingEnhancer.format_duration(perf_data['duration'])
                perf_parts.append(f"⏱️ {duration_str}")
            
            if 'memory_mb' in perf_data:
                memory_str = VisualLoggingEnhancer.format_memory_usage(perf_data['memory_mb'])
                perf_parts.append(f"🧠 {memory_str}")
            
            if perf_parts:
                formatted += f" [{' | '.join(perf_parts)}]"
        
        return formatted

# Global progress tracker
_global_progress_tracker = None

# Enhanced logging functions with visual improvements
def log_step_start(logger_or_step_name, message: str = None, step_number: int = None, 
                  total_steps: int = None, **metadata):
    """Enhanced step start logging with visual progress indicators."""
    if isinstance(logger_or_step_name, str):
        step_name = logger_or_step_name
        message = message or f"Starting {step_name}"
        logger = EnhancedPipelineLogger.get_logger(step_name)
    else:
        logger = logger_or_step_name
        message = message or "Starting step"
    
    # Add progress tracking if step numbers provided
    global _global_progress_tracker
    if step_number and total_steps:
        if not _global_progress_tracker:
            _global_progress_tracker = PipelineProgressTracker(total_steps)
        
        progress_header = _global_progress_tracker.start_step(step_number, step_name if isinstance(logger_or_step_name, str) else f"Step {step_number}")
        message = f"{progress_header}\n    {message}"
    
    EnhancedPipelineLogger.log_structured(
        logger, logging.INFO,
        message,
        event_type="step_start",
        step_number=step_number,
        total_steps=total_steps,
        **metadata
    )

def log_step_success(logger_or_step_name, message: str = None, step_number: int = None,
                    duration: float = None, **metadata):
    """Enhanced step success logging with visual indicators."""
    if isinstance(logger_or_step_name, str):
        step_name = logger_or_step_name
        message = message or f"{step_name} completed successfully"
        logger = EnhancedPipelineLogger.get_logger(step_name)
    else:
        logger = logger_or_step_name
        message = message or "Step completed successfully"
    
    # Add completion tracking
    global _global_progress_tracker
    if step_number and _global_progress_tracker:
        completion_summary = _global_progress_tracker.complete_step(step_number, "SUCCESS", duration)
        message = f"{completion_summary}\n    {message}"
    
    # Add performance context to log record
    performance_context = {}
    if duration:
        performance_context['duration'] = duration
    if 'memory_mb' in metadata:
        performance_context['memory_mb'] = metadata['memory_mb']
    
    record = logging.LogRecord(
        name=logger.name, level=logging.INFO, pathname="", lineno=0,
        msg=f"✅ {message}", args=(), exc_info=None
    )
    record.structured_data = {**metadata, "event_type": "step_success", "step_number": step_number}
    record.performance_context = performance_context
    logger.handle(record)

def log_step_warning(logger_or_step_name, message: str = None, step_number: int = None, **metadata):
    """Enhanced step warning logging with visual indicators."""
    if isinstance(logger_or_step_name, str):
        step_name = logger_or_step_name
        message = message or f"Warning in {step_name}"
        logger = EnhancedPipelineLogger.get_logger(step_name)
    else:
        logger = logger_or_step_name
        message = message or "Step warning"
    
    # Add completion tracking for warnings
    global _global_progress_tracker
    if step_number and _global_progress_tracker:
        completion_summary = _global_progress_tracker.complete_step(step_number, "SUCCESS_WITH_WARNINGS")
        message = f"{completion_summary}\n    {message}"
    
    EnhancedPipelineLogger.log_structured(
        logger, logging.WARNING,
        f"⚠️ {message}",
        event_type="step_warning",
        step_number=step_number,
        **metadata
    )

def log_step_error(logger_or_step_name, message: str = None, step_number: int = None, **metadata):
    """Enhanced step error logging with visual indicators."""
    if isinstance(logger_or_step_name, str):
        step_name = logger_or_step_name
        message = message or f"Error in {step_name}"
        logger = EnhancedPipelineLogger.get_logger(step_name)
    else:
        logger = logger_or_step_name
        message = message or "Step error"
    
    # Add completion tracking for errors
    global _global_progress_tracker
    if step_number and _global_progress_tracker:
        completion_summary = _global_progress_tracker.complete_step(step_number, "FAILED")
        message = f"{completion_summary}\n    {message}"
    
    # Extract event_type if provided, otherwise use default
    event_type = metadata.pop("event_type", "step_error") if metadata else "step_error"
    
    EnhancedPipelineLogger.log_structured(
        logger, logging.ERROR,
        f"❌ {message}",
        event_type=event_type,
        step_number=step_number,
        **metadata
    )
    # Return a small dict to satisfy tests expecting non-None
    return {"status": "ERROR", "message": message}

def log_pipeline_summary(logger: logging.Logger, summary_data: Dict[str, Any]):
    """Enhanced pipeline summary logging with sophisticated visual formatting."""
    
    # Extract summary statistics
    steps = summary_data.get('steps', [])
    total_steps = len(steps)
    successes = len([s for s in steps if s.get('status') == 'SUCCESS'])
    warnings = len([s for s in steps if s.get('status') == 'SUCCESS_WITH_WARNINGS'])
    failures = len([s for s in steps if 'FAILED' in s.get('status', '') or 'ERROR' in s.get('status', '')])
    
    # Calculate total duration
    total_duration = sum(s.get('duration_seconds', 0) for s in steps if s.get('duration_seconds'))
    
    # Determine overall status
    if failures > 0:
        overall_status = "FAILED"
        status_color = "RED"
        status_icon = "❌"
    elif warnings > 0:
        overall_status = "SUCCESS_WITH_WARNINGS"
        status_color = "YELLOW" 
        status_icon = "⚠️"
    else:
        overall_status = "SUCCESS"
        status_color = "GREEN"
        status_icon = "✅"
    
    # Create sophisticated visual summary box
    box_width = 85
    title = f"{status_icon} PIPELINE EXECUTION SUMMARY - {overall_status} {status_icon}"
    
    # Unicode box drawing characters
    top_border = "╔" + "═" * (box_width - 2) + "╗"
    bottom_border = "╚" + "═" * (box_width - 2) + "╝"
    middle_border = "╠" + "═" * (box_width - 2) + "╣"
    
    # Create title line with proper centering
    title_padding = (box_width - 2 - len(title)) // 2
    title_line = f"║{' ' * title_padding}{VisualLoggingEnhancer.colorize(title, status_color, True)}{' ' * (box_width - 2 - title_padding - len(title))}║"
    
    # Create content lines with enhanced formatting
    content_lines = []
    
    # Basic statistics
    stats_line = f"║ Total Steps: {VisualLoggingEnhancer.colorize(str(total_steps), 'WHITE', True)}"
    content_lines.append(stats_line.ljust(box_width - 1) + "║")
    
    # Success statistics with colors
    success_text = f"✅ Successful: {VisualLoggingEnhancer.colorize(str(successes), 'GREEN', True)}"
    warning_text = f"⚠️ Warnings: {VisualLoggingEnhancer.colorize(str(warnings), 'YELLOW', True)}"
    failure_text = f"❌ Failed: {VisualLoggingEnhancer.colorize(str(failures), 'RED', True)}"
    
    # Calculate line lengths accounting for ANSI color codes
    success_display_len = len(f"✅ Successful: {successes}")
    warning_display_len = len(f"⚠️ Warnings: {warnings}")
    failure_display_len = len(f"❌ Failed: {failures}")
    
    success_line = f"║ {success_text}"
    content_lines.append(success_line.ljust(box_width - 1 + (len(success_text) - success_display_len)) + "║")
    
    warning_line = f"║ {warning_text}"
    content_lines.append(warning_line.ljust(box_width - 1 + (len(warning_text) - warning_display_len)) + "║")
    
    failure_line = f"║ {failure_text}"
    content_lines.append(failure_line.ljust(box_width - 1 + (len(failure_text) - failure_display_len)) + "║")
    
    # Duration with enhanced formatting
    duration_text = f"⏱️ Total Time: {VisualLoggingEnhancer.format_duration(total_duration)}"
    duration_display_len = len(f"⏱️ Total Time: ") + len(VisualLoggingEnhancer.format_duration(total_duration).replace('\033[', '').split('m')[-1].replace('\033[0m', ''))
    duration_line = f"║ {duration_text}"
    content_lines.append(duration_line.ljust(box_width - 1 + (len(duration_text) - duration_display_len)) + "║")
    
    # Add performance insights if available
    if total_duration > 0:
        avg_step_time = total_duration / total_steps
        performance_line = f"║ Average Step Time: {VisualLoggingEnhancer.format_duration(avg_step_time)}"
        content_lines.append(performance_line.ljust(box_width - 1) + "║")
    
    # Success rate calculation
    success_rate = ((successes + warnings) / total_steps * 100) if total_steps > 0 else 0
    success_rate_color = "GREEN" if success_rate >= 90 else "YELLOW" if success_rate >= 70 else "RED"
    rate_text = f"📊 Success Rate: {VisualLoggingEnhancer.colorize(f'{success_rate:.1f}%', success_rate_color, True)}"
    rate_display_len = len(f"📊 Success Rate: {success_rate:.1f}%")
    rate_line = f"║ {rate_text}"
    content_lines.append(rate_line.ljust(box_width - 1 + (len(rate_text) - rate_display_len)) + "║")
    
    # Log the sophisticated formatted summary
    logger.info("")
    logger.info(top_border)
    logger.info(title_line)
    logger.info(middle_border)
    for line in content_lines:
        logger.info(line)
    logger.info(bottom_border)
    logger.info("")
    
    # Add step-by-step breakdown for failures/warnings
    if failures > 0 or warnings > 0:
        logger.info("🔍 " + VisualLoggingEnhancer.colorize("DETAILED STEP ANALYSIS:", "CYAN", True))
        for step in steps:
            status = step.get('status', 'UNKNOWN')
            step_name = step.get('script_name', 'Unknown')
            if 'FAILED' in status or 'ERROR' in status or 'WARNING' in status:
                duration = step.get('duration_seconds', 0)
                duration_str = f" ({VisualLoggingEnhancer.format_duration(duration)})" if duration else ""
                
                if 'FAILED' in status or 'ERROR' in status:
                    icon = "❌"
                    color = "RED"
                else:
                    icon = "⚠️"
                    color = "YELLOW"
                
                logger.info(f"  {icon} {VisualLoggingEnhancer.colorize(step_name, color)}: {status}{duration_str}")
        logger.info("")

def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary for the current pipeline run."""
    return performance_tracker.get_summary()

def reset_progress_tracker():
    """Reset the global progress tracker for a new pipeline run."""
    global _global_progress_tracker
    _global_progress_tracker = None

def get_progress_summary() -> str:
    """Get a summary of current progress."""
    return "Progress tracking not available in this context"

def setup_correlation_context(step_name: str, correlation_id: Optional[str] = None) -> str:
    """
    Setup correlation context for a pipeline step.
    
    Args:
        step_name: Name of the pipeline step
        correlation_id: Optional correlation ID (generated if not provided)
        
    Returns:
        The correlation ID that was set
    """
    return PipelineLogger.set_correlation_context(step_name, correlation_id) 

# --- Backwards compatibility alias ---
# Some modules import set_correlation_context directly from this module.
# Provide a thin wrapper that delegates to PipelineLogger.set_correlation_context.
def set_correlation_context(step_name: str, correlation_id: Optional[str] = None) -> str:
    """Compatibility alias. Prefer setup_correlation_context or PipelineLogger.set_correlation_context."""
    return PipelineLogger.set_correlation_context(step_name, correlation_id)