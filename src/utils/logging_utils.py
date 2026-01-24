#!/usr/bin/env python3
"""
Logging Utilities Compatibility Module for GNN Processing Pipeline.

This module provides the standard logging interface expected by pipeline modules.
It serves as the central import point for logging functionality.
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Import PerformanceTracker from the dedicated module
from .performance_tracker import PerformanceTracker, performance_tracker

# Thread-local storage for correlation context
_correlation_context = threading.local()


def setup_correlation_context(correlation_id: str = None, step_name: str = None):
    """Set up correlation context for logging."""
    if correlation_id:
        _correlation_context.correlation_id = correlation_id
    if step_name:
        _correlation_context.step_name = step_name


class PipelineLogger:
    """Centralized logger for the GNN pipeline with correlation support."""

    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False

    @classmethod
    def get_logger(cls, name: str, level: int = logging.INFO) -> logging.Logger:
        """Get or create a logger for the given name."""
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(level)

            if not logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                handler.setLevel(level)
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)

            cls._loggers[name] = logger

        return cls._loggers[name]

    @classmethod
    def setup(cls, log_dir: Path = None, verbose: bool = False):
        """Set up the pipeline logging system."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls._initialized = True


def setup_step_logging(step_name: str, verbose: bool = False) -> logging.Logger:
    """Set up logging for a pipeline step."""
    level = logging.DEBUG if verbose else logging.INFO
    logger = PipelineLogger.get_logger(step_name, level)
    setup_correlation_context(step_name=step_name)
    return logger


def setup_main_logging(verbose: bool = False) -> logging.Logger:
    """Set up main pipeline logging."""
    PipelineLogger.setup(verbose=verbose)
    return PipelineLogger.get_logger('pipeline')


def log_step_start(logger: logging.Logger, message: str):
    """Log the start of a step."""
    logger.info(f"🚀 {message}")


def log_step_success(logger: logging.Logger, message: str):
    """Log a successful step completion."""
    logger.info(f"✅ {message}")


def log_step_warning(logger: logging.Logger, message: str):
    """Log a warning during step execution."""
    logger.warning(f"⚠️ {message}")


def log_step_error(logger: logging.Logger, message: str):
    """Log an error during step execution."""
    logger.error(f"❌ {message}")


def log_section_header(logger: logging.Logger, title: str, char: str = "="):
    """Log a section header."""
    width = max(60, len(title) + 4)
    border = char * width
    logger.info(border)
    logger.info(f"  {title}")
    logger.info(border)


def get_performance_summary() -> Dict[str, Any]:
    """Get a summary of performance metrics."""
    return performance_tracker.get_summary()


# Export all public symbols
__all__ = [
    'PipelineLogger',
    'setup_step_logging',
    'setup_main_logging',
    'log_step_start',
    'log_step_success',
    'log_step_warning',
    'log_step_error',
    'log_section_header',
    'get_performance_summary',
    'PerformanceTracker',
    'performance_tracker',
    'setup_correlation_context',
]
