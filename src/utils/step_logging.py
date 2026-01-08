#!/usr/bin/env python3
"""
Step Logging - Lightweight Logging Functions for Pipeline Steps

This module provides simple, always-importable logging functions with visual prefixes.
These functions are intentionally minimal with no external dependencies to ensure
they can always be imported successfully.

Usage:
    from utils.step_logging import log_step_start, log_step_success, log_step_error, log_step_warning
"""

import logging
from typing import Optional


def log_step_start(logger: logging.Logger, message: str) -> None:
    """Log the start of a step with 🚀 prefix."""
    logger.info(f"🚀 {message}")


def log_step_success(logger: logging.Logger, message: str) -> None:
    """Log step success with ✅ prefix."""
    logger.info(f"✅ {message}")


def log_step_warning(logger: logging.Logger, message: str) -> None:
    """Log step warning with ⚠️ prefix."""
    logger.warning(f"⚠️ {message}")


def log_step_error(logger: logging.Logger, message: str) -> None:
    """Log step error with ❌ prefix."""
    logger.error(f"❌ {message}")


def setup_step_logging(name: str, verbose: bool = False) -> logging.Logger:
    """
    Set up a logger for a pipeline step.
    
    Args:
        name: Name for the logger
        verbose: If True, set DEBUG level; otherwise INFO
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger


__all__ = [
    "log_step_start",
    "log_step_success", 
    "log_step_warning",
    "log_step_error",
    "setup_step_logging",
]
