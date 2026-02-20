#!/usr/bin/env python3
"""
Step Logging - Thin Re-export from utils.logging.logging_utils

DEPRECATED: Import directly from utils.logging.logging_utils instead.

This module re-exports logging functions from the canonical source
(utils.logging.logging_utils) for backwards compatibility. New code
should import from utils.logging.logging_utils directly.
"""

try:
    from utils.logging.logging_utils import (
        log_step_start,
        log_step_success,
        log_step_warning,
        log_step_error,
        setup_step_logging,
    )
except ImportError:
    # Minimal fallback if the canonical module is unavailable.
    # This ensures step_logging is always importable.
    import logging
    from typing import Optional

    def log_step_start(logger: logging.Logger, message: str) -> None:
        """Log the start of a step with prefix."""
        logger.info(f"\U0001f680 {message}")

    def log_step_success(logger: logging.Logger, message: str) -> None:
        """Log step success with prefix."""
        logger.info(f"\u2705 {message}")

    def log_step_warning(logger: logging.Logger, message: str) -> None:
        """Log step warning with prefix."""
        logger.warning(f"\u26a0\ufe0f {message}")

    def log_step_error(logger: logging.Logger, message: str) -> None:
        """Log step error with prefix."""
        logger.error(f"\u274c {message}")

    def setup_step_logging(name: str, verbose: bool = False) -> logging.Logger:
        """Set up a logger for a pipeline step."""
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
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
