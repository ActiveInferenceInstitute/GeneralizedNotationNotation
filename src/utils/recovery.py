#!/usr/bin/env python3
"""
Utils Recovery module for GNN Processing Pipeline.

This module provides recovery implementations when core modules are not available.
"""

import logging
from pathlib import Path


class _DefaultStepArgs:
    """Default pipeline step arguments returned by RecoveryArgumentParser."""

    def __init__(self, step_name: str) -> None:
        self.verbose = False
        self.output_dir = Path("output")
        self.step_name = step_name


class RecoveryArgumentParser:
    """
    Recovery argument parser for when argument_utils is not available.
    """
    @staticmethod
    def parse_step_arguments(step_name):
        """Return default args (verbose=False, output_dir='output')."""
        return _DefaultStepArgs(step_name)

def setup_step_logging(step_name: str, verbose: bool = False):
    """Return a logger set to DEBUG if verbose else INFO."""
    logger = logging.getLogger(step_name)
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger
