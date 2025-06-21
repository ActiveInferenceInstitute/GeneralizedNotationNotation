#!/usr/bin/env python3
"""
Pipeline module for centralized configuration and utilities.
"""

from .config import (
    get_pipeline_config,
    set_pipeline_config,
    PipelineConfig,
    StepConfig,
    STEP_METADATA,
    get_output_dir_for_script
)

__all__ = [
    'get_pipeline_config',
    'set_pipeline_config', 
    'PipelineConfig',
    'StepConfig',
    'STEP_METADATA',
    'get_output_dir_for_script'
] 