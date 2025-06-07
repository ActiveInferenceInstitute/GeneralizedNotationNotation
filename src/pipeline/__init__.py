"""
Pipeline Module

Provides utilities for managing and executing the GNN processing pipeline.
Centralizes configuration, execution logic, and performance monitoring.
"""

from .config import (
    PIPELINE_STEP_CONFIGURATION,
    STEP_TIMEOUTS,
    CRITICAL_STEPS,
    ARG_PROPERTIES,
    SCRIPT_ARG_SUPPORT,
    OUTPUT_DIR_MAPPING,
    get_step_timeout,
    is_critical_step,
    get_output_dir_for_script
)

from .execution import (
    StepExecutionResult,
    get_memory_usage_mb,
    build_command_args,
    execute_pipeline_step
)

__all__ = [
    'PIPELINE_STEP_CONFIGURATION',
    'STEP_TIMEOUTS', 
    'CRITICAL_STEPS',
    'ARG_PROPERTIES',
    'SCRIPT_ARG_SUPPORT',
    'OUTPUT_DIR_MAPPING',
    'get_step_timeout',
    'is_critical_step',
    'get_output_dir_for_script',
    'StepExecutionResult',
    'get_memory_usage_mb',
    'build_command_args',
    'execute_pipeline_step'
] 