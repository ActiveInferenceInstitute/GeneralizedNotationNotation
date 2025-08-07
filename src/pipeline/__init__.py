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

from .execution import (
    run_pipeline,
    get_pipeline_status,
    validate_pipeline_config,
    get_pipeline_info,
    create_pipeline_config,
    execute_pipeline_step,
    execute_pipeline_steps,
    StepExecutionResult
)

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "GNN pipeline orchestration and execution"

# Feature availability flags
FEATURES = {
    'pipeline_orchestration': True,
    'step_execution': True,
    'configuration_management': True,
    'performance_monitoring': True,
    'error_handling': True,
    'dependency_validation': True
}

# Main API functions
__all__ = [
    # Configuration
    'get_pipeline_config',
    'set_pipeline_config', 
    'PipelineConfig',
    'StepConfig',
    'STEP_METADATA',
    'get_output_dir_for_script',
    
    # Execution
    'run_pipeline',
    'get_pipeline_status',
    'validate_pipeline_config',
    'get_pipeline_info',
    'create_pipeline_config',
    'execute_pipeline_step',
    'execute_pipeline_steps',
    'StepExecutionResult',
    
    # Metadata
    'FEATURES',
    '__version__'
] 