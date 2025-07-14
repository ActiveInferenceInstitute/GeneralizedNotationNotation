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


def get_module_info():
    """Get comprehensive information about the pipeline module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'pipeline_capabilities': [],
        'execution_modes': []
    }
    
    # Pipeline capabilities
    info['pipeline_capabilities'].extend([
        'Multi-step pipeline orchestration',
        'Dynamic step discovery and execution',
        'Configuration management',
        'Performance monitoring and tracking',
        'Error handling and recovery',
        'Dependency validation'
    ])
    
    # Execution modes
    info['execution_modes'].extend([
        'Sequential execution',
        'Parallel execution (where supported)',
        'Selective step execution',
        'Conditional execution based on dependencies'
    ])
    
    return info


def get_pipeline_options() -> dict:
    """Get information about available pipeline options."""
    return {
        'execution_modes': {
            'sequential': 'Execute steps in order',
            'parallel': 'Execute compatible steps in parallel',
            'selective': 'Execute only specified steps',
            'conditional': 'Execute steps based on conditions'
        },
        'monitoring_levels': {
            'basic': 'Basic execution monitoring',
            'detailed': 'Detailed performance monitoring',
            'comprehensive': 'Comprehensive monitoring with resource tracking'
        },
        'error_handling': {
            'stop_on_error': 'Stop pipeline on first error',
            'continue_on_error': 'Continue pipeline despite errors',
            'retry_on_error': 'Retry failed steps automatically'
        },
        'output_formats': {
            'json': 'JSON structured output',
            'text': 'Plain text output',
            'html': 'HTML report generation'
        }
    } 