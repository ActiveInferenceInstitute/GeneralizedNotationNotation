"""
Execute module for running rendered GNN simulators.

This package contains modules for executing:
- PyMDP scripts
- RxInfer.jl configurations
- DisCoPy diagrams
- ActiveInference.jl scripts
- JAX implementations
"""

# Import from submodules
from . import pymdp
from . import rxinfer
from . import discopy
from . import activeinference_jl
from . import jax

# Import from executor module
from .executor import (
    GNNExecutor,
    execute_gnn_model,
    run_simulation,
    generate_execution_report
)

# Module metadata
__version__ = "1.1.0"
__author__ = "Active Inference Institute"
__description__ = "GNN model execution and simulation"

# Feature availability flags
FEATURES = {
    'pymdp_execution': True,
    'rxinfer_execution': True,
    'discopy_execution': True,
    'activeinference_jl_execution': True,
    'jax_execution': True,
    'simulation_management': True,
    'report_generation': True
}

# Main API functions
__all__ = [
    # Submodules
    'pymdp',
    'rxinfer', 
    'discopy',
    'activeinference_jl',
    'jax',
    
    # Executor functions
    'GNNExecutor',
    'execute_gnn_model',
    'run_simulation',
    'generate_execution_report',
    
    # Test-compatible functions
    'execute_script_safely',
    'validate_execution_environment',
    'run_pymdp_simulation',
    'run_rxinfer_simulation',
    
    # Metadata
    'FEATURES',
    '__version__'
]


def get_module_info():
    """Get comprehensive information about the execute module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'execution_types': [],
        'supported_backends': []
    }
    
    # Execution types
    info['execution_types'].extend([
        'PyMDP script execution',
        'RxInfer.jl configuration execution',
        'DisCoPy diagram execution',
        'ActiveInference.jl script execution',
        'JAX implementation execution'
    ])
    
    # Supported backends
    info['supported_backends'].extend(['Python', 'Julia', 'JAX'])
    
    return info


def get_execution_options() -> dict:
    """Get information about available execution options."""
    return {
        'execution_modes': {
            'synchronous': 'Synchronous execution with blocking',
            'asynchronous': 'Asynchronous execution with callbacks',
            'batch': 'Batch execution of multiple models'
        },
        'timeout_options': {
            'short': 'Short timeout (30 seconds)',
            'medium': 'Medium timeout (5 minutes)',
            'long': 'Long timeout (30 minutes)',
            'unlimited': 'No timeout'
        },
        'output_formats': {
            'json': 'JSON structured output',
            'text': 'Plain text output',
            'binary': 'Binary output for large datasets'
        },
        'monitoring_options': {
            'basic': 'Basic execution monitoring',
            'detailed': 'Detailed execution monitoring',
            'profiling': 'Performance profiling'
        }
    }


# Test-compatible function aliases
def execute_script_safely(script_path, timeout=300, **kwargs):
    """Execute a script safely with timeout and error handling (test-compatible alias)."""
    executor = GNNExecutor()
    return executor.execute_script(script_path, timeout=timeout, **kwargs)

def validate_execution_environment():
    """Validate the execution environment (test-compatible alias)."""
    executor = GNNExecutor()
    return executor.validate_environment()

def run_pymdp_simulation(script_path, **kwargs):
    """Run a PyMDP simulation (test-compatible alias)."""
    executor = GNNExecutor()
    return executor.execute_script(script_path, **kwargs)

def run_rxinfer_simulation(script_path, **kwargs):
    """Run an RxInfer simulation (test-compatible alias)."""
    executor = GNNExecutor()
    return executor.execute_script(script_path, **kwargs) 