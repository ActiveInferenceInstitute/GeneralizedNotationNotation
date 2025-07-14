"""
Execute module for running rendered GNN simulators.

This package contains modules for executing:
- PyMDP scripts
- RxInfer.jl configurations
- DisCoPy diagrams
- ActiveInference.jl scripts
"""

# Import from submodules
from . import pymdp
from . import rxinfer
from . import discopy
from . import activeinference_jl

__all__ = [
    'pymdp',
    'rxinfer', 
    'discopy',
    'activeinference_jl'
] 