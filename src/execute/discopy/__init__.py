"""
DisCoPy execution module for validating and analyzing DisCoPy diagrams.

This module contains the DisCoPy executor for the GNN Processing Pipeline.
"""

from .discopy_executor import DisCoPyExecutor, run_discopy_analysis

__all__ = [
    'run_discopy_analysis',
    'DisCoPyExecutor'
]
