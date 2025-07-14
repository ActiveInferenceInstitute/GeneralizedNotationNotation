"""
DisCoPy execution module for validating and analyzing DisCoPy diagrams.

This module contains the DisCoPy executor for the GNN Processing Pipeline.
"""

from .discopy_executor import run_discopy_analysis, DisCoPyExecutor

__all__ = [
    'run_discopy_analysis',
    'DisCoPyExecutor'
] 