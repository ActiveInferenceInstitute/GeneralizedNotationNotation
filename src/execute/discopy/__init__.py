"""
DisCoPy execution module for validating and analyzing DisCoPy diagrams.

This module contains the DisCoPy executor for the GNN Processing Pipeline.
"""

from typing import Any

from .discopy_executor import DisCoPyExecutor, run_discopy_analysis

__all__: list[Any] = ["run_discopy_analysis", "DisCoPyExecutor"]
