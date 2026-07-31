"""Public API for the stan package.

Re-exports Any, render_stan from submodules.
"""

# Stan renderer for GNN
from typing import Any

from .stan_renderer import render_stan

__all__: list[Any] = ["render_stan"]
