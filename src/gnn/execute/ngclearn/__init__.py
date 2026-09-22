"""ngc-learn executor package for GNN pipeline."""

from typing import Any

from .ngclearn_runner import (
    execute_ngclearn_script,
    find_ngclearn_scripts,
    is_ngclearn_available,
    run_ngclearn_scripts,
)

__all__: list[Any] = [
    "is_ngclearn_available",
    "find_ngclearn_scripts",
    "execute_ngclearn_script",
    "run_ngclearn_scripts",
]
