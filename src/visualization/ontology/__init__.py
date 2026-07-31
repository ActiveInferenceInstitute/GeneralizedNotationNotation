"""Public API for the ontology package.

Re-exports Any, OntologyVisualizer from submodules.
"""

from typing import Any

from .visualizer import OntologyVisualizer

__all__: list[Any] = ["OntologyVisualizer"]
