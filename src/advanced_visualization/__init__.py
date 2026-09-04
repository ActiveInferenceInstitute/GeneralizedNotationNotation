"""
Advanced visualization package for GNN Processing Pipeline.

Exports real advanced visualization components including D2 diagram generation.
"""

from typing import Any

__version__ = "1.6.0"
FEATURES: dict[str, Any] = {
    "d2_diagrams": True,
    "interactive_dashboards": True,
    "network_visualization": True,
    "timeline_visualization": True,
    "heatmap_visualization": True,
    "data_extraction": True,
    "mcp_integration": True,
}

# Canonical set of ``viz_type`` values accepted by ``process_advanced_viz``.
# Duplicating the list inline, so the CLI choices stay in sync with the processor.
VIZ_TYPE_CHOICES: tuple[str, ...] = (
    "all",
    "3d",
    "interactive",
    "dashboard",
    "d2",
    "diagrams",
    "pipeline",
    "statistical",
    "pomdp",
    "network",
)

from .d2_visualizer import (
    D2DiagramSpec,
    D2GenerationResult,
    D2Visualizer,
    process_gnn_file_with_d2,
)
from .dashboard import (
    DashboardGenerator,
    generate_dashboard,
)
from .data_extractor import (
    VisualizationDataExtractor,
    extract_visualization_data,
)
from .visualizer import (
    AdvancedVisualizer,
    create_dashboard_section,
    create_default_visualization,
    create_heatmap_visualization,
    create_network_visualization,
    create_timeline_visualization,
    create_visualization_from_data,
)

D2_AVAILABLE = True

# Import main processor function for thin orchestrator
from .processor import process_advanced_viz

__all__: list[Any] = [
    "AdvancedVisualizer",
    "create_visualization_from_data",
    "create_dashboard_section",
    "create_network_visualization",
    "create_timeline_visualization",
    "create_heatmap_visualization",
    "create_default_visualization",
    "DashboardGenerator",
    "generate_dashboard",
    "VisualizationDataExtractor",
    "extract_visualization_data",
    "process_advanced_viz",  # Main processing function
    "D2Visualizer",  # D2 diagram generation
    "D2DiagramSpec",  # D2 diagram specifications
    "D2GenerationResult",  # D2 generation results
    "process_gnn_file_with_d2",  # Process GNN files with D2
    "D2_AVAILABLE",  # D2 availability flag
    "VIZ_TYPE_CHOICES",  # canonical viz_type values
    "probe_capabilities",  # live capability probe
]


def probe_capabilities() -> dict[str, Any]:
    """Probe the live runtime capabilities of the advanced_visualization module.

    Unlike the static :data:`FEATURES` map (which advertises designed features),
    this function checks the actual environment: whether the ``d2`` CLI is on
    ``PATH`` and whether ``plotly``, ``seaborn``, ``matplotlib``, ``numpy``,
    and ``networkx`` are importable. Returns a dict of ``{feature: bool}``
    suitable for the ``check_visualization_capabilities`` MCP tool.
    """
    import importlib.util
    import shutil

    def _importable(name: str) -> bool:
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    return {
        "d2_cli": shutil.which("d2") is not None,
        "matplotlib": _importable("matplotlib"),
        "numpy": _importable("numpy"),
        "plotly": importlib.util.find_spec("plotly") is not None,
        "seaborn": _importable("seaborn"),
        "networkx": _importable("networkx"),
    }


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "advanced_visualization",
        "version": __version__,
        "description": "Advanced 3D visualization and interactive dashboards for GNN models",
        "features": FEATURES,
    }
