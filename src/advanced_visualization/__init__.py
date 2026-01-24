"""
Advanced visualization package for GNN Processing Pipeline.

Exports real advanced visualization components including D2 diagram generation.
"""

__version__ = "1.1.3"
FEATURES = {
    "d2_diagrams": True,
    "interactive_dashboards": True,
    "network_visualization": True,
    "timeline_visualization": True,
    "heatmap_visualization": True,
    "data_extraction": True,
    "mcp_integration": True
}

from .visualizer import (
    AdvancedVisualizer,
    create_visualization_from_data,
    create_dashboard_section,
    create_network_visualization,
    create_timeline_visualization,
    create_heatmap_visualization,
    create_default_visualization,
)

from .dashboard import (
    DashboardGenerator,
    generate_dashboard,
)

from .data_extractor import (
    VisualizationDataExtractor,
    extract_visualization_data,
)

# Import D2 visualization components
try:
    from .d2_visualizer import (
        D2Visualizer,
        D2DiagramSpec,
        D2GenerationResult,
        process_gnn_file_with_d2,
    )
    D2_AVAILABLE = True
except ImportError:
    D2_AVAILABLE = False
    D2Visualizer = None
    D2DiagramSpec = None
    D2GenerationResult = None
    process_gnn_file_with_d2 = None

# Import main processor function for thin orchestrator
from .processor import process_advanced_viz_standardized_impl

__all__ = [
    'AdvancedVisualizer',
    'create_visualization_from_data',
    'create_dashboard_section',
    'create_network_visualization',
    'create_timeline_visualization',
    'create_heatmap_visualization',
    'create_default_visualization',
    'DashboardGenerator',
    'generate_dashboard',
    'VisualizationDataExtractor',
    'extract_visualization_data',
    'process_advanced_viz_standardized_impl',  # Main processing function
    'D2Visualizer',  # D2 diagram generation
    'D2DiagramSpec',  # D2 diagram specifications
    'D2GenerationResult',  # D2 generation results
    'process_gnn_file_with_d2',  # Process GNN files with D2
    'D2_AVAILABLE',  # D2 availability flag
]
