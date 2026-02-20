"""
Combined analysis visualization functions for GNN Processing Pipeline.

This module handles combined analysis plots, standalone panel generation,
generative model diagrams, and cross-file combined visualizations.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging
import json
import numpy as np

# Import visualization libraries with error handling
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except (ImportError, RecursionError):
    plt = None
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import shared helpers from processor
from .processor import _save_plot_safely, _safe_tight_layout, parse_gnn_content


def generate_combined_analysis(parsed_data: Dict[str, Any], output_dir: Path, model_name: str) -> List[str]:
    """
    Generate combined analysis visualizations.

    Args:
        parsed_data: Parsed GNN data
        output_dir: Output directory
        model_name: Name of the model

    Returns:
        List of generated visualization file paths
    """
    visualizations = []

    if not MATPLOTLIB_AVAILABLE:
        return visualizations

    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Variable type distribution
        variables = parsed_data.get("variables", [])
        if variables:
            var_types = []
            for var_info in variables:
                var_type = 'unknown'
                if isinstance(var_info, dict):
                    var_type = var_info.get('type', var_info.get('node_type', 'unknown'))
                var_types.append(var_type)

            if var_types:
                type_counts = {}
                for var_type in var_types:
                    type_counts[var_type] = type_counts.get(var_type, 0) + 1

                ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
                ax1.set_title("Variable Type Distribution")
            else:
                ax1.text(0.5, 0.5, "No variable type data", transform=ax1.transAxes, ha='center', va='center')
                ax1.set_title("Variable Type Distribution (No Data)")

        # 2. Connection count histogram
        connections = parsed_data.get("connections", [])
        if connections:
            source_counts = {}
            target_counts = {}
            for conn in connections:
                if isinstance(conn, dict):
                    source = conn.get("source_variables", [conn.get("source", "unknown")])[0] if conn.get("source_variables") else conn.get("source", "unknown")
                    target = conn.get("target_variables", [conn.get("target", "unknown")])[0] if conn.get("target_variables") else conn.get("target", "unknown")

                    source_counts[source] = source_counts.get(source, 0) + 1
                    target_counts[target] = target_counts.get(target, 0) + 1

            all_nodes = set(source_counts.keys()) | set(target_counts.keys())
            node_counts = [source_counts.get(node, 0) + target_counts.get(node, 0) for node in all_nodes]

            if node_counts:
                ax2.hist(node_counts, bins=min(10, len(set(node_counts))), alpha=0.7, color='skyblue', edgecolor='black')
                ax2.set_title("Node Connection Count Distribution")
                ax2.set_xlabel("Number of Connections")
                ax2.set_ylabel("Frequency")
            else:
                ax2.text(0.5, 0.5, "No connection data", transform=ax2.transAxes, ha='center', va='center')
                ax2.set_title("Node Connection Count Distribution (No Data)")

        # 3. Matrix statistics
        parameters = parsed_data.get("parameters", [])
        matrix_sizes = []
        if parameters:
            for param in parameters:
                if isinstance(param, dict) and "value" in param:
                    value = param["value"]
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        try:
                            arr = np.array(value)
                            matrix_sizes.append(arr.size)
                        except Exception:
                            def count_elements(obj):
                                if isinstance(obj, (int, float)):
                                    return 1
                                elif isinstance(obj, (list, tuple)):
                                    return sum(count_elements(item) for item in obj)
                                else:
                                    return 1
                            matrix_sizes.append(count_elements(value))

        if matrix_sizes:
            ax3.hist(matrix_sizes, bins=min(10, len(matrix_sizes)), alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_title("Matrix Size Distribution")
            ax3.set_xlabel("Matrix Size (elements)")
            ax3.set_ylabel("Frequency")
        else:
            ax3.text(0.5, 0.5, "No matrix data", transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title("Matrix Size Distribution (No Data)")

        # 4. Section content length
        raw_sections = parsed_data.get("raw_sections", {})
        if raw_sections:
            section_lengths = [len(str(content)) for content in raw_sections.values()]
            section_names = list(raw_sections.keys())

            ax4.bar(range(len(section_names)), section_lengths, alpha=0.7, color='orange')
            ax4.set_title("Section Content Length")
            ax4.set_xlabel("Sections")
            ax4.set_ylabel("Content Length (characters)")
            ax4.set_xticks(range(len(section_names)))
            ax4.set_xticklabels([name[:20] + "..." if len(name) > 20 else name for name in section_names], rotation=45, ha='right')
        else:
            ax4.text(0.5, 0.5, "No section data", transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title("Section Content Length (No Data)")

        plt.suptitle(f"{model_name} - Combined Analysis", fontsize=16)
        _safe_tight_layout()

        plot_file = output_dir / f"{model_name}_combined_analysis.png"
        _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        visualizations.append(str(plot_file))

        # Generate standalone panel visualizations
        standalone_files = _generate_standalone_panels(parsed_data, output_dir, model_name)
        visualizations.extend(standalone_files)

        # Generate generative model diagram
        gm_files = _generate_generative_model_diagram(parsed_data, output_dir, model_name)
        visualizations.extend(gm_files)

    except Exception as e:
        print(f"Error generating combined analysis: {e}")

    return visualizations


def _generate_standalone_panels(parsed_data: Dict[str, Any], output_dir: Path, model_name: str) -> List[str]:
    """Generate standalone visualization files for each panel in the combined analysis."""
    visualizations = []

    if not MATPLOTLIB_AVAILABLE:
        return visualizations

    try:
        # 1. Matrix Size Distribution (standalone)
        parameters = parsed_data.get("parameters", [])
        matrix_sizes = []
        matrix_names = []
        if parameters:
            for param in parameters:
                if isinstance(param, dict) and "value" in param:
                    value = param["value"]
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        try:
                            arr = np.array(value)
                            matrix_sizes.append(arr.size)
                            matrix_names.append(param.get("name", "Unknown"))
                        except Exception:
                            pass

        if matrix_sizes:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(matrix_names, matrix_sizes, alpha=0.7, color='lightgreen', edgecolor='black')
            ax.set_title(f"{model_name} - Matrix Size Distribution", fontsize=14, fontweight='bold')
            ax.set_xlabel("Matrix Parameter")
            ax.set_ylabel("Size (elements)")

            for bar, size in zip(bars, matrix_sizes):
                height = bar.get_height()
                ax.annotate(f'{size}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)

            _safe_tight_layout()
            plot_file = output_dir / f"{model_name}_matrix_size_distribution.png"
            _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append(str(plot_file))

        # 2. Section Content Length (standalone)
        raw_sections = parsed_data.get("raw_sections", {})
        if raw_sections:
            fig, ax = plt.subplots(figsize=(12, 6))
            section_lengths = [len(str(content)) for content in raw_sections.values()]
            section_names = list(raw_sections.keys())

            colors = plt.cm.viridis(np.linspace(0, 1, len(section_names)))
            bars = ax.bar(range(len(section_names)), section_lengths, alpha=0.7, color=colors, edgecolor='black')
            ax.set_title(f"{model_name} - Section Content Length", fontsize=14, fontweight='bold')
            ax.set_xlabel("Sections")
            ax.set_ylabel("Content Length (characters)")
            ax.set_xticks(range(len(section_names)))
            ax.set_xticklabels([name[:15] + "..." if len(name) > 15 else name for name in section_names],
                               rotation=45, ha='right', fontsize=9)

            _safe_tight_layout()
            plot_file = output_dir / f"{model_name}_section_content_length.png"
            _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append(str(plot_file))

        # 3. Variable Type Distribution (standalone pie chart)
        variables = parsed_data.get("variables", [])
        if variables:
            fig, ax = plt.subplots(figsize=(10, 8))
            var_types = []
            for var_info in variables:
                var_type = 'unknown'
                if isinstance(var_info, dict):
                    var_type = var_info.get('type', var_info.get('node_type', 'unknown'))
                var_types.append(var_type)

            if var_types:
                type_counts = {}
                for var_type in var_types:
                    type_counts[var_type] = type_counts.get(var_type, 0) + 1

                colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
                wedges, texts, autotexts = ax.pie(type_counts.values(), labels=type_counts.keys(),
                                                   autopct='%1.1f%%', colors=colors)
                ax.set_title(f"{model_name} - Variable Type Distribution", fontsize=14, fontweight='bold')

                _safe_tight_layout()
                plot_file = output_dir / f"{model_name}_variable_type_distribution.png"
                _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append(str(plot_file))

    except Exception as e:
        print(f"Error generating standalone panels: {e}")

    return visualizations


def _generate_generative_model_diagram(parsed_data: Dict[str, Any], output_dir: Path, model_name: str) -> List[str]:
    """Generate a generative model diagram showing the POMDP structure."""
    visualizations = []

    if not MATPLOTLIB_AVAILABLE:
        return visualizations

    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')

        positions = {
            'D': (2, 8),
            's': (2, 6),
            's\'': (5, 6),
            'A': (2, 4),
            'o': (2, 2),
            'B': (3.5, 7.5),
            'C': (8, 4),
            'E': (8, 7),
            '\u03c0': (6, 5),
            'G': (8, 5.5),
            'u': (5, 3),
        }

        node_colors = {
            'D': '#98D8C8',
            's': '#7EC8E3',
            's\'': '#7EC8E3',
            'A': '#F7DC6F',
            'o': '#82E0AA',
            'B': '#F1948A',
            'C': '#C39BD3',
            'E': '#F5B7B1',
            '\u03c0': '#FAD7A0',
            'G': '#D2B4DE',
            'u': '#ABEBC6',
        }

        for node_name, (x, y) in positions.items():
            color = node_colors.get(node_name, 'lightgray')
            circle = plt.Circle((x, y), 0.4, color=color, ec='black', linewidth=2, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, node_name, ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)

        edges = [
            ('D', 's', 'Prior'),
            ('s', 'A', 'Likelihood'),
            ('A', 'o', 'Observation'),
            ('s', 's\'', 'Transition'),
            ('B', 's\'', 'Control'),
            ('\u03c0', 'u', 'Selection'),
            ('C', 'G', 'Preferences'),
            ('E', '\u03c0', 'Habit'),
            ('G', '\u03c0', 'EFE'),
            ('u', 's\'', 'Effect'),
        ]

        for source, target, label in edges:
            if source in positions and target in positions:
                x1, y1 = positions[source]
                x2, y2 = positions[target]

                dx, dy = x2 - x1, y2 - y1
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    start_x = x1 + 0.4 * dx / dist
                    start_y = y1 + 0.4 * dy / dist
                    end_x = x2 - 0.4 * dx / dist
                    end_y = y2 - 0.4 * dy / dist

                    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                               zorder=1)

        ax.set_title(f"{model_name}\nGenerative Model Structure (POMDP)", fontsize=16, fontweight='bold', pad=20)

        legend_items = [
            ('D', 'Prior (Initial State Belief)', '#98D8C8'),
            ('s/s\'', 'Hidden State', '#7EC8E3'),
            ('A', 'Likelihood Matrix', '#F7DC6F'),
            ('o', 'Observation', '#82E0AA'),
            ('B', 'Transition Matrix', '#F1948A'),
            ('C', 'Preferences', '#C39BD3'),
            ('E', 'Habit (Policy Prior)', '#F5B7B1'),
            ('\u03c0', 'Policy', '#FAD7A0'),
            ('G', 'Expected Free Energy', '#D2B4DE'),
            ('u', 'Action', '#ABEBC6'),
        ]

        legend_y = 1.5
        for symbol, desc, color in legend_items:
            ax.add_patch(plt.Rectangle((6.5, legend_y), 0.3, 0.3, color=color, ec='black'))
            ax.text(6.95, legend_y + 0.15, f"{symbol}: {desc}", fontsize=8, va='center')
            legend_y -= 0.4

        _safe_tight_layout()
        plot_file = output_dir / f"{model_name}_generative_model.png"
        _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations.append(str(plot_file))

    except Exception as e:
        print(f"Error generating generative model diagram: {e}")
        import traceback
        traceback.print_exc()

    return visualizations


def generate_combined_visualizations(gnn_files: List[Path], results_dir: Path, verbose: bool = False) -> List[str]:
    """
    Generate combined visualizations across multiple GNN files.

    Args:
        gnn_files: List of GNN file paths
        results_dir: Results directory
        verbose: Enable verbose output

    Returns:
        List of generated visualization file paths
    """
    visualizations = []

    if not MATPLOTLIB_AVAILABLE:
        return visualizations

    try:
        # Import the proper GNN parser
        from .parser import GNNParser

        # Collect data from all files
        all_variables = []
        all_connections = []
        all_matrices = []

        parser = GNNParser()
        for gnn_file in gnn_files:
            try:
                parsed_data = parser.parse_file(str(gnn_file))

                variables = parsed_data.get("Variables", {})
                for var_name, var_info in variables.items():
                    all_variables.append({
                        "name": var_name,
                        "type": var_info.get('type', 'unknown'),
                        "dimensions": var_info.get('dimensions', []),
                        "comment": var_info.get('comment', '')
                    })

                connections = parsed_data.get("Edges", [])
                all_connections.extend(connections)

                for var_name, var_info in variables.items():
                    dimensions = var_info.get('dimensions', [])
                    if len(dimensions) >= 2 and all(isinstance(d, int) for d in dimensions[:2]):
                        all_matrices.append({
                            "name": var_name,
                            "shape": dimensions,
                            "size": dimensions[0] * dimensions[1] if len(dimensions) >= 2 else 0
                        })

            except Exception as e:
                logger.warning(f"Could not parse {gnn_file}: {e}")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Overall variable type distribution
        if all_variables:
            var_types = [v["type"] for v in all_variables]
            type_counts = {}
            for var_type in var_types:
                type_counts[var_type] = type_counts.get(var_type, 0) + 1

            ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
            ax1.set_title("Overall Variable Type Distribution")

        # 2. File comparison
        file_stats = []
        for gnn_file in gnn_files:
            with open(gnn_file, 'r', encoding='utf-8') as f:
                content = f.read()

            file_parsed_data = parse_gnn_content(content)
            file_stats.append({
                "name": gnn_file.stem,
                "variables": len(file_parsed_data.get("variables", [])),
                "connections": len(file_parsed_data.get("connections", [])),
                "matrices": len(file_parsed_data.get("matrices", []))
            })

        if file_stats:
            file_names = [stat["name"] for stat in file_stats]
            var_counts = [stat["variables"] for stat in file_stats]
            conn_counts = [stat["connections"] for stat in file_stats]

            x = range(len(file_names))
            width = 0.35

            ax2.bar([i - width/2 for i in x], var_counts, width, label='Variables', alpha=0.7)
            ax2.bar([i + width/2 for i in x], conn_counts, width, label='Connections', alpha=0.7)
            ax2.set_title("File Comparison")
            ax2.set_xlabel("Files")
            ax2.set_ylabel("Count")
            ax2.set_xticks(x)
            ax2.set_xticklabels(file_names, rotation=45, ha='right')
            ax2.legend()

        # 3. Matrix size distribution
        if all_matrices:
            matrix_sizes = []
            for m in all_matrices:
                if isinstance(m, dict) and "size" in m:
                    matrix_sizes.append(m["size"])
                elif hasattr(m, 'size'):
                    matrix_sizes.append(m.size)
            if matrix_sizes:
                ax3.hist(matrix_sizes, bins=min(15, len(matrix_sizes)), alpha=0.7, color='lightgreen', edgecolor='black')
                ax3.set_title("Overall Matrix Size Distribution")
                ax3.set_xlabel("Matrix Size (elements)")
                ax3.set_ylabel("Frequency")

        # 4. Connection type analysis
        if all_connections:
            connection_types = {}
            for conn in all_connections:
                conn_type = f"{conn['source']}->{conn['target']}"
                connection_types[conn_type] = connection_types.get(conn_type, 0) + 1

            if connection_types:
                top_connections = sorted(connection_types.items(), key=lambda x: x[1], reverse=True)[:10]
                conn_names, conn_counts = zip(*top_connections)

                ax4.barh(range(len(conn_names)), conn_counts, alpha=0.7, color='orange')
                ax4.set_title("Top Connection Types")
                ax4.set_xlabel("Count")
                ax4.set_yticks(range(len(conn_names)))
                ax4.set_yticklabels(conn_names)

        plt.suptitle("Combined Analysis Across All Files", fontsize=16)
        _safe_tight_layout()

        plot_file = results_dir / "combined_analysis.png"
        _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        visualizations.append(str(plot_file))

    except Exception as e:
        print(f"Error generating combined visualizations: {e}")

    return visualizations
