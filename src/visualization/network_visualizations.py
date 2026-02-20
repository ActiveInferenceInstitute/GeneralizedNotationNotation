"""
Network visualization functions for GNN Processing Pipeline.

This module handles network graph generation, interactive plotly networks,
3D surface plots, and network statistics for POMDP models.
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

try:
    import sys
    if sys.version_info >= (3, 13):
        import os
        os.environ.pop('NETWORKX_AUTOMATIC_BACKENDS', None)
        os.environ['NETWORKX_CACHE_CONVERTED_GRAPHS'] = '1'
    import networkx as nx
    NETWORKX_AVAILABLE = True
except (ImportError, RecursionError, AttributeError, ValueError):
    nx = None
    NETWORKX_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import shared helpers from processor
from .processor import _save_plot_safely, _safe_tight_layout


def generate_network_visualizations(parsed_data: Dict[str, Any], output_dir: Path, model_name: str) -> List[str]:
    """
    Generate network visualizations for POMDP models.

    Args:
        parsed_data: Parsed GNN data
        output_dir: Output directory
        model_name: Name of the model

    Returns:
        List of generated visualization file paths
    """
    visualizations = []

    if not NETWORKX_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        return visualizations

    try:
        # Create network graph
        G = nx.DiGraph()

        # Extract variables from parsed data
        variables = parsed_data.get("variables", [])

        # Ensure variables is a list of dictionaries
        if not isinstance(variables, list):
            variables = []

        # Extract connections from parsed data
        connections = parsed_data.get("connections", [])

        # Ensure connections is a list
        if not isinstance(connections, list):
            connections = []

        # Add nodes (variables) with proper type information
        for var_info in variables:
            if isinstance(var_info, dict):
                var_name = var_info.get("name", "unknown")
                var_type = var_info.get("var_type", "unknown")
                dimensions = var_info.get("dimensions", [])
                description = var_info.get("description", "")

                # Create comprehensive node attributes
                node_attrs = {
                    'type': var_type,
                    'dimensions': dimensions,
                    'description': description,
                    'size': max(1, min(10, len(dimensions) * 2)),
                }
                G.add_node(var_name, **node_attrs)

        # Add edges (connections) - handle both old and new connection formats
        for conn_info in connections:
            if isinstance(conn_info, dict):
                normalized_conn = _normalize_connection_format(conn_info)
                source_vars = normalized_conn.get("source_variables", [])
                target_vars = normalized_conn.get("target_variables", [])

                for source_var in source_vars:
                    for target_var in target_vars:
                        if source_var and target_var and source_var != target_var:
                            source_type = None
                            target_type = None

                            for var in variables:
                                if isinstance(var, dict) and var.get("name") == source_var:
                                    source_type = var.get("var_type", "unknown")
                                if isinstance(var, dict) and var.get("name") == target_var:
                                    target_type = var.get("var_type", "unknown")

                            conn_type = _determine_connection_type(source_var, target_var, source_type, target_type)

                            edge_attrs = {
                                'connection_type': conn_type,
                                'source_location': normalized_conn.get("source_location"),
                                'metadata': normalized_conn.get("metadata", {}),
                                'source_type': source_type,
                                'target_type': target_type,
                                'weight': 1.0,
                                'style': _get_edge_style(conn_type)
                            }
                            G.add_edge(source_var, target_var, **edge_attrs)

        if len(G.nodes()) > 0:
            plt.figure(figsize=(14, 12))

            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

            node_sizes = [G.nodes[node].get('size', 5) * 100 for node in G.nodes()]
            node_types = [G.nodes[node].get('type', 'unknown') for node in G.nodes()]

            type_colors = {
                'hidden_state': 'skyblue',
                'observation': 'lightgreen',
                'policy': 'lightcoral',
                'action': 'gold',
                'prior_vector': 'plum',
                'likelihood_matrix': 'orange',
                'transition_matrix': 'pink',
                'preference_vector': 'lightblue',
                'unknown': 'gray'
            }

            node_colors = [type_colors.get(node_type, 'gray') for node_type in node_types]

            edge_groups = {}
            for edge in G.edges(data=True):
                conn_type = edge[2].get('connection_type', 'generic_causal')
                if conn_type not in edge_groups:
                    edge_groups[conn_type] = []
                edge_groups[conn_type].append((edge[0], edge[1]))

            for conn_type, edges in edge_groups.items():
                style = _get_edge_style(conn_type)
                edge_list = [(u, v) for u, v in edges]

                if edge_list:
                    nx.draw_networkx_edges(G, pos,
                                         edgelist=edge_list,
                                         edge_color=style['color'],
                                         width=style['width'],
                                         alpha=style['alpha'],
                                         arrows=True,
                                         arrowsize=20,
                                         style=style['style'])

            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

            legend_elements = [plt.Rectangle((0,0),1,1, fc=color, label=var_type)
                              for var_type, color in type_colors.items()]
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))

            plt.title(f'Bayesian Graphical Model: {model_name}\nPOMDP Active Inference Network', fontsize=16, fontweight='bold')
            plt.axis('off')
            _safe_tight_layout()

            network_path = output_dir / f"{model_name}_network_graph.png"
            plt.savefig(network_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append(str(network_path))

            stats = _generate_network_statistics(variables, connections)
            stats_path = output_dir / f"{model_name}_network_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            visualizations.append(str(stats_path))

            if 'plotly' in globals() and plotly:
                try:
                    interactive_path = _generate_interactive_network(G, output_dir / f"{model_name}_network_interactive.html")
                    if interactive_path:
                        visualizations.append(str(interactive_path))
                except Exception as e:
                    print(f"Failed to generate interactive network: {e}")

        else:
            print(f"Warning: No valid nodes found for network visualization of {model_name}")

    except Exception as e:
        print(f"Error generating network visualizations for {model_name}: {e}")
        import traceback
        traceback.print_exc()

    return visualizations


def _normalize_connection_format(conn_info: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize connection format to handle both old and new formats."""
    if "source_variables" in conn_info and "target_variables" in conn_info:
        return conn_info
    elif "source" in conn_info and "target" in conn_info:
        return {
            "source_variables": [conn_info["source"]],
            "target_variables": [conn_info["target"]],
            **{k: v for k, v in conn_info.items() if k not in ["source", "target"]}
        }
    else:
        return conn_info


def _determine_connection_type(source_var: str, target_var: str, source_type: str = None, target_type: str = None) -> str:
    """Determine the semantic type of connection between variables."""
    if source_type and target_type:
        if source_type == "hidden_state" and target_type == "hidden_state":
            return "state_transition"
        elif source_type == "hidden_state" and target_type == "observation":
            return "observation_generation"
        elif source_type == "hidden_state" and "transition" in target_type:
            return "state_action_influence"
        elif source_type == "action" and target_type == "hidden_state":
            return "action_effect"
        elif source_type == "policy" and target_type == "action":
            return "policy_selection"
        elif source_type == "prior_vector" and target_type == "hidden_state":
            return "prior_influence"
        elif source_type == "hidden_state" and "likelihood" in target_type:
            return "likelihood_influence"
        elif "free_energy" in source_type or "free_energy" in target_type:
            return "energy_flow"

    if source_var == "s" and target_var in ["A", "o"]:
        return "state_observation"
    elif source_var in ["s", "s_prime"] and target_var == "B":
        return "state_transition_matrix"
    elif source_var == "C" and target_var == "G":
        return "preference_energy"
    elif source_var == "E" and target_var == "\u03c0":
        return "habit_policy"
    elif source_var == "\u03c0" and target_var == "u":
        return "policy_action"

    return "generic_causal"


def _get_edge_style(connection_type: str) -> Dict[str, Any]:
    """Get visual styling for different connection types."""
    style_map = {
        "state_transition": {"color": "blue", "width": 3, "alpha": 0.8, "style": "solid"},
        "observation_generation": {"color": "green", "width": 2, "alpha": 0.7, "style": "dashed"},
        "state_action_influence": {"color": "orange", "width": 2, "alpha": 0.7, "style": "dotted"},
        "action_effect": {"color": "red", "width": 3, "alpha": 0.8, "style": "solid"},
        "policy_selection": {"color": "purple", "width": 2, "alpha": 0.7, "style": "solid"},
        "prior_influence": {"color": "cyan", "width": 2, "alpha": 0.6, "style": "dashed"},
        "likelihood_influence": {"color": "magenta", "width": 2, "alpha": 0.6, "style": "dotted"},
        "energy_flow": {"color": "yellow", "width": 1, "alpha": 0.5, "style": "dashed"},
        "preference_energy": {"color": "lime", "width": 2, "alpha": 0.7, "style": "solid"},
        "habit_policy": {"color": "pink", "width": 2, "alpha": 0.7, "style": "solid"},
        "generic_causal": {"color": "gray", "width": 1, "alpha": 0.5, "style": "solid"}
    }

    return style_map.get(connection_type, style_map["generic_causal"])


def _generate_network_statistics(variables: list, connections: list) -> Dict[str, Any]:
    """Generate comprehensive network statistics."""
    stats = {
        "total_variables": len(variables),
        "total_connections": len(connections),
        "variable_types": {},
        "connection_types": {},
        "network_properties": {}
    }

    for var_info in variables:
        if isinstance(var_info, dict):
            var_type = var_info.get("var_type", "unknown")
            stats["variable_types"][var_type] = stats["variable_types"].get(var_type, 0) + 1

    for conn_info in connections:
        if isinstance(conn_info, dict):
            normalized_conn = _normalize_connection_format(conn_info)
            source_vars = normalized_conn.get("source_variables", [])
            target_vars = normalized_conn.get("target_variables", [])

            for source_var in source_vars:
                for target_var in target_vars:
                    if source_var != target_var:
                        conn_type = f"{source_var}->{target_var}"
                        stats["connection_types"][conn_type] = stats["connection_types"].get(conn_type, 0) + 1

    if NETWORKX_AVAILABLE:
        try:
            simple_G = nx.DiGraph()
            for conn_info in connections:
                normalized_conn = _normalize_connection_format(conn_info)
                source_vars = normalized_conn.get("source_variables", [])
                target_vars = normalized_conn.get("target_variables", [])
                for source_var in source_vars:
                    for target_var in target_vars:
                        if source_var != target_var:
                            simple_G.add_edge(source_var, target_var)

            if len(simple_G.nodes()) > 0:
                stats["network_properties"] = {
                    "num_nodes": simple_G.number_of_nodes(),
                    "num_edges": simple_G.number_of_edges(),
                    "density": nx.density(simple_G),
                    "is_strongly_connected": nx.is_strongly_connected(simple_G) if len(simple_G.nodes()) > 1 else True,
                    "is_weakly_connected": nx.is_weakly_connected(simple_G) if len(simple_G.nodes()) > 1 else True,
                    "average_clustering": nx.average_clustering(simple_G)
                }
        except Exception as e:
            print(f"Error calculating network properties: {e}")

    return stats


def _generate_interactive_network(G, output_path: Path) -> bool:
    """Generate an interactive network visualization using plotly."""
    if not PLOTLY_AVAILABLE or not go:
        return False

    try:
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_info = []
        node_types = nx.get_node_attributes(G, 'type')

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_info.append(f'{node}<br>Type: {node_types.get(node, "unknown")}<br>Connections: {G.degree(node)}')

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=list(G.nodes()),
            textposition="top center",
            hovertext=node_info,
            marker=dict(
                size=[G.degree(node) * 10 + 20 for node in G.nodes()],
                color=['lightblue' if node_types.get(node, 'unknown') == 'unknown' else 'orange' for node in G.nodes()],
                line=dict(width=2)
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Interactive Network Graph ({len(G.nodes())} nodes, {len(G.edges())} edges)',
                           title_font_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[dict(
                               text="Hover over nodes and edges for details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002)],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )

        fig.write_html(str(output_path))
        return True

    except Exception as e:
        print(f"Error generating interactive network: {e}")
        return False


def _generate_3d_surface_plot(matrix: np.ndarray, matrix_name: str, output_path: Path) -> bool:
    """Generate a 3D surface plot for a matrix."""
    if not MATPLOTLIB_AVAILABLE:
        return False

    try:
        if matrix.size > 1000:
            step = max(1, matrix.shape[0] // 20)
            x = np.arange(0, matrix.shape[1], step)
            y = np.arange(0, matrix.shape[0], step)
            X, Y = np.meshgrid(x, y)
            Z = matrix[::step, ::step]
        else:
            x = np.arange(matrix.shape[1])
            y = np.arange(matrix.shape[0])
            X, Y = np.meshgrid(x, y)
            Z = matrix

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        ax.set_title(f'3D Surface Plot: {matrix_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Value')

        _safe_tight_layout()
        _save_plot_safely(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        print(f"Error generating 3D surface plot for {matrix_name}: {e}")
        plt.close()
        return False
