#!/usr/bin/env python3
"""
Step 8: Visualization Processing

This step handles visualization processing for GNN files using the visualization module.
"""

import sys
import json
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

def create_network_graph(variables: List[Dict], connections: List[Dict]) -> nx.DiGraph:
    """Create a NetworkX graph from variables and connections."""
    G = nx.DiGraph()
    
    # Add nodes (variables) - handle both old and new field names
    for var in variables:
        var_name = var.get("name", "unknown")
        var_type = var.get("var_type", var.get("type", "unknown"))
        dimensions = var.get("dimensions", [1])
        data_type = var.get("data_type", "unknown")
        description = var.get("description", "")
        
        G.add_node(var_name, 
                  type=var_type,
                  dimensions=dimensions,
                  data_type=data_type,
                  description=description)
    
    # Add edges (connections) - handle both old and new field names
    for conn in connections:
        # Handle both old format (source/target) and new format (source_variables/target_variables)
        sources = conn.get("source_variables", conn.get("source", []))
        targets = conn.get("target_variables", conn.get("target", []))
        conn_type = conn.get("connection_type", conn.get("type", "unknown"))
        description = conn.get("description", "")
        
        for source in sources:
            for target in targets:
                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target, 
                              type=conn_type,
                              description=description)
    
    return G

def generate_network_plot(G: nx.DiGraph, output_path: Path, title: str = "GNN Model Network"):
    """Generate a network visualization plot."""
    if len(G.nodes()) == 0:
        # Create a placeholder plot if no nodes
        plt.figure(figsize=(12, 10))
        plt.text(0.5, 0.5, 'No variables found in model', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=16, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    plt.figure(figsize=(12, 10))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw nodes with different colors based on type
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        node_data = G.nodes[node]
        node_type = node_data.get("type", "unknown")
        
        # Color mapping
        color_map = {
            "hidden_state": "lightblue",
            "observation": "lightgreen", 
            "action": "lightcoral",
            "policy": "lightyellow",
            "likelihood_matrix": "lightpink",
            "transition_matrix": "lightgray",
            "preference_vector": "lightcyan",
            "prior_vector": "lightsteelblue"
        }
        
        node_colors.append(color_map.get(node_type, "white"))
        
        # Size based on dimensions
        dimensions = node_data.get("dimensions", [1])
        total_elements = 1
        for dim in dimensions:
            total_elements *= dim
        node_sizes.append(min(1000, max(100, total_elements * 10)))
    
    # Draw the graph
    nx.draw(G, pos, 
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=True,
            font_size=8,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            width=1,
            alpha=0.8)
    
    # Add title
    plt.title(title, fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = []
    for node_type, color in {
        "hidden_state": "lightblue",
        "observation": "lightgreen", 
        "action": "lightcoral",
        "policy": "lightyellow",
        "likelihood_matrix": "lightpink",
        "transition_matrix": "lightgray",
        "preference_vector": "lightcyan",
        "prior_vector": "lightsteelblue"
    }.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, 
                                        label=node_type.replace('_', ' ').title()))
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_variable_type_chart(variables: List[Dict], output_path: Path):
    """Generate a pie chart of variable types."""
    type_counts = {}
    for var in variables:
        # Handle both old and new field names
        var_type = var.get("var_type", var.get("type", "unknown"))
        type_counts[var_type] = type_counts.get(var_type, 0) + 1
    
    if not type_counts:
        # Create placeholder if no data
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, 'No variables found', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=16, fontweight='bold')
        plt.title('Variable Type Distribution', fontsize=16, fontweight='bold')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    plt.figure(figsize=(10, 8))
    
    labels = [t.replace('_', ' ').title() for t in type_counts.keys()]
    sizes = list(type_counts.values())
    colors = plt.cm.Set3(range(len(sizes)))
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Variable Type Distribution', fontsize=16, fontweight='bold')
    plt.axis('equal')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_dimension_analysis(variables: List[Dict], output_path: Path):
    """Generate dimension analysis visualization."""
    dimension_counts = {}
    for var in variables:
        dimensions = var.get("dimensions", [1])
        dim_key = f"{len(dimensions)}D"
        dimension_counts[dim_key] = dimension_counts.get(dim_key, 0) + 1
    
    if not dimension_counts:
        # Create placeholder if no data
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No variables found', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=16, fontweight='bold')
        plt.title('Variable Dimension Distribution', fontsize=16, fontweight='bold')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    plt.figure(figsize=(10, 6))
    
    labels = list(dimension_counts.keys())
    values = list(dimension_counts.values())
    
    bars = plt.bar(labels, values, color='skyblue', alpha=0.7)
    plt.title('Variable Dimension Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Number of Variables')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(value), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_connection_analysis(connections: List[Dict], output_path: Path):
    """Generate connection analysis visualization."""
    connection_types = {}
    for conn in connections:
        # Handle both old and new field names
        conn_type = conn.get("connection_type", conn.get("type", "unknown"))
        connection_types[conn_type] = connection_types.get(conn_type, 0) + 1
    
    if not connection_types:
        # Create placeholder if no data
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No connections found', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=16, fontweight='bold')
        plt.title('Connection Type Distribution', fontsize=16, fontweight='bold')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    plt.figure(figsize=(10, 6))
    
    labels = [t.replace('_', ' ').title() for t in connection_types.keys()]
    values = list(connection_types.values())
    
    bars = plt.bar(labels, values, color='lightcoral', alpha=0.7)
    plt.title('Connection Type Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Connection Type')
    plt.ylabel('Number of Connections')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(value), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_model_overview_chart(model_data: Dict, output_path: Path):
    """Generate a comprehensive model overview chart."""
    variables = model_data.get('variables', [])
    connections = model_data.get('connections', [])
    parameters = model_data.get('parameters', [])
    equations = model_data.get('equations', [])
    
    # Count different types
    var_types = {}
    for var in variables:
        var_type = var.get("var_type", var.get("type", "unknown"))
        var_types[var_type] = var_types.get(var_type, 0) + 1
    
    conn_types = {}
    for conn in connections:
        conn_type = conn.get("connection_type", conn.get("type", "unknown"))
        conn_types[conn_type] = conn_types.get(conn_type, 0) + 1
    
    # Create overview chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Variable types
    if var_types:
        labels = [t.replace('_', ' ').title() for t in var_types.keys()]
        values = list(var_types.values())
        ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Variable Types', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No variables', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Variable Types', fontweight='bold')
    
    # Connection types
    if conn_types:
        labels = [t.replace('_', ' ').title() for t in conn_types.keys()]
        values = list(conn_types.values())
        ax2.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Connection Types', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No connections', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Connection Types', fontweight='bold')
    
    # Component counts
    components = ['Variables', 'Connections', 'Parameters', 'Equations']
    counts = [len(variables), len(connections), len(parameters), len(equations)]
    bars = ax3.bar(components, counts, color=['skyblue', 'lightcoral', 'lightgreen', 'lightyellow'])
    ax3.set_title('Model Components', fontweight='bold')
    ax3.set_ylabel('Count')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    # Dimension distribution
    dim_counts = {}
    for var in variables:
        dimensions = var.get("dimensions", [1])
        dim_key = f"{len(dimensions)}D"
        dim_counts[dim_key] = dim_counts.get(dim_key, 0) + 1
    
    if dim_counts:
        labels = list(dim_counts.keys())
        values = list(dim_counts.values())
        ax4.bar(labels, values, color='lightsteelblue')
        ax4.set_title('Variable Dimensions', fontweight='bold')
        ax4.set_xlabel('Dimensions')
        ax4.set_ylabel('Count')
    else:
        ax4.text(0.5, 0.5, 'No variables', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Variable Dimensions', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main visualization processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("8_visualization.py")
    
    # Setup logging
    logger = setup_step_logging("visualization", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("8_visualization.py", Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, "Processing visualization")
        
        # Load parsed GNN data from previous step
        gnn_output_dir = get_output_dir_for_script("3_gnn.py", Path(args.output_dir))
        gnn_results_file = gnn_output_dir / "gnn_processing_results.json"
        
        if not gnn_results_file.exists():
            log_step_error(logger, "GNN processing results not found. Run step 3 first.")
            return 1
        
        with open(gnn_results_file, 'r') as f:
            gnn_results = json.load(f)
        
        logger.info(f"Loaded {len(gnn_results['processed_files'])} parsed GNN files")
        
        # Visualization results
        viz_results = {
            "timestamp": datetime.now().isoformat(),
            "source_directory": str(args.target_dir),
            "output_directory": str(output_dir),
            "files_visualized": [],
            "summary": {
                "total_files": 0,
                "successful_visualizations": 0,
                "failed_visualizations": 0,
                "total_images_generated": 0
            }
        }
        
        for file_result in gnn_results["processed_files"]:
            if not file_result["parse_success"]:
                continue
            
            file_name = file_result["file_name"]
            logger.info(f"Generating visualizations for: {file_name}")
            
            # Create file-specific output directory
            file_output_dir = output_dir / file_name.replace('.md', '')
            file_output_dir.mkdir(exist_ok=True)
            
            file_viz_result = {
                "file_name": file_name,
                "file_path": file_result["file_path"],
                "visualizations": {},
                "success": True
            }
            
            try:
                # Load the parsed model data from the individual file
                parsed_file_path = gnn_output_dir / file_name.replace('.md', '') / f"{file_name.replace('.md', '')}_parsed.json"
                if parsed_file_path.exists():
                    with open(parsed_file_path, 'r') as f:
                        model_data = json.load(f)
                else:
                    # Fallback to file_result data
                    model_data = {
                        "variables": file_result.get("variables", []),
                        "connections": file_result.get("connections", []),
                        "parameters": file_result.get("parameters", []),
                        "equations": file_result.get("equations", []),
                        "model_name": file_result.get("model_name", file_name)
                    }
                
                # Create NetworkX graph
                G = create_network_graph(model_data.get('variables', []), model_data.get('connections', []))
                
                # Generate network plot
                network_plot_path = file_output_dir / "network_plot.png"
                generate_network_plot(G, network_plot_path)
                file_viz_result["visualizations"]["network_plot"] = {
                    "path": str(network_plot_path),
                    "file_size": network_plot_path.stat().st_size
                }
                
                # Generate variable type chart
                var_type_chart_path = file_output_dir / "variable_type_chart.png"
                generate_variable_type_chart(model_data.get('variables', []), var_type_chart_path)
                file_viz_result["visualizations"]["variable_type_chart"] = {
                    "path": str(var_type_chart_path),
                    "file_size": var_type_chart_path.stat().st_size
                }
                
                # Generate dimension analysis
                dim_analysis_path = file_output_dir / "dimension_analysis.png"
                generate_dimension_analysis(model_data.get('variables', []), dim_analysis_path)
                file_viz_result["visualizations"]["dimension_analysis"] = {
                    "path": str(dim_analysis_path),
                    "file_size": dim_analysis_path.stat().st_size
                }
                
                # Generate connection analysis
                conn_analysis_path = file_output_dir / "connection_analysis.png"
                generate_connection_analysis(model_data.get('connections', []), conn_analysis_path)
                file_viz_result["visualizations"]["connection_analysis"] = {
                    "path": str(conn_analysis_path),
                    "file_size": conn_analysis_path.stat().st_size
                }
                
                # Generate model overview chart
                model_overview_path = file_output_dir / "model_overview.png"
                generate_model_overview_chart(model_data, model_overview_path)
                file_viz_result["visualizations"]["model_overview"] = {
                    "path": str(model_overview_path),
                    "file_size": model_overview_path.stat().st_size
                }
                
                logger.info(f"Generated {len(file_viz_result['visualizations'])} visualizations for {file_name}")
                
            except Exception as e:
                file_viz_result["success"] = False
                file_viz_result["error"] = str(e)
                logger.error(f"Visualization failed for {file_name}: {e}")
            
            viz_results["files_visualized"].append(file_viz_result)
            
            # Update summary
            viz_results["summary"]["total_files"] += 1
            if file_viz_result["success"]:
                viz_results["summary"]["successful_visualizations"] += 1
                viz_results["summary"]["total_images_generated"] += len(file_viz_result["visualizations"])
            else:
                viz_results["summary"]["failed_visualizations"] += 1
        
        # Save visualization results
        results_file = output_dir / "visualization_results.json"
        with open(results_file, 'w') as f:
            json.dump(viz_results, f, indent=2, default=str)
        
        # Save summary
        summary_file = output_dir / "visualization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(viz_results["summary"], f, indent=2)
        
        # Determine success
        success = viz_results["summary"]["successful_visualizations"] > 0
        
        if success:
            log_step_success(logger, f"Generated {viz_results['summary']['total_images_generated']} visualizations for {viz_results['summary']['successful_visualizations']} files")
            return 0
        else:
            log_step_error(logger, "Visualization processing failed")
            return 1
            
    except Exception as e:
        log_step_error(logger, f"Visualization processing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
