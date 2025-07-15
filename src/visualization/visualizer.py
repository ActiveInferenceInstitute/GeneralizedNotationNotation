"""
GNN Visualizer Module

This module provides the main visualization functionality for GNN models.
It generates comprehensive state-space visualizations of GNN files and models.
"""

import os
import json
import time
import datetime
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import re
import numpy as np

from .parser import GNNParser
from .matrix_visualizer import MatrixVisualizer
from .ontology_visualizer import OntologyVisualizer


class GNNVisualizer:
    """
    Visualizer for GNN models.
    
    This class provides methods to visualize GNN models from parsed GNN files.
    It generates various visualizations of the model's state space, connections,
    and other properties.
    """
    
    def __init__(self, output_dir: Optional[str] = None, project_root: Optional[Union[str, Path]] = None):
        """
        Initialize the GNN visualizer.
        
        Args:
            output_dir: Directory where output visualizations will be saved.
                        If None, creates a timestamped directory in the current working directory.
            project_root: Optional path to the project root for making file paths relative.
        """
        self.parser = GNNParser()
        self.matrix_visualizer = MatrixVisualizer()
        self.ontology_visualizer = OntologyVisualizer()
        
        # Create timestamped output directory if not provided
        if output_dir is None:
            # Default to project_root/output
            # Assumes script is run from a subdirectory of the project root (e.g. src/)
            # or that current working directory is project root.
            # Path.cwd() will be /path/to/GeneralizedNotationNotation/src
            # Path.cwd().parent will be /path/to/GeneralizedNotationNotation
            project_root_output_dir = Path.cwd().parent / 'output'
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Keep timestamp for uniqueness if multiple runs without specific output dir
            output_dir = project_root_output_dir / f'gnn_visualization_{timestamp}'
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.project_root = Path(project_root).resolve() if project_root else None
        
    def visualize_file(self, file_path: str) -> str:
        """
        Generate visualizations for a GNN file.
        
        Args:
            file_path: Path to the GNN file to visualize
            
        Returns:
            Path to the directory containing generated visualizations
        """
        try:
            # Parse the GNN file
            parsed_data = self.parser.parse_file(file_path)
            
            # Create subdirectory for this file
            file_name = Path(file_path).stem
            file_output_dir = self.output_dir / file_name
            file_output_dir.mkdir(exist_ok=True)
            
            # Generate and save model metadata
            self._save_model_metadata(parsed_data, file_output_dir)
            
            # Generate basic text-based visualizations regardless of parsing success
            self._create_basic_text_visualization(parsed_data, file_path, file_output_dir)
            
            # Generate different visualizations if we have parsed structured data
            print(f"Checking for variables in {file_path}...")
            if 'Variables' in parsed_data:
                print(f"Found {len(parsed_data['Variables'])} variables: {list(parsed_data['Variables'].keys())}")
                self._visualize_state_space(parsed_data, file_output_dir)
            else:
                print(f"No variables found in {file_path}")
                # Write available sections
                print(f"Available sections: {list(parsed_data.keys())}")
                
                # Try to extract state space from the StateSpaceBlock
                if 'StateSpaceBlock' in parsed_data:
                    print(f"StateSpaceBlock content: {parsed_data['StateSpaceBlock'][:100]}...")
                    self._process_state_space_and_visualize(parsed_data, file_output_dir)
            
            print(f"Checking for edges in {file_path}...")
            if 'Edges' in parsed_data:
                print(f"Found {len(parsed_data['Edges'])} edges")
                self._visualize_connections(parsed_data, file_output_dir)
            else:
                print(f"No edges found in {file_path}")
                # Try to extract connections from the Connections section
                if 'Connections' in parsed_data:
                    print(f"Connections content: {parsed_data['Connections'][:100]}...")
                    self._process_connections_and_visualize(parsed_data, file_output_dir)
            
            # Generate matrix visualizations
            if 'InitialParameterization' in parsed_data:
                print(f"[GNNVisualizer] Found 'InitialParameterization' section for {file_name}. Attempting matrix visualization.")
                if parsed_data['InitialParameterization'].strip(): # Check if content is not just whitespace
                    self.matrix_visualizer.visualize_all_matrices(parsed_data, file_output_dir)
                else:
                    print(f"[GNNVisualizer] 'InitialParameterization' section for {file_name} is empty. Skipping matrix visualization.")
            else:
                print(f"[GNNVisualizer] 'InitialParameterization' section NOT FOUND for {file_name}. Skipping matrix visualization.")
            
            # Generate ontology visualizations
            if 'ActInfOntologyAnnotation' in parsed_data:
                print(f"[GNNVisualizer] Found 'ActInfOntologyAnnotation' section for {file_name}. Attempting ontology visualization.")
                if parsed_data['ActInfOntologyAnnotation'].strip(): # Check if content is not just whitespace
                    self.ontology_visualizer.visualize_ontology(parsed_data, file_output_dir)
                else:
                    print(f"[GNNVisualizer] 'ActInfOntologyAnnotation' section for {file_name} is empty. Skipping ontology visualization.")
            else:
                print(f"[GNNVisualizer] 'ActInfOntologyAnnotation' section NOT FOUND for {file_name}. Skipping ontology visualization.")
            
            if 'Variables' in parsed_data and 'Edges' in parsed_data:
                self._visualize_combined(parsed_data, file_output_dir)
            
            return str(file_output_dir)
        except Exception as e:
            # Create a subdirectory even for failed files
            file_name = Path(file_path).stem
            file_output_dir = self.output_dir / file_name
            file_output_dir.mkdir(exist_ok=True)
            
            # Create a basic report of the error
            with open(file_output_dir / 'parsing_error.txt', 'w') as f:
                f.write(f"Error parsing {file_path}: {str(e)}\n")
            
            # Create a basic text visualization
            self._create_basic_text_visualization({}, file_path, file_output_dir)
            
            # Re-raise the exception for higher-level handling
            raise
    
    def _process_state_space_and_visualize(self, parsed_data: Dict[str, Any], output_dir: Path) -> None:
        """Process state space and generate visualization."""
        try:
            # Process state space
            self.parser._process_state_space(parsed_data)
            
            # Visualize if we have variables
            if 'Variables' in parsed_data and parsed_data['Variables']:
                print(f"Successfully processed state space, found {len(parsed_data['Variables'])} variables")
                self._visualize_state_space(parsed_data, output_dir)
        except Exception as e:
            print(f"Error processing state space: {e}")
    
    def _process_connections_and_visualize(self, parsed_data: Dict[str, Any], output_dir: Path) -> None:
        """Process connections and generate visualization."""
        try:
            # Process connections
            self.parser._process_connections(parsed_data)
            
            # Visualize if we have edges
            if 'Edges' in parsed_data and parsed_data['Edges']:
                print(f"Successfully processed connections, found {len(parsed_data['Edges'])} edges")
                self._visualize_connections(parsed_data, output_dir)
        except Exception as e:
            print(f"Error processing connections: {e}")
    
    def visualize_directory(self, dir_path: str) -> str:
        """
        Generate visualizations for all GNN files in a directory.
        
        Args:
            dir_path: Path to directory containing GNN files
            
        Returns:
            Path to the directory containing all generated visualizations
        """
        dir_path = Path(dir_path)
        
        # Process all markdown files in the directory
        for file_path in dir_path.glob('*.md'):
            try:
                self.visualize_file(str(file_path))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return str(self.output_dir)
    
    def _create_basic_text_visualization(self, parsed_data: Dict[str, Any], file_path: str, output_dir: Path) -> None:
        """Create a simple text-based visualization of the file."""
        # Read the raw file content
        raw_file_content = Path(file_path).read_text()
        
        # Determine display path for the report
        display_file_path = Path(file_path).name # Default to just name
        if self.project_root:
            try:
                display_file_path = Path(file_path).resolve().relative_to(self.project_root)
            except ValueError:
                # Keep as name if not under project_root for some reason
                pass 

        # Create a simple text report
        with open(output_dir / 'file_content.md', 'w') as f:
            f.write(f"# GNN File: {display_file_path}\\n\\n")
            f.write("## Raw File Content\\n\\n")
            f.write("```\\n")
            f.write(raw_file_content)
            f.write("\\n```\\n\\n")
            
            # Add parsed sections if available
            if parsed_data:
                f.write("## Parsed Sections\n\n")
                for section, content in parsed_data.items():
                    if section not in ['Variables', 'Edges']:  # Skip processed sections
                        f.write(f"### {section}\n\n")
                        f.write("```\n")
                        f.write(str(content))
                        f.write("\n```\n\n")
    
    def _save_model_metadata(self, parsed_data: Dict[str, Any], output_dir: Path) -> None:
        """Save model metadata as JSON for reference."""
        # Extract relevant metadata
        metadata = {
            'ModelName': parsed_data.get('ModelName', ''),
            'ModelAnnotation': parsed_data.get('ModelAnnotation', ''),
            'GNNVersionAndFlags': parsed_data.get('GNNVersionAndFlags', ''),
            'Time': parsed_data.get('Time', ''),
            'ActInfOntologyAnnotation': parsed_data.get('ActInfOntologyAnnotation', '')
        }
        
        # Save as JSON
        with open(output_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Also save full parsed data for reference
        with open(output_dir / 'full_model_data.json', 'w') as f:
            # Convert to serializable format
            try:
                serializable_data = {}
                for k, v in parsed_data.items():
                    if k not in ['Variables', 'Edges']:  # Skip complex objects
                        serializable_data[k] = str(v)
                json.dump(serializable_data, f, indent=2)
            except Exception as e:
                # Fallback to simple format
                json.dump({"error": f"Failed to serialize data: {str(e)}"}, f)
    
    def _visualize_state_space(self, parsed_data: Dict[str, Any], output_dir: Path) -> None:
        """Generate visualization of the state space variables."""
        if 'Variables' not in parsed_data or not parsed_data['Variables']:
            return
            
        variables = parsed_data['Variables']
        
        # Create figure and table
        fig, ax = plt.subplots(figsize=(10, max(5, len(variables) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        for var_name, var_info in variables.items():
            dimensions = 'x'.join(str(d) for d in var_info.get('dimensions', [])) if var_info.get('dimensions') else ''
            var_type = var_info.get('type', '') or ''
            comment = var_info.get('comment', '') or ''
            table_data.append([var_name, dimensions, var_type, comment])
        
        # Create the table
        if table_data:
            table = ax.table(
                cellText=table_data,
                colLabels=['Variable', 'Dimensions', 'Type', 'Description'],
                loc='center',
                cellLoc='left',
                colWidths=[0.15, 0.15, 0.15, 0.55]
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
        else:
            ax.text(0.5, 0.5, "No state space variables found", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12)
        
        # Add title
        plt.title('State Space Variables', fontsize=14, fontweight='bold', pad=20)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_dir / 'state_space.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"State space visualization saved to {output_dir / 'state_space.png'}")
    
    def _visualize_connections(self, parsed_data: Dict[str, Any], output_dir: Path) -> None:
        """Generate visualization of the connections/edges in the model."""
        if 'Edges' not in parsed_data or not parsed_data['Edges']:
            return
            
        edges = parsed_data['Edges']
        
        # Create directed graph
        G = nx.DiGraph()
        
        try:
            # Add nodes and edges
            for edge in edges:
                source = edge.get('source', '')
                target = edge.get('target', '')
                if not source or not target:
                    continue
                    
                directed = edge.get('directed', True)
                constraint = edge.get('constraint', None)
                comment = edge.get('comment', None)
                
                G.add_node(source)
                G.add_node(target)
                
                if directed:
                    G.add_edge(source, target, constraint=constraint, comment=comment)
                else:
                    # For undirected edges in a directed graph, add edges in both directions
                    G.add_edge(source, target, constraint=constraint, comment=comment)
                    G.add_edge(target, source, constraint=constraint, comment=comment)
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            if G.number_of_nodes() > 0:
                # Set node positions using spring layout
                pos = nx.spring_layout(G, seed=42)
                
                # Draw nodes
                nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.8)
                
                # Draw edges
                nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, arrowsize=20)
                
                # Draw labels
                nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
                
                # Add edge labels for constraints
                edge_labels = {(edge.get('source', ''), edge.get('target', '')): edge.get('constraint', '') 
                              for edge in edges if edge.get('constraint')}
                if edge_labels:
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
            else:
                plt.text(0.5, 0.5, "No connections found", 
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=14)
            
            # Set title
            plt.title('Model Connections', fontsize=14, fontweight='bold')
            
            # Remove axis
            plt.axis('off')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(output_dir / 'connections.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Connections visualization saved to {output_dir / 'connections.png'}")
        except Exception as e:
            # Create error text figure if visualization fails
            plt.figure(figsize=(10, 5))
            plt.text(0.5, 0.5, f"Error generating connections visualization: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, wrap=True)
            plt.axis('off')
            plt.savefig(output_dir / 'connections_error.png', dpi=150)
            plt.close()
    
    def _visualize_combined(self, parsed_data: Dict[str, Any], output_dir: Path) -> None:
        """Generate a combined visualization of the model."""
        try:
            # Create a comprehensive visualization that combines state space and connections
            if 'Variables' not in parsed_data or not parsed_data['Variables'] or 'Edges' not in parsed_data or not parsed_data['Edges']:
                return
                
            variables = parsed_data['Variables']
            edges = parsed_data['Edges']
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Left subplot: Variable details
            ax1.axis('tight')
            ax1.axis('off')
            
            # Prepare table data
            table_data = []
            for var_name, var_info in variables.items():
                dimensions = 'x'.join(str(d) for d in var_info.get('dimensions', [])) if var_info.get('dimensions') else ''
                var_type = var_info.get('type', '') or ''
                table_data.append([var_name, dimensions, var_type])
            
            # Create the table
            if table_data:
                table = ax1.table(
                    cellText=table_data,
                    colLabels=['Variable', 'Dimensions', 'Type'],
                    loc='center',
                    cellLoc='left'
                )
                
                # Style the table
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
            else:
                ax1.text(0.5, 0.5, "No state space variables found", 
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=12)
            
            ax1.set_title('State Space Variables', fontsize=14, fontweight='bold')
            
            # Right subplot: Connections graph
            ax2.axis('off')
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Add nodes and edges
            valid_edges = []
            for edge in edges:
                source = edge.get('source', '')
                target = edge.get('target', '')
                if not source or not target:
                    continue
                    
                G.add_node(source)
                G.add_node(target)
                G.add_edge(source, target, directed=edge.get('directed', True))
                valid_edges.append(edge)
            
            if G.number_of_nodes() > 0:
                # Set node positions using spring layout
                pos = nx.spring_layout(G, seed=42)
                
                # Draw nodes
                nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=700, node_color='lightblue', alpha=0.8)
                
                # Draw edges with different styles for directed and undirected
                directed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('directed', True)]
                undirected_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('directed', True)]
                
                if directed_edges:
                    nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=directed_edges, 
                                        width=1.5, alpha=0.7, arrowsize=20)
                if undirected_edges:
                    nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=undirected_edges, 
                                        width=1.5, alpha=0.7, arrowstyle='-')
                
                # Draw labels
                nx.draw_networkx_labels(G, pos, ax=ax2, font_size=12, font_family='sans-serif')
            else:
                ax2.text(0.5, 0.5, "No connections found", 
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=14)
            
            ax2.set_title('Model Connections', fontsize=14, fontweight='bold')
            
            # Set overall title
            model_name = self._extract_model_name(parsed_data)
            fig.suptitle(model_name, fontsize=16, fontweight='bold')
            
            # Save figure
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
            plt.savefig(output_dir / 'combined_visualization.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Combined visualization saved to {output_dir / 'combined_visualization.png'}")
        except Exception as e:
            # Create error text figure if visualization fails
            plt.figure(figsize=(10, 5))
            plt.text(0.5, 0.5, f"Error generating combined visualization: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, wrap=True)
            plt.axis('off')
            plt.savefig(output_dir / 'combined_visualization_error.png', dpi=150)
            plt.close()
        
    def _extract_model_name(self, parsed_data: Dict[str, Any]) -> str:
        """Extract a clean model name from the parsed data."""
        if 'ModelName' in parsed_data and parsed_data['ModelName']:
            # Remove Markdown formatting and clean up
            return parsed_data['ModelName'].replace('#', '').strip()
        return "GNN Model" 


def generate_graph_visualization(gnn_data: Dict[str, Any], output_path: str) -> bool:
    """
    Generate a graph visualization from GNN data.
    
    Args:
        gnn_data: Parsed GNN data dictionary
        output_path: Path where the visualization should be saved
    
    Returns:
        True if successful, False otherwise
    """
    try:
        visualizer = GNNVisualizer()
        visualizer._visualize_connections(gnn_data, Path(output_path).parent)
        return True
    except Exception as e:
        print(f"Error generating graph visualization: {e}")
        return False


def generate_matrix_visualization(gnn_data: Dict[str, Any], output_path: str) -> bool:
    """
    Generate matrix visualizations from GNN data.
    
    Args:
        gnn_data: Parsed GNN data dictionary
        output_path: Path where the visualization should be saved
    
    Returns:
        True if successful, False otherwise
    """
    try:
        visualizer = GNNVisualizer()
        visualizer.matrix_visualizer.visualize_all_matrices(gnn_data, Path(output_path).parent)
        return True
    except Exception as e:
        print(f"Error generating matrix visualization: {e}")
        return False


def create_visualization_report(gnn_file_path: str, output_dir: str) -> str:
    """
    Create a comprehensive visualization report for a GNN file.
    
    Args:
        gnn_file_path: Path to the GNN file
        output_dir: Output directory for visualizations
    
    Returns:
        Path to the generated report
    """
    try:
        visualizer = GNNVisualizer(output_dir=output_dir)
        result_path = visualizer.visualize_file(gnn_file_path)
        return result_path
    except Exception as e:
        print(f"Error creating visualization report: {e}")
        return ""


def visualize_gnn_model(gnn_content: str, model_name: str, output_dir: str) -> dict:
    """
    Visualize a GNN model from content string.
    
    Args:
        gnn_content: GNN model content as string
        model_name: Name of the model
        output_dir: Output directory for visualizations
    
    Returns:
        Dictionary with visualization result information
    """
    import tempfile
    
    try:
        # Create temporary file for parsing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(gnn_content)
            temp_path = f.name
        
        # Create visualizations
        visualizer = GNNVisualizer(output_dir=output_dir)
        result_path = visualizer.visualize_file(temp_path)
        
        return {
            "success": True,
            "model_name": model_name,
            "output_directory": result_path,
            "message": "Visualization generated successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "model_name": model_name,
            "error": str(e),
            "error_type": type(e).__name__
        }
    finally:
        # Clean up temporary file
        if 'temp_path' in locals():
            os.unlink(temp_path) 


def generate_visualizations(target_dir: Path, output_dir: Path, logger, recursive: bool = False):
    """Generate visualizations for GNN models."""
    log_step_start(logger, f"Generating visualizations for GNN files in: {target_dir}")
    
    # Use centralized output directory configuration
    viz_output_dir = get_output_dir_for_script("6_visualization.py", output_dir)
    
    try:
        # Create GNN visualizer instance
        gnn_visualizer = GNNVisualizer(output_dir=str(viz_output_dir))
        
        # Initialize results dictionary
        results = {'success': False, 'files_processed': 0}
        
        # Use performance tracking for visualization generation
        with performance_tracker.track_operation("generate_all_visualizations"):
            # Find GNN files
            if recursive:
                gnn_files = list(target_dir.rglob("*.md"))
            else:
                gnn_files = list(target_dir.glob("*.md"))
            
            log_step_success(logger, f"Found {len(gnn_files)} GNN files to visualize")
            
            # Process each file
            processed_count = 0
            for gnn_file in gnn_files:
                try:
                    output_path = gnn_visualizer.visualize_file(str(gnn_file))
                    log_step_success(logger, f"Generated visualization for {gnn_file.name}: {output_path}")
                    processed_count += 1
                except Exception as e:
                    log_step_warning(logger, f"Failed to visualize {gnn_file.name}: {e}")
            
            results['files_processed'] = processed_count
            results['success'] = processed_count > 0
        
        # Generate matrix visualizations if available
        if MatrixVisualizer:
            try:
                with performance_tracker.track_operation("generate_matrix_visualizations"):
                    matrix_viz = MatrixVisualizer()
                    matrix_results = matrix_viz.visualize_directory(
                        target_dir=target_dir,
                        output_dir=viz_output_dir / "matrices"
                    )
                    results.update(matrix_results)
                log_step_success(logger, "Matrix visualizations completed")
            except Exception as e:
                log_step_warning(logger, f"Matrix visualization failed: {e}")
        
        # Generate ontology visualizations if available
        if OntologyVisualizer:
            try:
                with performance_tracker.track_operation("generate_ontology_visualizations"):
                    ontology_viz = OntologyVisualizer()
                    ontology_results = ontology_viz.visualize_directory(
                        target_dir=target_dir,
                        output_dir=viz_output_dir / "ontology"
                    )
                    results.update(ontology_results)
                log_step_success(logger, "Ontology visualizations completed")
            except Exception as e:
                log_step_warning(logger, f"Ontology visualization failed: {e}")
        
        # Log results summary
        if results.get('success', False):
            log_step_success(logger, f"Visualization generation completed successfully. Files processed: {results.get('files_processed', 0)}")
        else:
            log_step_warning(logger, f"Visualization generation completed with issues. Files processed: {results.get('files_processed', 0)}")
        
        return results.get('success', False)
        
    except Exception as e:
        log_step_error(logger, f"Visualization generation failed: {e}")
        return False 