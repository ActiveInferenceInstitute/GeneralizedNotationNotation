"""
Matrix Visualization Module

This module provides specialized functionality for visualizing matrices from GNN models.
It generates heatmap and other matrix-related visualizations using real numpy and matplotlib
functionality.
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MatrixVisualizer:
    """
    A class for visualizing matrices extracted from GNN models.
    
    This visualizer provides methods to create heatmaps and other
    visualizations of matrices in the model using real numpy and matplotlib
    functionality.
    """
    
    def __init__(self):
        """Initialize the matrix visualizer."""
        self.figure_size = (10, 8)  # Default figure size
        self.dpi = 150  # Default DPI for output files
        self.cmap = 'viridis'  # Default colormap
        self.font_size = {
            'title': 14,
            'subtitle': 12,
            'labels': 10,
            'values': 8
        }
    
    def visualize_directory(self, input_dir: Path, output_dir: Path) -> List[str]:
        """
        Visualize all matrices found in GNN files in a directory.
        
        Args:
            input_dir: Directory containing GNN files
            output_dir: Directory to save visualizations
            
        Returns:
            List of paths to saved visualization files
        """
        saved_files = []
        
        try:
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all GNN files in directory
            for gnn_file in input_dir.glob('**/*.md'):
                try:
                    # Create subdirectory for this file's visualizations
                    file_output_dir = output_dir / gnn_file.stem
                    file_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Read and parse file content
                    with open(gnn_file, 'r') as f:
                        content = f.read()
                    
                    # Extract matrices from content
                    matrices = self._extract_matrices_from_content(content)
                    
                    if matrices:
                        # Create individual matrix visualizations
                        for matrix_name, matrix_data in matrices.items():
                            viz_path = self.create_heatmap(matrix_name, matrix_data, file_output_dir)
                            if viz_path:
                                saved_files.append(viz_path)
                        
                        # Create combined visualization
                        combined_path = self.create_combined_matrix_visualization(matrices, file_output_dir)
                        if combined_path:
                            saved_files.append(combined_path)
                    
                except Exception as e:
                    logger.error(f"Error processing file {gnn_file}: {e}")
                    continue
                    
            return saved_files
            
        except Exception as e:
            logger.error(f"Error processing directory {input_dir}: {e}")
            return saved_files
    
    def _extract_matrices_from_content(self, content: str) -> Dict[str, List[List[float]]]:
        """
        Extract all matrices from GNN file content.
        
        Args:
            content: Raw content of GNN file
            
        Returns:
            Dictionary mapping matrix names to their data
        """
        matrices = {}
        
        # Look for InitialParameterization section
        init_param_match = re.search(r'## InitialParameterization\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
        if init_param_match:
            section_content = init_param_match.group(1)
            
            # Extract matrix definitions
            matrix_pattern = r'([A-E])\s*=\s*{([^}]*)}'
            for match in re.finditer(matrix_pattern, section_content, re.DOTALL):
                matrix_name = match.group(1)
                matrix_content = match.group(2).strip()
                
                try:
                    # Handle 3D matrices (like B matrix)
                    if matrix_name == 'B':  # Special handling for B matrix
                        # Extract the first slice only
                        first_slice_match = re.search(r'\(\s*\(([^)]+)\)\s*,\s*\(([^)]+)\)\s*,\s*\(([^)]+)\)\s*\)', matrix_content)
                        if first_slice_match:
                            current_slice = []
                            for i in range(1, 4):  # We know it's 3 rows
                                row_str = first_slice_match.group(i)
                                values = [float(x.strip()) for x in row_str.split(',')]
                                current_slice.append(values)
                            matrices[matrix_name] = current_slice
                    else:  # Regular 2D matrix
                        rows = []
                        for row_match in re.finditer(r'\((.*?)\)', matrix_content):
                            try:
                                row_values = [float(val.strip()) for val in row_match.group(1).split(',')]
                                rows.append(row_values)
                            except ValueError:
                                logger.warning(f"Skipping invalid row in matrix {matrix_name}")
                                continue
                        if rows:
                            matrices[matrix_name] = rows
                except Exception as e:
                    logger.error(f"Error parsing matrix {matrix_name}: {e}")
                    continue
        
        return matrices
    
    def create_heatmap(self, matrix_name: str, matrix_data: List[List[float]], output_dir: Path) -> Optional[str]:
        """
        Create a heatmap visualization for a matrix.
        
        Args:
            matrix_name: Name of the matrix
            matrix_data: 2D list of matrix values
            output_dir: Directory to save visualization
            
        Returns:
            Path to saved visualization file, or None if failed
        """
        try:
            # Convert to numpy array for visualization
            matrix_array = np.array(matrix_data)
            
            # Create figure and axis
            plt.figure(figsize=self.figure_size)
            ax = plt.subplot(111)
            
            # Create heatmap
            im = ax.imshow(matrix_array, cmap=self.cmap)
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Value', fontsize=self.font_size['labels'])
            
            # Add labels and title
            plt.title(f'Matrix: {matrix_name}', fontsize=self.font_size['title'], fontweight='bold')
            
            # Add row and column labels if matrix is small enough
            if matrix_array.shape[0] <= 10:
                row_labels = [str(i) for i in range(matrix_array.shape[0])]
                ax.set_yticks(np.arange(matrix_array.shape[0]))
                ax.set_yticklabels(row_labels, fontsize=self.font_size['labels'])
            
            if matrix_array.shape[1] <= 10:
                col_labels = [str(i) for i in range(matrix_array.shape[1])]
                ax.set_xticks(np.arange(matrix_array.shape[1]))
                ax.set_xticklabels(col_labels, fontsize=self.font_size['labels'])
            
            # Add grid
            ax.set_xticks(np.arange(-.5, matrix_array.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-.5, matrix_array.shape[0], 1), minor=True)
            ax.grid(which="minor", color="black", linestyle='-', linewidth=1, alpha=0.2)
            
            # Add value annotations if matrix is small enough
            if matrix_array.shape[0] <= 10 and matrix_array.shape[1] <= 10:
                for i in range(matrix_array.shape[0]):
                    for j in range(matrix_array.shape[1]):
                        ax.text(j, i, f"{matrix_array[i, j]:.2f}", 
                               ha="center", va="center", fontsize=self.font_size['values'],
                               color="black" if matrix_array[i, j] > 0.5 else "white")
            
            # Save figure
            output_path = output_dir / f"matrix_{matrix_name.replace('_', '')}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Matrix visualization saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating heatmap for matrix {matrix_name}: {e}")
            return None
    
    def create_combined_matrix_visualization(self, matrices: Dict[str, List[List[float]]], output_dir: Path) -> Optional[str]:
        """
        Create a combined visualization of all matrices.
        
        Args:
            matrices: Dictionary of matrix names to matrix data
            output_dir: Directory to save visualization
            
        Returns:
            Path to saved visualization file, or None if failed
        """
        if not matrices:
            return None
            
        try:
            # Calculate grid size based on number of matrices
            num_matrices = len(matrices)
            grid_size = int(np.ceil(np.sqrt(num_matrices)))
            
            # Create figure with subplots
            fig = plt.figure(figsize=(grid_size * 5, grid_size * 4))
            
            # Create subplots
            axes = []
            for i in range(grid_size * grid_size):
                ax = fig.add_subplot(grid_size, grid_size, i + 1)
                axes.append(ax)
            
            # Plot each matrix
            im = None  # Store last imshow result for colorbar
            for i, (matrix_name, matrix_data) in enumerate(matrices.items()):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                matrix_array = np.array(matrix_data)
                
                # Create heatmap
                im = ax.imshow(matrix_array, cmap=self.cmap)
                
                # Add title
                ax.set_title(matrix_name, fontsize=self.font_size['subtitle'])
                
                # Add labels if matrix is small enough
                if matrix_array.shape[0] <= 8 and matrix_array.shape[1] <= 8:
                    # Row labels
                    ax.set_yticks(np.arange(matrix_array.shape[0]))
                    ax.set_yticklabels([str(i) for i in range(matrix_array.shape[0])],
                                     fontsize=self.font_size['labels'])
                    
                    # Column labels
                    ax.set_xticks(np.arange(matrix_array.shape[1]))
                    ax.set_xticklabels([str(i) for i in range(matrix_array.shape[1])],
                                     fontsize=self.font_size['labels'])
                    
                    # Add grid
                    ax.set_xticks(np.arange(-.5, matrix_array.shape[1], 1), minor=True)
                    ax.set_yticks(np.arange(-.5, matrix_array.shape[0], 1), minor=True)
                    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.2)
                    
                    # Add value annotations
                    for y in range(matrix_array.shape[0]):
                        for x in range(matrix_array.shape[1]):
                            ax.text(x, y, f"{matrix_array[y, x]:.2f}", 
                                  ha="center", va="center", fontsize=self.font_size['values'],
                                  color="black" if matrix_array[y, x] > 0.5 else "white")
            
            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
            
            # Add colorbar if we have matrices
            if im is not None:
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                fig.colorbar(im, cax=cbar_ax)
            
            # Add title
            fig.suptitle('All Model Matrices', fontsize=self.font_size['title'], fontweight='bold')
            
            # Save figure
            output_path = output_dir / 'combined_matrices.png'
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Combined matrix visualization saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating combined matrix visualization: {e}")
            return None 