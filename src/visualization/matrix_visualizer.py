"""
Matrix Visualization Module

This module provides specialized functionality for visualizing matrices from GNN models.
It generates heatmap and other matrix-related visualizations.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple


class MatrixVisualizer:
    """
    A class for visualizing matrices extracted from GNN models.
    
    This visualizer provides methods to create heatmaps and other
    visualizations of matrices in the model.
    """
    
    def __init__(self):
        """Initialize the matrix visualizer."""
        pass
    
    def visualize_all_matrices(self, parsed_data: Dict[str, Any], output_dir: Path) -> List[str]:
        """
        Generate visualizations for all matrices in the model.
        
        Args:
            parsed_data: Parsed GNN model data
            output_dir: Directory to save visualizations
            
        Returns:
            List of paths to saved visualization files
        """
        saved_files = []
        
        # Extract matrices from InitialParameterization
        if 'InitialParameterization' in parsed_data:
            matrices = self._extract_matrices(parsed_data['InitialParameterization'])
            
            # Create individual heatmaps
            for matrix_name, matrix_data in matrices.items():
                file_path = self.create_heatmap(matrix_name, matrix_data, output_dir)
                if file_path:
                    saved_files.append(file_path)
            
            # Create combined matrix visualization
            if matrices:
                combined_path = self.create_combined_matrix_visualization(matrices, output_dir)
                if combined_path:
                    saved_files.append(combined_path)
        
        return saved_files
    
    def _extract_matrices(self, init_params: str) -> Dict[str, List[List[float]]]:
        """
        Extract matrices from the InitialParameterization section.
        
        Args:
            init_params: Raw content of the InitialParameterization section
            
        Returns:
            Dictionary mapping matrix names to their data
        """
        matrices = {}
        
        # Look for patterns like A_π1={...}, B={...}, etc.
        matrix_pattern = r'(\w+(?:_\w+)?)\s*=\s*{([^}]*)}'
        
        for match in re.finditer(matrix_pattern, init_params, re.DOTALL):
            matrix_name = match.group(1)
            matrix_content = match.group(2).strip()
            
            # Parse matrix content
            # Look for row patterns like (0.1,0.2,0.3,0.4)
            rows = []
            for row_match in re.finditer(r'\((.*?)\)', matrix_content):
                try:
                    row_values = [float(val.strip()) for val in row_match.group(1).split(',')]
                    rows.append(row_values)
                except ValueError:
                    # Skip rows with non-numeric values
                    continue
            
            if rows:
                matrices[matrix_name] = rows
        
        return matrices
    
    def create_heatmap(self, matrix_name: str, matrix_data: List[List[float]], 
                       output_dir: Path) -> str:
        """
        Create a heatmap visualization for a matrix.
        
        Args:
            matrix_name: Name of the matrix
            matrix_data: 2D list of matrix values
            output_dir: Directory to save visualization
            
        Returns:
            Path to saved visualization file, or empty string if failed
        """
        try:
            # Convert to numpy array for heatmap
            matrix_array = np.array(matrix_data)
            
            # Create heatmap visualization
            plt.figure(figsize=(10, 8))
            ax = plt.subplot(111)
            
            # Create heatmap
            im = ax.imshow(matrix_array, cmap='viridis')
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Value')
            
            # Add labels and title
            plt.title(f'Matrix: {matrix_name}', fontsize=14, fontweight='bold')
            
            # Add row and column labels if available
            # Row labels on y-axis
            if matrix_array.shape[0] <= 10:  # Only add labels for reasonably sized matrices
                row_labels = [str(i) for i in range(matrix_array.shape[0])]
                ax.set_yticks(np.arange(matrix_array.shape[0]))
                ax.set_yticklabels(row_labels)
            
            # Column labels on x-axis
            if matrix_array.shape[1] <= 10:
                col_labels = [str(i) for i in range(matrix_array.shape[1])]
                ax.set_xticks(np.arange(matrix_array.shape[1]))
                ax.set_xticklabels(col_labels)
            
            # Add grid to make it easier to read values
            ax.set_xticks(np.arange(-.5, matrix_array.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-.5, matrix_array.shape[0], 1), minor=True)
            ax.grid(which="minor", color="black", linestyle='-', linewidth=1, alpha=0.2)
            
            # Add text annotations with the values
            if matrix_array.shape[0] <= 10 and matrix_array.shape[1] <= 10:
                for i in range(matrix_array.shape[0]):
                    for j in range(matrix_array.shape[1]):
                        ax.text(j, i, f"{matrix_array[i, j]:.2f}", 
                               ha="center", va="center", 
                               color="black" if matrix_array[i, j] > 0.5 else "white")
            
            # Save figure
            matrix_filename = f"matrix_{matrix_name.replace('_', '')}.png"
            output_path = output_dir / matrix_filename
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            print(f"Matrix visualization saved to {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"Error creating heatmap for matrix {matrix_name}: {e}")
            return ""
            
    def create_combined_matrix_visualization(self, matrices: Dict[str, List[List[float]]], 
                                            output_dir: Path) -> str:
        """
        Create a combined visualization of all matrices.
        
        Args:
            matrices: Dictionary of matrix names to matrix data
            output_dir: Directory to save visualization
            
        Returns:
            Path to saved visualization file, or empty string if failed
        """
        if not matrices:
            return ""
            
        try:
            # Calculate grid size based on number of matrices
            num_matrices = len(matrices)
            grid_size = int(np.ceil(np.sqrt(num_matrices)))
            
            # Create a figure with subplots
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 5, grid_size * 4))
            
            # Make axes a 2D array even if it's a single subplot
            if num_matrices == 1:
                axes = np.array([[axes]])
            elif grid_size == 1:
                axes = np.array([axes])
            
            # Flatten axes for easy iteration
            axes_flat = axes.flatten()
            
            # Plot each matrix in its own subplot
            for i, (matrix_name, matrix_data) in enumerate(matrices.items()):
                if i >= len(axes_flat):
                    break
                    
                # Get current axis
                ax = axes_flat[i]
                
                # Convert to numpy array
                matrix_array = np.array(matrix_data)
                
                # Create heatmap
                im = ax.imshow(matrix_array, cmap='viridis')
                
                # Add matrix name as title
                ax.set_title(matrix_name, fontsize=12)
                
                # Add row and column labels if matrix is small enough
                if matrix_array.shape[0] <= 8 and matrix_array.shape[1] <= 8:
                    # Row labels
                    ax.set_yticks(np.arange(matrix_array.shape[0]))
                    ax.set_yticklabels([str(i) for i in range(matrix_array.shape[0])])
                    
                    # Column labels
                    ax.set_xticks(np.arange(matrix_array.shape[1]))
                    ax.set_xticklabels([str(i) for i in range(matrix_array.shape[1])])
                    
                    # Add grid
                    ax.set_xticks(np.arange(-.5, matrix_array.shape[1], 1), minor=True)
                    ax.set_yticks(np.arange(-.5, matrix_array.shape[0], 1), minor=True)
                    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.2)
                    
                    # Add text annotations with values
                    for y in range(matrix_array.shape[0]):
                        for x in range(matrix_array.shape[1]):
                            ax.text(x, y, f"{matrix_array[y, x]:.2f}", 
                                  ha="center", va="center", fontsize=8,
                                  color="black" if matrix_array[y, x] > 0.5 else "white")
            
            # Hide unused subplots
            for j in range(i+1, len(axes_flat)):
                axes_flat[j].axis('off')
            
            # Add a common colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            
            # Add overall title
            fig.suptitle('All Model Matrices', fontsize=16, fontweight='bold')
            
            # Save figure
            output_path = output_dir / 'combined_matrices.png'
            plt.tight_layout()  # Make room for colorbar and title
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Combined matrix visualization saved to {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"Error creating combined matrix visualization: {e}")
            return "" 