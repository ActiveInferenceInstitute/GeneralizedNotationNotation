"""
Ontology Visualization Module

This module provides specialized functionality for visualizing ontology annotations
from GNN models using real matplotlib functionality.
"""

import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class OntologyVisualizer:
    """
    A class for visualizing ontology annotations extracted from GNN models.
    
    This visualizer provides methods to create table-based and other
    visualizations of ontology mappings using real matplotlib functionality.
    """
    
    def __init__(self):
        """Initialize the ontology visualizer."""
        self.figure_size = (10, 8)  # Default figure size
        self.dpi = 150  # Default DPI for output files
        self.font_size = {
            'title': 14,
            'subtitle': 12,
            'labels': 10,
            'values': 10
        }
        self.colors = {
            'header': '#E6E6E6',  # Light gray
            'alternate': '#F5F5F5',  # Very light gray
            'text': '#000000',  # Black
            'border': '#CCCCCC'  # Medium gray
        }
    
    def visualize_directory(self, input_dir: Path, output_dir: Path) -> List[str]:
        """
        Visualize ontology annotations from all GNN files in a directory.
        
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
                    
                    # Extract ontology section
                    ontology_match = re.search(r'## ActInfOntologyAnnotation\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
                    if ontology_match:
                        # Extract mappings
                        mappings = self._extract_ontology_mappings(ontology_match.group(1))
                        
                        if mappings:
                            # Create visualization
                            viz_path = self._create_ontology_table(mappings, file_output_dir)
                            if viz_path:
                                saved_files.append(viz_path)
                    
                except Exception as e:
                    logger.error(f"Error processing file {gnn_file}: {e}")
                    continue
                    
            return saved_files
            
        except Exception as e:
            logger.error(f"Error processing directory {input_dir}: {e}")
            return saved_files
    
    def visualize_ontology(self, parsed_data: Dict[str, Any], output_dir: Path) -> Optional[str]:
        """
        Generate visualization of the ontology annotations.
        
        Args:
            parsed_data: Parsed GNN model data
            output_dir: Directory to save visualization
            
        Returns:
            Path to saved visualization file, or None if failed
        """
        if 'ActInfOntologyAnnotation' not in parsed_data:
            return None
            
        ontology = parsed_data['ActInfOntologyAnnotation']
        
        # Extract ontology mappings
        mappings = self._extract_ontology_mappings(ontology)
        
        if not mappings:
            return None
            
        # Create visualization
        try:
            return self._create_ontology_table(mappings, output_dir)
        except Exception as e:
            logger.error(f"Error creating ontology visualization: {e}")
            return None
    
    def _extract_ontology_mappings(self, ontology_content: str) -> List[Tuple[str, str]]:
        """
        Extract variable-concept mappings from ontology content.
        
        Args:
            ontology_content: Raw content of the ActInfOntologyAnnotation section
            
        Returns:
            List of (variable, concept) tuples
        """
        mappings = []
        
        # Split content into lines and process each line
        for line in str(ontology_content).split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
                
            # Split on equals sign
            parts = line.split('=', 1)
            if len(parts) == 2:
                variable = parts[0].strip()
                concept = parts[1].strip()
                
                # Skip empty or invalid mappings
                if not variable or not concept:
                    continue
                
                # Handle comments after mapping
                if '#' in concept:
                    concept = concept.split('#')[0].strip()
                
                mappings.append((variable, concept))
        
        return mappings

    def extract_ontology_mappings(self, ontology_content: str) -> List[Tuple[str, str]]:
        return self._extract_ontology_mappings(ontology_content)
    
    def _create_ontology_table(self, mappings: List[Tuple[str, str]], output_dir: Path) -> Optional[str]:
        """
        Create a table visualization of ontology mappings.
        
        Args:
            mappings: List of (variable, concept) tuples
            output_dir: Directory to save visualization
            
        Returns:
            Path to saved visualization file, or None if failed
        """
        try:
            # Calculate figure dimensions based on content
            num_rows = len(mappings) + 1  # +1 for header
            row_height = 0.5  # inches per row
            figure_height = max(self.figure_size[1], num_rows * row_height)
            
            # Create figure
            plt.figure(figsize=(self.figure_size[0], figure_height))
            ax = plt.subplot(111)
            ax.axis('tight')
            ax.axis('off')
            
            # Prepare table data with header
            table_data = [['Variable', 'Ontological Concept']]  # Header row
            table_data.extend([[var, concept] for var, concept in mappings])
            
            # Create table with custom styling
            table = ax.table(
                cellText=table_data[1:],  # Data rows
                colLabels=table_data[0],  # Header row
                loc='center',
                cellLoc='left',
                colWidths=[0.3, 0.7]
            )
            
            # Style the table
            table.auto_set_font_size(False)
            
            # Style header cells
            for j, cell in enumerate(table._cells[(0, j)] for j in range(2)):
                cell.set_facecolor(self.colors['header'])
                cell.set_text_props(weight='bold', size=self.font_size['subtitle'])
                cell.set_edgecolor(self.colors['border'])
            
            # Style data cells
            for i in range(len(mappings)):
                for j in range(2):
                    cell = table._cells[(i+1, j)]
                    cell.set_facecolor(self.colors['alternate'] if i % 2 else 'white')
                    cell.set_text_props(size=self.font_size['values'])
                    cell.set_edgecolor(self.colors['border'])
            
            # Scale table
            table.scale(1, 1.5)
            
            # Add title
            plt.title('Ontological Annotations', 
                     fontsize=self.font_size['title'], 
                     fontweight='bold',
                     pad=20)
            
            # Save figure
            output_path = output_dir / 'ontology_annotations.png'
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Ontology visualization saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to create ontology table: {e}")
            return None 

    # Public wrappers expected by tests
    def extract_ontology_mappings(self, ontology_content: "str | dict") -> List[Tuple[str, str]]:
        return self._extract_ontology_mappings(ontology_content)

    def create_ontology_table(self, mappings: List[Tuple[str, str]], output_dir: Path) -> Optional[str]:
        return self._create_ontology_table(mappings, output_dir)