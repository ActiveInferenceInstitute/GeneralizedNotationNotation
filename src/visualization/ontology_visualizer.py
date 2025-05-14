"""
Ontology Visualization Module

This module provides specialized functionality for visualizing ontology annotations
from GNN models.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple


class OntologyVisualizer:
    """
    A class for visualizing ontology annotations extracted from GNN models.
    
    This visualizer provides methods to create table-based and other
    visualizations of ontology mappings.
    """
    
    def __init__(self):
        """Initialize the ontology visualizer."""
        pass
    
    def visualize_ontology(self, parsed_data: Dict[str, Any], output_dir: Path) -> str:
        """
        Generate visualization of the ontology annotations.
        
        Args:
            parsed_data: Parsed GNN model data
            output_dir: Directory to save visualization
            
        Returns:
            Path to saved visualization file, or empty string if failed
        """
        if 'ActInfOntologyAnnotation' not in parsed_data:
            return ""
            
        ontology = parsed_data['ActInfOntologyAnnotation']
        
        # Extract ontology mappings
        mappings = self._extract_ontology_mappings(ontology)
        
        if not mappings:
            return ""
            
        # Create visualization
        try:
            return self._create_ontology_table(mappings, output_dir)
        except Exception as e:
            print(f"Error creating ontology visualization: {e}")
            return ""
    
    def _extract_ontology_mappings(self, ontology_content: str) -> List[Tuple[str, str]]:
        """
        Extract variable-concept mappings from ontology content.
        
        Args:
            ontology_content: Raw content of the ActInfOntologyAnnotation section
            
        Returns:
            List of (variable, concept) tuples
        """
        mappings = []
        
        for line in ontology_content.split('\n'):
            line = line.strip()
            if not line or '=' not in line:
                continue
                
            parts = line.split('=', 1)
            if len(parts) == 2:
                variable = parts[0].strip()
                concept = parts[1].strip()
                mappings.append((variable, concept))
        
        return mappings
    
    def _create_ontology_table(self, mappings: List[Tuple[str, str]], output_dir: Path) -> str:
        """
        Create a table visualization of ontology mappings.
        
        Args:
            mappings: List of (variable, concept) tuples
            output_dir: Directory to save visualization
            
        Returns:
            Path to saved visualization file
        """
        # Create figure
        plt.figure(figsize=(10, max(5, len(mappings) * 0.5)))
        ax = plt.subplot(111)
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table_data = [[var, concept] for var, concept in mappings]
        table = ax.table(
            cellText=table_data,
            colLabels=['Variable', 'Ontological Concept'],
            loc='center',
            cellLoc='left',
            colWidths=[0.3, 0.7]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Add title
        plt.title('Ontological Annotations', fontsize=14, fontweight='bold', pad=20)
        
        # Save figure
        output_path = output_dir / 'ontology_annotations.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Ontology visualization saved to {output_path}")
        return str(output_path) 