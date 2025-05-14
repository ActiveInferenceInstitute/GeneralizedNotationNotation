"""
GNN Visualization CLI

Command-line interface for generating visualizations of GNN models.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

from .visualizer import GNNVisualizer


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate visualizations for GNN models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    parser.add_argument(
        'input',
        help='Path to a GNN file or directory containing GNN files'
    )
    
    # Output directory
    parser.add_argument(
        '-o', '--output-dir',
        help='Directory to save visualizations. If not provided, creates a timestamped directory in ../output.',
        default='../output'  # Defaults to output folder in the parent of current scripts (e.g. project_root/output)
    )
    
    # Visualization options
    parser.add_argument(
        '--recursive',
        help='Recursively process directories',
        action='store_true'
    )
    parser.add_argument(
        '--project-root',
        help='Absolute path to the project root, for relative path generation in reports'
    )
    
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for GNN visualization CLI."""
    parsed_args = parse_args(args)
    
    # Create visualizer
    visualizer = GNNVisualizer(output_dir=parsed_args.output_dir, project_root=parsed_args.project_root)
    
    # Get input path
    input_path = Path(parsed_args.input)
    
    # Process input
    if input_path.is_file():
        # Single file
        output_dir = visualizer.visualize_file(str(input_path))
        print(f"Visualizations generated in {output_dir}")
    elif input_path.is_dir():
        # Directory
        if parsed_args.recursive:
            # Process all md files recursively
            for file_path in input_path.glob('**/*.md'):
                try:
                    visualizer.visualize_file(str(file_path))
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            print(f"Visualizations generated in {visualizer.output_dir}")
        else:
            # Process only md files in the top directory
            output_dir = visualizer.visualize_directory(str(input_path))
            print(f"Visualizations generated in {output_dir}")
    else:
        print(f"Error: Input path '{input_path}' does not exist")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 