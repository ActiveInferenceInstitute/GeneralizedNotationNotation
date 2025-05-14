#!/usr/bin/env python3
"""
GNN Visualization Runner

This script runs the GNN visualization module on all examples in the src/gnn/examples directory.
It generates comprehensive visualizations and saves them to the output directory.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to sys.path to allow importing from src
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from visualization import GNNVisualizer


def main():
    """Run the visualization on GNN examples."""
    parser = argparse.ArgumentParser(description='Generate visualizations for GNN examples.')
    parser.add_argument('--input', '-i', type=str, default=str(parent_dir / 'gnn' / 'examples'),
                        help='Directory containing GNN example files')
    parser.add_argument('--output', '-o', type=str, default=str(parent_dir.parent / 'output' / 'gnn_examples_visualization'),
                        help='Directory to save visualizations')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Recursively process all subdirectories')
    args = parser.parse_args()
    
    print(f"Processing GNN examples from {args.input}")
    print(f"Saving visualizations to {args.output}")
    
    # Create visualizer
    visualizer = GNNVisualizer(output_dir=args.output)
    
    # Process examples
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix.lower() == '.md':
        # If input is a single file
        md_files = [input_path]
        print(f"Processing single file: {input_path}")
    else:
        # If input is a directory
        if not input_path.exists():
            print(f"Error: Input path {input_path} does not exist")
            return 1
        
        # Find all markdown files
        if args.recursive:
            md_files = list(input_path.glob('**/*.md'))
        else:
            md_files = list(input_path.glob('*.md'))
        
        if not md_files:
            print(f"No Markdown files found in {input_path}")
            return 1
    
    print(f"Found {len(md_files)} Markdown files")
    
    # Process each file
    success_count = 0
    for md_file in md_files:
        print(f"\nProcessing {md_file}...")
        try:
            output_dir = visualizer.visualize_file(str(md_file))
            print(f"Visualizations saved to {output_dir}")
            success_count += 1
        except Exception as e:
            print(f"Error processing {md_file}: {e}")
    
    print(f"\nProcessed {len(md_files)} files, {success_count} succeeded")
    return 0


if __name__ == '__main__':
    sys.exit(main()) 