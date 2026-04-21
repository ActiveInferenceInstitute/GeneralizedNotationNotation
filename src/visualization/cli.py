"""
GNN Visualization CLI

Command-line interface for generating visualizations of GNN models.
"""

import argparse
import sys
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
        help='Path to a GNN file or directory containing GNN files (e.g., input/gnn_files)'
    )

    # Output directory — resolve lazily so importing this module doesn't
    # touch pipeline.config. Phase 2.2: replaces the fragile '../output'
    # relative default that broke when cwd differed from the CLI invocation
    # directory.
    parser.add_argument(
        '-o', '--output-dir',
        help='Directory to save visualizations. If not provided, uses the pipeline-standard output dir for step 8.',
        default=None,
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


def _resolve_output_dir(explicit: Optional[str]) -> str:
    """Resolve the CLI output directory, defaulting to the pipeline-standard
    step-8 location when no explicit value is given."""
    if explicit:
        return explicit
    try:
        from pipeline.config import get_output_dir_for_script
        return str(get_output_dir_for_script("8_visualization.py", Path("output")))
    except Exception:
        return "output/8_visualization_output"


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for GNN visualization CLI."""
    parsed_args = parse_args(args)
    output_dir = _resolve_output_dir(parsed_args.output_dir)

    # Create visualizer
    visualizer = GNNVisualizer(output_dir=output_dir, project_root=parsed_args.project_root)

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
