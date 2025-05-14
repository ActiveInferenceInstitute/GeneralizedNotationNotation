"""
GNN Visualization Module Main Entry Point

This module allows the GNN visualization package to be executed directly.
Usage: python -m gnn.visualization <input_path> [options]
"""

import sys
from .cli import main

if __name__ == '__main__':
    sys.exit(main()) 