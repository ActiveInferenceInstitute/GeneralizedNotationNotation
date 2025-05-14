"""
GNN Type Checker Main entry point.

Allows the module to be executed directly with python -m gnn_type_checker
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main()) 