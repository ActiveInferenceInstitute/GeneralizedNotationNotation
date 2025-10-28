#!/usr/bin/env python3
"""
Pipeline Step 24: oxdraw Visual Interface Integration

Provides visual diagram-as-code interface for GNN Active Inference model
construction through bidirectional GNN ↔ Mermaid ↔ oxdraw synchronization.

Usage:
    python src/24_oxdraw.py --target-dir input/gnn_files --output-dir output
    python src/24_oxdraw.py --target-dir input/gnn_files --mode interactive --launch-editor
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from oxdraw.processor import process_oxdraw


def main():
    """Main entry point for oxdraw integration step."""
    
    # Create standardized pipeline script following GNN patterns
    run_script = create_standardized_pipeline_script(
        script_name="24_oxdraw.py",
        processing_function=process_oxdraw,
        description="oxdraw visual interface for GNN model construction",
        
        # Additional argument specifications
        additional_args=[
            {
                "name": "--mode",
                "type": str,
                "default": "headless",
                "choices": ["interactive", "headless"],
                "help": "Processing mode: interactive (launch editor) or headless (convert only)"
            },
            {
                "name": "--launch-editor",
                "action": "store_true",
                "help": "Launch oxdraw interactive editor (requires --mode interactive)"
            },
            {
                "name": "--port",
                "type": int,
                "default": 5151,
                "help": "Port for oxdraw server (default: 5151)"
            },
            {
                "name": "--host",
                "type": str,
                "default": "127.0.0.1",
                "help": "Host for oxdraw server (default: 127.0.0.1)"
            },
            {
                "name": "--auto-convert",
                "action": "store_true",
                "default": True,
                "help": "Automatically convert GNN files to Mermaid"
            },
            {
                "name": "--validate-on-save",
                "action": "store_true",
                "default": True,
                "help": "Validate models when converting back from Mermaid"
            }
        ]
    )
    
    return run_script()


if __name__ == "__main__":
    sys.exit(main())

