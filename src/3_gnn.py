#!/usr/bin/env python3
"""
Step 3: GNN File Discovery and Parsing (Thin Orchestrator)

Delegates discovery, parsing, and multi-format serialization to
`gnn/multi_format_processor.py` using the standardized pipeline wrapper.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from gnn.multi_format_processor import process_gnn_multi_format


run_script = create_standardized_pipeline_script(
    "3_gnn.py",
    process_gnn_multi_format,
    "GNN discovery, parsing, and multi-format serialization",
)


def main() -> int:
    return run_script()


if __name__ == "__main__":
    sys.exit(main()) 