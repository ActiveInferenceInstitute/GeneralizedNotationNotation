#!/usr/bin/env python3
"""
Step 3: GNN File Discovery and Parsing (Thin Orchestrator)

Delegates discovery, parsing, and multi-format serialization to
`gnn/multi_format_processor.py` using the standardized pipeline wrapper.
"""

from typing import cast

from gnn.processing.multi_format_processor import process_gnn_multi_format
from gnn.utils.pipeline_orchestration.pipeline_template import (
    create_standardized_pipeline_script,
)

run_script = create_standardized_pipeline_script(
    "3_gnn.py",
    process_gnn_multi_format,
    "GNN discovery, parsing, and multi-format serialization",
)


def main() -> int:
    """Provide main behavior."""
    return cast("int", run_script())


if __name__ == "__main__":
    raise SystemExit(main())
