#!/usr/bin/env python3
"""
Step 7: Multi-format Export Generation (Thin Orchestrator)

This step generates exports in multiple formats (JSON, XML, GraphML, GEXF, Pickle).

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/export/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the export module.

Pipeline Flow:
    main.py → 7_export.py (this script) → export/ (modular implementation)

How to run:
  python src/7_export.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Multi-format exports in the specified output directory
  - JSON, XML, GraphML, GEXF, and Pickle format files
  - Comprehensive export reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that export dependencies are installed
  - Check that src/export/ contains export modules
  - Check that the output directory is writable
  - Verify export configuration and format requirements
"""

import sys
from pathlib import Path
from typing import Any, cast

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from export import process_export
from export.registry import DEFAULT_PIPELINE_FORMATS
from utils.pipeline_template import create_standardized_pipeline_script

geo_arguments = {
    "geo_step_seconds": {
        "type": float,
        "default": None,
        "help": "GEO-INFER export: explicit step seconds (mandatory, positive)",
    },
    "geo_state_ids": {
        "type": str,
        "default": None,
        "help": "GEO-INFER export: path to a state IDs file",
    },
    "geo_space_kind": {
        "type": str,
        "choices": ["categorical", "h3"],
        "default": None,
        "help": "GEO-INFER export: space kind (categorical or h3)",
    },
}


def _export_with_geo(**kwargs: Any) -> bool:
    """Opt-in GEO-INFER export via the registered --geo-* CLI flags."""
    step_seconds = kwargs.pop("geo_step_seconds", None)
    state_ids_path = kwargs.pop("geo_state_ids", None)
    space_kind = kwargs.pop("geo_space_kind", None)
    if step_seconds is None:
        if state_ids_path or space_kind:
            raise ValueError(
                "geo_infer export requires explicit geo_infer options with a "
                "mandatory, positive 'step_seconds' key"
            )
        return process_export(**kwargs)
    kwargs["geo_infer"] = {
        "step_seconds": float(step_seconds),
        "state_ids_path": state_ids_path,
        "space_kind": space_kind or "categorical",
    }
    formats = kwargs.get("formats") or list(DEFAULT_PIPELINE_FORMATS)
    if "geo_infer" not in formats:
        kwargs["formats"] = [*formats, "geo_infer"]
    return process_export(**kwargs)


# Create the standardized pipeline script
run_script = create_standardized_pipeline_script(
    "7_export.py",
    _export_with_geo,
    "Multi-format export generation",
    additional_arguments=geo_arguments,
)


def main() -> int:
    """Main entry point for the export step."""
    return cast("int", run_script())


if __name__ == "__main__":
    raise SystemExit(main())
