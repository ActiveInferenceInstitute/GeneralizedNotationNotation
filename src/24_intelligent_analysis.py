#!/usr/bin/env python3
"""
Step 24: Intelligent Pipeline Analysis

This step performs an intelligent analysis of the pipeline execution using
the available LLM infrastructure. It reads the pipeline summary and logs,
analyzes failures or performance bottlenecks, and generates an executive report.

This script uses the intelligent_analysis module for all processing logic.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from intelligent_analysis.processor import process_intelligent_analysis

# Create the runnable script using the standardized template
run_script = create_standardized_pipeline_script(
    "24_intelligent_analysis.py",
    process_intelligent_analysis,
    "Intelligent analysis of pipeline execution logs",
    additional_arguments={
        "analysis_model": {"type": str, "help": "Specific LLM model to use for analysis", "default": None},
        "skip_llm": {"type": bool, "help": "Skip LLM-powered analysis (use only rule-based)", "default": False},
        "bottleneck_threshold": {"type": float, "help": "Duration threshold (seconds) for bottleneck detection", "default": 60.0}
    }
)


def main() -> int:
    """Main entry point."""
    return run_script()


if __name__ == "__main__":
    sys.exit(main())
