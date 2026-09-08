#!/usr/bin/env bash
# MAJ-06 (MCP dispatcher consolidation) benchmark harness.
#
# Deterministic and offline: static AST scan of the process_*_mcp wrapper
# surface plus a live MCP registry pin against the committed audit report
# (same initialize/settle sequence as the CI tool-count gate).
#
# Metrics:
#   METRIC wrapper_loc=<int>     (primary; lower is better)
#   METRIC wrapper_count=<int>
#   METRIC mcp_tool_count=<int>
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python bench/mcp_wrapper_metric.py
