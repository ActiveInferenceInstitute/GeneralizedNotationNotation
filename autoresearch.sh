#!/usr/bin/env bash
# Canonical benchmark entrypoint for the deep-horizon wave-2 session
# (MCP surface + execute stack).
#
# Runs the deterministic MCP/execute workload
# (scripts/run_autoresearch_bench.py) against the current source tree and
# emits one `METRIC <name>=<value>` line per metric on stdout.
# Exits 0 only when every deterministic surface check passes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# Test the current tree, not a stale installed wheel (repo convention:
# `gnn.*` resolves from the repo with PYTHONPATH=src).
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --extra dev python scripts/run_autoresearch_bench.py "$@"
