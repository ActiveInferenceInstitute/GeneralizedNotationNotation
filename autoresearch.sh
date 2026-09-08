#!/usr/bin/env bash
# Canonical autoresearch benchmark entrypoint for the GNN render backends.
#
# Runs the deterministic render-backend conformance workload (see
# scripts/bench_render_backends.py) and prints METRIC lines. Exits 0 when the
# workload completes and emits its metrics; non-zero on harness failure.
#
# Note: main's copy of this file flip-flops between wave-2 sessions (each
# active autoresearch session keeps its own entrypoint here). The MCP/execute
# workload lives at scripts/run_autoresearch_bench.py and the utility/analysis
# suite harness in that session's history; this entrypoint is owned by the
# render-backend conformance session.
set -euo pipefail
cd "$(dirname "$0")"

exec uv run --frozen python scripts/bench_render_backends.py
