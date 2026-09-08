#!/usr/bin/env bash
# Canonical autoresearch benchmark entrypoint for the GNN render backends.
#
# Runs the deterministic render-backend conformance workload (see
# scripts/bench_render_backends.py) and prints METRIC lines. Exits 0 when the
# workload completes and emits its metrics; non-zero on harness failure.
set -euo pipefail
cd "$(dirname "$0")"

exec uv run --frozen python scripts/bench_render_backends.py
