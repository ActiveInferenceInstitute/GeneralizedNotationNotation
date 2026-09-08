#!/usr/bin/env bash
# autoresearch.sh — benchmark harness for MAJ-04 module decomposition.
#
# Goal context: MAJ-04 (TO-DO.md "Open Scoped Roadmap") — decompose the six
# >2000-line modules (integration/meta_analysis/visualizer.py 2871,
# analysis/visualizations.py 2412, testing/test_round_trip.py 2214,
# render/jax/jax_renderer.py 2200, render/discopy/translator.py 2150,
# analysis/analyzer.py 2031) following the 3.3.0 execute/processor.py split
# pattern (mechanical extraction into sibling modules, facade re-exports
# preserved, one module per PR).
#
# Primary metric:
#   oversized_module_lines — total line count of the six MAJ-04 target files
#   with more than 2000 lines. Lower is better; 0 when MAJ-04 is complete.
#
# Secondary metrics:
#   oversized_modules    — count of MAJ-04 target files still >2000 lines
#   largest_module_lines — line count of the largest remaining target file
#   gnn_python_lines     — total lines across all tracked src/gnn .py files
#                          (guard: decomposition MOVES code, it must not
#                          delete it; this total should stay roughly constant)
#   ruff_violations      — `ruff check src/gnn scripts` violations (gate: 0)
#   mypy_errors          — `mypy src/gnn` errors (gate: 0)
#   import_failures      — MAJ-04 module import paths that no longer resolve
#                          (gate: 0; enforces "no import-path changes for
#                          consumers")
#
# Scope note: src/gnn/mcp/mcp.py (2002 lines) is intentionally excluded —
# TO-DO.md scopes that file to MAJ-06 (MCP wrapper collapse), not MAJ-04.
#
# Determinism: pure filesystem census + lockfile-pinned toolchain
# (uv sync --frozen). No network after a warm uv cache, no time-of-day input,
# sorted file order. Import checks execute real package imports inside the
# pinned venv.
set -uo pipefail
cd "$(dirname "$0")" || exit 2

fail=0

# --- pinned toolchain --------------------------------------------------------
if ! uv sync --extra dev --frozen --quiet; then
    echo "GATE-FAIL: uv sync --extra dev --frozen"
    exit 2
fi
run() { uv run --no-sync "$@"; }

# --- primary metric: MAJ-04 oversized-module census ---------------------------
if ! run python - <<'PYEOF'
import subprocess
from pathlib import Path

files = sorted(
    f
    for f in subprocess.run(
        ["git", "ls-files", "src/gnn"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    if f.endswith(".py")
)
# MAJ-04 target set; mcp/mcp.py excluded (scoped to MAJ-06, see header note).
MAJ04_EXCLUDED = {"src/gnn/mcp/mcp.py"}
total = oversized_sum = oversized_n = largest = 0
for f in files:
    data = Path(f).read_bytes()
    n = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
    total += n
    if f in MAJ04_EXCLUDED:
        continue
    if n > 2000:
        print(f"OVERSIZED {n:6d}  {f}")
        oversized_sum += n
        oversized_n += 1
        largest = max(largest, n)
print(f"METRIC oversized_module_lines={oversized_sum}")
print(f"METRIC oversized_modules={oversized_n}")
print(f"METRIC largest_module_lines={largest}")
print(f"METRIC gnn_python_lines={total}")
PYEOF
then
    echo "GATE-FAIL: module census"
    fail=1
fi

# --- ruff gate -----------------------------------------------------------------
echo "--- ruff check src/gnn scripts ---"
ruff_out=$(run ruff check src/gnn scripts 2>&1)
ruff_rc=$?
ruff_v=$(printf '%s\n' "$ruff_out" | sed -n 's/^Found \([0-9][0-9]*\) error.*/\1/p' | tail -n 1)
if [ "$ruff_rc" -eq 0 ]; then
    ruff_v=0
elif [ -z "$ruff_v" ]; then
    ruff_v=1  # failed without a parseable count; treat as >=1 violation
fi
echo "METRIC ruff_violations=$ruff_v"
printf '%s\n' "$ruff_out" | tail -n 3
[ "$ruff_rc" -eq 0 ] || { echo "GATE-FAIL: ruff"; fail=1; }

# --- mypy gate -----------------------------------------------------------------
echo "--- mypy src/gnn ---"
mypy_out=$(run mypy src/gnn --show-error-codes 2>&1)
mypy_rc=$?
mypy_e=$(printf '%s\n' "$mypy_out" | sed -n 's/^Found \([0-9][0-9]*\) error.*/\1/p' | tail -n 1)
if [ "$mypy_rc" -eq 0 ]; then
    mypy_e=0
elif [ -z "$mypy_e" ]; then
    mypy_e=1  # failed without a parseable count; treat as >=1 error
fi
echo "METRIC mypy_errors=$mypy_e"
printf '%s\n' "$mypy_out" | tail -n 3
[ "$mypy_rc" -eq 0 ] || { echo "GATE-FAIL: mypy"; fail=1; }

# --- import-path stability gate -------------------------------------------------
echo "--- import-path stability (MAJ-04 consumer surfaces) ---"
if ! run python - <<'PYEOF'
import importlib.util
import sys

MODULES = [
    "gnn.integration.meta_analysis.visualizer",
    "gnn.analysis.visualizations",
    "gnn.testing.test_round_trip",
    "gnn.render.jax.jax_renderer",
    "gnn.render.discopy.translator",
    "gnn.analysis.analyzer",
]
failed = 0
for m in MODULES:
    try:
        spec = importlib.util.find_spec(m)
    except Exception as exc:  # parent-package import failure also breaks consumers
        print(f"IMPORT-FAIL {m}: {exc!r}")
        failed += 1
        continue
    if spec is None or not spec.origin:
        print(f"IMPORT-FAIL {m}")
        failed += 1
    else:
        print(f"IMPORT-OK   {m}")
print(f"METRIC import_failures={failed}")
sys.exit(1 if failed else 0)
PYEOF
then
    echo "GATE-FAIL: import paths"
    fail=1
fi

# --- verdict --------------------------------------------------------------------
if [ "$fail" -ne 0 ]; then
    echo "RESULT: FAIL (gate violation)"
    exit 1
fi
echo "RESULT: OK"
exit 0
