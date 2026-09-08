#!/usr/bin/env bash
# =============================================================================
# autoresearch.sh — deterministic benchmark harness for the GNN pipeline
# "utility + analysis" territory (deep horizon wave 2).
#
# Territory: src/gnn/{analysis,utils,advanced_visualization,type_checker,schemas,api}
#
# Workload (deterministic, offline, fixed environment):
#   1. ruff check over the territory          -> ruff_errors
#   2. mypy --strict over the territory       -> mypy_strict_errors
#   3. deterministic pytest subset over the
#      territory's own test directories       -> passed/failed/skipped counts
#
# Primary metric:
#   quality_violations = mypy_strict_errors + ruff_errors   (lower is better)
#
# Success: exit 0 with all METRIC lines emitted.
# Failure: any stage crashes without producing a measurable summary -> exit 1.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

TERRITORY=(
  src/gnn/analysis
  src/gnn/utils
  src/gnn/advanced_visualization
  src/gnn/type_checker
  src/gnn/schemas
  src/gnn/api
)

# Territory-owned test directories (deterministic, offline, no Ollama/Julia/browser).
TEST_DIRS=(
  tests/analysis
  tests/advanced_visualization
  tests/api
  tests/type_checker
  tests/utils
)

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

count_or_zero() { # count_or_zero <pattern> <file>
  grep -cE "$1" "$2" 2>/dev/null || true
}

# --- 1. ruff over the territory ----------------------------------------------
RUFF_LOG="$OUT/ruff.log"
if ! uv run --extra dev ruff check "${TERRITORY[@]}" --output-format concise \
    >"$RUFF_LOG" 2>&1; then
  echo "ruff completed with findings (expected) — counting" >&2
fi
RUFF_ERRORS="$(count_or_zero ': [A-Z]+[0-9]+ ' "$RUFF_LOG")"
[ -n "$RUFF_ERRORS" ] || RUFF_ERRORS=0

# --- 2. mypy --strict over the territory --------------------------------------
MYPY_LOG="$OUT/mypy.log"
if ! uv run --extra dev mypy --strict --follow-imports=silent "${TERRITORY[@]}" --config-file pyproject.toml \
    >"$MYPY_LOG" 2>&1; then
  echo "mypy completed with findings (expected) — counting" >&2
fi
MYPY_ERRORS="$(count_or_zero 'error:' "$MYPY_LOG")"
[ -n "$MYPY_ERRORS" ] || MYPY_ERRORS=0

# --- 3. deterministic territory test subset ------------------------------------
PYTEST_LOG="$OUT/pytest.log"
set +e
uv run --extra dev python -m pytest "${TEST_DIRS[@]}" \
  -q -n 4 --tb=no -p no:cacheprovider \
  -m "not pipeline and not mcp" \
  >"$PYTEST_LOG" 2>&1
PYTEST_RC=$?
set -e

PASSED="$(grep -oE '[0-9]+ passed' "$PYTEST_LOG" | tail -1 | grep -oE '[0-9]+' || true)"
FAILED="$(grep -oE '[0-9]+ failed' "$PYTEST_LOG" | tail -1 | grep -oE '[0-9]+' || true)"
SKIPPED="$(grep -oE '[0-9]+ skipped' "$PYTEST_LOG" | tail -1 | grep -oE '[0-9]+' || true)"
ERRORS="$(grep -oE '[0-9]+ errors?' "$PYTEST_LOG" | tail -1 | grep -oE '[0-9]+' || true)"
PASSED="${PASSED:-0}"; FAILED="${FAILED:-0}"; SKIPPED="${SKIPPED:-0}"; ERRORS="${ERRORS:-0}"

# A collection error / internal error yields no summary: harness failure.
if [ "$PASSED" -eq 0 ] && [ "$FAILED" -eq 0 ] && [ "$SKIPPED" -eq 0 ]; then
  echo "HARNESS FAILURE: pytest produced no summary (rc=$PYTEST_RC)" >&2
  tail -30 "$PYTEST_LOG" >&2 || true
  exit 1
fi

QUALITY=$((MYPY_ERRORS + RUFF_ERRORS))
echo "METRIC quality_violations=$QUALITY"
echo "METRIC mypy_strict_errors=$MYPY_ERRORS"
echo "METRIC ruff_errors=$RUFF_ERRORS"
echo "METRIC territory_tests_passed=$PASSED"
echo "METRIC territory_tests_failed=$FAILED"
echo "METRIC territory_tests_errors=$ERRORS"
echo "METRIC territory_tests_skipped=$SKIPPED"
