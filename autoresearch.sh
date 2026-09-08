#!/usr/bin/env bash
# autoresearch.sh — deterministic verification-layer benchmark for the GNN pipeline.
#
# Workload (identical every run):
#   1. `uv sync --frozen` with the exact extras `dev + ml-ai + torch`
#      (lock-pinned versions; the ml-ai/torch extras unlock the 12
#      environment-skipped tests: 11 sklearn cases in
#      tests/ml_integration/test_ml_integration_inference.py and 1 torch
#      execution case in tests/render/test_continuous_renderers.py).
#   2. The CI coverage-parity test selection (just test-cov / ci.yml main run):
#      -m "not pipeline and not mcp", both Ollama files ignored, executed
#      offline under pytest-xdist with a FIXED -n 4 (deterministic
#      distribution; CI itself uses -n auto --dist worksteal, which is not
#      reproducible run-to-run).
#   3. Coverage over the installed `gnn` package (--cov=gnn), JSON report.
#
# Metrics (printed as `METRIC name=value` lines):
#   coverage_percent — line coverage of the gnn package (PRIMARY, higher=better)
#   tests_passed / tests_failed / tests_skipped — junit-aggregated counts
#   suite_seconds    — wall-clock seconds for the pytest run
#
# Artifacts (all under .gitignore'd paths, tree stays clean):
#   pytest-junit-autoresearch/junit.xml, pytest-junit-autoresearch/pytest-run.log
#   coverage.json (repo root; ignored by .gitignore)
set -euo pipefail
cd "$(dirname "$0")"

# Deterministic, CI-parity environment (PYTHONPATH/PYTHONHASHSEED mirror ci.yml env).
export PYTHONPATH=src
export PYTHONHASHSEED=0
export TZ=UTC
export MPLBACKEND=Agg

EXTRAS=(--extra dev --extra ml-ai --extra torch)
if ! uv sync --frozen --quiet "${EXTRAS[@]}"; then
    echo "autoresearch.sh: uv sync failed" >&2
    uv sync --frozen "${EXTRAS[@]}" || exit 3   # retry un-quieted so the error is visible
    exit 3
fi

ART=pytest-junit-autoresearch   # ignored via .gitignore 'pytest-junit-*/'
mkdir -p "$ART"

START=$SECONDS
set +e
uv run --no-sync python -m pytest tests/ \
  -n 4 \
  -q \
  -m "not pipeline and not mcp" \
  --ignore=tests/llm/test_llm_ollama.py \
  --ignore=tests/llm/test_llm_ollama_integration.py \
  --cov=gnn \
  --cov-report=json:coverage.json \
  --junitxml="$ART/junit.xml" \
  > "$ART/pytest-run.log" 2>&1
PYTEST_RC=$?
set -e
WALL_SECONDS=$((SECONDS - START))

# pytest exit codes 3 (internal error) and 4 (usage error) are harness
# failures, not measurements. Exit code 2 (interrupted / collection error)
# still yields a parseable junit whose error count surfaces via tests_failed.
if [ "$PYTEST_RC" -ge 3 ]; then
    echo "autoresearch.sh: pytest exited with $PYTEST_RC; last log lines:" >&2
    tail -n 40 "$ART/pytest-run.log" >&2 || true
    exit 2
fi

uv run --no-sync python - "$ART/junit.xml" coverage.json "$WALL_SECONDS" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET

junit_path, coverage_path, wall = sys.argv[1], sys.argv[2], int(sys.argv[3])

try:
    suites = list(ET.parse(junit_path).getroot().iter("testsuite"))
except Exception as exc:
    print(f"autoresearch.sh: cannot parse junit report: {exc}", file=sys.stderr)
    sys.exit(2)

tests = sum(int(s.get("tests", 0)) for s in suites)
errors = sum(int(s.get("errors", 0)) for s in suites)
failures = sum(int(s.get("failures", 0)) for s in suites) + errors
skipped = sum(int(s.get("skipped", 0)) for s in suites)
passed = tests - failures - skipped

try:
    with open(coverage_path, encoding="utf-8") as fh:
        percent = json.load(fh)["totals"]["percent_covered"]
except Exception as exc:
    print(f"autoresearch.sh: cannot parse coverage report: {exc}", file=sys.stderr)
    sys.exit(2)

if tests == 0:
    print("autoresearch.sh: zero tests collected", file=sys.stderr)
    sys.exit(2)

print(f"METRIC coverage_percent={percent:.2f}")
print(f"METRIC tests_passed={passed}")
print(f"METRIC tests_failed={failures}")
print(f"METRIC tests_skipped={skipped}")
print(f"METRIC suite_seconds={wall}")
PY
