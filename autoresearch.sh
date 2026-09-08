#!/usr/bin/env bash
# =============================================================================
# autoresearch.sh — orchestration-gate benchmark for the GNN pipeline
# (deep-horizon wave 2: pipeline orchestration territory)
#
# Deterministic, offline workload:
#   1. scripts/run_v3_orchestration_acceptance.py --strict
#      (v3.0.0 long-running orchestration contracts: 19 fail-closed checks)
#   2. Targeted orchestration pytest subset (36 files, 464 tests, -n 4):
#      step registry, orchestrators, pipeline config/context/scripts, health
#      surface, preflight, diagnostics, runtime validator, durable streams,
#      run manifests/sessions, container plans, wave-2 wiring.
#
# Primary metric: orchestration_gate_seconds (lower is better) — total wall
# time of the full gate. Secondary metrics decompose it and verify greenness:
#   acceptance_seconds, acceptance_checks, pytest_seconds,
#   tests_passed, tests_failed, tests_skipped, tests_total
#
# Determinism: fixed test selection, PYTHONHASHSEED=0, lockfile-pinned env
# (uv --frozen), pytest.ini already pins -p no:randomly and per-test timeouts.
# Exit 0 only when every check/test is green; on hard failure, diagnostics go
# to stderr and the script exits non-zero (METRIC lines are printed when the
# pytest phase completes).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

export UV_NO_PROGRESS=1
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1

ACC_LOG=/tmp/gnn-pipe-w2-acceptance.txt
PYT_LOG=/tmp/gnn-pipe-w2-pytest.txt

num() { python3 -c 'import time; print(time.time())'; }
sec() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f", b - a }'; }
emit() { echo "METRIC $1=$2"; }

# Untimed: ensure the dev environment is materialized so timed sections
# measure the workload, not dependency resolution.
uv sync --frozen --extra dev >/dev/null 2>&1

TOTAL_START=$(num)

# --- 1) v3 orchestration acceptance gate ------------------------------------
ACC_START=$(num)
set +e
uv run --no-sync python scripts/run_v3_orchestration_acceptance.py --strict \
    > "$ACC_LOG" 2>&1
ACC_STATUS=$?
set -e
if [ "$ACC_STATUS" -ne 0 ]; then
    echo "ORCHESTRATION ACCEPTANCE GATE FAILED (exit $ACC_STATUS)" >&2
    tail -n 40 "$ACC_LOG" >&2
    exit 1
fi
ACC_END=$(num)

CHECK_LINE=$(grep -o '[0-9]*/[0-9]* checks passed' "$ACC_LOG" | tail -n 1)
CHECKS_PASSED=${CHECK_LINE%%/*}
CHECKS_TOTAL=${CHECK_LINE#*/}
CHECKS_TOTAL=${CHECKS_TOTAL%% *}
if [ "$CHECKS_PASSED" != "$CHECKS_TOTAL" ]; then
    echo "ACCEPTANCE GATE INCOMPLETE: $CHECK_LINE" >&2
    exit 1
fi

# --- 2) orchestration pytest subset ------------------------------------------
SUBSET="
tests/pipeline/test_pipeline_overall.py
tests/pipeline/test_pipeline_orchestration.py
tests/pipeline/test_pipeline_config.py
tests/pipeline/test_pipeline_context.py
tests/pipeline/test_pipeline_scripts.py
tests/pipeline/test_pipeline_main.py
tests/pipeline/test_main_orchestrator.py
tests/pipeline/test_step_registry.py
tests/pipeline/test_step_timeouts.py
tests/pipeline/test_preflight_behavior.py
tests/pipeline/test_preflight_diagnostics.py
tests/pipeline/test_health_check.py
tests/pipeline/test_diagnostic_enhancer.py
tests/pipeline/test_pipeline_runtime_validator.py
tests/pipeline/test_pipeline_schemas.py
tests/pipeline/test_run_manifest.py
tests/pipeline/test_run_session.py
tests/pipeline/test_session_acceptance.py
tests/pipeline/test_wave2_manifests.py
tests/pipeline/test_wave2_sessions.py
tests/pipeline/test_wave2_container_plans.py
tests/pipeline/test_wave2_identity.py
tests/pipeline/test_container_plan.py
tests/pipeline/test_durable_streams.py
tests/pipeline/test_hasher.py
tests/pipeline/test_pipeline_logging_config.py
tests/pipeline/test_pipeline_refactor_contracts.py
tests/pipeline/test_pipeline_stage_hardening_review.py
tests/pipeline/test_autonomous_contract.py
tests/pipeline/test_pipeline_infrastructure.py
tests/pipeline/test_pipeline_integration.py
tests/pipeline/test_pipeline_functionality.py
tests/pipeline/test_pipeline_error_scenarios.py
tests/pipeline/test_pipeline_improvements_validation.py
tests/pipeline/test_pipeline_recovery.py
tests/pipeline/test_pipeline_performance.py
"
# shellcheck disable=SC2046
SUBSET_ARGS=$(printf '%s ' $SUBSET)

PYT_START=$(num)
set +e
uv run --no-sync python -m pytest $SUBSET_ARGS \
        -q --tb=no -n 4 > "$PYT_LOG" 2>&1
PYT_STATUS=$?
set -e
if [ "$PYT_STATUS" -ne 0 ]; then
    echo "ORCHESTRATION PYTEST SUBSET FAILED (exit $PYT_STATUS)" >&2
    grep -E '^(FAILED|ERROR)' "$PYT_LOG" | head -n 20 >&2
fi
summary=$(tail -n 1 "$PYT_LOG")
count() { echo "$summary" | grep -o "[0-9]* $1" | head -n 1 | grep -o '^[0-9]*' || echo 0; }
PYT_END=$(num)
PYT_EXIT=0
PASSED=$(count passed)
FAILED=$(count failed)
if [ "$PYT_STATUS" -ne 0 ]; then PYT_EXIT=1; fi
SKIPPED=$(count skipped)
DESELECTED=$(count deselected)
TESTS_TOTAL=$((PASSED + FAILED + SKIPPED + DESELECTED))

TOTAL_END=$(num)

# --- metrics ------------------------------------------------------------------
emit orchestration_gate_seconds "$(sec "$TOTAL_START" "$TOTAL_END")"
emit acceptance_seconds "$(sec "$ACC_START" "$ACC_END")"
emit acceptance_checks "$CHECKS_PASSED"
emit pytest_seconds "$(sec "$PYT_START" "$PYT_END")"
emit tests_passed "$PASSED"
emit tests_failed "$FAILED"
emit tests_skipped "$SKIPPED"
emit tests_total "$TESTS_TOTAL"

if [ "$PYT_EXIT" -ne 0 ]; then
    echo "BENCHMARK FAILED: $FAILED failing tests in orchestration subset" >&2
    exit 1
fi
echo "BENCHMARK OK: $CHECKS_PASSED/$CHECKS_TOTAL acceptance checks, $PASSED tests passed"
