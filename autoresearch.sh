#!/usr/bin/env bash
# MAJ-05 benchmark: de-duplication of the validate_gnn* public surface.
#
# Deterministic workload: AST-based audit of src/gnn (no network, no clock,
# no randomness) plus — once the alias regression tests exist — a runtime
# behavior check that old names still work identically and emit
# DeprecationWarning.
#
# Primary metric:   validate_noncanonical_defs   (lower is better; 0 = done)
# Secondary:        validate_defs_total, validate_aliases_ok,
#                   validate_aliases_bad, internal_callers_noncanonical,
#                   doc_refs_noncanonical, alias_tests_passed
set -euo pipefail
cd "$(dirname "$0")"

# 1) Static surface audit (stdlib-only AST scan).
uv run --extra dev python scripts/audit_validate_surface.py

# 2) Runtime alias-behavior proof (present from the first migration commit).
if [ -f tests/test_validate_surface_aliases.py ]; then
    uv run --extra dev python -m pytest tests/test_validate_surface_aliases.py -q --tb=short -p no:cacheprovider
    echo "METRIC alias_tests_passed=1"
fi
