# REPORT — Analysis-FrameworkRenderers (analysis/numpyro, analysis/pytorch, generate_cross_model_report)

Scope owner: Analysis-FrameworkRenderers (gnn-swarm-2)
Date: 2026-08-24
Status: Complete — no commits/stages made (per HARD RULE).

## Summary

Added direct pytest coverage for the previously-uncovered numpyro/pytorch
analyzers and the cross-model comparison report generator, and fixed one real
bug uncovered by the new tests (the `plots_generated` flag lied when matplotlib
was unavailable).

## Changes

### Source (bug fix — root cause)

- `src/analysis/numpyro/analyzer.py`
  - `_generate_plots` now returns `bool` (True iff at least one plot artifact was
    written) instead of `None`.
  - The caller sets `analysis["plots_generated"]` from that return value instead
    of hardcoding `True`.
  - Root cause fixed: previously `_generate_plots` swallowed the matplotlib
    `ImportError` internally and returned normally, so `plots_generated` was
    always `True` even when no plots were produced — a misleading documented flag.
- `src/analysis/pytorch/analyzer.py`
  - Identical fix (PyTorch action-bar color `#4285F4` preserved).

### Tests added (new files only, under mirror scope)

- `src/tests/analysis/test_numpyro_pytorch_analyzers.py` (14 parametrized tests:
  8 scenarios x numpyro+pytorch)
  - end-to-end on realistic runner-shaped `simulation_results.json` (beliefs /
    actions / efe_history / validation): asserts documented analysis JSON
    structure (framework, model_name, num_timesteps, num_states, validation,
    metrics) with pinned metric values (mean_confidence 0.9, final_confidence
    1.0, action_distribution, mean_efe 0.6); when matplotlib is present asserts
    `plots_generated is True` and the three PNGs exist (regression for the bug).
  - graceful degradation: matplotlib absent => `plots_generated is False`,
    analysis still written.
  - empty / missing results dir => `[]`.
  - malformed JSON result => skipped, returns `[]`.
  - disjoint-scope discovery: analyzer ignores the *other* framework's results.
  - root-level `simulation_results.json` recovery, and default output_dir.
- `src/tests/analysis/test_generate_cross_model_report.py` (14 tests)
  - end-to-end report generation from a realistic multi-model/multi-framework
    `12_execute_output`: asserts valid markdown structure (Summary Matrix, EFE,
    Entropy, Execution Time, Per-Model Details, Cross-Model Observations) and
    pinned rendered metrics (validation + confidence `✅ 0.900`, EFE `0.5333`,
    execution times from the summary file).
  - empty / missing execution dir => returns `""` (graceful, no file written).
  - `allowed_frameworks` / `allowed_model_names` scope filtering.
  - metric extractor behavior on nested `simulation_trace` schema and empty
    inputs; `_validation_status` truth/failure/mixed combos.
  - collection helpers: malformed file skipped; nested `summaries/` times;
    missing summary handled.

No production API signatures changed (`_generate_plots` is private, internal
only). No docs touched (AGENTS/docs out of scope).

## Scoped verification (all green)

- `uv run --extra dev ruff check src/analysis/numpyro src/analysis/pytorch src/analysis/generate_cross_model_report.py` -> All checks passed
- `uv run --extra dev ruff check <both new test files>` -> All checks passed
- `uv run --extra dev mypy src/analysis/numpyro src/analysis/pytorch src/analysis/generate_cross_model_report.py --config-file pyproject.toml` -> Success, no issues in 5 source files
- `uv run --extra dev pytest src/tests/analysis -q --tb=short` -> 220 passed
  (192 baseline + 28 new), 0 failed. Baseline suite re-run to confirm the bug
  fix introduced no regressions.

## Notes

- No commits/pushes/stages; only scoped files modified. Sibling agents' changes
  in the working tree (gnn, gui, ontology, research, security, type_checker,
  validation, docs) were left untouched.
