# REPORT — ml_integration-worker (GNN module fleet 3, 2026-09-04)

Scope: `src/gnn/ml_integration/` (+ AGENTS/README/SPEC/SKILL) and `src/gnn/14_ml_integration.py`.
Head started: `f64ac9085`. No git commands used; all edits in place.

## Files changed + why

| File | Change | Why |
|---|---|---|
| `src/gnn/ml_integration/processor.py` | Rewritten (582→645 lines, logic decomposed): `process_ml_integration` split into `_check_training_dependencies` / `_discover_gnn_files` / `_collect_features` / `_structural_analysis_entries`; new **public pure helpers** `feature_vector`, `complexity_label`, `summarize_features`; new **public constants** `NUMERIC_FEATURE_NAMES`, `COMPLEXITY_THRESHOLDS`, `COMPLEXITY_LABELS`, `SUMMARY_STATISTIC_KEYS`; training matrix now built via `feature_vector` (single source of column order); internal types tightened (`dict[str, list[int]]` etc.); stale docstring claim `model_family_encoded` removed; **dead `MODEL_FAMILY_ENCODING` deleted** (repo-wide grep: zero consumers) | Composability: pure, typed, reusable units; dedup of the 14-entry feature list that existed twice (name list + hand-rolled row builder); dedup of the two copy-pasted structural-analysis fallback blocks |
| `src/gnn/ml_integration/frameworks.py` | **New.** `check_ml_frameworks()` moved here from `__init__.py`, rewritten data-driven (`SIMPLE_FRAMEWORK_PROBES` + `_probe_module` / `_probe_pytorch`) | Package init stays declarative; framework detection isolated and independently importable |
| `src/gnn/ml_integration/inference.py` | **New.** `load_classifier`, `predict_with_model`, `predict_batch`, `InferenceError(RuntimeError)`; `ImportError` (incl. `ModuleNotFoundError`) caught on artifact load so a sklearn-pickled model without sklearn raises `InferenceError`, never a raw import error | `FEATURES["model_inference"]=True` previously had **no backing API**; closes the loop on the `.pkl` artifacts using the exact canonical training column order |
| `src/gnn/ml_integration/__init__.py` | Rewritten: re-exports expanded, `__all__` fixed (`get_module_info` was missing from it), `__version__` 1.7.0 | Documented public API (`extract_gnn_features` was documented but not exported from package root) |
| `src/gnn/ml_integration/mcp.py` | **Unchanged** | 4 MCP tools preserved — `src/gnn/mcp/audit_report.json` (outside my scope) pins them |
| `src/gnn/14_ml_integration.py` | **Unchanged** (55 lines) | Thin-orchestrator contract intact; verified end-to-end |
| `src/gnn/ml_integration/{AGENTS,README,SPEC,SKILL}.md` | Updated for 1.7.0: new API + Inference section + public constants, unified degradation wording, new test files listed, Last Updated 2026-09-04; zero stale refs (`model_family_encoded`/`MODEL_FAMILY_ENCODING` gone) | Docs of record must match API |
| `tests/ml_integration/test_ml_integration_features.py` | **New**, 23 tests | Pins pure-function behavior (extraction, connectivity math, family detection, feature_vector order/defaults, complexity boundaries, summarize, structural entries) |
| `tests/ml_integration/test_ml_integration_degradation.py` | **New**, 4 tests, env-branching on `find_spec("sklearn")` | Pins degradation contract incl. the doc-aligned `feature_statistics`-always-saved fix; passes with sklearn present OR absent |
| `tests/ml_integration/test_ml_integration_inference.py` | **New**, 8 tests (3 error-path + 1 missing-dependency monkeypatch regression + 4 `pytest.importorskip("sklearn")` full train→predict roundtrips) | Pins InferenceError paths and the training/artifact/predict loop for CI runs with the `ml-ai` extra |

## API deltas (additive; no breaking changes)

- New package-root exports: `extract_gnn_features`, `feature_vector`, `complexity_label`, `summarize_features`, `predict_with_model`, `predict_batch`, `load_classifier`, `InferenceError`, `NUMERIC_FEATURE_NAMES`, `COMPLEXITY_THRESHOLDS`, `COMPLEXITY_LABELS`, `SUMMARY_STATISTIC_KEYS`; `get_module_info` added to `__all__`.
- Removed: `MODEL_FAMILY_ENCODING` (dead code; grep-proved no callers).
- Behavior deltas (both doc-aligned): (1) `feature_statistics`/`model_families` now saved in the results JSON for **both** degradation cases (sklearn-missing AND <2 files) — AGENTS.md of record already claimed this; (2) framework probe reports `available=True, version=None` instead of crashing `AttributeError` for a module without `__version__`.
- Preserved exactly: all existing signatures, log messages, degradation note strings, exit codes, output filenames (`ml_integration_results.json`, `gnn_*.pkl`), JSON key contracts.

## Verification output tails

```
uv run ruff check src/gnn/ml_integration tests/ml_integration   -> "All checks passed!"
uv run --extra dev mypy src/gnn/ml_integration --config-file pyproject.toml
                                                                -> "Success: no issues found in 5 source files"
uv run pytest tests/ml_integration/ -q                      -> "54 passed, 4 skipped in 0.48s"
python src/gnn/14_ml_integration.py --target-dir … --output-dir …   -> exit 0; standard logging; 14_ml_integration_output/ written
smoke: InferenceError on missing artifact + garbage pickle; feature_vector len 14 in
      NUMERIC_FEATURE_NAMES order; complexity_label(50/500/5000)=small/medium/large;
      summarize min/max/mean exact; frameworks keys {jax,pytorch,sklearn,tensorflow}
```

sklearn is **absent** in the dev environment (not installed, per fleet rules). The 4 gated
roundtrip tests skip locally and execute wherever the `ml-ai` extra is synced.

## Doc/ or manuscript/ follow-ups (other workers own those)

- `src/gnn/pipeline/pipeline_validation.py` expects `ml_integration_summary.json` for step 14, but the module (always has) writes `ml_integration_results.json` — pre-existing cross-module discrepancy for the pipeline/worker to reconcile; untouched by me.
- `src/gnn/mcp/audit_report.json` + `validate-mcp-manifest`: a future 5th tool (e.g. `predict_ml_label`) would expose the new inference API; needs that file, which is outside my scope.
- If `doc/` or `manuscript/` enumerates module files, they may want `frameworks.py`/`inference.py` mentioned (module-local docs already updated).

## Follow-up ideas

- `predict_proba_with_model` when the artifact exposes `predict_proba`.
- Self-describing sidecar JSON per `.pkl` (feature order + label_names + training snapshot) so artifacts travel without the results JSON.
- GNN markdown parsing here duplicates parts of `src/gnn/` — a shared extraction surface would remove a second parser (needs a cross-module decision).
- Consider one CI leg with the `ml-ai` extra so the 4 gated tests actually execute in CI.
