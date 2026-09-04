# pipeline-worker REPORT — src/pipeline/ (GNN fleet 3, 2026-09-04)

Scope: `src/pipeline/` entirely (18 files touched, 2 created) + `src/tests/pipeline/`.
Head at start: f64ac9085 (main). No git operations performed; edits in place only.

## Files changed + why

### New files (src/pipeline/)
| File | Why |
|---|---|
| `pipeline/_io.py` | Shared atomic-write primitives (`atomic_write_text` / `atomic_write_bytes`: mkstemp in same dir + `os.replace`). Was implemented 3× independently (durable_streams, run_session, run_manifest). |
| `pipeline/_version.py` | Single version source. Kills the `__init__.__version__="1.6.0"` vs `execution.get_pipeline_info()="1.0.0"` drift. |

### Modified (src/pipeline/)
| File | Change |
|---|---|
| `durable_streams.py` | `_atomic_write_text` delegates to `pipeline._io` (private name kept for internal callers). |
| `run_session.py` | `checkpoint` writes through `pipeline._io`; dropped now-unused `os`/`tempfile` imports. |
| `run_manifest.py` | `_write_index` delegates to `pipeline._io` (comment-documented duplication removed). |
| `hasher.py` | `index_run` now writes the history index atomically (was plain `open(...,"w")` — torn writes on concurrent runs); tighter `index` typing. |
| `model_family_acceptance.py` | New public `select_model_families(families, family_names)` — the single family-filter rule (manifest order, whitespace-strip, `KeyError("Unknown model families: …")`). `run_model_family_acceptance` uses it. |
| `semantic_fidelity.py` | `_select_families` delegates to the shared rule (was copy #2). |
| `session_acceptance.py` | `_selected_family_names` delegates (was copy #3). |
| `dag.py` | `total_steps` default now derived from `step_registry.STEPS` (was hardcoded 25); modern annotations (`set[int] \| None`, `list[int]`, `list[str]`); **new `find_circular_dependencies(step_dependencies, nodes=None) -> Set[int]`** (Kahn-peel; cycle members + everything transitively depending on them; unknown deps ignored — same semantics as `resolve_execution_order`). |
| `mcp.py` | `validate_pipeline_dependencies` now computes **real** `circular_dependencies` via `dag.find_circular_dependencies` (was a stubbed always-`[]`). Output keys unchanged. |
| `execution.py` | `_SUCCESS_STATUSES` frozenset dedups the two inline `{"SUCCESS","SUCCESS_WITH_WARNINGS","SKIPPED"}` sets; **new public `resolve_step_numbers(steps, pipeline_data=None)`** (typed wrapper over `_coerce_steps`, now also supports comma-separated lists like the `--only-steps` CLI — `"3,5"` previously resolved silently to `[]`, an error path); `get_pipeline_status`/`get_pipeline_info` derive step counts from the registry; `get_pipeline_info`/`create_pipeline_config` use the package version; path fallbacks use the new constants; `execute_pipeline_steps` result list properly typed. |
| `config.py` | New `DEFAULT_TARGET_DIR` / `DEFAULT_OUTPUT_DIR` constants (single source for the `"input/gnn_files"` / `"output"` literals repeated in ≥4 modules). |
| `context.py` | `PipelineContext` defaults use the shared constants (same values). |
| `__init__.py` | Version from `_version`; imports constants; re-exports `PipelineContext`, `StepRecord`, `StepStatus`; `__all__: list[str]` and now complete (adds `discover_pipeline_steps`, `get_module_info`, `validate_pipeline_step`, `resolve_step_numbers`, `DEFAULT_TARGET_DIR`, `DEFAULT_OUTPUT_DIR`). |
| `preflight.py` | `add_issue` return annotation `Any`→`None`; `to_markdown` `lines: list[str]`; **`validate_config` now validates `pipeline.skip_steps`** with the canonical `pipeline_container_plan.read_skip_steps` parser (bad values are a preflight *error* with a fix hint instead of a mid-run/container-plan failure). |
| `pipeline_validation.py` | `get_pipeline_modules` looped `range(1, 15)` (missed steps 15–24) and had a dead glob statement; now iterates `range(len(STEPS))` from the registry, dead line removed. |
| `AGENTS.md` | Fixed stale API docs (dag/execution signatures were wrong — `resolve_execution_order(["render",...])` etc.); documented all new APIs; added "2026-09-04 — Composability refactor" changelog section; fixed test-suite paths; stamps updated. |
| `README.md` | Same drift fixes + new API docs + corrected usage examples. |

## Tests
- New: `src/tests/pipeline/test_pipeline_refactor_contracts.py` — 20 deterministic, network-free tests pinning: atomic-write round-trip + failure-preservation contract (no `.tmp` residue, original intact), dag cycle/self-loop/downstream/unknown-node semantics + registry-derived default tiering, `resolve_step_numbers` (aliases, `.py` forms, comma lists, `pipeline_data` fallback, dedup/sort), `select_model_families` (order/strip/empty/KeyError), version pin (`get_pipeline_info()["version"] == pipeline.__version__`), context defaults vs constants, `get_output_dir_for_script` contracts (known/unknown/nesting-guard), `index_run` return-path + update-in-place persistence, preflight `skip_steps` gate (valid pass / out-of-range error).
- All 449 pre-existing tests in `src/tests/pipeline/` still pass.

## API deltas
**Additive (no callers broken):** `dag.find_circular_dependencies`, `execution.resolve_step_numbers`, `model_family_acceptance.select_model_families`, `config.DEFAULT_TARGET_DIR`/`DEFAULT_OUTPUT_DIR`, `pipeline._io.atomic_write_text/atomic_write_bytes`, `pipeline._version.__version__`, package-root re-exports + `__all__` completions.
**Behavioral (defect fixes, grep-verified no consumer depends on old behavior):** `get_pipeline_info()["version"]` 1.0.0→1.6.0 (drift fix); mcp `circular_dependencies` now real (was always `[]`); `"3,5"`-style step lists now resolve (previously silent-`[]` → "No valid pipeline steps requested" error); `index_run` write is atomic; `get_pipeline_modules` covers steps 15–24.
**Preserved:** every externally-imported signature (`get_output_dir_for_script` ×~20 consumers, hasher/preflight/dag/step_registry/context/durable_streams/run_session/container_plan/model_family_acceptance sets), exit codes, logging conventions, output contracts.

## Verification output tails
```
uv run ruff check src/pipeline src/tests/pipeline
  → All checks passed!
uv run ruff format --check src/pipeline src/tests/pipeline
  → 71 files already formatted
uv run --extra dev mypy src/pipeline --config-file pyproject.toml
  → Success: no issues found in 31 source files
uv run --extra dev python -m pytest src/tests/pipeline/ -q   (≡ just test-mod pipeline)
  → 469 passed, 2 warnings in 109.96s
```
Note: three transient fleet churn windows hit `import pipeline`/mypy via peer files (`src/utils/pipeline.py`, `src/utils/arg_parsing.py`, `src/gnn/parsers/common.py` syntax errors mid-edit); all were fixed by their owners and final gates ran clean.

## Follow-ups for other workers (not my scope)
- `doc/api/comprehensive_api_reference.md` documents a non-existent `gnn.pipeline` module (`Pipeline`, `StepResult`) and `doc/troubleshooting/api_error_reference.md` shows an outdated `run_pipeline` signature — doc owners should sync with the corrected signatures now in `src/pipeline/{AGENTS,README}.md`.
- `src/utils/pipeline_validator.py` class-name collision (`PipelineValidator` exists in both `utils/` and `pipeline/pipeline_validator.py`) — utils worker.
- Health-rating thresholds (excellent/good/fair/poor) still triplicated with three different formulas across `pipeline_validator`/`diagnostic_enhancer`/`health_check` (all in-scope files, but unifying changes scoring output; deferred as a deliberate behavior-preserving call — needs an owner decision on the canonical formula).
- `mcp.get_pipeline_status` still reads summaries via hardcoded candidate paths; a shared "latest summary locator" helper could serve `mcp.py`, `diagnostic_enhancer.py`, and `verify_pipeline.py`.

## Ideas (beyond this pass)
- Tier-parallel execution of independent step clusters (AGENTS roadmap item) — `dag` already returns tiers.
- Publish per-step streaming metrics via MCP (roadmap item) — `PipelineContext` callbacks (`on_step_start/on_step_complete/on_error`) are the natural hook points.
