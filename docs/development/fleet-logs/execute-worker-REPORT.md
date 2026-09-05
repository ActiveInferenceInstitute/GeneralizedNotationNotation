# execute-worker — Fleet Report (fleet3, 2026-09-04)

Scope: `src/execute/` (entire package incl. AGENTS.md/README/SKILL/SPEC) + numbered orchestrator `src/12_execute.py`. Module considered separately and alone.

## Files changed + why

### Refactor (composability / dedup / pure helpers)
| File | Why |
|---|---|
| `src/execute/processor.py` | Decomposed the two ~450-line functions. `process_execute` now delegates to pure helpers `_init_execution_summary`, `_update_framework_status`, `_classify_execute_outcome` (returns new `ExecutionOutcome` dataclass), `_write_execution_summaries`. Unified the four per-script envelope factories (`_make_skipped_result`, `_make_local_worker_pool_failure_result`, `_make_distributed_dispatch_failure_result`, `_new_execution_result`) onto a shared `_base_execution_envelope` + `_model_framework_from_path` (kills ~80 lines of near-identical 15-key dicts; key sets preserved). Removed the inline per-call `ErrorResult` class in `execute_single_script` — failure carriers now use `subprocess.CompletedProcess`, so success and failure paths share one return-code/stdout/stderr shape. Hoisted the bottom-of-file `data_extractors` imports to the top import block. |
| `src/execute/types.py` | Added `ExecutionOutcome` frozen dataclass (pure classification result) and `ExecutionPlan` TypedDict (the dry-run planner's typed return). Leaf module — no new imports of execute siblings. |
| `src/execute/executor.py` | Made `_initialize_execution_results` data-driven off `_framework_specs()` (the 7 `*_executions` keys now derive from the registry, so adding a framework is one edit, not four). Added public `list_frameworks()` registry introspection. |
| `src/execute/detection.py` | Added `RENDER_OUTPUT_DIR_NAME = "11_render_output"` constant and routed the 10× literal in `_resolve_render_output_dir` through it. |
| `src/execute/pymdp/pymdp_simulation.py` | Replaced the process-global `warnings.filterwarnings("ignore")` (which silenced ALL warnings process-wide at import) with a scoped `warnings.catch_warnings()` context around only the pymdp-backed `.simulation` import. User warning filters are now preserved. |
| `src/execute/rxinfer/rxinfer_runner.py` | Removed the silent `try/except ImportError` fallback that redefined `is_julia_available` with a PATH-only check (dropping version checks). `execute.julia_setup` is a core module — the fallback could never trigger and silently degraded behavior if it did. |
| `src/execute/activeinference_jl/activeinference_runner.py` | Same silent-fallback removal as rxinfer. |

### New functionality (additive, typed, documented)
| File | Why |
|---|---|
| `src/execute/planning.py` (new) | `plan_execute(target_dir, output_dir, frameworks="all", **config) -> ExecutionPlan` — dry-run Step 12 planner. Composes the same `detection`/`metadata`/`framework_availability` primitives as `process_execute` but runs **no scripts and no Julia package probing** (only a PATH-only `julia` lookup). Returns a typed `ExecutionPlan` with per-script dispositions (`would_execute` / `would_skip_dependency` / `unknown_framework_scripts`), `missing_render_scripts`, `render_failures`, and `status`. For preflight, CI gates, debugging. |
| `src/execute/executor.py` | `list_frameworks() -> list[dict]` — public introspection over the `ExecutorFrameworkSpec` registry (framework, result_key, available, operation). |
| `src/execute/__init__.py` | Exported `plan_execute` and `list_frameworks` from the package facade. |

### Bug fix
| File | Why |
|---|---|
| `src/execute/mcp.py` | `check_execute_dependencies_mcp` returned non-JSON-serializable `ValidationResult` dataclass objects (`check_dependencies()` returns `List[ValidationResult]`, never a dict, so the `isinstance(result, dict)` branch was dead). Now serializes each result via `dataclasses.asdict` so the MCP payload is JSON-serializable. Docstring corrected (it claimed find_spec probes for jax/torch that don't exist). |

### Tests (new, deterministic, no network)
| File | Tests |
|---|---|
| `src/tests/execute/test_execute_outcome_classification.py` | 14 tests pinning the pure `_classify_execute_outcome` truth table (every reason branch + exit-code derivation + `attempted` math). |
| `src/tests/execute/test_execute_envelope_factories.py` | 8 tests pinning the exact key set + discriminating values of each envelope factory (the unification cannot silently drift a key). |
| `src/tests/execute/test_execute_plan.py` | 7 tests for `plan_execute` using the real `render_processing_summary.json` contract schema (file_results → framework_results): no render output, no executable scripts, pymdp classification, missing render scripts, render failures, unsupported framework omission, invalid frameworks. |
| `src/tests/execute/test_execute_introspection.py` | 3 tests: `list_frameworks` registry shape + result-key mapping; `check_execute_dependencies_mcp` returns JSON-serializable plain-dict payload. |

### Docs of record
| File | Why |
|---|---|
| `src/execute/AGENTS.md` | Added `plan_execute` and `list_frameworks` API Reference entries with signatures, descriptions, and a usage example. |
| `src/execute/README.md` | Added `list_frameworks` to `executor.py` Core Components; added a `planning.py` Core Components section. |
| `src/execute/SKILL.md` | Added `plan_execute`/`list_frameworks` to the import block, usage examples, and Key Exports. |

## API deltas

**Added (public, exported from `execute`):**
- `execute.plan_execute(target_dir, output_dir, frameworks="all", **config) -> ExecutionPlan`
- `execute.list_frameworks() -> list[dict]`
- `execute.types.ExecutionOutcome` (frozen dataclass)
- `execute.types.ExecutionPlan` (TypedDict)

**Added (package-internal, underscore-private but importable for tests):**
- `execute.processor._base_execution_envelope`, `_model_framework_from_path`, `_init_execution_summary`, `_update_framework_status`, `_classify_execute_outcome`, `_write_execution_summaries`

**Changed (behavior-preserving):**
- `execute.processor.process_execute` — decomposed into helpers; return values, exit codes, summary JSON shape, log messages, and output contracts unchanged.
- `execute.processor.execute_single_script` — `ErrorResult` local class replaced with `subprocess.CompletedProcess`; observable result dict unchanged.
- `execute.executor._initialize_execution_results` — `*_executions` keys now derived from `_framework_specs()`; same keys, same order.
- `execute.mcp.check_execute_dependencies_mcp` — `dependencies` entries are now plain dicts instead of dataclass objects (the MCP contract was already "plain dicts"; this makes it true).

**Removed:**
- Inline `ErrorResult` class in `execute_single_script` (replaced by stdlib `CompletedProcess`).
- Silent `is_julia_available` ImportError fallbacks in `rxinfer_runner.py` and `activeinference_runner.py`.
- Process-global `warnings.filterwarnings("ignore")` in `pymdp_simulation.py`.

**No public entry point removed; no public signature changed.** All pre-existing external consumers (`src/12_execute.py`, `render/jax/jax_renderer.py`, `render/pymdp/pymdp_templates.py`, `tests/execute/*`, `tests/pipeline/*`, `tests/render/*`, `tests/security/test_sandbox.py`) keep working — verified by the full execute suite + grep of every imported symbol.

## Verification output tails

```
$ uv run ruff check src/execute src/tests/execute
All checks passed!

$ uv run --extra dev mypy src/execute --config-file pyproject.toml
Success: no issues found in 45 source files

$ just test-mod execute   (= uv run pytest src/tests/execute/ -v)
```
257 passed in 407.10s (0:06:47)   # 225 pre-existing + 32 new; 0 failures
```

## Doc / manuscript follow-ups (other workers own these)

- **`docs/` site**: the `plan_execute` / `list_frameworks` API should be added to the mkdocs module page for `execute` (owned by the docs worker). I updated the in-scope `src/execute/AGENTS.md`/`README.md`/`SKILL.md` only.
- **MCP manifest / `check_mcp_skills_health`**: I did **not** register a new MCP tool for `plan_execute` (adding a 6th tool could disturb the MCP manifest gate and `test_check_mcp_skills_health`). A `plan_execute_mcp` wrapper is a clean follow-up for the MCP worker.
- **`src/execute/SPEC.md`**: I did not edit SPEC.md (it is a short architectural spec that does not enumerate the public API, so no drift). The SPEC worker may reference `plan_execute` if they enumerate the surface.

## Follow-up ideas (not done; lower priority / higher risk)

1. **Runner log-writing dedup** (scout-5 finding): `execute_jax_script` / `execute_numpyro_script` / `execute_pytorch_script` / `execute_discopy_script` each repeat a ~120-line "execute via `execute_script_safely` + unpack + write stdout/stderr/execution_log.json" block. A shared `execute/runner_support.py` with `write_execution_logs()` (atomic tempfile variant) + `discover_scripts()` would collapse ~300 lines. I scoped this out: the four blocks have subtle per-runner differences (numpyro atomic, jax device env, pytorch CUDA) and the log JSON field sets may have downstream parsers in `analysis/` — needs a dedicated, careful pass.
2. **`GNNExecutor._execute_*` subprocess envelope dedup** (scout-2 finding): the 4 private methods + `execute_script_safely` reimplement the same try/subprocess.run→envelope. One `_run_command_envelope(argv, timeout)` helper would remove ~120 lines. Medium risk (the methods return slightly different envelopes); defer.
3. **`_write_execution_report` framework-section loop** (scout-2): still hardcodes 7 `_write_framework_report_section` calls; could be driven off `_framework_specs()`. Defer (report text is framework-specific).
4. **`data_extractors._load_first_json`** (scout-3): 4 extractors repeat the "glob `*simulation_results.json` → json.load → whitelist copy" pattern. A shared helper + `RXINFER_RESULT_KEYS`/`COLLECTED_CATEGORIES` constants would dedup ~60 lines and fix a latent `open()` without `encoding="utf-8"` platform-dependence. Defer (behavior change risk).

## Notes on fleet concurrency

- `src/advanced_visualization/network_viz.py` was broken by a peer mid-run, temporarily blocking 5 execute test files that import `analysis.pymdp.visualizer` (viz chain). The peer fixed it; the full suite then passed cleanly. No execute-side code was affected.
- I edited only files under `src/execute/`, `src/tests/execute/`, `src/12_execute.py` (unchanged — already a thin orchestrator), and the docs-of-record inside `src/execute/`. No `pyproject.toml`/`uv.lock`/dependency changes. No git add/commit.
