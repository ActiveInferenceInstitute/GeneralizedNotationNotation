# utils-worker REPORT — fleet 3, 2026-09-04

Scope: `src/utils/` (module considered separately and alone) + tests in `src/tests/utils/`.
Branch `main` @ f64ac9085. No git state-changing operations performed. No pyproject/uv.lock/deps touched.

## Method

1. Read the whole scope (`AGENTS.md`, `README.md`, `SPEC.md`, `SKILL.md`, `__init__.py` lazy-PEP562 surface: 118 exports / 19 modules) and fanned out 6 read-only scouts (4 cluster mappers + 1 repo-wide consumer-grep mapper).
2. Consumer-grep grounded every decision: hottest surfaces `utils.pipeline_template` (~27 importers), `log_step_*` (~15), `setup_step_logging` (~12), `utils.logging.logging_utils` (~10), root `performance_tracker` module-attr style (5) — all left behavior-identical. Zero wildcard imports repo-wide.
3. Refactor pass executed as behavior-preserving dedupe + latent-bug fixes; additive typed helpers; new test file pinning everything.

## Files changed + why

| File | Change |
|---|---|
| `src/utils/io_utils.py` | **ADD** `verify_directory_writable(directory, probe_name=".write_probe") -> None` — single create-rename-cleanup write probe (raises `OSError` on failure, cleans tmp on rename failure). |
| `src/utils/pipeline.py` | `validate_output_directory` now calls the shared probe (identical log messages/exit contract preserved). |
| `src/utils/pipeline_validator.py` | `check_pipeline_readiness` now calls the shared probe (identical `blocking_issues` message preserved). |
| `src/utils/resource_manager.py` | **ADD** canonical `get_memory_usage` (alias of `get_current_memory_usage`); `with_resource_limits` no longer masks body exceptions (body error propagates; limit `RuntimeError`s only when body succeeded); return type `Iterator[None]`. |
| `src/utils/test_utils.py` | `get_memory_usage` deleted (was verbatim psutil copy) — re-imported from `resource_manager`. Name/behavior preserved. |
| `src/utils/visualization_optimizer.py` | Same delegation; dead `str(e)` statement removed; `except Exception:` (unused `e`). |
| `src/utils/arg_parsing.py` | Two ~70-line fallback-default if/elif ladders (`ArgumentParser.parse_step_arguments`, `create_default_namespace`) collapsed onto one module-level `_FALLBACK_DEFAULTS` mapping + pure `fallback_default_for(arg_name)` (tuple→list copy on access). |
| `src/utils/step_config.py` | `validate_step_args(step_name, args, project_root=None)` — injectable project root for missing-input-path repair; legacy `sys._getframe(1)` heuristic preserved verbatim when omitted (extracted to `_resolve_missing_input_path`); `list[Any]`→`list[str]`. |
| `src/utils/pipeline_monitor.py` | **BUGFIX**: `_initialize_health_thresholds["duration_variance"]` gains `"critical": 3.0` (previously `KeyError` on the >3x-baseline alert path); warning branch now compares the degraded band (2–3x) so the ladder is reachable; no-op `self.step_metrics[step_name]` expression removed. |
| `src/utils/mcp.py` | **ADD** `SENSITIVE_ENV_KEY_MARKERS`, `is_sensitive_env_key(key) -> bool`, `redact_environment() -> dict[str, str]`; `get_environment_info` redaction widened (`credential`, `passwd`, `auth` added); `dict[str, str]` typing. |
| `src/utils/performance_tracking.py` | Typing polish: `track_operation_standalone -> Iterator[None]`, `metadata: dict[str, Any] | None`, `get_performance_metrics/stop_performance_monitoring/generate_performance_report -> dict[str, Any]`, `__all__: list[str]`. |
| `src/utils/AGENTS.md` / `README.md` / `SPEC.md` / `SKILL.md` | Documented all API deltas (AGENTS.md new "Composability Notes" section); fixed stale double-signature doc for `validate_output_directory`; SKILL.md stale `generate_pipeline_health_report()` no-arg example replaced with a valid call (signature verified `(pipeline_summary, logger)`). |
| `src/tests/utils/test_shared_helpers.py` | **NEW** — 63 tests pinning: shared probe (roundtrip/missing-dir/file-target/read-only + both former callers' parity), memory-probe alias identity, `with_resource_limits` masking semantics (3 cases), fallback defaults (25-row parametrize + mutable-copy + `create_default_namespace` contract + `parse_step_arguments` attribute guarantee), `validate_step_args` project-root injection/repair/report, monitor alert bands (critical exists, CRITICAL >3x, WARNING 2–3x, nothing within baseline), env redaction (markers + `redact_environment`). All deterministic, tmp_path-based, network-free. |

## API deltas

- **Additive**: `io_utils.verify_directory_writable`; `resource_manager.get_memory_usage`; `arg_parsing.fallback_default_for` + `_FALLBACK_DEFAULTS`; `mcp.SENSITIVE_ENV_KEY_MARKERS` / `is_sensitive_env_key` / `redact_environment`; `StepConfiguration.validate_step_args(..., project_root=None)` optional kwarg.
- **Behavioral fixes** (previously broken/masking paths): monitor `duration_variance.critical` KeyError gone; `with_resource_limits` no longer replaces a body exception with `RuntimeError` when limits were not exceeded; `create_default_namespace` now returns contract defaults for `advanced_stats` (False) / `simulation_params` (`"{}"`) instead of `None` (zero external callers found — recovery-only path).
- **Preserved**: every existing public entry point (verified via consumer map + root-surface smoke test incl. `from utils import performance_tracker` module-attr style). No renames, no removals.

## Verification output tails

```
$ uv run ruff check src/utils src/tests/utils
All checks passed!

$ uv run --extra dev mypy src/utils --config-file pyproject.toml
Success: no issues found in 51 source files

$ just test-mod utils   # `just` binary not installed on host; ran the recipe's expansion:
$ uv run pytest src/tests/utils/ -v
211 passed in 0.35s     # 148 pre-existing (all kept passing) + 63 new

Baseline pre-change: 148 passed in 0.52s.
Mid-session: 1 pre-existing test (test_readiness_uses_registered_gnn_extensions)
transiently failed due to a WAVE-PEER's mid-edit src/gnn/parsers/common.py
(SyntaxError→NameError→healed). Not caused by and not fixed in this scope;
green again at turn end.
```

Root-surface smoke: lazy-PEP562 resolution verified for all touched submodules (`validate_output_directory`, `get_current_memory_usage`, `PerformanceTracker`, `performance_tracker`, …) + a real `validate_output_directory(tmp)` roundtrip.

## Doc/ or manuscript/ follow-ups (other workers own those)

- `doc/` architecture references still describe the (pre-existing) tripled dependency catalogs and dual error frameworks — update when/if those consolidations happen.
- `utils/SKILL.md` MCP tools list could name the new redaction helpers if docs workers want MCP-surface parity.
- `REPO_COHERENCE_CHECK.md` / `architecture_reference.md` hardcode `from utils.pipeline_template import ...` examples — unchanged, still valid.

## Follow-up ideas (out of scope today, ranked)

1. Consolidate the THREE correlation-context thread-locals (`logging/logging_utils`, `structured_logging`, `diagnostic_logging`) onto one.
2. Collapse the four performance-tracking implementations onto `performance_tracking.PerformanceTracker` (incl. duplicate `get_performance_summary` shim vs impl).
3. Merge `error_handling`/`error_recovery` retry/backoff math (not verbatim — needs a semantic decision) and dedupe the three dependency catalogs.
4. Split `logging/logging_utils.py` (1334 lines) and pick one visual layer (Rich-based `visual_logging` vs ANSI `VisualLoggingEnhancer`).
5. `simulation_monitor.py` import-time side effects (mkdir + FileHandler + `logging.basicConfig()` root hijack) → lazy/injectable.
6. `test_utils.py` stub farm (empty bodies / return-True fakes) and its `sys.path` mutation; `timeout_manager` LLM config in-place mutation.
7. Route `network_utils` (real HTTP, no timeout defaults) and subprocess-heavy `dependency_validator` through injectable transports for hermetic tests.
