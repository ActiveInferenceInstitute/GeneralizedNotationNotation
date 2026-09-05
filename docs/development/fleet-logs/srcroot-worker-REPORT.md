# srcroot-worker REPORT — GNN module fleet 3 (2026-09-04)

Worker scope: `src/main.py`, `src/manuscript_variables.py`, `src/__init__.py`,
`src/README.md`, `src/SPEC.md`, `src/STEP_INDEX.md`, `src/AGENTS.md`,
`src/doc/`. Branch main @ f64ac9085 (start). No git mutations performed.

## Files changed + why

| File | Change |
|---|---|
| `src/main.py` | Composability refactor (see API deltas). 1734→~1730 lines. Step selection extracted into a pure typed core; serial/parallel result recording deduplicated into `_record_step_result`; YAML config loading deduplicated into `_read_input_config`; startup errors now fail through the pipeline exit contract. |
| `src/manuscript_variables.py` | `generate_variables` inline table loops extracted into four pure helpers (`_render_step_table`, `_render_family_table`, `_render_backend_table`, `_cross_framework_selection`) — byte-identical output. Added `load_variables` + `token_checksum` (exported). Public names `generate_variables`/`save_variables` and all token keys unchanged. |
| `src/__init__.py` | `__all__` typed `list[Any]` → `list[str]`. No API change. |
| `src/AGENTS.md` | New "Programmatic step selection" section documenting the selection API + fail-fast behavior; Last Updated → 2026-09-04. |
| `src/README.md` | Fail-fast step-selection note under "Running the pipeline". |
| `src/SPEC.md` | "Behavior" bullet documenting invalid-selection fail-fast semantics. |
| `src/tests/main/` (new) | `__init__.py`, `AGENTS.md`, `README.md`, `test_main_step_selection.py` (18 tests), `test_manuscript_variables_api.py` (6 tests). |
| `src/STEP_INDEX.md`, `src/doc/` | Unchanged — no API/behavior surface they document changed. |

## API deltas

New public (all additive; no existing name removed or resignatured):

- `main.StepSelection` — frozen dataclass: `selected`, `skipped`,
  `added_dependencies`, `requested_only`, `unknown_requested`.
- `main.select_pipeline_steps(pipeline_steps, only_steps=None, cli_skip_steps=None, config_skip_steps=None) -> StepSelection` — pure, no logging/globals; step list injected (DI over the `PIPELINE_STEPS` global).
- `main.parse_step_list_strict(step_input) -> list[int]` — raises `ValueError` on non-numeric tokens (lenient `parse_step_list` unchanged for back-compat).
- `main.step_number_from_script_name(script_name) -> int` — `N_*.py` → step number, `-1` otherwise (extracted from `execute_pipeline_step`'s inline regex).
- `manuscript_variables.load_variables(in_path) -> dict[str, str]` — validated inverse of `save_variables` (`ValueError` on non-flat-JSON).
- `manuscript_variables.token_checksum(variables) -> str` — sha256 over the canonical JSON body (matches `save_variables` bytes minus trailing newline).

Behavior deltas (error paths only; valid-input CLI behavior and all log
lines preserved — pinned by tests):

- Non-numeric tokens in `--only-steps`/`--skip-steps` (CLI or config) now fail fast: `ValueError` → clean startup error, exit 1 (previously silently ran zero steps and exited 0/SUCCESS).
- An `only_steps` request resolving to zero executable steps raises `ValueError` → exit 1 (previously silent SUCCESS with 0 steps).
- Out-of-range `only_steps` numbers are logged (`Ignoring unknown step number(s)…`) and dropped — previously dropped silently.
- Exceptions during context preparation (`_prepare_pipeline_context`) now return 1 via `_fail_pipeline_startup` instead of a raw traceback; exit code unchanged.

Consumer safety: verified by grep — `main()` consumers (`src/cli/__init__.py`,
`src/pipeline/execution.py` via `execute_pipeline_step`, tests pinning
`_step_has_actionable_warning`, `_update_performance_summary`,
`_finalize_pipeline_summary`, `_pipeline_exit_code`,
`_status_from_step_exit_code`, `CRITICAL_SCRIPTS`, `parse_step_list`,
`get_environment_info`, `validate_pipeline_summary`) all keep their contracts;
`manuscript_variables` consumers (`scripts/z_generate_manuscript_variables.py`,
`scripts/check_manuscript_tokens.py`) unaffected.

## Verification output tails

```
ruff check (main.py, manuscript_variables.py, __init__.py) ... All checks passed!
ruff format --check (same 3 files) .......................... 3 files already formatted
python src/main.py --help ................................... help_exit=0 (usage unchanged)
pytest src/tests/main/ src/tests/test_manuscript_variables.py -q
............................................................. 36 passed in 0.42s
pytest src/tests/pipeline/test_main_orchestrator.py -m unit -q
............................................................. 8 passed, 17 deselected
mypy src/main.py src/manuscript_variables.py --config-file pyproject.toml
............................................................. Success: no issues found in 2 source files
```

⚠️ The final mypy invocation at turn end is currently blocked by a **foreign**
syntax error in `src/gnn/parsers/common.py` (line 793, later 868 — the gnn
module worker is mid-edit). Evidence my scope is clean: the full-config mypy
run above passed with the identical logic (the only later change to my files
was `ruff format`, which is AST-preserving and followed by green ruff/pytest/
help). Re-run the gate command once `src/gnn/parsers/common.py` parses.

## Follow-ups needed (other workers own those)

1. `src/gnn/parsers/common.py` — restore parseable state so the repo mypy gate runs again.
2. `src/tests/tests/` (created 10:09 today, before my session) has `.py` files but no `AGENTS.md` — `doc/development/docs_audit.py --strict` fails on it (`src/ tests with .py but no AGENTS.md: src/tests/tests`). Tests-tree owner should add `AGENTS.md` (+`README.md` per audit pairing) or relocate the contract tests.
3. `doc/modules/main.md` (doc worker) — optionally document the new `select_pipeline_steps`/`StepSelection`/`parse_step_list_strict`/`step_number_from_script_name` API.
4. `manuscript/` tooling (manuscript worker) — `scripts/check_manuscript_tokens.py` could use `token_checksum` to detect token drift cheaply; `manuscript_fig_repo_metrics.py`/`manuscript_fig_triple_play.py` could use `load_variables` for validated JSON reads.

## Follow-up ideas

- Move the parallel-tier memory estimates (500/300/200/150/100/50 MB table) out of `main()` into a step-metadata field on `StepConfiguration` so tiers self-describe their footprint.
- `execute_pipeline_step` is still ~230 lines; a natural next split is matrix-mode vs standard-mode execution into two functions around the shared command builder.
- `_build_main_args` silently maps `--skip-llm` onto skip_steps; exposing that merge inside `select_pipeline_steps` inputs would make the selection fully observable in `StepSelection`.


## Closeout addendum (post-advisory pass)

- New test files linted with the repo gate surface: `ruff check
  src/tests/main/` → clean; `ruff format src/tests/main/` applied (2 files
  normalized). `pytestmark = pytest.mark.unit` restored to
  `test_manuscript_variables_api.py` (lost in the `_PROJECT_ROOT` edit
  shuffle) — marker hygiene now matches the selection test file. Re-ran:
  24/24 tests pass.
 - Transient fleet collisions observed (both self-healed by peers, no action
  taken): `src/pipeline/__init__.py` briefly missing
  `DEFAULT_TARGET_DIR`/`DEFAULT_OUTPUT_DIR` imports (broke conftest chain
  mid-run; peer landed `.config` imports) and `src/gnn/parsers/common.py`
  syntax error blocking mypy (above).
 - Advisory-claimed `cli/__init__.py` import of `_finalize_pipeline_summary`
  re-verified: cli imports only `from main import main as pipeline_main`
  (lines 416, 971). Regardless, `_finalize_pipeline_summary` was not
  renamed or resignatured — all pinned private helpers retain their
  contracts.
 - Git tree: read-only inspection only; no add/commit/push performed, per
  fleet rules ("push updates" interpreted as manifest/report sync below).
- REPORT re-mirrored to
  `~/.omp/agent/fleet-manifests/gnn-module-fleet3-2026-09-04/reports/`.

## Checkpoint log

See `docs/development/fleet-logs/srcroot-worker.md` (6 entries).
