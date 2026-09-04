# tests-worker REPORT — fleet 3, 2026-09-04

Scope: `src/tests/` shared plumbing (conftest, categories, runner + mode modules,
run_fast_tests, helpers/, infrastructure/, docs of record, top-level test_*.py)
and `src/2_tests.py`. No files outside scope were edited. No git operations, no
dependency changes.

## Files changed + why

| File | Change | Why |
|---|---|---|
| `src/tests/infrastructure/test_runner.py` | Canonical `TestRunner`: added thread-safe history append (`_history_lock`) and `--log-cli-level=WARNING` (both present only in the runner.py copy); docstring declares it the single source | Dedup of the known TestRunner drift |
| `src/tests/runner.py` | Deleted the duplicated `TestRunner` class (−295 lines) + dead `_psutil` block + `project_root`; now re-exports `TestRunner` from `.infrastructure` | Single source; `tests.runner.TestRunner` import path preserved |
| `src/tests/categories.py` | `TestCategory` TypedDict; typed accessors; `get_all_test_files()` now sorted+deduped; new `missing_category_files()` drift detector; **remapped all 86 `files` entries from stale root-level names to actual subdirectory paths** (e.g. `gnn/test_gnn_overall.py`); zero entries dropped | Routing table pointed at pre-reorg root paths → comprehensive-mode discovery found almost nothing; now `missing_category_files() == {}` |
| `src/tests/helpers/script_loader.py` (new) | `load_module_from_path(name, path, *, sys_path=None)` — typed importlib loader with sibling-dir injection/cleanup | 6 root test files duplicated the same loader boilerplate |
| `src/tests/helpers/gnn_samples.py` (new) | `SAMPLE_GNN_CONTENT` + `write_sample_gnn_markdown()` (byte-identical to conftest originals; verified programmatically) | Canonical GNN sample content; single source for fixtures |
| `src/tests/helpers/mcp_stubs.py` (new) | `MCPTools` registry stub (ex-conftest `_MCPTools`) | 7 module dirs re-declare near-identical stubs; conftest fixture now exposes the shared class |
| `src/tests/helpers/__init__.py` | Re-exports the new symbols; docstring table | Package surface |
| `src/tests/conftest.py` | Imports sample content + MCPTools from helpers; private duplicates removed; fixture names/behavior unchanged | Dedup into helpers |
| `src/tests/run_fast_tests.py` | `--timeout` flag now added only when `pytest_timeout` is importable | Without dev extra the unconditional flag made pytest exit 4 |
| `src/tests/test_runner_modes.py` | Recovery-mode file list: `test_main_orchestrator.py` → `pipeline/test_main_orchestrator.py` | The old root path never existed → recovery mode silently ran 2 of its 3 documented files |
| `src/tests/test_doc_contracts.py`, `test_docs_audit.py`, `test_check_external_links.py`, `test_check_mcp_skills_health.py`, `test_add_module_docstrings.py`, `test_run_pymdp_gnn_scaling_estimate.py` | Migrated onto `tests.helpers.load_module_from_path` | Dedup of importlib boilerplate |
| `src/tests/tests/` (new package) | `__init__.py`, `test_categories_contract.py`, `test_testrunner_unified.py`, `test_helpers_contract.py`, `test_infrastructure_exports.py`, `test_step2_wrapper_contract.py` — 26 fast, deterministic tests | Pin the refactored plumbing |
| `src/tests/AGENTS.md` | TestRunner API entry; category-system docs (subdir paths + drift detector); new "Shared Test Helpers (helpers/)" section; recovery-mode file list; date | Docs of record |
| `src/tests/infrastructure/AGENTS.md`, `src/tests/helpers/AGENTS.md` | Canonical-TestRunner note; new helper modules/exports | Docs of record |
| `src/tests/SPEC.md` | Components (runner split + infrastructure/ + tests/), categories section (live table authority), Key Exports | Docs of record |
| `src/tests/TEST_SUITE_SUMMARY.md` | Component tree (real layout incl. helpers/infrastructure/tests/), marker section now lists only markers that exist (removed e2e/safe_to_fail/requires_gpu/requires_network which were never registered) | Doc accuracy |

## API deltas (all backward compatible)

- `tests.runner.TestRunner` — now the re-exported canonical class; behavior of the
  live copy preserved (thread-safe history + `--log-cli-level=WARNING` were the
  copy-A behaviors already in use via `src/utils/test_utils.py:116`).
- `tests.categories`: annotation-only typing (`TestCategory`), `get_all_test_files()`
  order now deterministic (sorted — previously `list(set(...))`), `files` entries are
  subdirectory paths (only consumer, `discover_test_files()`, resolves them correctly).
- New: `missing_category_files(test_dir=None)`, `tests.helpers.{load_module_from_path,
  SAMPLE_GNN_CONTENT, write_sample_gnn_markdown, MCPTools}`.
- `run_fast_reliable_tests` now actually runs `pipeline/test_main_orchestrator.py`
  (documented contract, previously silently skipped).

## Verification (tails)

- `uv run ruff check src/tests/conftest.py src/tests/categories.py src/tests/runner.py src/tests/helpers src/tests/infrastructure` → **All checks passed!**
- `uv run --extra dev python -m pytest src/tests/test_runner_helper.py src/tests/test_fast_suite.py src/tests/test_tests_package_imports.py -q` → **23 passed**
- Extended: `src/tests/tests/` + infra-stats + output-isolation + zero-skip + unit-overall + 6 migrated files → **81 passed** (26 new tests included)
- `missing_category_files() == {}`; `tests.runner.TestRunner is tests.infrastructure.TestRunner`
- `python src/tests/test_runner_helper.py --help` → exit 0
- Transient failures seen mid-run (`src/gnn/parsers/common.py` IndentationError;
  `ClassVar` NameError via render/visualization imports) were concurrent peers'
  in-flight edits in their own dirs; both re-ran green moments later.

## Follow-ups for other workers (not my scope)

1. **Module dirs should adopt the shared fixtures**: gui + pipeline still shadow
   conftest's `sample_gnn_file`; execute shadows `temp_output_dir`; 6 module dirs
   could use `tests.helpers.mcp_stubs.MCPTools` instead of local `_CapturingMCP`/
   `StubMCP`/`_FakeMCP`; inline GNN blobs (10+ dirs) can import
   `tests.helpers.gnn_samples`. Behavior-neutral cleanups in their dirs.
2. **doc/gnn/modules/02_tests.md** still describes pre-refactor runner internals
   (it documents the API accurately, but the "runner.py contains TestRunner" wording
   should be refreshed by the docs worker).
3. `test_runner_output_isolation.py` + `test_tests_package_imports.py` reach into
   sibling test modules as libraries — acceptable, but a future pass could move
   `_isolated_pipeline_output_dir` to a public home.
4. The unused-marker long tail in conftest `PYTEST_MARKERS` (destructive, external,
  recovery, …) could be pruned in sync with pytest.ini (repo-root file, not mine).

## Follow-up ideas (my scope, next pass)

- Wire `missing_category_files()` into `_ModularTestRunner` startup logging.
- Expose category execution as a pure function to make `run_all_categories`
  testable without subprocesses.
- Parameterize recovery-mode file list via env for CI tuning.

## Post-push addendum (2026-09-04T18:20Z)

- Pushed `12556df51` (`12a565b2f..12556df51 main -> main`); staged set verified
  against the 30-path whitelist (`comm` exact match — no peer files committed).
- `src/tests/README.md` (doc of record) had two stale spots the push missed:
  flat `test_main_orchestrator.py` / `test_comprehensive_api.py` entries and a
  pre-remap `MODULAR_TEST_CATEGORIES` excerpt with bare basenames. Fixed to
  subdir-relative paths + `missing_category_files()` note; committed as a
  follow-up.
- `src/visualization/analysis/combined_analysis.py` circular-import failure in
  `test_fast_suite.py::TestFastVisualization::test_visualization_module_import`
  was peer-owned mid-edit churn; re-checked after the push and it now passes
  (peer fixed it). No out-of-scope breakage remains.

## Agent fan-out addendum (2026-09-04, post-71aa9e34f)

User directive: spin up agents for all remaining improvements. 8 parallel task
agents (herdr requested but HERDR_ENV unset -> omp-native task agents), each
with per-dir ownership and the behavior-preservation contract.

- **Adopted shared helpers** (verified: 64 passed on the 7 touched files):
  - `render/test_render_mcp_wiring.py`, `execute/test_execute_mcp_wiring.py`:
    local `_CapturingMCP` -> `tests.helpers.MCPTools` (call conventions verified
    against src/{render,execute}/mcp.py; peer's render_spec_to_format addition preserved)
  - `security/`, `research/`, `ml_integration/` test_security/research/ml_integration_mcp_tools.py:
    local stub classes -> MCPTools
  - `execute/test_execute_pymdp_visualizer.py`, `..._module.py`: two `temp_output_dir`
    shadow fixtures removed (conftest's fixture is functionally equivalent for all consumers)
- **MarkerPruner**: 10 zero-reference markers removed from conftest PYTEST_MARKERS
  (destructive, external, utilities, environment, render, export, parsers,
  type_checking, sapf, visualization); pytest.ini untouched (all 9 used);
  registries consistent; independently re-verified (0 refs, fast suite green).
- **DocsRefresher**: doc/gnn/modules/02_tests.md refreshed to the unified
  architecture (doc trio 15->15 green, orchestrator citation + links valid).
- **Disciplined negatives** (zero edits, every candidate failed the
  behavior-preservation bar): McpVizAdvanced (4 dirs), SetupOntologyTail (9 dirs),
  GuiPipelineFixtures (2 dirs). All "shadow" fixtures proved behavior-different;
  all inline GNN blobs are deliberately-varied parser fixtures.
- **Drift fixed at integration**: src/tests/README.md maintained-dir count
  34/32 -> 36/34; `docs/test_capability_contracts.py` now green (2 passed).
- **New plumbing**: `_ModularTestRunner` warns at startup when
  `missing_category_files()` is non-empty (`_warn_stale_category_files`); 2 new
  contract tests pin the discovery warning + startup warning (28 tests total in
  src/tests/tests/).
- Pre-existing reds left for owners (documented, not mine): untracked peer test
  files in execute/export/render/type_checker/ontology/utils; 3 order-dependent
  pipeline failures tied to src/5_type_checker.py peer WIP; render
  test_framework_availability.py format drift (tracked, untouched).

## Correction (2026-09-04T19:40Z)

`fe54b4faa` misshaped the startup drift warning: `_warn_stale_category_files`
treated `missing_category_files()`'s `Dict[str, List[str]]` as a list (warning
listed category names under an "entries" claim; `return sorted(missing)` broke
the documented shape; the contract test's fake returned a list so it passed
coincidentally). Fixed to the real contract — dict-aware warning
(`category: entries` details + entry count), `return missing` unchanged shape,
test fakes/asserts the dict. tests/tests 28/28, ruff clean.

Drift attribution: `fe54b4faa` committed `_warn_stale_category_files` with
`return sorted(missing)` although the pre-commit worktree read showed
`return missing` — an uncommitted modification landed in this shared checkout
between the repair and staging (35-tab fleet shares one tree; no agent claimed
the file; actor unattributable). `git diff fe54b4faa..d6e77879c` confirms the
correction bundled ONLY the intended hunks. Lesson recorded: re-read a file
immediately before `git add` when a shared tree is live.
