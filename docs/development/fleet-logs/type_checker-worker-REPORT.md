# type_checker-worker — Fleet 3 Report

**Worker**: type_checker-worker · **Scope**: `src/type_checker/**` + `src/5_type_checker.py` · **Date**: 2026-09-04

## Files changed + why

### New files
| File | Why |
|---|---|
| `src/tests/type_checker/test_type_checker_content_validation.py` | 19 tests pinning `sections.*`, `summarize_type_check_results`, `validate_content`, `strict_mode` plumbing, the real-newline summary fix, the `estimate_resources` flag, and the `validate_single_gnn_file` never-raises contract. |
| `src/type_checker/checking/summary.py` | `ValidationSummary` TypedDict + `summarize_type_check_results` — typed aggregation of a directory run (counts, complexity tiers, totals). Pure, no I/O; consumed by `validate_gnn_files` to write `type_check_summary.json` and by MCP/reports. |
| `src/tests/type_checker/test_type_checker_estimator_cli_mcp.py` | 10 tests pinning the `## Time` classifier, estimator `.gnn` discovery, the CLI end-to-end path (previously crashed), and MCP strict passthrough. |
| `docs/development/fleet-logs/type_checker-worker.md` | Per-phase checkpoint log (this worker's home). |

### Modified files
| File | Why |
|---|---|
| `src/type_checker/checking/core.py` | Removed the three duplicated private helpers (now imported from `sections.py`); **fixed a literal-`\n` rendering bug** in `_generate_type_check_summary` (5 lines emitted `\n` text instead of newlines); **fixed `GNNTypeChecker.__init__` silently swallowing `strict_mode`** (cli.py passed it; it was ignored) → `__init__(self, strict_mode: bool = False)`; split `validate_single_gnn_file` into a read + a new pure `validate_content(content, *, source_name, strict)` **with the `validate_content` delegation wrapped in try/except so a parser failure surfaces as an invalid-file dict (recoverable exit-2) instead of propagating (hard exit-1)**; thread `strict` through `validate_gnn_files` kwargs (defaults to `self.strict_mode`); honour the `estimate_resources` kwarg by running the estimator + writing `resource_estimates/`; write `type_check_summary.json` (was documented but never emitted); enrich the validation dict with additive `variables`/`connections`(+`is_temporal`)/`sections`/`model_complexity`/`type_distribution`/`time_dynamics` so the CLI report renderers have structured data; drop a redundant local `import json`. |
| `src/type_checker/checking/__init__.py` | Re-export the additive surface (`ResourceEstimate`, `VALID_TYPES`, `parse_state_variables`, `extract_b_matrix_evidence`, `extract_gnn_dimensions_with_diagnostics`, `sections.*`, `ValidationSummary`, `summarize_type_check_results`, `CANONICAL_GNN_SECTIONS`). |
| `src/type_checker/__init__.py` | Bump `__version__` 1.6.0 → 1.7.0; extend `FEATURES`; add a curated additive subset of the checking surface (`estimate_file_resources`, `extract_gnn_dimensions`, `summarize_type_check_results`, `ValidationEstimate`, `validate_dimension_compatibility`) while keeping the package import light (no estimator/matplotlib import at package import time). |
| `src/type_checker/processor.py` | Facade re-exports the additive surface (`ResourceEstimate`, `ValidationSummary`, `extract_b_matrix_evidence`, `extract_gnn_dimensions`, `extract_gnn_dimensions_with_diagnostics`, `parse_state_variables`, `summarize_type_check_results`). |
| `src/type_checker/estimation/estimator.py` | **Fixed `time_spec = "Dynamic" if "t" in content`** (true for almost every spec — read the `## Time` section via `_classify_time_spec` instead → Static/Dynamic/Hierarchical); replaced naive whole-content `re.findall` edge/equation parsing with section-scoped `parse_resource_connections` / `extract_markdown_section`; `estimate_from_directory` now walks every registered non-binary spec extension (mirroring the type-checker discovery fix) instead of only `*.md`; dropped the dead `is_hierarchical = any("hierarchical" in key.lower() for key in content)` heuristic (the content-dict keys never contain "hierarchical" — `time_spec` now carries Hierarchical). |
| `src/type_checker/estimation/report_html.py` | **Fixed a pre-existing `TypeError`**: `f"{metrics.get('flops_estimate', 0):.2e}"` / `f"{metrics.get('inference_time_estimate', 0) * 1000:.4f}"` formatted **dict** values (`flops_estimate`/`inference_time_estimate` are dicts from the strategies) → `unsupported format string passed to dict.__format__`. Added `_metric_scalar` to reach into the dicts; the HTML report now renders instead of crashing on any real input. |
| `src/type_checker/cli.py` | **Fixed the live `KeyError: 'is_valid'` crash** (`per_file_markdown_report` indexed `result['is_valid']` but `check_file` returns `valid`; the CLI exited 1 on every run before writing reports) by merging `is_valid` into each details entry; directory mode now uses the checker's registered-extension discovery instead of `*.md`/`**/*.md`; `--strict` now actually threads through the fixed constructor. |
| `src/type_checker/mcp.py` | `validate_gnn_files_mcp` / `validate_single_gnn_file_mcp` now construct `GNNTypeChecker(strict_mode=strict)` and pass `strict`/`estimate_resources` through (previously the params were accepted and silently dropped). |
| `src/type_checker/AGENTS.md`, `README.md`, `SPEC.md`, `SKILL.md`, `checking/AGENTS.md`, `estimation/AGENTS.md` | Docs of record updated for the new modules, `validate_content`, `strict_mode`, `estimate_resources`, `type_check_summary.json`, version 1.7.0 / 3.3.0. |

## API deltas (additive; no breaking changes)

**New public symbols** (all re-exported from `type_checker.checking` and the relevant facades):
- `type_checker.checking.sections`: `CANONICAL_GNN_SECTIONS`, `extract_markdown_section`, `connection_group`, `parse_resource_connections`, `section_presence`, `detect_time_dynamics`
- `type_checker.checking.summary`: `ValidationSummary` (TypedDict), `summarize_type_check_results`
- `type_checker.checking`: `ResourceEstimate`, `VALID_TYPES`, `parse_state_variables`, `extract_b_matrix_evidence`, `extract_gnn_dimensions_with_diagnostics` (newly re-exported; `extract_gnn_dimensions` already public)
- `GNNTypeChecker.validate_content(content, *, source_name="<content>", strict=None) -> dict` — pure, no filesystem
- `GNNTypeChecker.__init__(self, strict_mode: bool = False)` — **signature changed** from `(*args, **kwargs)` to a typed kwarg. Consumer grep confirmed `GNNTypeChecker` is constructed **zero-arg everywhere** in the repo except `cli.py` (`strict_mode=`), which now works. No positional-arg consumer exists.

**Behavior preserved**: all 41 pre-existing tests still pass. `validate_single_gnn_file`/`validate_gnn_files`/`check_file` signatures and return shapes are unchanged (additive keys only). `_discover_gnn_files` (pinned by `test_type_checker_discovery.py`) is retained. MCP tool names `validate_gnn_files` / `validate_single_gnn_file` are retained.

**Behavior fixes (regressions corrected, not contract changes)**:
1. `GNNTypeChecker(strict_mode=True)` + `--strict` now promote `[GNN-E002]` B-orientation contradictions from warnings to errors (the flag was silently ignored before).
2. `validate_gnn_files(..., estimate_resources=True)` now runs the resource estimator and writes `resource_estimates/resource_data.json` + `resource_report.md` (the documented `--estimate-resources` Step 5 option was silently ignored before).
3. `type_check_summary.json` is now written (documented at `doc/gnn/modules/05_type_checker.md:111` but never emitted before).
4. The CLI (`python -m type_checker.cli`) no longer crashes with `KeyError: 'is_valid'`; it writes per-file reports + CSV artifacts with real data.
5. The Markdown summary uses real newlines (the 5 `\\n`-literal lines emitted `\n` text before).
6. The estimator classifies `## Time` as Static/Dynamic/Hierarchical (the old `"Dynamic" if 't' in content` was true for ~every spec).
7. The HTML resource report renders (the `dict.__format__` TypeError on `flops_estimate`/`inference_time_estimate` is fixed).

## Verification output tails

```
$ uv run ruff check src/type_checker src/tests/type_checker
All checks passed!

$ uv run --extra dev mypy src/type_checker --config-file pyproject.toml
Success: no issues found in 21 source files

$ uv run pytest src/tests/type_checker/ -v   # ≡ `just test-mod type_checker` (just binary absent on host)
============================== 74 passed in 1.06s ==============================
```
Baseline was 41 passing (0 regressions). New total: **74** (41 existing + 33 new).

## Post-check comprehensive round (2026-09-04, after fleet re-check)

With the orchestrator's go-ahead (doc/ and CHANGELOG.md were untouched by any
other worker), the flagged follow-ups were resolved in this same scope:

1. **Phase 1.1 exit-code contract aligned.** Repo-wide evidence
   (`test_pipeline_render_execute_analyze.py:117`, analysis/execute/render
   regression tests, `doc/gnn/testing/SPEC.md:51`) shows "no input" must be
   exit-2 warning, not exit-1. `validate_gnn_files` no longer sets
   `hard_failure` on no-files → returns 2; artifacts still written; MCP
   message distinguishes the warning outcome. Regression test:
   `test_validate_gnn_files_no_files_is_warning_exit_2`. (My initial
   "preserve exit 1" call was wrong — the doc was right, the code was the
   outlier.)
2. **`doc/gnn/modules/05_type_checker.md` reconciled**: strict wording
   narrowed to the real semantics (B-orientation `[GNN-E002]` promotion);
   exit-code paragraph rewritten (0/1/2 with Phase 1.1 semantics); Testing
   section now references real tests (the previously listed
   `test_type_checker_strict_mode_promotes_warnings` never existed).
3. **`doc/gnn/tutorials/quickstart_tutorial.md`**: stale ".gnn silently not
   found" note replaced (discovery now walks registered extensions).
4. **`CHANGELOG.md`**: `### Fixed (2026-09-04 — type checker contract
   alignment)` entry added under `[Unreleased]`.
5. **Single read per file**: directory loop reads content once and passes it
   to `validate_single_gnn_file(content=...)` and `_analyze_types(content=...)`;
   new `_invalid_file_result` helper dedups the invalid-dict shape (3 copies → 1).
6. **Enrichment**: `validate_content` now sets `model_type` (via new shared
   `checking.sections.classify_time_spec`, replacing the estimator's private
   `_classify_time_spec` — dedup) and the full granular `model_complexity`
   metrics (`state_space_complexity`, `graph_density`, `cyclic_complexity`,
   `temporal_complexity`, `equation_complexity`, `overall_complexity` via
   `calculate_complexity`); `output_utils.per_file_markdown_report` renders
   the real keys (was zero-only placeholders).
7. **`report_html.py` matplotlib guard**: import is now try/except +
   `MATPLOTLIB_AVAILABLE` (parity with `visualizer.py`); plots are skipped
   when matplotlib is absent; HTML tables still render.
8. **Strict semantics decision**: `validate_dimension_compatibility` keeps its
   narrow documented semantics (orientation contradictions only) — the module
   docstring, AGENTS.md, and SPEC.md already agree; the broad "all warnings"
   claim existed only in `doc/gnn/modules/05_type_checker.md` and is fixed.
9. **`classify_time_spec` unifies its Dynamic marker set with
   `detect_time_dynamics`** (delegates to it): a "continuous-time" spec now
   classifies `Dynamic` instead of contradicting `time_dynamics.is_dynamic=True`
   with `model_type="Static"`; agreement pinned by
   `test_classify_time_spec_agrees_with_detect_time_dynamics`.
10. **Ownership check before doc edits**: `doc/` + `CHANGELOG.md` were and remain
   claimed by no worker (dirty surface = this worker's 3 files only); the newly
   appeared `DocsRefresher` worker owns `doc/gnn/modules/02_tests.md` — no overlap.
   `sections.py` is an untracked new file of this worker (no foreign edit).

Note: `doc/development/docs_audit.py --strict` currently flags 1 issue —
`src/tests/tests` (tests-worker's new directory) lacks an AGENTS.md. That is
tests-worker scope, not this module's.

## 5-line summary
- Audited `src/type_checker/**` + `5_type_checker.py`; baseline 41 tests, ruff+mypy clean; proved live bugs (CLI `KeyError`, literal `\n` summary, `strict_mode` swallowed, `--estimate-resources` ignored, `## Time` misclassified, HTML report `dict.__format__` crash); verified `visualizer` charts are NOT dead (probe disproved the matplotlib 3.9-cmap-removal assumption; no visualizer change made).
- Added `checking/sections.py` (shared section-scoped parsing) + `checking/summary.py` (`ValidationSummary` TypedDict) and re-exported the additive surface; dedup'd the checker/estimator connection parsing.
- Post-check comprehensive round: Phase 1.1 exit-2 alignment (no-files), doc/tutorial/CHANGELOG reconciliation, single-read-per-file + `_invalid_file_result` dedup, `model_type` + granular `model_complexity` enrichment (shared `classify_time_spec`, Dynamic markers unified with `detect_time_dynamics`), `report_html` matplotlib guard — see "Post-check comprehensive round" above. Final gate green: `ruff` clean, `mypy` 21 files 0 errors, `pytest src/tests/type_checker/` **74 passed** (41 existing + 33 new, 0 regressions).
- Added 33 deterministic tests (`test_type_checker_content_validation.py` 23, `test_type_checker_estimator_cli_mcp.py` 10), incl. the `validate_single_gnn_file` never-raises regression, the Phase 1.1 no-files exit-2 regression, and the classify/detect time-marker agreement test; updated AGENTS/README/SPEC/SKILL/subpackage docs to v1.7.0/3.3.0.