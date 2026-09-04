# advanced_visualization worker — fleet report

**Scope:** `src/advanced_visualization/` (entire module incl. AGENTS.md/README) + numbered orchestrator `src/9_advanced_viz.py`
**Repo:** `/Users/hum/Documents/GitHub/HumOS/projects/outside_of_hum/GeneralizedNotationNotation` (branch main, HEAD `f64ac9085`)
**Worker:** advanced_visualization-worker
**Date:** 2026-09-04

---

## Summary

Raised composability, functionality, and internal quality of `src/advanced_visualization` considered separately and alone. Fixed a 100%-silent dashboard-generation bug, deduplicated triplicated attempt-accounting and connection-expansion logic, deleted dead code across four files, added a live capability probe + canonical `viz_type` choice set, made the MCP tool's `generate_d2` parameter honest, and added 26 deterministic tests pinning the new behavior. All three verification gates green: ruff clean, mypy 12/12, 82/82 tests passed (was 56 baseline + 26 new).

## Files changed + why

### `src/advanced_visualization/dashboard.py` — **bug fix**
- **Imported `datetime`** and prefixed the footer HTML chunk with `f`. Previously the footer chunk was a plain `"""` string (no `f` prefix) and `datetime` was never imported, so `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}` shipped as **literal text** in every generated dashboard. The broad `except Exception` in `generate_dashboard` swallowed the `NameError`, so dashboard generation silently returned `None` 100% of the time. Verified live: before the fix the output contained `{datetime.now().strftime` literally; after, it contains `Generated on 20YY-MM-DD HH:MM:SS`.
- **Regression test:** `test_dashboard_timestamp_renders` in `test_advanced_visualization_public_api_refactor.py`.

### `src/advanced_visualization/_shared.py` — **composability helpers + determinism**
- **Added `record_attempt(results, attempt, *, optional_message_filter=None)`** — pure aggregate bookkeeping for `AdvancedVisualizationAttempt` → `AdvancedVisualizationResults` (success/failed/skipped counts, output_files/errors/warnings). The `optional_message_filter` suppresses warning entries whose message mentions an optional dependency marker (e.g. `"D2 CLI"`) so optional CLI absence is not surfaced as a hard warning. Replaces the triplicated attempt-accounting in `processor.py` (the inline `_track` closure + two manual D2 blocks).
- **Added `_conn_endpoints(conn_info) -> (source_variables, target_variables)`** — normalizes legacy `{"source", "target"}` and new `{"source_variables", "target_variables"}` connection formats in one call. Replaces the triplicated normalize-then-extract dance in `network_viz.py` (3D, dashboard adjacency, dashboard 3D scatter, network metrics).
- **Added `VAR_TYPE_COLORS` / `VAR_TYPE_UNKNOWN_COLOR`** — canonical var-type → hex color palette, replacing the two inlined `type_color_map` dicts in `network_viz.py`.
- **Added named layout constants** `FORCE_LAYOUT_SEED` (=42), `LAYOUT_SEED`, `LAYOUT_SPAN` (=10.0), `LAYOUT_ITERATIONS` (=50), `LAYOUT_STEP` (=0.01). Replaced the process-global `np.random.seed(42)` mutation with a local `np.random.default_rng(42)` so importing the module no longer mutates the global RNG state.
- Deduplicated the connection-expansion loop inside `_calculate_semantic_positions` (now uses `_conn_endpoints`).

### `src/advanced_visualization/processor.py` — **dedup + dead-code removal**
- **Dropped the duplicated matplotlib/numpy/seaborn import-guard blocks** (lines 28-40, 88-95 in the original) — these are the single source of truth in `_shared.py`; processor now imports `MATPLOTLIB_AVAILABLE`, `SEABORN_AVAILABLE`, `np`, `plt` from `._shared`.
- **Deleted the dead `global MATPLOTLIB_AVAILABLE, SEABORN_AVAILABLE`** statement in `_check_dependencies` (never assigned).
- **Replaced the per-model `_track` closure + two manual D2-accounting blocks with `record_attempt`** — three duplicated accounting sites → one pure helper. The empty-input contract is preserved: `process_advanced_viz` still returns `2` with `summary["warnings"] == ["No GNN models found"]` and `total_attempts == 0` (verified by the existing `test_process_advanced_viz_empty_input_returns_warning_code`).

### `src/advanced_visualization/network_viz.py` — **dedup + dead-code removal + O(V²)→O(V)**
- **Deleted the dead `_module_logger`** (created, never used — all functions take a `logger` param).
- **Replaced the eager `from visualization.matrix_visualizer import MatrixVisualizer as _MatrixVisualizer`** with the lazy `_MatrixVisualizer` factory from `_shared` (matching `statistical_viz`/`interactive_viz`). The eager import defeated the lazy factory and shadowed its name.
- **Replaced three `var_names.index(s)` / O(V²) linear name scans with `name_to_idx` dict lookups** in the 3D viz, dashboard adjacency, and dashboard 3D scatter.
- **Routed all four connection-expansion sites through `_conn_endpoints`** (3D, dashboard adjacency, dashboard 3D scatter, network metrics).
- **Replaced the two inlined `type_color_map` dicts with `VAR_TYPE_COLORS`/`VAR_TYPE_UNKNOWN_COLOR`**.
- **Dropped dead `len(variables)` / `len(connections)` bare-expression statements** (results discarded).
- *Note:* an advisor caught that my initial import edit had dropped `SEABORN_AVAILABLE` from the import block — re-added before moving on; the POMDP transition/policy paths that use `SEABORN_AVAILABLE and sns` are now exercised by a smoke test.

### `src/advanced_visualization/d2_visualizer.py` — **dead code + temp-leak fix + constants**
- **Deleted the dead numpy try/except import block** (lines 29-36 in the original) — `np`/`NUMPY_AVAILABLE` were never referenced anywhere else in the file.
- **Moved the local `import re`** in `_sanitize_name` to the module top.
- **Hoisted `D2_COMPILE_TIMEOUT_S` (=30), `D2_MISSING_MESSAGE`, `VALID_D2_FORMATS` (=("svg","png","pdf"))** as named constants. The compile loop now uses `D2_COMPILE_TIMEOUT_S` and the missing-CLI message uses `D2_MISSING_MESSAGE`.
- **Fixed a temp-file leak** in `compile_d2_diagram`: on `os.replace` failure the `NamedTemporaryFile(delete=False)` was never cleaned up; now unlinks it in the except path.
- **Added format validation**: unsupported formats are dropped before the CLI is invoked (previously arbitrary suffixes were passed to `d2`).

### `src/advanced_visualization/visualizer.py` — **dedup + determinism + real data**
- **Collapsed three near-identical wrapper methods** (`_generate_statistical_visualizations`, `_generate_network_visualizations`, `_generate_matrix_visualizations`) — each was a try/except + inner `import matplotlib`/`import numpy` + vacuous `if matplotlib:` truthiness check + call to `_create_*`. Replaced with a single `_run_stage(label, create_fn, ...)` helper and a stage-dispatch loop in `generate_visualizations`. Removed the redundant inner imports and dead `if matplotlib:` checks.
- **Seeded the RNG** in `_create_network_graph` and `_create_matrix_heatmap` with `np.random.default_rng(42)` (was `np.random.rand` — unseeded, non-deterministic outputs).
- **Wired `_create_matrix_heatmap` to real matrix data** from `extracted_data["parameters"]` when available, with a clearly-labeled deterministic fallback matrix (was: `np.random.rand(5, 5)` — random placeholder data unrelated to the model, written to disk as if real).

### `src/advanced_visualization/__init__.py` — **new public functionality**
- **Added `VIZ_TYPE_CHOICES`** — canonical tuple of `viz_type` values accepted by `process_advanced_viz`. The `9_advanced_viz.py` orchestrator now imports this instead of hand-maintaining a duplicate list (single source of truth).
- **Added `probe_capabilities()`** — a live runtime probe (`d2` CLI on PATH, `plotly`/`seaborn`/`matplotlib`/`numpy`/`networkx` importability), distinct from the static `FEATURES` map. Used by the `check_visualization_capabilities` MCP tool so its docstring ("Probes D2 availability, dashboard generation support, and network visualization backends") is honest for the first time.
- Exported both in `__all__`.

### `src/advanced_visualization/mcp.py` — **honest params + reuse**
- **`process_advanced_visualization_mcp` now honors `generate_d2`**: `False` → `viz_type="network"` (non-D2); `True` → `viz_type="all"`. Previously `generate_d2` was accepted and documented in the schema but **silently ignored**. The return payload now includes `generate_d2` and `viz_type` in the message.
- **Stopped passing the lying `verbose=verbose` kwarg** to `process_advanced_viz` (which has no `verbose` parameter — it was swallowed into `**kwargs`). `verbose` remains in the signature for schema stability.
- **`check_visualization_capabilities_mcp` now calls `probe_capabilities()`** and returns a `capabilities` dict + `d2_cli_available` (live probe) alongside the static `FEATURES`.
- **`get_advanced_visualization_module_info_mcp` now reuses `get_module_info()`** instead of the `importlib.import_module(__package__)` + `getattr` dance.

### `src/9_advanced_viz.py` — **single source of truth for choices**
- Imports `VIZ_TYPE_CHOICES` from the package and uses `list(VIZ_TYPE_CHOICES)` for the argparse `choices`, instead of a hand-maintained duplicate list (10 entries). Still 53 lines (well under the 150-line thin-orchestrator ceiling).

### `src/advanced_visualization/AGENTS.md` — **docs of record**
- Updated the Test Files list (was 2 files; now 8, including the 2 new test files).
- Added a **Composability Helpers** section documenting `record_attempt`, `_conn_endpoints`, `VAR_TYPE_COLORS`, `LAYOUT_*` constants, `VIZ_TYPE_CHOICES`, `probe_capabilities`, `D2_COMPILE_TIMEOUT_S`/`D2_MISSING_MESSAGE`/`VALID_D2_FORMATS`.
- Added a **Dashboard Footer Timestamp (Fixed)** section documenting the bug + fix + regression test.
- Updated the MCP `process_advanced_visualization_mcp` endpoint doc to reflect `generate_d2` honoring + `verbose` no longer passed through.

### `src/advanced_visualization/README.md` — **docs of record**
- Updated the Module Structure block (was 5 files; now 14, reflecting `_shared.py`, `processor.py`, `network_viz.py`, `statistical_viz.py`, `interactive_viz.py`, `d2_visualizer.py`, `mcp.py`, `AGENTS.md`).

### `src/tests/advanced_visualization/test_advanced_visualization_composability.py` — **NEW** (13 tests)
- `TestRecordAttempt` (7 tests): success/failed/skipped counts, output_files/errors/warnings extension, `optional_message_filter` behavior, aggregate across multiple attempts.
- `TestConnEndpoints` (4 tests): new format, legacy format, empty, extra-keys-preserved.
- `TestSharedConstants` (2 tests): palette coverage, layout constant values.

### `src/tests/advanced_visualization/test_advanced_visualization_public_api_refactor.py` — **NEW** (13 tests)
- `TestVizTypeChoices` (3): is-tuple-of-strings, includes documented values, orchestrator sources from it.
- `TestProbeCapabilities` (3): returns dict of bools, `d2_cli` reflects `shutil.which`, numpy/matplotlib true in test env.
- `TestRecordAttemptReExport` (1): importable from package.
- `TestMcpGenerateD2Honored` (2): `generate_d2=True` → `viz_type=all`; `False` → `viz_type=network`.
- `TestDashboardTimestampRenders` (1): regression for the silent `{datetime.now()}` placeholder bug.
- `TestD2ConstantsAndFormatValidation` (3): constants exposed, missing-CLI error path, unsupported-format drop.

## API deltas

### Added (additive, no breaking changes)
- `advanced_visualization.VIZ_TYPE_CHOICES` — `tuple[str, ...]` (public)
- `advanced_visualization.probe_capabilities() -> dict[str, bool]` (public)
- `advanced_visualization._shared.record_attempt(results, attempt, *, optional_message_filter=None) -> None` (internal but re-exported)
- `advanced_visualization._shared._conn_endpoints(conn_info) -> tuple[list, list]` (internal)
- `advanced_visualization._shared.VAR_TYPE_COLORS` / `VAR_TYPE_UNKNOWN_COLOR` (internal)
- `advanced_visualization._shared.FORCE_LAYOUT_SEED` / `LAYOUT_SEED` / `LAYOUT_SPAN` / `LAYOUT_ITERATIONS` / `LAYOUT_STEP` (internal)
- `advanced_visualization.d2_visualizer.D2_COMPILE_TIMEOUT_S` / `D2_MISSING_MESSAGE` / `VALID_D2_FORMATS` (module constants)

### Changed (behavior preserved or made honest)
- `process_advanced_visualization_mcp(..., generate_d2=False)` now routes to `viz_type="network"` (was: silently ignored). `generate_d2=True` unchanged.
- `check_visualization_capabilities_mcp()` now returns `capabilities` (live probe) + `d2_cli_available` alongside `features`.
- `get_advanced_visualization_module_info_mcp()` now reuses `get_module_info()` (same payload shape).
- `9_advanced_viz.py` argparse `choices` now sourced from `VIZ_TYPE_CHOICES` (same values). *Advisory-sweep correction:* this holds at import/config level only — at runtime the enhanced parser (`utils.ArgumentParser.ARGUMENT_DEFINITIONS["viz_type"]`) enforces its own copy of the choices, and the orchestrator's `choices=` only feeds the recovery/fallback parser. See Follow-up #8.
- `_calculate_semantic_positions` now uses a local `np.random.default_rng(42)` instead of mutating the process-global RNG.
- `_create_matrix_heatmap` now uses real matrix data from `extracted_data["parameters"]` when available (deterministic fallback otherwise).

### Removed (dead code)
- `advanced_visualization.network_viz._module_logger` (dead — never used)
- `advanced_visualization.d2_visualizer` numpy import block (dead — never referenced)
- `advanced_visualization.processor` dead `global MATPLOTLIB_AVAILABLE, SEABORN_AVAILABLE` statement
- `advanced_visualization.visualizer` three redundant wrapper methods (consolidated into `_run_stage`)

### No breaking changes to any public entry point
- `process_advanced_viz` signature + return contract preserved (empty → `2` + `warnings==["No GNN models found"]`; success → `True`; hard fail → `False`).
- `generate_dashboard(content, model_name, output_dir) -> Optional[Path]` signature preserved; behavior **fixed** (was silently `None`; now returns the path).
- All `create_*` free functions, `AdvancedVisualizer`, `DashboardGenerator`, `VisualizationDataExtractor`, `D2Visualizer`, `D2DiagramSpec`, `D2GenerationResult`, `process_gnn_file_with_d2` signatures unchanged.
- MCP tool names/schemas unchanged (pinned in `src/mcp/audit_report.json`).

## Verification output tails

```
--- ruff (module + tests + orchestrator) ---
All checks passed!

--- mypy (module, default traversal) ---
Success: no issues found in 12 source files

--- just test-mod advanced_visualization (recipe = uv run pytest src/tests/advanced_visualization/ -v) ---
`just` is not installed on this host; ran the recipe's exact command directly:
============================= 82 passed in 4.21s ==============================
```

Baseline before any edits: 56 passed. After: 82 passed (+26 new tests, 0 regressions).

## Fleet-coincidence note

Mid-run, `test_generates_html_dashboard` briefly failed with a circular-import error in `src/visualization/analysis/combined_analysis.py` — a fleet peer's in-flight file (dirty ` M`, syntax error visible mid-refactor). My module has zero references to `combined_analysis` (verified via grep). The peer fixed their syntax error before my final verification run; the test now passes. No action needed from me, but the `visualization/` module is a shared dependency and concurrent edits there can transiently break my module's import graph.

## doc/ or manuscript/ follow-ups needed (other workers own those)

- **`docs/` owner:** RESOLVED in the advisory sweep (2026-09-04): AGENTS.md Version History bumped 3.0.0 → 3.2.0 to match the header, and both stale "Last Updated" stamps refreshed to 2026-09-04 (within this module's scope; no external release process consulted).
- **`doc/` owner:** the D2_README.md `docs/development/fleet-logs/` README convention doc is owned by the fleet coordinator; my checkpoint log follows it.
- **`manuscript/` owner:** none — no manuscript references my module's API.

## Follow-up ideas (not done; out of scope or higher-risk)

1. **`visualizer.py` `_create_network_graph`** plots connections as `ax.plot([positions[0,...], positions[1,...]], ...)` — it always connects node 0 to node 1 regardless of which connection is being plotted (a known simplification, comment present). Wiring real source/target node indices would make the graph accurate.
2. **`visualizer.py` `VIS_PROCESSOR_AVAILABLE` global** could become dependency injection (`__init__(self, extractor=None, logger=None)`); left as-is to avoid changing the public `AdvancedVisualizer()` constructor contract.
3. **`network_viz.py` `_MatrixVisualizer is None` guards** — RESOLVED in the advisory sweep: option (b) implemented (`_LazyMatrixVisualizer.__call__` catches ImportError → `None`; all four sites check the result). See "Item 5" above.
4. **`mcp.py` `probe_capabilities()` is called twice** in `check_visualization_capabilities_mcp` (once for `d2_cli_available`, once for `capabilities`) — micro-caching would avoid the double probe. Left for a future pass; correctness over micro-perf.
5. **`d2_visualizer.py` `process_gnn_file_with_d2`** hardcodes `Path("output/3_gnn_output")` — should be a parameter. Left unchanged to avoid breaking the `process_gnn_file_with_d2(test_file, output_dir)` call shape pinned by tests.
6. **`dashboard.py` / `html_generator.py`** share ~350 lines of inline CSS (font stack, gradients, card styles). A shared `_theme.py` with `BASE_CSS` would deduplicate, but extracting it safely requires diffing both files' CSS byte-for-byte; deferred as a larger refactor.
7. **`data_extractor.py`** `extract_visualization_data` writes per-model `extracted_data.json` + `statistics.json` + a top-level `extraction_summary.json` — the `import json` / `from pathlib import Path` are inside the function (E402-ignored per repo config). Could hoist to module top, but the repo's ruff config allows E402 and the function is self-contained.

---

## Advisory sweep — 2026-09-04 (follow-up worker)

Verified all 10 advisory items live; fixes applied where warranted. No git
operations, no dependency changes, scope respected
(`src/advanced_visualization/` + `src/9_advanced_viz.py` +
`src/tests/advanced_visualization/` only).

### Item 1 — Dashboard JS/CSS braces: VERIFIED FIXED (was real, now resolved)
The footer chunk (`dashboard.py` ~line 521) is now an f-string with `{{`
escapes. Generated HTML verified live: contains
`function showTab(tabName) {` (single brace), zero lines with literal `{{`,
and a real timestamp. The original pre-f-prefix state would have shipped
`{{` into the CSS/JS; the previous worker's f-prefix fix closed it.
`test_footer_contains_real_timestamp_not_placeholder` extended to assert the
single-brace `showTab` and assert no `{{` ships anywhere in the document.

### Item 2 — Import blocks vs usages: 2 DEAD IMPORTS FOUND + FIXED
Repo ruff rule set has no F821/F401, so ran them explicitly:
- `processor.py`: removed unused `cast` (typing) and `plt` (`._shared`) —
  both dead after the refactor dedup. No external importer references
  `processor.plt`/`processor.cast` (checked all `from ...processor import`
  sites repo-wide: only `process_advanced_viz`).
- Verified live: `SEABORN_AVAILABLE` used at `network_viz.py:437/469`
  (pomdp paths), `PerformanceTracker` used in
  `SafeAdvancedVisualizationManager.__init__` (processor.py:48), `Callable`
  used in `visualizer.py:103/218` stage annotations,
  `create_standardized_pipeline_script` used in `9_advanced_viz.py:20`.
- Live smoke: orchestrator module loads; `process_advanced_viz` empty-input
  → exit 2 with "No GNN models found"; `_generate_pomdp_transition_analysis`
  and `_generate_network_metrics` succeed on a real model dict (both new
  `source_variables` and legacy `source`/`target` scalar formats).

### Items 3+4 — network_viz duplicate connections: VERIFIED CLEAN
Zero references to `normalize_connection_format` anywhere in
`network_viz.py`; all four connection-expansion sites (3D ~line 122,
dashboard adjacency ~282, dashboard 3D scatter ~338, network metrics ~638)
route through `_conn_endpoints`. Legacy-format smoke through
`_generate_network_metrics` succeeds (no double-draw, no NameError).

### Item 5 — `_MatrixVisualizer` dead guards: OPTION (b) IMPLEMENTED
`_LazyMatrixVisualizer.__call__` now catches `ImportError` and returns
`None`; all four guard sites (network_viz ×2, statistical_viz,
interactive_viz) rewritten to check the RESULT:
`mv = _MatrixVisualizer()` + `if mv is None: skipped/failed with
"MatrixVisualizer not available"`. A missing
`visualization.matrix_visualizer` now surfaces as the intended skip instead
of a raw ImportError. Doc'ed in AGENTS.md Composability Helpers.

### Item 6 — d2 format-filter fallback: FIXED
`compile_d2_diagram` empty-after-filter fallback changed from
`list(spec.output_formats)` (could re-admit arbitrary suffixes) to
`list(VALID_D2_FORMATS[:2])` (svg+png). The
`result = subprocess.run(  # nosec B603` opener line verified intact.

### Item 7 — Test quality:
- `test_orchestrator_choices_match`: dropped the brittle
  `"list(VIZ_TYPE_CHOICES)" in text` source-pin; replaced with the real
  contract — `list(VIZ_TYPE_CHOICES) ==
  utils.ArgumentParser.ARGUMENT_DEFINITIONS["viz_type"].choices` (the
  choices the actual CLI parser enforces).
- MCP audit reports (`src/mcp/audit_report.json`,
  `src/tests/mcp_audit_report.json`) pin only tool `name`/`fn`/`description`
  for `check_visualization_capabilities` — no payload-shape pins exist; the
  new `capabilities`/`d2_cli_available` keys break nothing.
- Dashboard regression test now also covers the JS braces (item 1).

### Item 8 — Report corrections: THIS SECTION + the composability test label
(13, not 26 — 26 was the combined total of both new files) + the corrected
single-source-of-truth claim on the `9_advanced_viz.py` choices line.

### Item 9 — Docs drift: FIXED IN SCOPE
- SKILL.md API + Key Exports now list `VIZ_TYPE_CHOICES` and
  `probe_capabilities()`.
- AGENTS.md Version History "Current Version" 3.0.0 → 3.2.0 (matches the
  header); both stale "Last Updated" dates → 2026-09-04.
- README Module Structure block now lists all 11 Python submodules + the
  SPEC/D2_README/SKILL/AGENTS/README docs (was missing SPEC.md,
  D2_README.md, SKILL.md).

### Item 10 — Final gates (verbatim tails)

```
--- ruff (module + tests + orchestrator) ---
All checks passed!
--- ruff format --check (same scope) ---
22 files already formatted
--- mypy (module, default traversal) ---
Success: no issues found in 12 source files
--- pytest src/tests/advanced_visualization/ -v ---
============================== 82 passed in 9.83s ==============================
```

Note: the prior report's gate section predated `ruff format`; the sweep adds
the format gate (repo `gates` includes `ruff-format`). All 8 drifted files
were inside this module's scope and are now formatted; suite re-proven green
after formatting. Live smoke (throwaway script, deleted after run): empty
input → 2; dashboard braces/timestamp assertions; pomdp + network-metrics
success on new and legacy connection formats; MCP capabilities payload
contains `capabilities` + `d2_cli_available`.

## Follow-up ideas — amended

8. **`viz_type` choices are duplicated between `src/utils/arg_parsing.py`
   and the orchestrator — and the enhanced parser wins at runtime.**
   `utils.ArgumentParser.ARGUMENT_DEFINITIONS["viz_type"]` (arg_parsing.py
   ~393) carries its own hardcoded choices list; `_parse_step_args` only
   routes to the orchestrator's fallback parser (where
   `choices=list(VIZ_TYPE_CHOICES)` lives) if the enhanced parser raises.
   So "single source of truth" holds at import/config level, not at
   runtime parse level. `utils/` is outside this module's ownership — if
   the lists ever drift, the CLI silently enforces the `utils` copy. The
   new test pins them equal so drift fails CI; unification belongs to a
   `utils` owner.
9. **`_generate_*` viz functions trust the caller to `mkdir` the output
   dir** (only `process_advanced_viz` and the D2 helpers create
   directories). Pre-existing contract, unchanged by the refactor; callers
   invoking the free functions directly must mkdir first.

---

## Polish pass — 2026-09-04 (follow-up worker)

All five deferred follow-ups implemented (user-approved). Same constraints:
no git operations, no dependency changes, edits confined to owned scope.

### 1. mcp.py probe caching — DONE
`check_visualization_capabilities_mcp` now calls `probe_capabilities()` once
and reuses the dict for both `d2_cli_available` and `capabilities` (was two
full probes per call).

### 2. process_gnn_file_with_d2 parameterized lookup — IMPLEMENTED
New optional keyword `parsed_json_dir: Optional[Path] = None`; when None the
historical cwd-relative `output/3_gnn_output` lookup runs unchanged, so the
existing `process_gnn_file_with_d2(test_file, output_dir)` call shape and its
test stay valid. Docstring documents the semantics. Purely additive (the
earlier consumer grep found no other callers).

### 3. network graph real node indices — IMPLEMENTED
`_create_network_graph` builds a name→index map from `blocks` and draws one
line per resolvable `source_variables`/`target_variables` pair (legacy
`source`/`target` scalars accepted as singletons). Unresolvable pairs are
skipped silently (module convention). Seeded `default_rng(42)` positions
unchanged. Notably, the old code read `conn.get("from")/("to")` — keys the
extractor never emits — so connections previously never drew at all; this
fix is both an accuracy and an effectiveness fix.

### 4. AdvancedVisualizer dependency injection — IMPLEMENTED
`__init__(self, logger=None, extractor=None)` + `_get_extractor()` (lazy
build of a real extractor; `None` → recovery path, preserving degraded
behavior). Every observed call shape still works; `VIS_PROCESSOR_AVAILABLE`
kept as a module symbol (grep proved zero external readers, retained for
back-compat). New test injects a stub extractor — no module-global
monkeypatching. README's pre-existing invalid example
`AdvancedVisualizer(strict_validation=True)` (would TypeError) corrected to
the real constructor surface.

### 5. Shared theme `_theme.py` — EXTRACTED + PARITY PROVEN
Byte-normalized diff of both `<style>` blocks found exactly **4 genuinely
identical rules** (`*` reset, `.header h2`, `.parameter-name`,
`.stat-label`) plus the shared `FONT_STACK` and `BODY_GRADIENT` fragments.
`_theme.py` holds exactly those (constants + `__all__`, no logic); both
emitters interpolate them and keep their unique rules inline.
**Parity proof (honest):** the true pre-theme baseline was reconstructed
from `git show HEAD:` (the prior worker never touched these style blocks, so
HEAD is the legitimate pre-refactor template). After whitespace
normalization: `html_generator.py` output **byte-identical**;
`dashboard.py` CSS **byte-identical** — the only remaining deltas are the
prior worker's intentional, separately-pinned fixes (footer timestamp
rendering, JS `{{`→`{` f-string escapes). Throwaway before/after harnesses
ran, proved, and were deleted.

### Tests added (`test_advanced_visualization_polish.py`, 8 tests)
d2 optional-keyword signature + Step-3-artifact consumption; network graph
resolvable-pair filtering + legacy scalar format; stub-extractor injection
proving the seam (stub consumed, real extractor never built); theme
constants purity + dashboard single-brace/theme-value parity guard.

### Docs updated
AGENTS.md Composability Helpers (4 new bullets), README module tree
(`_theme.py`, `SPEC.md` restored, duplicate `dashboard.py` removed),
constructor example corrected. No new public API surface (module-private
`_theme`), so SKILL.md is unchanged.

### Final gates (verbatim tails)

```
--- ruff format (module + tests) ---
23 files left unchanged
--- ruff check (module + tests + orchestrator) ---
All checks passed!
--- mypy (module, default traversal) ---
Success: no issues found in 13 source files
--- pytest src/tests/advanced_visualization/ -v ---
============================== 90 passed in 5.15s ==============================
```

## Follow-up ideas — amended (post-polish)

The five deferred items are now implemented; the follow-up list carries only
the previously recorded cross-boundary items (#8 `utils` `viz_type` choices
duplication, #9 caller-mkdir contract).

**Post-pass correction (2026-09-04):** SKILL.md stale examples fixed —
`create_network_visualization(data)` / `create_heatmap_visualization(data)`
(the dict-returning helpers never had an `output_path` param) and
`generate_dashboard(gnn_content, "model_name", Path("output/"))` (real
signature). SKILL.md needs no further change for the polish pass: `_theme.py`
is module-private and the DI/`parsed_json_dir` seams carry no new importable
names.
