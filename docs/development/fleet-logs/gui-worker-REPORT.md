# gui-worker REPORT — 2026-09-04 (fleet 3)

Scope: `src/gui/` entirely + `src/22_gui.py`. No git operations; no dependency changes.

## Files changed + why

| File | Change | Why |
|---|---|---|
| `src/gui/runner.py` | **NEW** — `resolve_output_root()`, `load_first_markdown()`, `launch_gradio_in_thread()` | Output-root normalization, starter-markdown discovery, and the background-Gradio launch pattern were copy-pasted across gui_1/gui_2/gui_3 processors; now one shared, typed module. |
| `src/gui/backend.py` | Added `write_text_atomically()`; `write_json_atomically()` now delegates to it | Text sibling of the existing JSON atomic writer; removes the 6 duplicated temp-file+`os.replace` dances. |
| `src/gui/processor.py` | Rewritten: typed (`GUISummary` TypedDict), pure helpers `normalize_gui_types()` / `summarize_gui_results()` / `collect_pipeline_outputs()`, constants `PIPELINE_OUTPUT_SECTIONS` (25 steps) / `MAX_FILES_PER_SECTION` (20) / `DEFAULT_GUI_TYPES`; `html.escape` on all dynamic HTML; honors `logger=` kwarg; summary via `write_json_atomically` | Composability + a real correctness fix (unescaped filenames could break navigation.html); the documented `logger` param was silently ignored before. |
| `src/gui/gui_1/processor.py` | Deduped onto runner+backend; removed dead `# noqa: F401` re-export and `_load_first_markdown` | Thin domain logic only; module-level `_GUI_STATUS/_GUI_BACKEND/_GUI_BACKEND_REASON` kept (tests reload with a stubbed gradio). |
| `src/gui/gui_2/processor.py` | Same treatment; POMDP preference via `prefer_patterns=("*pomdp*.md", "*POMDP*.md")` | Same dedup; sorted-glob makes "first template" deterministic. |
| `src/gui/gui_3/processor.py` | Shared `detect_gradio_backend()` instead of private try/except import; atomic artifact writes; shared launch helper | A gradio install lacking `Blocks` now degrades to headless artifacts (same recovery contract as gui_1/gui_2) instead of crashing at interactive build. |
| `src/gui/oxdraw/processor.py` | `oxdraw_processing_results.json` written via `write_json_atomically` | Was the only non-atomic write in the module (torn file risk on crash). |
| `src/gui/__init__.py` | v1.7.0; exports `normalize_gui_types`, `summarize_gui_results`, `collect_pipeline_outputs`, `DEFAULT_GUI_TYPES`, `MAX_FILES_PER_SECTION`, `PIPELINE_OUTPUT_SECTIONS` | New typed public surface, additive only. |
| `src/gui/AGENTS.md`, `src/gui/README.md` | Composability-helpers API docs, Shared Building Blocks section, test inventory, stamps | Docs of record kept in lock-step. |
| `src/tests/gui/test_gui_composability.py` | **NEW** — 24 tests | Pins the new surface + reload contracts + escaping + caller-logger behavior. |
| `src/22_gui.py` | **UNCHANGED** (96 lines) | Already a correct thin orchestrator; no edits needed. |

## API deltas (additive unless noted)

- New exports (from `gui`): `normalize_gui_types`, `summarize_gui_results` + `GUISummary`, `collect_pipeline_outputs`, `DEFAULT_GUI_TYPES`, `MAX_FILES_PER_SECTION`, `PIPELINE_OUTPUT_SECTIONS`.
- `process_gui(..., logger=<Logger>)` is now honored (was ignored; docs already promised it). Non-Logger values still ignored safely.
- New shared module `gui.runner` (`resolve_output_root`, `load_first_markdown`, `launch_gradio_in_thread`); removed private `_load_first_markdown` / `_load_template_markdown` (grep proved no external consumers).
- Behavioral fixes: blank `gui_types` entries dropped (previously became "unknown GUI" failures); navigation.html escapes names/paths; deterministic markdown/template discovery order; gui_3 broken-gradio → headless artifacts; atomic writes for all GUI JSON/markdown artifacts.

## Verification (tails)

```
uv run ruff check src/gui src/tests/gui   -> All checks passed!
uv run --extra dev mypy src/gui --config-file pyproject.toml
                                          -> Success: no issues found in 26 source files
uv run pytest src/tests/gui/ -v           -> 94 passed in 3.37s
  (includes new test_gui_composability.py 24 tests; `just` binary absent on this
   host, ran the justfile's underlying command directly)
End-to-end smoke: python src/22_gui.py --target-dir input/gnn_files/discrete \
  --output-dir .cache/gui-smoke-output --headless
  -> exit 0; 22_gui_output/{constructed_model_gui1.md, visual_model_gui2.md,
     visual_matrices.json, gui_status.json, gui_processing_summary.json,
     navigation.html} all generated; smoke dir removed after.
```

Exit codes, logging conventions, and output contracts unchanged (all 70 pre-existing gui tests still pass unmodified).

## Follow-ups needed (other workers own these)

- None blocking. Mid-run, `src/mcp/mcp.py` transiently broke `test_oxdraw_integration.py` (missing `import logging` — foreign scope); the owning peer fixed it and the full gui module is green again.
- `doc/` / `manuscript/`: no changes required; if the manuscript references GUI internals, note the new `gui.runner` module and v1.7.0 exports.

## Follow-up ideas

1. **gui_3 output-root unification**: gui_3 writes `designed_model_gui_3.md`/`design_analysis.json` to `output/` while gui_1/gui_2 write inside `output/22_gui_output/`. Deliberately left as-is (behavior preservation); a unification should be verified against the full pipeline.
2. Downstream consumers (website/report modules) could embed `summarize_gui_results(gui_processing_summary.json["results"])` for typed GUI status sections.
3. `gui_2/ui.py` (55 KB) and `gui_3/ui_designer.py` (24 KB) are large single files; further decomposition is possible but their callbacks are already behavior-pinned by tests.
