# export-worker Fleet Report

**Module**: `src/export/` + `src/7_export.py`
**Date**: 2026-09-04
**Worker**: export-worker
**Branch**: main (HEAD f64ac9085)

---

## Files Changed

| File | Change | Why |
|---|---|---|
| `src/export/registry.py` | **NEW** | Single source of truth for 7 export formats: names, extensions, writer callables, categories. Eliminates 3 duplicated format lists. |
| `src/export/formatters.py` | Refactored | Added `_write_pretty_xml` and `_dump_pickle` helpers; 3 XML writers and 2 pickle writers now delegate to them. Removed ~40 lines of duplicated serialization code. |
| `src/export/processor.py` | Refactored + extended | `export_model`/`export_gnn_model`/`export_single_gnn_file`/`process_export` dispatch via registry tables (`_MODEL_FORMAT_FILES`, `_GNN_MODEL_WRITERS`, `_PIPELINE_WRITERS`) instead of if/elif chains. **Bug fix**: `export_gnn_model` no longer appends bogus `"No valid formats requested"` to `errors` on the all-success path; default formats changed from `["json","xml","graphml","gexf","pickle"]` (3 always failed) to `["json","xml","pickle","txt","dsl"]` (the formats it actually supports). `process_export` format dispatch now uses `_PIPELINE_WRITERS` from registry. **New function**: `validate_export_outputs(output_dir, expected_formats=None)` — post-run artifact validation. |
| `src/export/core.py` | Bug fix + refactor | **Silent-failure bug fixed**: `export_gnn_files` now unwraps `Tuple[bool, str]` returns from `format_exporters` via `_writer_success`/`_writer_error` helpers instead of treating truthy tuples as success. Dispatch is data-driven (`writer_table` list). |
| `src/export/__init__.py` | Refactored | `HAS_NETWORKX` now truthfully imported from `format_exporters` (was hardcoded `True`). `get_supported_formats`/`get_supported_formats_dict`/`validate_export_format` rebased on registry. `validate_export_outputs` and `get_export_registry` added to `__all__`. |
| `src/export/utils.py` | Refactored | `get_module_info`/`get_supported_formats` rebased on registry — format lists and categories are no longer duplicated. |
| `src/export/mcp.py` | Unchanged | No new MCP tools added (adding tools would break the `test_mcp_audit` allowlist owned by another worker). |
| `src/export/AGENTS.md` | Updated | Added `validate_export_outputs` to API reference, documented registry as single source of truth, added `logger` kwarg, listed new test file, updated date. |
| `src/export/README.md` | Updated | Added `registry.py` to module tree. |
| `src/tests/export/test_export_registry_and_validate.py` | **NEW** | 15 tests: 8 registry invariants + 6 `validate_export_outputs` scenarios + 1 core silent-failure propagation pin. |

## API Deltas

### New public names
- `export.validate_export_outputs(output_dir, expected_formats=None) -> dict` — manifest-driven artifact validation
- `export.get_export_registry() -> dict[str, ExportFormatSpec]` — introspectable format registry
- `export.registry.DEFAULT_PIPELINE_FORMATS` — canonical 5-tuple of pipeline format names
- `export.registry.resolve_format_writer(name) -> FormatWriter | None` — writer lookup
- `export.registry.get_format_categories() -> dict` — categories (data/graph/text)
- `export.registry.is_supported_format(name) -> bool`
- `export.registry.get_format_spec(name) -> ExportFormatSpec | None`

### Preserved (no signature change)
All 24 previously-public names preserved with identical signatures and return shapes (except the two bug-fix behavior changes noted below).

### Behavior changes (bug fixes, not regressions)
1. **`core.export_gnn_files`**: Writer failures returning `(False, msg)` tuples are now correctly counted as failures. Previously they were silently treated as successes because the truthy tuple was not unwrapped. This is a **bug fix** — the documented contract says writers return `Tuple[bool, str]`.
2. **`processor.export_gnn_model`**: Default formats changed from `["json","xml","graphml","gexf","pickle"]` (where graphml/gexf always failed as unsupported) to `["json","xml","pickle","txt","dsl"]` (the formats the function actually supports). The bogus `"No valid formats requested"` error appended on the all-success path is removed. Both changes make `export_gnn_model()` succeed by default — clearly the intended behavior.
3. **`export.HAS_NETWORKX`**: Now truthfully reflects whether `networkx` is importable (was hardcoded `True`).

## Verification Output

```
$ uv run ruff check src/export src/tests/export
All checks passed!

$ uv run --extra dev mypy src/export --config-file pyproject.toml
Success: no issues found in 8 source files

$ uv run --extra dev python -m pytest src/tests/export/ -q
64 passed in 0.27s
```

Cross-cutting tests: 63/64 pass. The 1 failure is a pre-existing `test_zero_skip_contracts` issue in `advanced_visualization` (unrelated to export).

## Doc / Manuscript Follow-ups Needed (other workers own these)

- **`doc/`** references to export API: The docs_audit and check_doc_links gates should be run by the doc worker to verify no stale references to the old format lists. The module-level docstrings are updated; `doc/` prose may reference old dispatch patterns.
- **`src/mcp/audit_report.json`** or equivalent: If the repo has a generated MCP manifest that records per-module tool counts, the manifest worker should regenerate it (no tools were added or removed, but the `__all__` grew by 2).
- **`src/export/SPEC.md`** and **`src/export/SKILL.md`**: Light updates to mention the registry could be done by a doc-focused worker. I updated AGENTS.md/README.md only.

## Follow-up Ideas

1. **DI logger in `process_export`** — DONE (late close-out): `process_export` now honors an injected `logger` kwarg (`kwargs.pop("logger", None)`), falling back to the module logger. `verbose` only widens the level on the module-owned logger; an injected logger's level stays owned by its configurator. Verified: injected capture logger receives all 13 log records of a full run. AGENTS.md `logger` kwarg documentation is now accurate.
2. **Extend registry with `format_exporters` tuple-returning writers**: The registry currently maps to the `formatters`-family (bool-returning) writers. A second registry for the `format_exporters`-family (tuple-returning) writers would let `core.py`'s dispatch table be fully data-driven.
3. **TypedDict `ExportResult`**: The result dicts from `export_model`/`export_gnn_model`/`process_export` could be formalized as TypedDicts for mypy strictness.
4. **`validate_export_outputs` MCP tool**: Expose validation as an MCP tool once the `test_mcp_audit` allowlist is updated (coordinated with the MCP worker).
5. **Remove `format_exporters._gnn_model_to_dict` path-based parser**: It re-parses GNN files from disk; the content-based `processor._gnn_model_to_dict` is preferred. Migrate `core.py` to use the processor version.


## Late-Fix Doc Corrections (post-adversarial-review)

- **README.md Module Structure fence**: The code fence around the directory tree was missing its closing ` ``` `, causing the Export Workflow heading and mermaid diagram to render as literal code. Fixed by adding the closing fence.
- **AGENTS.md test list**: `test_export_overall.py` was omitted from the Test Files list. Added.
- **README.md test list**: `test_export_registry_and_validate.py` was missing. Added.

## Post-Review Concern Fixes (final pass)

- `registry.py`: dropped unused `Union` import; `get_format_categories` typing tightened to `Dict[str, list[str]]`.
- `test_export_registry_and_validate.py`: removed redundant manual monkeypatch restore (monkeypatch auto-undoes); `test_missing_export_file` now targets `*_json.json` so it cannot accidentally delete the `export_results.json` manifest.
- Verified pinned external contracts intact: `export.core.HAS_NETWORKX` / `export.core.FORMAT_EXPORTERS_LOADED` module globals (monkeypatch targets), `parse_gnn_content` still delegates to `gnn.parse_gnn_file` (zero-skip contract), all pinned package attributes present.

## Transient Cross-Worker Artifacts (not export scope, observed at turn end)

- `src/tests/test_zero_skip_contracts.py::test_default_suite_does_not_reintroduce_skips_or_xfails` fails on `src/tests/advanced_visualization/test_advanced_visualization_public_api_refactor.py` containing `pytest.skip` — advanced_visualization is a fleet peer's file; both export-scoped zero-skip assertions pass.
- `doc/development/docs_audit.py --strict --no-write` reports 1 issue: `src/tests/tests/` lacks an AGENTS.md — that directory is another worker's active workspace (modified files present); all `src/export/` doc invariants pass.