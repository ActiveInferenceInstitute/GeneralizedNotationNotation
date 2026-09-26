# Website Tests

Pytest coverage for `src/gnn/website/`.

This folder contains module-focused tests for static website generation and artifact discovery.

## Test Files

- `test_website_collection.py` — the pipeline-artifact collection seam in `gnn/website/collection.py`: `collect_website_data` and its private collectors, the package re-export identity, and `WebsiteGenerator._collect_all_data` delegation.
- `test_website_index_dashboard.py` — the folded dashboard data on the generated index page: canonical-summary badge/header (status, end time, duration, peak memory), the registry+filesystem artifact browser with capped previews, and per-step memory receipts with graceful empty states.
- `test_website_generator_units.py` — generator internals: the step-registry-derived typed catalogue, step statuses sourced from `pipeline_execution_summary.json` with the dir-heuristic fallback, pure `collect_website_data` sourcing the MCP page from the step-21 artifacts (`mcp_processing_summary.json` + `registered_tools.json`) with truthful empty states, gallery asset-collision handling, truncation markers, HTML escaping, resilient per-page writes, and the `website_results.json` manifest contract.
- `test_website_inspection.py` — `inspect_website` / `list_website_pages` contracts shared by the Python API and the MCP tools, on complete, partial, and missing websites.
- `test_website_overall.py` — module-level aggregate contract for the website folder.
- `test_website_public_api.py` — public API surface: `embed_*` functions, `generate_html_report`, `process_website`, `FEATURES`, `SUPPORTED_FILE_TYPES`, `get_supported_file_types`, and `generate_website`.
- `test_website_steps.py` — the static step catalogue in `gnn/website/steps.py`: `StepInfo` fields/frozen dataclass behavior, the derived `script_name`/`output_dir_name` round-trips to the registry, display-name derivation, and the package re-exports (2026-09-26).
- `test_website_composability.py` — the dict-driven composability seams (2026-09-26): the `collection → steps` cycle direction (source-level pins + one `PIPELINE_STEPS` object across collection/generator/package), the pure no-filesystem `website_data_from_dict` defaults/merge contract and `generate_website(filesystem=False)` zero-collector generation, and the `SUPPORTED_FILE_TYPES` single-source pin (`renderer.py` definition, `__init__` re-export, derived inventories).

Run:

```bash
uv run --extra dev python -m pytest tests/website/ -q
```
