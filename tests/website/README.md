# Website Tests

Pytest coverage for `src/gnn/website/`.

This folder contains module-focused tests for static website generation and artifact discovery.

## Test Files

- `test_website_dashboard.py` — the self-contained HTML dashboard builder in `gnn/website/dashboard.py`.
- `test_website_generator_units.py` — generator internals: the typed step catalogue, pure `collect_website_data` with an injected MCP-tools provider, HTML escaping, resilient per-page writes, and the `website_results.json` manifest contract.
- `test_website_inspection.py` — `inspect_website` / `list_website_pages` contracts shared by the Python API and the MCP tools, on complete, partial, and missing websites.
- `test_website_overall.py` — module-level aggregate contract for the website folder.
- `test_website_public_api.py` — public API surface: `embed_*` functions, `generate_html_report`, `process_website`, `FEATURES`, `SUPPORTED_FILE_TYPES`, `get_supported_file_types`, and `generate_website`.

Run:

```bash
uv run --extra dev python -m pytest tests/website/ -q
```
