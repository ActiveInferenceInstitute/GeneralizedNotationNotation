# website-worker REPORT — fleet 3, 2026-09-04

Scope: `src/gnn/website/` entirely + orchestrator `src/gnn/20_website.py`. Branch main @ f64ac9085. No git operations, no dependency changes, no files outside scope edited (checkpoint logs excepted, per convention).

## Files changed + why

| File | Change |
|---|---|
| `src/gnn/website/generator.py` | (1) 25-step tuple-of-tuples → typed `StepInfo` frozen dataclass + `PIPELINE_STEPS` + public `get_pipeline_steps()` (with `script_name` display property); (2) data collection extracted from `WebsiteGenerator._collect_all_data` into pure module-level `collect_website_data()` with **injectable `mcp_tools_provider`** (the old code mutated `sys.path` and live-imported the MCP registry inline — now isolated in `_load_live_mcp_tools()` as the default provider, guarded insert); (3) all page builders HTML-escape pipeline-derived values via shared `_esc()` (analysis values, MCP tool fields, report content, viz titles/paths, GNN source text — previously only 2 of 7 pages escaped, via duplicated manual `.replace()` chains); (4) per-page resilience: each page is rendered + written independently via `_write_atomic()` (temp file + `os.replace`, cleanup on failure) — one bad page records an error instead of destroying the whole build (previously an exception in any page builder aborted ALL pages); (5) removed dead statement at old line 861 (f-string computed and discarded); deduped the badge class/label maps duplicated across `_page_index`/`_page_pipeline`; hoisted inline `os`/`tempfile` imports; result dict now also carries `pages` (written filenames). |
| `src/gnn/website/renderer.py` | (1) The five `embed_*` functions deduplicated into one `_write_embed_page()` skeleton (each ~20 lines → ~10; identical wrapper semantics, per-type extra CSS preserved); (2) `embed_markdown_file`/`embed_text_file`/`embed_json_file` now HTML-escape file content (previously raw injection into `<pre>` — an HTML/script break-out in generated artifacts); `embed_image` escapes the src attribute; `embed_html_file` stays verbatim by design; (3) all file I/O is explicit `encoding="utf-8"`; (4) `process_website` writes the manifest through `_write_results_manifest()`; (5) `get_module_info()["version"]` now reports the package `__version__` (was hardcoded `"1.0.0"` vs `1.6.0` — drift). |
| `src/gnn/website/inspection.py` | **NEW** — pure site-query core: `inspect_website()`, `list_website_pages()`, `KEY_PAGES` (7 canonical pages). Extracted from mcp.py so the Python API and MCP tools share one implementation. |
| `src/gnn/website/mcp.py` | `get_website_status_mcp` and `list_generated_pages_mcp` are now one-line delegations to `website.inspection` (response shapes byte-identical). Still exactly 5 registered tools (inventory is pinned by tests — no additions). |
| `src/gnn/website/dashboard.py` | `_render_step_details` traversed each artifact dir with `rglob` **twice** (files + recount); now one pass. |
| `src/gnn/website/__init__.py` | Additive re-exports: `render_dashboard`, `collect_website_data`, `get_pipeline_steps`, `PIPELINE_STEPS`, `StepInfo`, `inspect_website`, `list_website_pages`. `__version__` 1.6.0 → **1.7.0** (additive API). |
| `src/gnn/website/AGENTS.md`, `src/gnn/website/README.md` | New API documented (incl. `mcp_tools_provider` DI), inspection.py in structure block, manifest keys, page-resilience + escaping semantics, test-file list, version/last-updated. |
| `tests/website/test_website_generator_units.py` | **NEW** — 20 tests: step catalogue (coverage/frozen/script names), `collect_website_data` (provider injection replaces live registry, GNN discovery, step statuses, per-dir report cap, malformed-analysis skip), escaping per page (gnn/analysis/mcp/reports), resilient page writes (bad MCP data → `success False`, 6 pages survive, error message pinned), manifest contract (keys incl. `pages`/`generated_at`; orchestrator-style kwargs absorbed), embed escaping/missing-source parity, `get_module_info` version alignment. |
| `tests/website/test_website_inspection.py` | **NEW** — 8 tests: `KEY_PAGES` order, missing-dir errors, complete site (`pages_count == 7`, `all_key_pages_present`), partial site flags, listing metadata (`size_bytes`>0, ISO-8601 `modified`, sorted). |

`src/gnn/20_website.py`: **unchanged** (64 lines, thin-orchestrator compliant; `**kwargs` absorbs `logger`/`recursive`/`website_html_filename` — pinned by a new test).

## API deltas

- **Additive exports**: `render_dashboard`, `collect_website_data`, `get_pipeline_steps`, `PIPELINE_STEPS`, `StepInfo`, `inspect_website`, `list_website_pages` (package level); `website.inspection.{inspect_website, list_website_pages, KEY_PAGES}` (new module). All prior exports unchanged; all 5 MCP tool names and schemas unchanged.
- **`WebsiteGenerator.__init__(*, mcp_tools_provider=None)`** — new optional kwarg; zero-arg construction unchanged.
- **`generate_website(...)` result dict** — adds `pages: list[str]`.
- **Behavior deltas (deliberate, documented, tested)**:
  1. `success` is now `True` **only when no errors occurred**. Before, a page *write* failure left `success=True` (site silently incomplete); a page *render* failure aborted everything with `success=False`. Now: partial site → `success=False` with `pages_created` recording what landed, other pages intact. Rationale: step-20's exit code must reflect a broken publication; no test pinned the old partial-failure bool.
  2. HTML escaping of pipeline-derived values (fixes malformed/broken markup, incl. `<script>` break-out in embeds and pages).
  3. `embed_*` helpers `mkdir` the output parent (previously only `generate_html_report` did); undecodable sources still return `False`.

## Verification (tails)

```
$ uv run ruff check src/gnn/website tests/website
All checks passed!

$ uv run --extra dev mypy src/gnn/website --config-file pyproject.toml
Success: no issues found in 7 source files

$ uv run pytest tests/website/ -v        # = `just test-mod website` recipe
============================== 82 passed in 0.12s ==============================
```

`just` binary is **not installed** on this host (`command -v just` empty); the recipe's exact underlying command (`uv run pytest tests/{{MODULE}}/ -v`) was run instead.

Zero-regression spot checks (external consumers of the website API):
```
tests/test_core_modules.py -k website            → 3 passed
tests/pipeline/test_pipeline_scripts.py -k step20 → 2 passed
tests/api/test_comprehensive_api.py -k website   → 5 passed
tests/test_fast_suite.py -k website              → 1 passed
```

## doc/ or manuscript/ follow-ups (other workers own those)

- `doc/gnn/modules/20_website.md` (if it exists in maintained docs): may want the manifest-key and resilience notes; I did not touch `doc/`.
- `src/gnn/mcp/audit_report.json` / `tests/mcp/mcp_audit_report.json`: unchanged and still accurate (no tool added/removed).

## Follow-up ideas (out of scope today)

1. **Derive the step catalogue from `pipeline.step_registry.STEPS`** (canonical single source of truth, fixes approximate script names like `3_gnn_processing.py` → `3_gnn.py`). Blocked on a policy call: `pipeline/__init__` imports config/execution/health_check, which would break website's documented stdlib-only import guarantee — needs a docs/decision note or a registry split (`step_registry` is import-light but the package init is not).
2. Client-side search over `website_results.json` + a page keyword index (no new page; additive artifact).
3. Optional base64 data-URI embedding mode for `embed_image` (currently path-reference only, despite the name).
4. `render_dashboard` MCP tool — pinned 5-tool inventory blocks it; would require updating the pinned audit fixtures in the same change.

## Reduced-confidence note

GitNexus MCP tools were unavailable in this session, so the mandated impact analysis was done via repo-wide grep of every external consumer (20_website.py, 6 test files, step_registry string, audit JSONs) instead; all consumer tests were re-run green post-change.
