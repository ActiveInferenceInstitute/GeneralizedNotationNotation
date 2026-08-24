# Integration, MCP, API, CLI, GUI, and Website Report

Dispatch: `.agents/dispatch/gnn-improve/mission-integration.md`  
Repository: `/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation`

## Outcome

The scoped mission is complete. MCP inventories now match their live registrations
and every covered tool executes through the real dispatcher with explicit failures.
Both FastAPI surfaces use the exact JSON envelope `{status, data, error, meta}` and
validate invalid requests without leaking internal 500 details. CLI exits follow
`0` success, `1` error, and `2` warning/usage semantics. GUI 3's previously inert
controls are wired and its default model validates cleanly. All requested scoped
verification gates pass.

## Files changed

- API implementation: `src/api/{app.py,auth.py,mcp.py,models.py,path_utils.py,processor.py,rate_limit.py,responses.py,server.py}`.
- API documentation: `src/api/{AGENTS.md,README.md,SPEC.md}`.
- CLI implementation and documentation: `src/cli/{__init__.py,mcp.py,AGENTS.md,README.md,SKILL.md,SPEC.md}`.
- MCP and integration surfaces: `src/mcp/{mcp.py,server_core.py,AGENTS.md}` and `src/integration/AGENTS.md`.
- GUI implementation and documentation: `src/gui/gui_3/ui_designer.py`, `src/gui/oxdraw/mcp.py`, `src/gui/{AGENTS.md,README.md,SKILL.md}`, and `src/gui/gui_3/AGENTS.md`.
- Website implementation and documentation: `src/website/{mcp.py,AGENTS.md}`.
- API tests: `src/tests/api/{test_api_endpoints.py,test_api_mcp_tools.py,test_api_response_contract.py,test_auth.py}`.
- CLI tests: `src/tests/cli/{test_cli.py,test_cli_public_api.py}`.
- MCP/integration tests: `src/tests/mcp/test_mcp_functional.py` and `src/tests/integration/test_integration_mcp.py`.
- GUI tests: `src/tests/gui/{test_gui_functionality.py,test_oxdraw_integration.py}`.
- Website tests: `src/tests/website/test_website_overall.py`.
- Report: `.agents/dispatch/gnn-improve/REPORT-integration.md`.

## Fixes

- Added one canonical API response implementation and exception handlers for
  request validation, HTTP errors, authentication, rate limits, and sanitized
  unexpected failures. Tightened step, path, extra-field, pagination, and
  include/skip validation, including direct processor callers.
- Replaced the alternate API app's simulated pipeline completion with the real
  `src/main.py` subprocess path and canonical execution-summary events.
- Consolidated API MCP metadata into one five-tool inventory used by static and
  live registration. Corrected schemas and preserved explicit domain errors.
- Hardened the MCP dispatcher with parameter-object, required-field, and strict
  JSON-schema type checks; corrected health metrics and concurrency accounting;
  and stopped structured tool errors from being rewritten as internal errors.
- Corrected JSON-RPC parameter rejection and human-readable server errors.
  Repaired Oxdraw's direct registrations so keyword execution and legacy mapping
  calls both work. Corrected Integration, GUI, and Website inventory docs.
- Normalized CLI parsing and forwarding for step lists, combined skips, global
  verbosity, ports, overlaps, preflight health, interrupts, and unexpected
  failures. Removed a no-op option and added `--only-steps`.
- Wired GUI 3 add/remove/mapping/validation/layout/export/preview actions, made
  callbacks tolerant of Gradio/Pandas/list values, normalized dimensions and
  loaded parameters, escaped previews, wrote exports atomically, and repaired
  the default model's connections and state spaces.
- Isolated Website tests in temporary output directories so verification no
  longer mutates tracked generated artifacts.

## Tests added or expanded

- Exact API envelope checks for successes, 404/422/nonstandard HTTP errors,
  sanitized 500s, invalid fields/steps/paths, middleware failures, real subprocess
  dispatch, and warning exits.
- Static/live parity plus successful execution for all five API MCP tools and all
  four Integration MCP tools.
- MCP strict-type, malformed-params, structured-error, metrics, and JSON-RPC
  regressions.
- CLI exit-code, skip serialization, step overlap, verbosity, and handler-error
  regressions.
- GUI callback, validation, Unicode variable, escaping, atomic export, control
  binding, and direct Oxdraw MCP execution regressions.
- Website tool-inventory and output-isolation regressions.

## Scoped verification

- `uv run python -m pytest src/tests/integration src/tests/mcp src/tests/api src/tests/cli src/tests/gui src/tests/website -q --tb=no -x` — **640 passed in 11.89s**.
- `uv run ruff check src/integration src/mcp src/api src/cli src/gui src/website` — **All checks passed**.
- `uv run ruff format --check src/integration src/mcp src/api src/cli src/gui src/website` — **74 files already formatted**.
- `uv run mypy src/integration src/mcp src/api src/cli src/gui src/website --config-file pyproject.toml` — **Success: no issues found in 74 source files**.
- Additional mirror-test lint: `uv run ruff check src/tests/integration src/tests/mcp src/tests/api src/tests/cli src/tests/gui src/tests/website` — **All checks passed**.
- Scoped `git diff --check` — **clean**. Tracked Website output artifacts are
  unchanged after verification.

## Repository state

All mission changes are left uncommitted and unstaged. No commit, push, stage,
stash, reset, or clean operation was performed. Concurrent changes outside the
owned paths were preserved and are intentionally not included in this report.
