# Integration, MCP, API, CLI, GUI, Website Scope — mission-integration.md

You own these paths ONLY within the GNN repo at
`/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation`:

- src/gnn/integration/
- src/gnn/mcp/  (Model Context Protocol tools)
- src/gnn/api/  (FastAPI)
- src/gnn/cli/
- src/gnn/gui/  (gui_1/2/3, oxdraw)
- src/gnn/website/
- mirror tests: tests/integration/, tests/mcp/, tests/api/,
  tests/cli/, tests/gui/, tests/website/

DO NOT touch anything outside (see the shared no-touch list).

GOAL (shallow→deep):
1. MCP tool registration/health: each documented tool should register,
   execute, and report errors explicitly. Fix tool-surface drift
   (registered-vs-implemented mismatches). Keep the MCP inventory
   contract test green.
2. API/FastAPI surface: responses must be consistent {status,data,error,
   meta}; fix any 500s and missing validation.
3. CLI entry points: argument parsing consistency, helpful errors, exit
   codes (0 success / 1 error / 2 warning).
4. GUI polish: functional fixes, no-op buttons, crashed default states.
5. Remove mypy strict smells where genuinely better.

VERIFY (scoped only):
- `uv run python -m pytest tests/integration tests/mcp tests/api tests/cli tests/gui tests/website -q --tb=no -x`
- `uv run ruff check src/gnn/integration src/gnn/mcp src/gnn/api src/gnn/cli src/gnn/gui src/gnn/website`
- `uv run ruff format --check` (same tree)
- `uv run mypy src/gnn/integration src/gnn/mcp src/gnn/api src/gnn/cli src/gnn/gui src/gnn/website --config-file pyproject.toml`

HARD RULE: leave ALL changes uncommitted; no commit/push/stage.

## Finish
Write a concise report to
/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation/.agents/dispatch/gnn-improve/REPORT-integration.md
files changed, fixes, tests added, scoped verification. Reply with only
the report's absolute path.