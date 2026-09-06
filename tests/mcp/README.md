# MCP Tests

Pytest coverage for `src/gnn/mcp/`.

This folder contains module-focused tests for MCP registry discovery, tool behavior, and audit checks.

## Test Files

- `test_check_mcp_skills_health.py` — regression tests for `scripts/check_mcp_skills_health.py` (every `SKILL.md` surface resolves against the live codebase).
- `test_mcp_audit.py` — MCP audit-report behavior.
- `test_mcp_configurability.py` — MCP server configurability.
- `test_mcp_functional.py` — end-to-end MCP tool behavior.
- `test_mcp_http_auth.py` — HTTP transport authentication.
- `test_mcp_overall.py` — module-level aggregate contract for the MCP folder.
- `test_mcp_performance.py` — MCP performance characteristics.
- `test_mcp_standalone_modules.py` — standalone MCP module imports.
- `test_mcp_tools.py` — MCP tool registration and dispatch.
- `test_transport_reliability.py` — transport reliability behavior.

Run:

```bash
uv run --extra dev python -m pytest tests/mcp/ -q
```
