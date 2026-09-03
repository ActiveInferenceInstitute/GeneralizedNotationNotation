# Workflows Agent Guide

## Purpose

Defines behavior and guardrails for workflows in this directory. Human index of all `.github/` automation: [../README.md](../README.md).

## Workflow set

| File | Role |
|------|------|
| `ci.yml` | Matrix test with JUnit + coverage + artifact + summary; Ruff/mypy/doc audits on 3.12 only (merged into test job); MCP tool count ≥ 140 on 3.12 (`tests.mcp.test_mcp_audit.count_mcp_tools`); v3 orchestration acceptance gate on 3.12; Bandit SARIF → `upload-sarif` + artifact; job fails on findings. No path filter — runs on doc-only changes too. |
| `mcp-audit.yml` | MCP tool count ≥ 140 audit on push/PR to `main`. |
| `full-extras.yml` | Weekly all-extras suite: `uv sync --frozen --all-extras`, optional-import validation, full pytest (Python 3.12). |
| `docs-audit.yml` | Strict Markdown audit when docs or `docs_audit.py` change. |
| `actionlint.yml` | Lint workflow YAML when `.github/workflows/**` changes. |
| `dependency-review.yml` | PR gate: high-severity failures; AGPL deny list; PR comment summary on failure. |
| `codeql.yml` | Python CodeQL: `init` → `uv sync --frozen --extra dev` → `analyze`; skips doc-only paths on push/PR; weekly Monday 04:28 UTC cron + `workflow_dispatch`. |
| `supply-chain-audit.yml` | Scheduled `pip-audit` on frozen exports (core + all extras, no dev); bash `set -euo pipefail`; job summary. |

## Standards

- Use official actions pinned by major version.
- Use explicit `timeout-minutes`.
- Apply least-privilege `permissions` globally and per job.
- Use deterministic dependency operations (`uv sync --frozen`, `uv export --frozen`).
