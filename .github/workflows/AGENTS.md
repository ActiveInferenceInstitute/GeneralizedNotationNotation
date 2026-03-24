# Workflows Agent Guide

## Purpose

Defines behavior and guardrails for workflows in this directory. Human index of all `.github/` automation: [../README.md](../README.md).

## Workflow set

| File | Role |
|------|------|
| `ci.yml` | Matrix test (`pytest -m "not pipeline and not mcp"`), MCP tool count on 3.12, Ruff, Bandit (medium+). Skips pure `*.md` / `doc/**` paths. |
| `docs-audit.yml` | Strict Markdown audit when docs or `docs_audit.py` change. |
| `actionlint.yml` | Lint workflow YAML when `.github/workflows/**` changes. |
| `dependency-review.yml` | PR gate: high-severity failures; AGPL deny list. |
| `codeql.yml` | Python CodeQL on push/PR/schedule. |
| `supply-chain-audit.yml` | Scheduled `pip-audit` on frozen exports (core + all extras, no dev). |

## Standards

- Use official actions pinned by major version.
- Use explicit `timeout-minutes`.
- Apply least-privilege `permissions` globally and per job.
- Use deterministic dependency operations (`uv sync --frozen`, `uv export --frozen`).
