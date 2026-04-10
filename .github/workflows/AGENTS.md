# Workflows Agent Guide

## Purpose

Defines behavior and guardrails for workflows in this directory. Human index of all `.github/` automation: [../README.md](../README.md).

## Workflow set

| File | Role |
|------|------|
| `ci.yml` | Matrix test with JUnit XML + artifact per Python version, job summary; MCP tool count on 3.12; Ruff; Bandit (medium+) with JSON artifact on pass or fail. Skips pure `*.md` / `doc/**` paths. |
| `docs-audit.yml` | Strict Markdown audit when docs or `docs_audit.py` change. |
| `actionlint.yml` | Lint workflow YAML when `.github/workflows/**` changes. |
| `dependency-review.yml` | PR gate: high-severity failures; AGPL deny list; PR comment summary on failure. |
| `codeql.yml` | Python CodeQL: `init` → `uv sync --frozen --extra dev` → `analyze`; skips doc-only paths on push/PR; weekly schedule + dispatch unchanged. |
| `supply-chain-audit.yml` | Scheduled `pip-audit` on frozen exports (core + all extras, no dev); bash `set -euo pipefail`; job summary. |

## Standards

- Use official actions pinned by major version.
- Use explicit `timeout-minutes`.
- Apply least-privilege `permissions` globally and per job.
- Use deterministic dependency operations (`uv sync --frozen`, `uv export --frozen`).
