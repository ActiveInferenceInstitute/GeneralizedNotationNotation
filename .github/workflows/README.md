# GitHub workflows

YAML workflows for CI, documentation audit, workflow lint, dependency review, CodeQL, and scheduled supply-chain checks. Parent hub (Dependabot + full index): [../README.md](../README.md). Agent guide: [AGENTS.md](AGENTS.md).

## Workflow files

| File | Triggers | Jobs / behavior |
|------|----------|-----------------|
| [ci.yml](ci.yml) | `push` / `pull_request` → `main`; ignores `**/*.md`, `doc/**`; `workflow_dispatch` | **test**: Python 3.11–3.13, `uv sync --frozen --extra dev`, `pytest -m "not pipeline and not mcp"`; MCP tool count ≥ 131 on 3.12. **lint**: `ruff check src/`. **security**: Bandit medium+ on `src`, JSON artifact. |
| [docs-audit.yml](docs-audit.yml) | `push` / `pull_request` when `*.md`, `doc/**`, root `AGENTS.md`/`CLAUDE.md`/`README.md`/`SKILL.md`, or `doc/development/docs_audit.py` change; `workflow_dispatch` | `docs_audit.py --strict` |
| [actionlint.yml](actionlint.yml) | Changes under `.github/workflows/**`; `workflow_dispatch` | `rhysd/actionlint@v1.7.11` |
| [dependency-review.yml](dependency-review.yml) | `pull_request` → `main`; `workflow_dispatch` | Dependency review: fail on high severity; AGPL license deny list |
| [codeql.yml](codeql.yml) | `push` / `pull_request` → `main`, weekly cron, `workflow_dispatch` | CodeQL Python |
| [supply-chain-audit.yml](supply-chain-audit.yml) | Weekly cron Monday 06:00 UTC, `workflow_dispatch` | Two `pip-audit` jobs: frozen core export and frozen all-extras (no dev), OSV |

## Local validation

```bash
actionlint .github/workflows/*.yml
```

Run from repo root (paths relative to root).
