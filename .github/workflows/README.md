# GitHub workflows

YAML workflows for CI, documentation audit, workflow lint, dependency review, CodeQL, and scheduled supply-chain checks. Parent hub (Dependabot + full index): [../README.md](../README.md). Agent guide: [AGENTS.md](AGENTS.md).

## Workflow files

| File | Triggers | Jobs / behavior |
|------|----------|-----------------|
| [ci.yml](ci.yml) | `push` / `pull_request` → `main`; ignores `**/*.md`, `doc/**`; `workflow_dispatch` | **test**: matrix 3.11–3.13, Ruff on 3.12, JUnit + artifact + summary, MCP ≥ 131 on 3.12. **security**: Bandit SARIF → code scanning + artifact. |
| [docs-audit.yml](docs-audit.yml) | `push` / `pull_request` when `*.md`, `doc/**`, root `AGENTS.md`/`CLAUDE.md`/`README.md`/`SKILL.md`, or `doc/development/docs_audit.py` change; `workflow_dispatch` | `docs_audit.py --strict` |
| [actionlint.yml](actionlint.yml) | Changes under `.github/workflows/**`; `workflow_dispatch` | `rhysd/actionlint@v1.7.11` |
| [dependency-review.yml](dependency-review.yml) | `pull_request` → `main`; `workflow_dispatch` | High severity + AGPL deny; PR comment summary on failure. Fork PRs may get limited review. |
| [codeql.yml](codeql.yml) | `push` / `pull_request` (skips doc-only paths), weekly cron, `workflow_dispatch` | Init → `uv sync --extra dev` → analyze (Python). |
| [supply-chain-audit.yml](supply-chain-audit.yml) | Weekly cron Monday 06:00 UTC, `workflow_dispatch` | Two `pip-audit` jobs (OSV); strict shell; job summary. |

## Local validation

```bash
actionlint .github/workflows/*.yml
```

Run from repo root (paths relative to root).
