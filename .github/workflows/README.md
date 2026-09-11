# GitHub workflows

YAML workflows for CI, MCP tool-count audit, weekly all-extras suite, documentation audit, workflow lint, dependency review, CodeQL, and scheduled supply-chain checks. Parent hub (Dependabot + full index): [../README.md](../README.md). Agent guide: [AGENTS.md](AGENTS.md).

## Workflow files

| File | Triggers | Jobs / behavior |
|------|----------|-----------------|
| [ci.yml](ci.yml) | `push` / `pull_request` → `main` (no path filter); `workflow_dispatch` | **test**: matrix 3.11–3.13; Python 3.12 also runs Ruff format/check over `src scripts`, terminology audits, docs audit, doc contract audit, GNN doc patterns, mypy, collect-only, focused PyMDP/POMDP tests, MCP ≥ 140 (see `tests/mcp_audit_report.json`), and the v3 orchestration acceptance gate. All matrix entries run JUnit + coverage + artifact + summary. **security**: Bandit SARIF → code scanning + artifact. |
| [mcp-audit.yml](mcp-audit.yml) | `push` / `pull_request` → `main`; `workflow_dispatch` | MCP tool count ≥ 140 via `tests.mcp.test_mcp_audit.count_mcp_tools`. |
| [full-extras.yml](full-extras.yml) | Weekly cron Sunday 06:00 UTC (`0 6 * * 0`); `workflow_dispatch` | `uv sync --frozen --all-extras`, optional-import checks (audio, GUI, research/scaling), full pytest suite under all extras (Python 3.12). |
| [docs-audit.yml](docs-audit.yml) | `push` / `pull_request` when `*.md`, `docs/**`, root `AGENTS.md`/`CLAUDE.md`/`README.md`/`SKILL.md`, or `docs/development/docs_audit.py` change; `workflow_dispatch` | Strict docs audit with anchors plus repository/doc terminology and GNN doc-pattern audits. |
| [actionlint.yml](actionlint.yml) | Changes under `.github/workflows/**`; `workflow_dispatch` | `rhysd/actionlint@v1.7.12` |
| [dependency-review.yml](dependency-review.yml) | `pull_request` → `main`; `workflow_dispatch` | High severity + AGPL deny; PR comment summary on failure. Fork PRs may get limited review. |
| [codeql.yml](codeql.yml) | `push` / `pull_request` (skips doc-only paths), weekly cron, `workflow_dispatch` | Init → `uv sync --frozen --extra dev` → analyze (Python). |
| [supply-chain-audit.yml](supply-chain-audit.yml) | Weekly cron Monday 06:00 UTC, `workflow_dispatch` | Two `pip-audit` jobs (OSV); strict shell; job summary. |
| [custody-re-render.yml](custody-re-render.yml) | Daily cron 07:14 UTC (`14 7 * * *`); `workflow_dispatch` | Report-only fresh manuscript render from the `docxology/template` checkout (symlinked at `projects/active/`): template `stage_03_render` → record render-custody manifest → strict token gate → `tests/test_manuscript_latex_log.py`; manifest + rendered evidence uploaded as artifact, nothing committed back. |

## Local validation

```bash
actionlint .github/workflows/*.yml
```

Run from repo root (paths relative to root).

CI sets `UV_PYTHON` to the test matrix version (and 3.12 for security and
documentation jobs), overriding the local `.python-version` pin. XML export
validation uses `defusedxml` and rejects entity declarations.
