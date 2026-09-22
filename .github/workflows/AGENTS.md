# Workflows Agent Guide

## Purpose

Defines behavior and guardrails for workflows in this directory. Human index of all `.github/` automation: [../README.md](../README.md).

## Workflow set

| File | Role |
|------|------|
| `ci.yml` | Matrix test with JUnit + coverage + artifact + summary; Ruff/mypy/doc audits on 3.12 only (merged into test job); MCP tool count ≥ `MCP_TOOL_FLOOR` (140, defined in `tests/mcp/test_mcp_audit.py` — the single source shared with `mcp-audit.yml` and the justfile gate); v3 orchestration acceptance gate on 3.12; Bandit SARIF → `upload-sarif` + artifact; job fails on findings. No path filter — runs on doc-only changes too. |
| `mcp-audit.yml` | MCP tool count ≥ `MCP_TOOL_FLOOR` audit on push/PR to `main`. |
| `full-extras.yml` | Weekly all-extras suite: `uv sync --frozen --all-extras`, optional-import validation, full pytest (Python 3.12). |
| `docs-audit.yml` | Strict Markdown audit when docs or `docs_audit.py` change. |
| `actionlint.yml` | Lint workflow YAML when `.github/workflows/**` changes. |
| `dependency-review.yml` | PR gate: high-severity failures; AGPL deny list; PR comment summary on failure. |
| `codeql.yml` | Python CodeQL: `init` → `uv sync --frozen --extra dev` → `analyze`; skips doc-only paths on push/PR; weekly Monday 04:28 UTC cron + `workflow_dispatch`. |
| `supply-chain-audit.yml` | Scheduled `pip-audit` on frozen exports (core + all extras, no dev); bash `set -euo pipefail`; job summary. |
| `custody-re-render.yml` | Daily cron 07:14 UTC + `workflow_dispatch`; report-only (no commit back). Fresh manuscript render via the `docxology/template` checkout (symlinked at `projects/active/`): template `stage_03_render` → record render-custody manifest → strict token gate → `tests/test_manuscript_latex_log.py`; receipts + rendered evidence uploaded as artifact. |
| `fep-lean-paired-revision.yml` | Paired-revision CI for the fep_lean bridge pair: validates `.github/fep-lean-pair.json`, checks out fep_lean at the pinned SHA, runs fep_lean's read-only bridge surface (status, emit `--check` finite/continuous) against this GNN checkout; blocking. Canonical custody ordering: [docs/development/fep_lean_paired_revision.md](../../docs/development/fep_lean_paired_revision.md). |
| `pair-pin-freshness.yml` | Nightly pair-pin freshness gate (BC-12, scope-comp-consumers §5.3): validates both committed pair pins and asserts each pinned companion revision is ancestor-or-equal of the companion default-branch HEAD (`scripts/check_pair_pin_freshness.py`); nightly cron + `workflow_dispatch`. Exit 2 "re-pin required" names the stale pair file; a red run means the pair pin needs a bump. |

## Standards

- Pin every third-party action to a full commit SHA with the version as a trailing comment (e.g. `uses: actions/checkout@<sha> # v7.0.1`) for supply-chain hardening; Dependabot reads the version comment. Resolve SHAs with `git ls-remote` — never guess.
- Pin `astral-sh/setup-uv` steps with `version: "0.12"` (the Dockerfile `UV_VERSION` bootstrap floor minor series).
- Use explicit `timeout-minutes`.
- Apply least-privilege `permissions` globally and per job.
- Use deterministic dependency operations (`uv sync --frozen`, `uv export --frozen`).

CI sets `UV_PYTHON` to the test matrix version (and 3.12 for security and
documentation jobs), overriding the local `.python-version` pin. XML export
validation uses `defusedxml` and rejects entity declarations.
