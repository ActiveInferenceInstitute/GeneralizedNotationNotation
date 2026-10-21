# Specification: .github

## Design Requirements
The `.github` directory owns repository presentation, automation, compliance,
security orchestration and issue tracking configuration. It connects the GNN
repository to GitHub's landing page and orchestration platform.

Its core design requirement is to facilitate real-implementation test coverage through automated, reliable, and secure Continuous Integration workflows. It natively scaffolds standard GitHub integrations such as `Dependabot` version bumping, Action Linting, Pytest Runner orchestration, Docs Audits, and Python CodeQL scanning with small explicit Python helpers for paired-runtime checks and release-status collection.

## Components
The integration footprint includes YAML configuration, documentation and standard-library Python helpers. `release_gate.py` collects exact-source workflow/job status; `fep_paired_runtime.py` validates paired runtime inputs. Their adjacent tests cover their contracts.

[README.md](README.md) is the GitHub repository landing page. Its generated
[overview artwork](gnn-overview-2026-10-21.png) and
[generation prompt](gnn-overview-2026-10-21.prompt.txt) live beside it. This
conceptual illustration is separate from source-bound scientific figures and
published release assets.

Core configuration components include:
1. **`dependabot.yml`**: Supply chain updates routing, responsible for maintaining dependency freshness for pip and GitHub Actions ecosystems with strict scheduled cadences.
2. **`workflows/ci.yml`**: Headless end-to-end multi-version execution matrix for running Pytest, Ruff linting, and Bandit security scanning.
3. **`workflows/docs-audit.yml`**: CI structural validation of the GNN documentation system, enforcing Markdown integrity and logical inter-linking.
4. **`workflows/codeql.yml`**: Static analysis tracking delegating semantic code security audits to GitHub's native engine.
5. **`workflows/supply-chain-audit.yml`**: Active OSV pipeline vulnerability tracking via `pip-audit`.
6. **`workflows/actionlint.yml`**: Static validation of workflow YAML schemas on changes under `.github/workflows/`.
7. **`workflows/dependency-review.yml`**: PR gate denying AGPL licenses and failing on high-severity vulnerability introductions.
8. **`workflows/mcp-audit.yml`**: Fast MCP tool-count regression gate (≥ 140 registered tools) on push/PR to `main`.
9. **`workflows/full-extras.yml`**: Weekly scheduled matrix validating optional dependency groups and running the full pytest suite under `--all-extras`.

The complete current workflow inventory and trigger semantics live in
[workflows/README.md](workflows/README.md); the list above highlights core automation.

## Technical Rules
- **Syntax Validation**: Ensure 100% structural adherence to GitHub Actions schema formats; require successful execution of `actionlint` prior to merging structural changes.
- **Local Parity**: Workflow modifications must guarantee complete execution parity with local developer testing logic (e.g., exclusively orchestrating the `uv` environment layer using robust CLI mappings).
- **Least-Privilege Security**: Workflows should request the smallest possible set of GitHub Token permissions to mitigate supply chain risks.
