# Specification: .github

## Design Requirements
The `.github` directory is structurally dedicated to repository automation, compliance, security orchestration, and issue tracking configurations within the GNN ecosystem. It serves as the primary declarative interface between the GNN repository's source code and the GitHub orchestration platform. 

Its core design requirement is to facilitate 100% Zero-Mock compliance through automated, reliable, and secure Continuous Integration workflows. It natively scaffolds standard GitHub integrations such as `Dependabot` version bumping, Action Linting, Pytest Runner orchestration, Docs Audits, and Python CodeQL scanning without introducing runtime code logic.

## Components
Expected available types: No exported Python classes are native to this location. The integration footprint consists exclusively of YAML configuration files and associated documentation.

Core configuration components include:
1. **`dependabot.yml`**: Supply chain updates routing, responsible for maintaining dependency freshness for Piper/Python and GitHub Actions ecosystems with strict scheduled cadences.
2. **`workflows/ci.yml`**: Headless end-to-end multi-version execution matrix for running Zero-Mock Pytest, Ruff linting, and Bandit security scanning.
3. **`workflows/docs-audit.yml`**: Pre-commit structural validation of the GNN documentation system, enforcing Markdown integrity and logical inter-linking.
4. **`workflows/codeql.yml`**: Static analysis tracking delegating semantic code security audits to GitHub's native engine.
5. **`workflows/supply-chain-audit.yml`**: Active OSV pipeline vulnerability tracking via `pip-audit`.

## Technical Rules
- **Syntax Validation**: Ensure 100% structural adherence to GitHub Actions schema formats; require successful execution of `actionlint` prior to merging structural changes.
- **Local Parity**: Workflow modifications must guarantee complete execution parity with local developer testing logic (e.g., exclusively orchestrating the `uv` environment layer using robust CLI mappings).
- **Least-Privilege Security**: Workflows should request the smallest possible set of GitHub Token permissions to mitigate supply chain risks.
