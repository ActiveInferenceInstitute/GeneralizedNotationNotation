# Specification: Workflows

## Design Requirements
The `workflows` directory serves as the execution layer for the GNN repository's Continuous Integration and Continuous Deployment (CI/CD) pipeline. Its primary design requirement is to enforce 100% Zero-Mock compliance, maintain repository stability, and ensure code quality via automated execution graphs triggered by GitHub events.

The workflow system must:
1. Support deterministic isolation of testing environments utilizing `uv`.
2. Securely handle permission scopes and artifact generation.
3. Facilitate strict parallelized static analysis (CodeQL, Bandit, Actionlint) and security audits.
4. Offer resilient and debuggable job summaries.

## Components
The module defines strict automation routines via YAML configuration:

1. **`ci.yml`**: The primary test matrix (Python 3.11, 3.12, 3.13) that executes the Zero-Mock Pytest suite, Ruff linting, MCP tool count assertions, and Bandit security scans, halting on critical code health violations.
2. **`actionlint.yml`**: Safeguards workflow logic by running static analysis on the YAML files themselves before integration.
3. **`codeql.yml`**: Triggers GitHub's advanced CodeQL semantic scanning engine for deep security auditing.
4. **`dependency-review.yml`**: A supply chain guardrail enforcing license compliance (denying AGPL) and halting high-severity CVE introductions on Pull Requests.
5. **`docs-audit.yml`**: An orchestration layer executing the `docs_audit.py` strict enforcement of Markdown integrity.
6. **`supply-chain-audit.yml`**: A weekly scheduled audit explicitly exporting headless `uv` lockfiles and executing `pip-audit` via OSV vulnerability tracking.

## Technical Rules
- **Syntax**: 100% strict adherence to GitHub Actions YAML schemas.
- **Resilience**: Every workflow must declare explicit `timeout-minutes` and granular `permissions`.
- **Parallelism**: Implement appropriate `concurrency` groups to aggressively cancel redundant pipeline executions and conserve CI runner time.
- **Determinism**: Dependency installation must strictly use `uv sync --frozen` or equivalent zero-drift package resolution commands to guarantee reproducible CI environments.
