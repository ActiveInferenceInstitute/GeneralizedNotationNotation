# TO-DO — GNN Pipeline Roadmap

**Last Updated**: 2026-03-15
**Current Version**: 1.3.0
**Next Target**: v1.4.0

---

## v1.4.0 — Test Coverage Milestone

> **Scope**: Achieve ≥ 85% line coverage across all 31 modules. This is a minor release milestone.

- [x] **Coverage baseline** — Run `uv run pytest --cov=src --cov-report=term-missing` and record per-module coverage
- [x] **Identify gaps** — List all modules below 80% coverage with specific uncovered functions
- [ ] **Core modules** — Ensure `gnn/`, `render/`, `execute/`, `validation/`, `type_checker/` each exceed 85%
- [ ] **Infrastructure modules** — Ensure `pipeline/`, `utils/`, `cli/`, `api/`, `lsp/` each exceed 80%
- [ ] **CI enforcement** — Add `--cov-fail-under=80` to CI workflow (`ci.yml`)

### v1.4.0 Acceptance

- [ ] `uv run pytest --cov=src --cov-fail-under=80` passes
- [ ] CI pipeline enforces coverage floor

---

## v1.5.0 — Pipeline Observability & Structured Logging

> **Scope**: Replace raw `print()` calls with structured logging and add pipeline metrics dashboard.

- [ ] **Structured logging** — Audit and replace raw `print()` calls in non-test production code with `logger.info()` / `logger.debug()`
- [ ] **JSON log format** — Ensure `--log-format json` produces valid JSON-lines output from all 25 steps
- [ ] **Performance dashboard** — Generate `output/00_pipeline_summary/performance_dashboard.html` with step timing, memory, and throughput charts
- [ ] **Memory profiling** — Add optional `--profile-memory` flag that records peak RSS per step
- [ ] **Step dependency graph** — Generate live Mermaid diagram of step DAG execution in pipeline summary

### v1.5.0 Acceptance

- [ ] Zero raw `print()` calls in non-test `src/` files (all use `logging`)
- [ ] `gnn run --log-format json` produces valid JSONL
- [ ] Pipeline summary includes HTML performance dashboard

---

## v1.6.0 — Renderer Parity & Execution Coverage

> **Scope**: Ensure all 8 renderers produce runnable code and execute module has matching runners.

- [ ] **Stan renderer** — Verify `render/stan/` produces valid `.stan` files; add smoke test
- [ ] **DisCoPy renderer** — Verify `render/discopy/` produces valid DisCoPy circuits; add smoke test
- [ ] **Execute parity** — Create `execute/stan/` runner (or document as render-only with rationale)
- [ ] **Execute parity** — Create `execute/discopy/` runner (or document as render-only with rationale)
- [ ] **Cross-framework test** — Add integration test that renders + executes the same GNN model across PyMDP, JAX, and PyTorch
- [ ] **Renderer benchmarks** — Generate timing comparison table across all 8 renderers for `actinf_pomdp_agent.md`

### v1.6.0 Acceptance

- [ ] All 8 renderers pass smoke tests
- [ ] `execute/` has runners for ≥ 6 of 8 frameworks (remaining 2 documented as render-only)
- [ ] Cross-framework integration test passes

---

## v1.7.0 — Documentation Quality & Discoverability

> **Scope**: Raise all module documentation to comprehensive quality and improve cross-linking.

- [ ] **Docstring coverage** — Increase from current level to ≥ 80% of all public functions having docstrings
- [ ] **`doc/` index** — Create `doc/INDEX.md` that lists all 580+ documentation files with one-line descriptions
- [ ] **Module READMEs** — Ensure all 31 module README.md files have ≥ 50 lines with usage examples
- [ ] **API reference** — Auto-generate API docs from docstrings using `pdoc` or `sphinx-autodoc` for top 10 modules
- [ ] **Search index** — Add `doc/search_index.json` for documentation search tooling
- [ ] **Broken link audit** — Run link checker across all `doc/` files and fix any broken references

### v1.7.0 Acceptance

- [ ] Docstring coverage ≥ 80%
- [ ] `doc/INDEX.md` exists with ≥ 500 entries
- [ ] Zero broken internal links in `doc/`

---

## v1.8.0 — Developer Experience & Tooling

> **Scope**: Improve the development workflow with pre-commit hooks, linting, and automation.

- [ ] **Pre-commit hooks** — Add `.pre-commit-config.yaml` with ruff, black, mypy, and markdownlint
- [ ] **Ruff configuration** — Add `[tool.ruff]` section to `pyproject.toml` with project-specific rules
- [ ] **Mypy configuration** — Add `[tool.mypy]` section to `pyproject.toml` with gradual typing config
- [ ] **Makefile / justfile** — Create `justfile` with common developer commands (`just test`, `just lint`, `just run`, `just docs`)
- [ ] **VS Code settings** — Add `.vscode/settings.json` + `.vscode/extensions.json` for recommended dev environment
- [ ] **Dev containers** — Add `.devcontainer/devcontainer.json` for GitHub Codespaces / VS Code remote containers

### v1.8.0 Acceptance

- [ ] `pre-commit run --all-files` passes
- [ ] `just test` runs test suite
- [ ] `.devcontainer/` works in GitHub Codespaces

---

## Conventions

- Versions follow [SemVer](https://semver.org/) — `MAJOR.MINOR.PATCH`
- Patch releases (1.3.x) target a single file or narrow focus area, completable in 1 session
- Minor releases (1.x.0) are milestone releases requiring multiple sessions
- Each release has concrete acceptance criteria with verifiable commands
