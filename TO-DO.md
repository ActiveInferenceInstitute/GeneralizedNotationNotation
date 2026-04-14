# TO-DO — GNN Pipeline Roadmap

**Last Updated**: 2026-04-13
**Current Version**: 1.3.0
**Next Target**: v1.4.0

---

## v1.4.0 — Test Coverage Milestone

> **Scope**: Achieve ≥ 85% line coverage across all 31 modules.

- [ ] **CI enforcement** — Add `--cov-fail-under=80` to CI workflow (`ci.yml`)
- [ ] **Core modules ≥ 85%** — `gnn/`, `render/`, `execute/`, `validation/`, `type_checker/`
- [ ] **Infrastructure ≥ 80%** — `pipeline/`, `utils/`, `cli/`, `api/`, `lsp/`

### Acceptance

```bash
uv run pytest --cov=src --cov-fail-under=80  # must pass
```

---

## v1.5.0 — Structured Logging & Print Cleanup

> **Scope**: ~105 raw `print()` calls remain in non-test production code. Replace with `logging` and add pipeline metrics export.

- [ ] **Print audit** — Replace raw `print()` in `src/` production files with `logger.info()` / `logger.debug()` (currently ~105 occurrences)
- [ ] **JSON log format** — Ensure `--log-format json` produces valid JSON-lines output from all 25 steps
- [ ] **Performance dashboard** — Generate `output/00_pipeline_summary/performance_dashboard.html` with step timing, memory, and throughput charts

### Acceptance

```bash
grep -rn "print(" src/ --include="*.py" | grep -v test_ | grep -v __pycache__ | wc -l  # should be 0
```

---

## v1.6.0 — Renderer & Executor Parity

> **Scope**: All 8 renderers produce runnable code; execute module has matching runners.
>
> **Current state**: `render/stan/` exists but no `execute/stan/` runner. DisCoPy has both. PyMDP, JAX, PyTorch, NumPyro, RxInfer, ActiveInference.jl, bnlearn all operational.

- [ ] **Stan executor** — Create `execute/stan/` runner or document as render-only with rationale
- [ ] **Renderer smoke tests** — Ensure all 8 renderers pass output-validation smoke tests
- [ ] **Cross-framework test** — Integration test that renders + executes the same GNN model across ≥ 3 frameworks

### Acceptance

```bash
uv run pytest src/tests/test_render*.py src/tests/test_execute*.py -v  # all pass
```

---

## v1.7.0 — Documentation Quality

> **Scope**: Raise docstring coverage and eliminate broken links.
>
> **Current state**: `doc/gnn/modules/[00-24].md` freshly synced from `src/*/AGENTS.md`. Several production modules still have functions without docstrings.

- [ ] **Docstring coverage ≥ 80%** — Prioritise `visualization/`, `ml_integration/`, `research/` (currently weakest)
- [ ] **Broken link audit** — Run link checker across `doc/` and fix any broken references
- [ ] **API reference** — Auto-generate API docs from docstrings using `pdoc` for top 10 modules

### Acceptance

```bash
# Zero broken internal links verified by link checker
```

---

## v1.8.0 — Developer Experience

> **Scope**: Pre-commit hooks, justfile, and dev containers.
>
> **Current state**: `[tool.ruff]` and `[tool.mypy]` already configured in `pyproject.toml`. No `.pre-commit-config.yaml`, `justfile`, `.vscode/`, or `.devcontainer/` exist yet.

- [ ] **Pre-commit hooks** — Add `.pre-commit-config.yaml` with ruff, black, mypy, and markdownlint
- [ ] **justfile** — Create `justfile` with common commands (`just test`, `just lint`, `just run`, `just docs`)
- [ ] **VS Code settings** — Add `.vscode/settings.json` + `.vscode/extensions.json`
- [ ] **Dev containers** — Add `.devcontainer/devcontainer.json` for GitHub Codespaces

### Acceptance

```bash
pre-commit run --all-files  # passes
just test                   # runs test suite
```

---

## Conventions

- Versions follow [SemVer](https://semver.org/) — `MAJOR.MINOR.PATCH`
- Patch releases (1.3.x) target a single file or narrow focus area, completable in 1 session
- Minor releases (1.x.0) are milestone releases requiring multiple sessions
- Each release has concrete acceptance criteria with verifiable commands
