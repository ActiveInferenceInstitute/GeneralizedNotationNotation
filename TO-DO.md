# TO-DO — GNN Pipeline Roadmap

**Last Updated**: 2026-03-06  
**Current Version**: 2.0.0  
**Pipeline Steps**: 25 (0–24) · **Modules**: 38 · **MCP Tools**: 131 · **Tests**: 1,522+ · **Renderers**: 8/8

---

<details>
<summary><strong>Changelog</strong> (completed releases)</summary>

- **v2.0.0** (2026-03-06) — `src/api/app.py` (FastAPI, 6 endpoints, SSE streaming), `src/render/health.py` (8/8 renderer health check), `src/gnn/parse_cache.py` (section-level incremental cache), `src/pipeline/preflight.py` (config + environment validation).
- **v1.9.0** (2026-03-06) — `src/gnn/multimodel.py` (multi-model file support), `src/render/stan/stan_renderer.py` (Stan code generation), `src/lsp/__init__.py` (LSP diagnostics + hover).
- **v1.8.0** (2026-03-06) — `src/cli/__init__.py` (6 subcommands), `src/pipeline/hasher.py` (content-addressable hashing), `src/gnn/frontmatter.py` (YAML front-matter with fallback).
- **v1.7.0** (2026-03-06) — `src/pipeline/schemas.py` (Pydantic models), `src/gnn/contracts.py` (framework validation), `src/intelligent_analysis/remediation.py` (auto-fix suggestions).
- **v1.6.0** (2026-03-06) — `src/pipeline/context.py` (PipelineContext), `src/pipeline/dag.py` (Kahn topological sort), `src/pipeline/step_registry.py` (`@pipeline_step` decorator, 25-step auto-discovery).
- **v1.5.0** (2026-03-06) — `src/report/pipeline_report.py` (6-section report), `src/website/dashboard.py` (self-contained HTML SPA), `src/report/diff_report.py` (run comparison + archival).
- **v1.4.0** (2026-03-06) — GNN v1.1 syntax spec, `src/gnn/schema.py`, TextMate grammar.
- **v1.3.2** (2026-03-06) — Test markers, `--durations=20`, CI workflow.
- **v1.3.1** (2026-03-06) — LLM pre-pull guard, timeouts, `--skip-llm`, content-hash caching.
- **v1.3.0** (2026-03-02) — MCP deadlock fix, LLM recursive glob, ML class-imbalance cap.
- **v1.2.0** (2026-02-23) — ActiveInference.jl renderer bugs, unified test counts.

</details>

---

## v2.1.0a — CLI Polish & `gnn preflight`

> **Scope**: Add preflight check subcommand and wire lsp/api subcommands into CLI.

- [ ] **`gnn preflight`** subcommand — Runs `run_preflight()` from `src/pipeline/preflight.py`. Outputs Markdown report.
- [ ] **`gnn lsp`** subcommand — Launches `start_server()` from `src/lsp/__init__.py` on stdio.
- [ ] **`gnn serve`** subcommand — Starts `start_server()` from `src/api/app.py` with `--host` and `--port`.
- [ ] **`gnn health`** subcommand — Runs `check_renderers()` + `check_environment()` and prints summary.
- [ ] **pyproject.toml** — Update entrypoint to `gnn = "src.cli:main"`.

### v2.1.0a Acceptance

- [ ] `gnn preflight` produces Markdown report with 🟢/🔴 status
- [ ] `gnn health` shows 8/8 renderers

---

## v2.1.0b — Pipeline Event Hooks

> **Scope**: Wire PipelineContext event callbacks for SSE integration.

- [ ] **`PipelineContext.on_step_start`** callback — Optional callable invoked at step start.
- [ ] **`PipelineContext.on_step_complete`** callback — Optional callable invoked at step end.
- [ ] **`PipelineContext.on_error`** callback — Optional callable invoked on step failure.
- [ ] **API integration** — Wire callbacks to SSE event broadcasting in `api/app.py`.

### v2.1.0b Acceptance

- [ ] SSE stream emits `step_start` / `step_complete` events during run

---

## v2.1.0c — Structured Logging & JSON Log Output

> **Scope**: Machine-readable logs for pipeline observability.

- [ ] **`src/pipeline/logging_config.py`** [NEW] — Configures structured JSON logging (stdlib `logging`). Fields: timestamp, level, step, message, duration.
- [ ] **`--log-format json`** CLI flag — Switches to JSON line output for piping to log aggregators.
- [ ] **Log rotation** — Configured via `logging.handlers.RotatingFileHandler`, 10 MB per file, 5 backups.

### v2.1.0c Acceptance

- [ ] `gnn run --log-format json 2>&1 | python -m json.tool` parses each line

---

## v2.2.0a — Watcher Mode & Auto-Reparse

> **Scope**: File-watching for live re-validation during development.

- [ ] **`src/gnn/watcher.py`** [NEW] — Uses `watchdog` (or `inotify` fallback) to monitor GNN files.
- [ ] **`gnn watch <dir>`** subcommand — Monitors `input/gnn_files/` and re-runs validate on change.
- [ ] **Debouncing** — 250ms debounce to avoid rapid-fire re-validation.
- [ ] **Integration** — On change, runs `validate_required_sections()` + `parse_state_space()` and prints results.

### v2.2.0a Acceptance

- [ ] Editing a `.md` file triggers re-validation within 500ms

---

## v2.2.0b — Model Dependency Graph Visualization

> **Scope**: Generate visual dependency graph from multi-model files.

- [ ] **`src/gnn/dep_graph.py`** [NEW] — Builds networkx/mermaid graph from inter-model connections.
- [ ] **`gnn graph <file.md>`** subcommand — Outputs Mermaid diagram to stdout or `.svg` file.
- [ ] **Dashboard integration** — Embed dependency graph in `dashboard.html`.

### v2.2.0b Acceptance

- [ ] `gnn graph multi_model.md` outputs valid Mermaid diagram

---

## v2.3.0 — Deep Roadmap (Unscheduled)

> Major features requiring significant effort.

- [ ] Full VSCode extension (beyond LSP diagnostics)
- [ ] Distributed Ray/Dask execution for parallel parameter sweeps
- [ ] GPU-accelerated JAX on cloud instances
- [ ] Content-addressable model registry with `gnn reproduce <run-hash>` CLI

---

## Conventions

- Versions follow [SemVer](https://semver.org/) — `MAJOR.MINOR.PATCH`
- Sub-patches (`x.y.za`, `x.y.zb`) denote incremental shipments within a patch
- Patch releases completable in 1 focused session
- Minor releases completable in 1–3 focused sessions
- Deep roadmap items tracked for visibility but not scheduled
