# TO-DO — GNN Pipeline Roadmap

**Last Updated**: 2026-03-13  
**Current Version**: 2.2.0  
**Pipeline Steps**: 25 (0–24) · **Modules**: 38 · **MCP Tools**: 131 · **Tests**: 1,522+ · **Renderers**: 8/8

---

<details>
<summary><strong>Changelog</strong> (completed releases)</summary>

- **v2.2.0** (2026-03-13) — `src/gnn/watcher.py` (Live re-validation Watcher mode), `src/gnn/dep_graph.py` (Model Dependency Graph & Dashboard integration).
- **v2.1.0** (2026-03-13) — `src/cli/__init__.py` (CLI Polish: health, preflight, lsp, serve), `src/pipeline/context.py` (Pipeline Event Hooks + SSE), `src/utils/logging/logging_utils.py` (Structured JSON Logging).
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

- [x] **`gnn preflight`** subcommand — Runs `run_preflight()` from `src/pipeline/preflight.py`. Outputs Markdown report.
- [x] **`gnn lsp`** subcommand — Launches `start_server()` from `src/lsp/__init__.py` on stdio.
- [x] **`gnn serve`** subcommand — Starts `start_server()` from `src/api/app.py` with `--host` and `--port`.
- [x] **`gnn health`** subcommand — Runs `check_renderers()` + `check_environment()` and prints summary.
- [x] **pyproject.toml** — Update entrypoint to `gnn = "src.cli:main"`.

### v2.1.0a Acceptance

- [x] `gnn preflight` produces Markdown report with 🟢/🔴 status
- [x] `gnn health` shows 8/8 renderers

---

## v2.1.0b — Pipeline Event Hooks

> **Scope**: Wire PipelineContext event callbacks for SSE integration.

- [x] **`PipelineContext.on_step_start`** callback — Optional callable invoked at step start.
- [x] **`PipelineContext.on_step_complete`** callback — Optional callable invoked at step end.
- [x] **`PipelineContext.on_error`** callback — Optional callable invoked on step failure.
- [x] **API integration** — Wire callbacks to SSE event broadcasting in `api/app.py`.

### v2.1.0b Acceptance

- [x] SSE stream emits `step_start` / `step_complete` events during run

---

## v2.1.0c — Structured Logging & JSON Log Output

> **Scope**: Machine-readable logs for pipeline observability.

- [x] **`src/pipeline/logging_config.py`** [NEW] — Configures structured JSON logging (stdlib `logging`). Fields: timestamp, level, step, message, duration.
- [x] **`--log-format json`** CLI flag — Switches to JSON line output for piping to log aggregators.
- [x] **Log rotation** — Configured via `logging.handlers.RotatingFileHandler`, 10 MB per file, 5 backups.

### v2.1.0c Acceptance

- [x] `gnn run --log-format json 2>&1 | python -m json.tool` parses each line

---

## v2.2.0a — Watcher Mode & Auto-Reparse

> **Scope**: File-watching for live re-validation during development.

- [x] **`src/gnn/watcher.py`** [NEW] — Uses `watchdog` (or `inotify` fallback) to monitor GNN files.
- [x] **`gnn watch <dir>`** subcommand — Monitors `input/gnn_files/` and re-runs validate on change.
- [x] **Debouncing** — 250ms debounce to avoid rapid-fire re-validation.
- [x] **Integration** — On change, runs `validate_required_sections()` + `parse_state_space()` and prints results.

### v2.2.0a Acceptance

- [x] Editing a `.md` file triggers re-validation within 500ms

---

## v2.2.0b — Model Dependency Graph Visualization

> **Scope**: Generate visual dependency graph from multi-model files.

- [x] **`src/gnn/dep_graph.py`** [NEW] — Builds networkx/mermaid graph from inter-model connections.
- [x] **`gnn graph <file.md>`** subcommand — Outputs Mermaid diagram to stdout or `.svg` file.
- [x] **Dashboard integration** — Embed dependency graph in `dashboard.html`.

### v2.2.0b Acceptance

- [x] `gnn graph multi_model.md` outputs valid Mermaid diagram

---

## v2.3.0 — Deep Roadmap (Unscheduled)

> Major computational scale-out and developer experience features. Scoped and ready for unblocking.

### v2.3.0a - Content-Addressable Model Registry (`gnn reproduce`)

- [x] Update `src/pipeline/hasher.py` to recursively capture `.gnn` file shasums into `index.json`.
- [x] Implement `src/cli/__init__.py::_cmd_reproduce` to read `.history/index.json`.
- [x] Wire the captured configuration parameters (including `testing_matrix` states) back into `PipelineArguments`.
- [x] Trigger execution bypassing normal CLI arg parsing.

### v2.3.0b - Distributed Parameter Sweeps (Ray/Dask integration)

- [x] Create `src/execute/distributed.py` module.
- [x] Migrate `12_execute.py` to optionally wrap standard grid search with `@ray.remote`.
- [x] Create robust retry semantics for node failure in external cloud instances.

### v2.3.0c - GPU-accelerated JAX

- [x] Create `src/render/jax/gpu_utils.py` to inspect available CUDA/TPU cores.
- [x] Automatically modify JAX code generation in `jax_renderer.py` to specify parallel execution contexts (`jax.pmap`, `jax.vmap`).
- [x] Adjust Dockerfiles/Setup phase to test for XLA compile compatibility.

### v2.3.0d - Visual Studio Code Extension

- [x] Bootstrapped extension scaffolding via `yo code`.
- [x] Integrate existing GNN TextMate syntax file (`package.json/contributes/grammars`).
- [x] Write LanguageClient wrapper `extension.ts` connecting via stdin/stdout to `gnn lsp`.

---

## Conventions

- Versions follow [SemVer](https://semver.org/) — `MAJOR.MINOR.PATCH`
- Sub-patches (`x.y.za`, `x.y.zb`) denote incremental shipments within a patch
- Patch releases completable in 1 focused session
- Minor releases completable in 1–3 focused sessions
- Deep roadmap items tracked for visibility but not scheduled
