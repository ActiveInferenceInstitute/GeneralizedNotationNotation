# .agent_rules — GNN Pipeline Development Guidelines

**Version**: 3.0.0 | **Status**: ✅ Production Ready | **Updated**: March 2026

> **Quick Start**: Run `uv run python src/main.py --verbose` from the project root. All tooling uses `uv`.

---

## Navigation by Task

| Task | File |
|------|------|
| Writing a pipeline script | [architecture.md](architecture.md) |
| Creating a new module | [module_patterns.md](module_patterns.md) |
| Writing tests | [testing.md](testing.md) |
| Adding MCP tools | [mcp.md](mcp.md) |
| Understanding GNN specs | [gnn_standards.md](gnn_standards.md) |
| Error handling / safe-to-fail | [error_handling.md](error_handling.md) |
| Optional dependencies | [dependencies.md](dependencies.md) |
| Render frameworks (PyMDP/JAX/etc.) | [render_frameworks.md](render_frameworks.md) |
| Debugging issues | [troubleshooting.md](troubleshooting.md) |
| Performance tuning | [performance.md](performance.md) |
| Code quality standards | [quality.md](quality.md) |

---

## Critical Standards (Must Know)

### 1. Thin Orchestrator Pattern ⚠️
Numbered scripts (`N_module.py`) are **thin orchestrators only**:
- All domain logic lives in `src/module_name/`
- Scripts only: argument parsing, logging setup, output directory, exit codes
- See: [architecture.md](architecture.md)

### 2. No Mocks Policy ⚠️
All tests use **real implementations** — no `unittest.mock` or monkeypatching.
- Skip gracefully when optional deps unavailable: `pytest.skip(...)`
- See: [testing.md](testing.md)

### 3. Safe-to-Fail Steps ⚠️
Steps 8, 9, 12 (Visualization/Execute) **never stop the pipeline**:
- Always return exit code `0`
- Multiple fallback levels with HTML reports on failure
- See: [error_handling.md](error_handling.md)

### 4. Exit Codes
| Code | Meaning | Pipeline Behavior |
|------|---------|-------------------|
| `0` | Success | Continue |
| `1` | Critical Error | Stop pipeline |
| `2` | Success with Warnings | Continue |

### 5. Environment
Always use `uv` for all operations:
```bash
uv run python src/main.py --verbose    # Run pipeline
uv run pytest src/tests/ -v           # Run tests
uv pip install -e .                   # Install deps
```

---

## Pipeline Overview — 25 Steps (0–24)

| Steps | Area |
|-------|------|
| 0–2 | Template, Setup, Tests |
| 3–9 | GNN Parse, Registry, Type Check, Validation, Export, Visualization |
| 10–16 | Ontology, Render, Execute, LLM, ML, Audio, Analysis |
| 17–24 | Integration, Security, Research, Website, MCP, GUI, Report, Intelligent Analysis |

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `README.md` | This file — navigation and critical standards |
| `architecture.md` | Pipeline architecture, thin orchestrator, orchestration patterns |
| `module_patterns.md` | Module directory structure, `__init__.py`, MCP patterns |
| `gnn_standards.md` | GNN file format, validation levels, multi-format support |
| `testing.md` | Test framework, import patterns, fixtures, markers |
| `quality.md` | Code quality, type hints, documentation, linting standards |
| `error_handling.md` | Exit codes, safe-to-fail, graceful degradation, recovery |
| `performance.md` | Benchmarks, memory optimization, caching, profiling |
| `mcp.md` | MCP tool registration, tool structure, error handling |
| `dependencies.md` | Core vs optional deps, detection patterns, fallbacks |
| `render_frameworks.md` | PyMDP, JAX (no Flax), RxInfer.jl, DisCoPy patterns |
| `troubleshooting.md` | Common issues, diagnostics, recovery procedures |

---

**Pipeline Version**: 1.3.0 | **Steps**: 25 | **Tests**: 1,522+ passing | **MCP Tools**: 131
