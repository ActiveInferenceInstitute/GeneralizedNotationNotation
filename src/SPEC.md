# GNN Source Specification

**Version**: 3.2.0 (Specification) — Release version: `pyproject.toml` `version = "3.2.0"`. The in-package strings `src/__init__.py::__version__` and `src/mcp/__init__.py::__version__` still read `1.6.0` and have not been bumped with the release; treat `pyproject.toml` as authoritative until they are.  
**Last Updated**: 2026-09-02  
**Status**: Maintained

---

## Overview

The `src/` directory contains the complete implementation of the GNN (Generalized Notation Notation) processing pipeline. This specification defines the architectural requirements, module standards, and development guidelines for all source code.

## Architecture Requirements

### Thin Orchestrator Pattern

All numbered pipeline scripts (0-24) MUST follow the thin orchestrator pattern:

```
┌─────────────────────────────────────────┐
│  N_module.py (Thin Orchestrator)        │
│  - Argument parsing                      │
│  - Logging setup                         │
│  - Output directory management           │
│  - Delegates ALL logic to module/        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  src/module/                            │
│  ├── __init__.py  (Public API)          │
│  ├── processor.py (Core logic, or accepted alternative) │
│  ├── mcp.py       (MCP tools)           │
│  ├── AGENTS.md    (Documentation)       │
│  └── README.md    (Usage guide)         │
└─────────────────────────────────────────┘
```

**Requirements:**

- Orchestrator scripts: <150 lines
- No domain logic in orchestrators
- All processing in module directories
- Standardized exit codes (0=success, 1=error, 2=success with warnings/skipped)

### Module Structure

Every module directory MUST contain:

| File | Purpose | Required |
|------|---------|----------|
| `__init__.py` | Public API exports | ✅ |
| `processor.py` (or accepted alternative) | Core processing logic | ✅ |
| `AGENTS.md` | Module documentation | ✅ |
| `README.md` | Usage documentation | ✅ |
| `mcp.py` | MCP tool registration | ⚠️ If applicable |
| `SPEC.md` | Technical specification | ⚠️ Optional |

Accepted alternatives for core processing file organization:

- `setup/`, `tests/`, `validation/`: processing logic in `__init__.py`
- `model_registry/`: processing logic in `registry.py`
- `website/`: processing logic split across `renderer.py` and `generator.py` (`processor.py` remains a thin facade)

### Exit Code Standards

| Code | Meaning | Action |
|------|---------|--------|
| `0` | Success | Continue pipeline |
| `1` | Critical error | Log and continue (configurable) |
| `2` | Success with warnings | Continue pipeline |

---

## Pipeline Structure

### 25-Step Pipeline (0-24)

**Core Processing (Steps 0-9)**

| Step | Script | Module | Purpose |
|------|--------|--------|---------|
| 0 | `0_template.py` | `template/` | Pipeline initialization |
| 1 | `1_setup.py` | `setup/` | Environment setup |
| 2 | `2_tests.py` | `tests/` | Test suite execution |
| 3 | `3_gnn.py` | `gnn/` | GNN parsing |
| 4 | `4_model_registry.py` | `model_registry/` | Model versioning |
| 5 | `5_type_checker.py` | `type_checker/` | Type validation |
| 6 | `6_validation.py` | `validation/` | Consistency checking |
| 7 | `7_export.py` | `export/` | Multi-format export |
| 8 | `8_visualization.py` | `visualization/` | Graph visualization |
| 9 | `9_advanced_viz.py` | `advanced_visualization/` | Interactive plots |

**Simulation & Analysis (Steps 10-16)**

| Step | Script | Module | Purpose |
|------|--------|--------|---------|
| 10 | `10_ontology.py` | `ontology/` | Ontology processing |
| 11 | `11_render.py` | `render/` | Code generation |
| 12 | `12_execute.py` | `execute/` | Simulation execution |
| 13 | `13_llm.py` | `llm/` | LLM analysis |
| 14 | `14_ml_integration.py` | `ml_integration/` | ML integration |
| 15 | `15_audio.py` | `audio/` | Audio generation |
| 16 | `16_analysis.py` | `analysis/` | Statistical analysis |

**Integration & Output (Steps 17-24)**

| Step | Script | Module | Purpose |
|------|--------|--------|---------|
| 17 | `17_integration.py` | `integration/` | System integration |
| 18 | `18_security.py` | `security/` | Security validation |
| 19 | `19_research.py` | `research/` | Research tools |
| 20 | `20_website.py` | `website/` | Website generation |
| 21 | `21_mcp.py` | `mcp/` | MCP processing |
| 22 | `22_gui.py` | `gui/` | GUI interface |
| 23 | `23_report.py` | `report/` | Report generation |
| 24 | `24_intelligent_analysis.py` | `intelligent_analysis/` | AI-powered analysis |

### Infrastructure Modules

| Module | Purpose |
|--------|---------|
| `utils/` | Shared utilities |
| `pipeline/` | Orchestration config |
| `api/` | REST API server (FastAPI) |
| `cli/` | CLI entry point |
| `lsp/` | Language Server Protocol support |
| `doc/` | In-repo technical documentation subtree |
| `tests/` | Test suite |
| `sapf/` | SAPF public entry point (`audio/sapf/`) |

Pipeline artifacts are written to the repository-level `output/` directory by default (`io.output_dir` in `input/config.yaml`). That tree is ignored except for its marker file, so generated artifacts should be regenerated rather than hand-edited. The `src/output/` directory is not a Python package; see [`doc/pipeline/README.md`](../doc/pipeline/README.md) for generated-output coverage exclusions.

---

## Testing Matrix Configuration

The pipeline supports **staged, folder-based execution** via a testing matrix defined in `input/config.yaml`. This allows different categories of GNN files to be processed by different subsets of pipeline steps.

> **📋 For the complete 20-column step reference, see [`STEP_INDEX.md`](STEP_INDEX.md).**

### Configuration

The shipped `input/config.yaml` currently reads (abridged):

```yaml
# input/config.yaml (shipped values)
testing_matrix:
  enabled: true

  # Global steps — toggled independently (not folder-routed)
  global_steps:
    0_template: true       # Pipeline template & initialization
    1_setup: true          # Environment setup & dependency install
    2_tests: true          # Test suite execution

  # Default steps for folders not listed under `folders`
  default_steps: [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    # 13 omitted: LLM runs once globally; disable via pipeline.skip_steps or --skip-llm
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
  ]

  # Per-folder overrides — empty, so every folder runs default_steps
  folders: {}
```

A hypothetical per-folder override (not the shipped configuration) looks like:

```yaml
folders:
  discrete: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
  basics: [3, 5, 6, 10]
```

### Behavior

- **Global steps** (0, 1, 2): Run once before folder-specific steps. Each can be toggled to `true`/`false` independently. Disabled steps are `SKIPPED`.
- **Processing steps** (3–24): When `enabled: true`, `main.py` iterates over subdirectories in `input/gnn_files/` and runs each step only on folders whose config includes that step number.
- Folders not explicitly listed use `default_steps`.
- A step that no folder lists (Step 13 in the shipped config) is not skipped: `main.py` falls back to a single invocation over the whole target directory.
- Continuous linear-Gaussian models (`input/gnn_files/continuous/`) render and execute only on frameworks with `supports_continuous` in `render/framework_registry.py` (JAX, NumPyro, PyTorch, Stan, RxInfer.jl); the others report status `unsupported`, which Step 12 skips.
- Results are aggregated across all folder executions per step.
- **Invalid step selections fail fast**: non-numeric tokens in
  `--only-steps` / `--skip-steps` (or `pipeline.only_steps` /
  `pipeline.skip_steps`) raise `ValueError` at startup, and an
  `only_steps` request that resolves to no executable step exits 1 with a
  startup error. Out-of-range step numbers are logged and dropped
  (`main.py::select_pipeline_steps`).

### Orchestrator Implementation

The matrix logic lives in `execute_pipeline_step()` in `main.py`. It loads the matrix from `input/config.yaml` via PyYAML, checks `global_steps` for steps 0–2, and dynamically sets `--target-dir` per subfolder for steps 3–24.

---

## Technical Requirements

### Python Version

- **Minimum**: Python 3.11+
- **Recommended**: Python 3.12.x or 3.13.x

### Core Dependencies

Pins are owned by `pyproject.toml` `[project] dependencies`; the current floors there include:

```
numpy>=1.21.0
networkx>=2.6.0
pyyaml>=6.0
```

(`jsonschema` is not a declared dependency — it appears only as a mypy override.)

### Optional Dependencies

Extras are declared in `pyproject.toml` `[project.optional-dependencies]`. pymdp, JAX, NumPyro, DisCoPy, and the LLM clients (openai, ollama) are **core** dependencies installed by plain `uv sync`.

| Group | Purpose |
|-------|---------|
| `dev` | Development tooling (pytest, ruff, mypy, docs, notebooks); also pulls in `cmdstanpy` |
| `api` | FastAPI/uvicorn server |
| `ml-ai` | Machine-learning extensions (transformers, scipy, scikit-learn) |
| `audio` | Audio processing (librosa, soundfile, pedalboard, pydub) |
| `gui` | GUI frameworks (gradio, streamlit) |
| `graphs` | Graphviz bindings |
| `stan` | `cmdstanpy` for the Stan backend (CmdStan toolchain installed separately); also folded into `dev` and `all` |
| `research` | Research tools (jupyterlab, sympy, numba, cython) |
| `scaling` | Scaling (dask, distributed, ray) |
| `all` | Every functionally distinct optional group combined |

---

## Framework Integration

### Supported Frameworks

| Framework | Language | Location | Purpose |
|-----------|----------|----------|---------|
| PyMDP | Python | `render/pymdp/`, `execute/pymdp/` | POMDP simulation |
| RxInfer.jl | Julia | `render/rxinfer/`, `execute/rxinfer/` | Bayesian inference |
| ActiveInference.jl | Julia | `render/activeinference_jl/`, `execute/activeinference_jl/` | Active Inference |
| DisCoPy | Python | `render/discopy/`, `execute/discopy/` | Category theory |
| JAX | Python | `render/jax/`, `execute/jax/` | GPU acceleration |
| PyTorch | Python | `render/pytorch/`, `execute/pytorch/` | Deep learning inference |
| NumPyro | Python | `render/numpyro/`, `execute/numpyro/` | Probabilistic programming |
| Stan | Stan | `render/stan/`, `execute/stan/` | HMM / LGSSM Stan programs plus a cmdstanpy driver (`<stem>_stan.py`) |
| bnlearn | Python | `render/generators.py` (`generate_bnlearn_code`) | Bayesian-network structure/parameter learning — **render-only**; no Step 12 executor |

---

## MCP Integration

All modules with external interfaces MUST provide MCP (Model Context Protocol) tools:

```python
# src/module/mcp.py
@server.tool()
def module_operation(input: str, output_path: str) -> dict:
    """Tool description for MCP clients."""
    return process_module_main(input, output_path)
```

---

## Quality Standards

### Code Requirements

- Type hints for all public functions
- Docstrings for all classes and methods
- >80% test coverage per module
- No syntax errors (validated on commit)

### Documentation Requirements

- AGENTS.md: Module capabilities and API reference
- README.md: Usage examples and quick start
- Inline comments for complex logic

### Testing Requirements

- Unit tests in `src/tests/test_{module}_*.py`
- Integration tests for cross-module flows
- No simulated implementations in production code

---

## Performance Targets

Performance and reliability targets should be validated by current benchmark/test runs in CI or local execution, rather than fixed values in static docs.

---

## Versioning

> **Dual Versioning Policy**: This repository uses two version numbers:
> - **Pipeline version** (src/): Corresponds to `src/__init__.py::__version__` (currently the string `"1.6.0"`, behind the `3.2.0` release in `pyproject.toml`)
> - **MCP version** (mcp/): Independent MCP subsystem versioning via `src/mcp/__init__.py::__version__` (currently `"1.6.0"`)
>
> MCP (Model Context Protocol) has its own version because it represents an extended protocol implementation that evolved beyond the main pipeline versioning.

---

## References

- **[AGENTS.md](AGENTS.md)**: Master module registry
- **[README.md](README.md)**: Pipeline safety documentation  
- **[main.py](main.py)**: Pipeline orchestrator
- **[../doc/gnn/README.md](../doc/gnn/README.md)**: GNN documentation index
- **[../ARCHITECTURE.md](../ARCHITECTURE.md)**: System architecture
