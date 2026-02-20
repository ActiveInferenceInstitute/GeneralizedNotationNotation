# GNN Source Specification

**Version**: 1.1.4  
**Last Updated**: 2026-02-20  
**Status**: ✅ Production Ready

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
│  ├── processor.py (Core logic)          │
│  ├── mcp.py       (MCP tools)           │
│  ├── AGENTS.md    (Documentation)       │
│  └── README.md    (Usage guide)         │
└─────────────────────────────────────────┘
```

**Requirements:**

- Orchestrator scripts: <150 lines
- No domain logic in orchestrators
- All processing in module directories
- Standardized exit codes (0=success, 1=error, 2=warnings)

### Module Structure

Every module directory MUST contain:

| File | Purpose | Required |
|------|---------|----------|
| `__init__.py` | Public API exports | ✅ |
| `processor.py` | Core processing logic | ✅ |
| `AGENTS.md` | Module documentation | ✅ |
| `README.md` | Usage documentation | ✅ |
| `mcp.py` | MCP tool registration | ⚠️ If applicable |
| `SPEC.md` | Technical specification | ⚠️ Optional |

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

| Module | Purpose | Files |
|--------|---------|-------|
| `utils/` | Shared utilities | 42 |
| `pipeline/` | Orchestration config | 20 |
| `tests/` | Test suite | 89 |
| `sapf/` | Audio framework | 1 |
| `output/` | Generated artifacts | — |

---

## Technical Requirements

### Python Version

- **Minimum**: Python 3.11+
- **Recommended**: Python 3.12.x or 3.13.x

### Core Dependencies

```
numpy>=1.24.0
networkx>=3.0
pyyaml>=6.0
jsonschema>=4.0
```

### Optional Dependencies

| Group | Packages | Purpose |
|-------|----------|---------|
| `pymdp` | pymdp, jax | Active Inference simulation |
| `viz` | matplotlib, plotly | Visualization |
| `llm` | openai, anthropic, ollama | LLM integration |
| `audio` | soundfile, pedalboard | Audio generation |
| `gui` | tkinter, customtkinter | GUI interface |

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
- No mock implementations in production code

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Full pipeline execution | <120s | ~90s ✅ |
| Peak memory usage | <100MB | 36MB ✅ |
| Test pass rate | 100% | 100% ✅ |
| Syntax validity | 100% | 100% ✅ |

---

## References

- **[AGENTS.md](AGENTS.md)**: Master module registry
- **[README.md](README.md)**: Pipeline safety documentation  
- **[main.py](main.py)**: Pipeline orchestrator
- **[../doc/gnn/README.md](../doc/gnn/README.md)**: GNN documentation index
- **[../ARCHITECTURE.md](../ARCHITECTURE.md)**: System architecture
