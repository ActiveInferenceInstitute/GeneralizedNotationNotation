# GNN Pipeline - Master Agent Scaffolding

## Overview

The GNN (Generalized Notation Notation) Pipeline is a comprehensive 25-step system for processing Active Inference generative models. Each module follows the **thin orchestrator pattern** where numbered scripts delegate to modular implementations.

## 📚 GNN Documentation

The GNN system is fully documented in `doc/gnn/`.

> **[GNN Documentation Index](../doc/gnn/README.md)** - Start here for all GNN guides.

### Specialized Documentation Agents

See **[doc/gnn/AGENTS.md](../doc/gnn/AGENTS.md)** for the registry of all 25 documentation agents, including:

- **Syntax & DSL**: `gnn_syntax.md`, `gnn_dsl_manual.md`
- **Modeling**: `quickstart_tutorial.md`, `gnn_examples_doc.md`
- **Integration**: `framework_integration_guide.md`, `gnn_implementation.md`
- **Troubleshooting**: `gnn_troubleshooting.md`

---

## Module Registry

### Core Processing Modules (Steps 0-9)

- **Step 0**: **[template/](template/AGENTS.md)** - Pipeline template and initialization
- **Step 1**: **[setup/](setup/AGENTS.md)** - Environment setup and dependency management
- **Step 2**: **[tests/](tests/AGENTS.md)** - Comprehensive test suite execution
- **Step 3**: **[gnn/](gnn/AGENTS.md)** - GNN file discovery, parsing, and multi-format serialization
- **Step 4**: **[model_registry/](model_registry/AGENTS.md)** - Model versioning and registry management
- **Step 5**: **[type_checker/](type_checker/AGENTS.md)** - Type checking and validation
- **Step 6**: **[validation/](validation/AGENTS.md)** - Advanced validation and consistency checking
- **Step 7**: **[export/](export/AGENTS.md)** - Multi-format export generation
- **Step 8**: **[visualization/](visualization/AGENTS.md)** - Graph and matrix visualization
- **Step 9**: **[advanced_visualization/](advanced_visualization/AGENTS.md)** - Advanced visualization and interactive plots

### Simulation & Analysis Modules (Steps 10-16)

- **Step 10**: **[ontology/](ontology/AGENTS.md)** - Active Inference ontology processing
- **Step 11**: **[render/](render/AGENTS.md)** - Code generation for simulation frameworks
- **Step 12**: **[execute/](execute/AGENTS.md)** - Execute rendered simulation scripts
- **Step 13**: **[llm/](llm/AGENTS.md)** - LLM-enhanced analysis and interpretation
- **Step 14**: **[ml_integration/](ml_integration/AGENTS.md)** - Machine learning integration
- **Step 15**: **[audio/](audio/AGENTS.md)** - Audio generation and sonification
- **Step 16**: **[analysis/](analysis/AGENTS.md)** - Advanced statistical analysis

### Integration & Output Modules (Steps 17-24)

- **Step 17**: **[integration/](integration/AGENTS.md)** - System integration and coordination
- **Step 18**: **[security/](security/AGENTS.md)** - Security validation and access control
- **Step 19**: **[research/](research/AGENTS.md)** - Research tools and experimental features
- **Step 20**: **[website/](website/AGENTS.md)** - Static HTML website generation
- **Step 21**: **[mcp/](mcp/AGENTS.md)** - Model Context Protocol processing
- **Step 22**: **[gui/](gui/AGENTS.md)** - Interactive GUI for model construction (includes gui_1, gui_2, gui_3, oxdraw)
- **Step 23**: **[report/](report/AGENTS.md)** - Comprehensive analysis report generation
- **Step 24**: **[intelligent_analysis/](intelligent_analysis/AGENTS.md)** - AI-powered pipeline analysis and executive reports

### Step Index

- **📋 [STEP_INDEX.md](STEP_INDEX.md)** — Comprehensive 20-column reference table for all 25 steps
  - Covers: script, module, phase, input, output, frameworks, timeouts, dependencies, recovery behavior, data flow, matrix routing, criticality, and category

### Infrastructure Modules

- **[utils/](utils/AGENTS.md)** - Shared utilities and helper functions
- **[pipeline/](pipeline/AGENTS.md)** - Pipeline orchestration and configuration
- **[api/](api/AGENTS.md)** - REST API server (FastAPI)
- **[cli/](cli/AGENTS.md)** - CLI entry point
- **[lsp/](lsp/AGENTS.md)** - Language Server Protocol support
- **[sapf/](sapf/AGENTS.md)** - SAPF compatibility shim (re-exports from `audio/sapf/`)
- **[doc/](doc/AGENTS.md)** - In-repo technical documentation subtree (`src/doc/`)

---

## Architectural Pattern

### Thin Orchestrator Design

**Numbered Scripts** (`N_module.py`):

- Handle argument parsing
- Setup logging and output directories
- Call module processing functions
- Return standardized exit codes

**Module Implementation** (`src/module/`):

- Contains all domain logic
- Provides public API for orchestrators
- Implements error handling and fallbacks
- Exports functions via `__init__.py`

### Example Structure

```
src/
├── 11_render.py              # Thin orchestrator (< 150 lines)
├── render/                   # Module implementation
│   ├── __init__.py          # Public API exports
│   ├── AGENTS.md            # This documentation
│   ├── processor.py         # Core logic
│   ├── pymdp/               # Framework-specific code
│   ├── rxinfer/
│   └── mcp.py               # MCP tool registration
```

---

## Pipeline Execution Flow

This diagram shows nominal full-run order. Matrix-driven folder routing and dependency-based step inclusion are documented in `src/main.py` and `src/STEP_INDEX.md`.

```mermaid
flowchart TD
    Main[main.py Orchestrator] --> Step0[Step 0: Template]
    Step0 --> Step1[Step 1: Setup]
    Step1 --> Step2[Step 2: Tests]
    Step2 --> Step3[Step 3: GNN]
    Step3 --> Step4[Step 4: Registry]
    Step4 --> Step5[Step 5: Type Check]
    Step5 --> Step6[Step 6: Validation]
    Step6 --> Step7[Step 7: Export]
    Step7 --> Step8[Step 8: Visualization]
    Step8 --> Step9[Step 9: Advanced Viz]
    Step9 --> Step10[Step 10: Ontology]
    Step10 --> Step11[Step 11: Render]
    Step11 --> Step12[Step 12: Execute]
    Step12 --> Step13[Step 13: LLM]
    Step13 --> Step14[Step 14: ML Integration]
    Step14 --> Step15[Step 15: Audio]
    Step15 --> Step16[Step 16: Analysis]
    Step16 --> Step17[Step 17: Integration]
    Step17 --> Step18[Step 18: Security]
    Step18 --> Step19[Step 19: Research]
    Step19 --> Step20[Step 20: Website]
    Step20 --> Step21[Step 21: MCP]
    Step21 --> Step22[Step 22: GUI]
    Step22 --> Step23[Step 23: Report]
    Step23 --> Step24[Step 24: Intelligent Analysis]

    Step24 --> Output[output/ Directory]
    Output --> Summary[pipeline_execution_summary.json]
```

### Data Dependencies

```mermaid
graph TD
    Step3[Step 3: GNN Parse] -->|Parsed Models| Step5[Step 5: Type Check]
    Step3 -->|Parsed Models| Step6[Step 6: Validation]
    Step3 -->|Parsed Models| Step7[Step 7: Export]
    Step3 -->|Parsed Models| Step8[Step 8: Visualization]
    Step3 -->|Parsed Models| Step10[Step 10: Ontology]
    Step3 -->|Parsed Models| Step11[Step 11: Render]
    Step3 -->|Parsed Models| Step13[Step 13: LLM]
    
    Step11 -->|Generated Code| Step12[Step 12: Execute]
    Step12 -->|Execution Results| Step16[Step 16: Analysis]
    
    Step5 -->|Type Info| Step6
    Step6 -->|Validation Results| Step7
    Step7 -->|Exported Data| Step8
    Step8 -->|Visualizations| Step16
    Step13 -->|LLM Insights| Step16
    Step16 -->|Analysis Results| Step23[Step 23: Report]
```

---

## Performance Characteristics

### Status Notes

- The pipeline contains 25 ordered steps (0-24).
- Modules follow the thin orchestrator pattern.
- MCP integration and documentation coverage are tracked by repository audits.
- Use step outputs and tests as the ground-truth status indicators.

### Recent Validation (March 2026)

- **MCP Deadlock Resolved**: Fixed a multithreading deadlock in `discover_modules` that caused silent timeouts, restoring full pipeline summaries with 131 tools registered perfectly within 5 seconds.
- **LLM Glob Fixed**: Resolved recursive path issues during LLM processing logic.
- **ML Class Warning Fixed**: Updated cross-validation fold logic `min(5, len(X), min_class_count)` to dynamically avoid target class sparsity warnings.
- **Confirmed**: Full pipeline execution with 100% success rate and enhanced visual logging.
- **Performance**: All 25 steps complete rapidly with comprehensive progress tracking.
- **Tests (local `uv run pytest src/tests/ -q --tb=no --ignore=src/tests/test_llm_ollama.py --ignore=src/tests/test_llm_ollama_integration.py`)**: 1,922 passed, 29 skipped (2026-04-12). Re-include those modules when `ollama` is installed and responsive.
- **LLM Default Model**: `smollm2:135m-instruct-q4_K_S` via Ollama (`llm.defaults.DEFAULT_OLLAMA_MODEL`; configurable).
- **Visual Accessibility**: All pipeline steps now include enhanced visual indicators and progress tracking.

---

## 25-Step Pipeline Structure (CURRENT)

The pipeline consists of exactly 25 steps (steps 0-24), executed in order:

0. **0_template.py** → `src/template/` - Pipeline template and initialization
1. **1_setup.py** → `src/setup/` - Environment setup, virtual environment management, dependency installation
2. **2_tests.py** → `src/tests/` - Comprehensive test suite execution
3. **3_gnn.py** → `src/gnn/` - GNN file discovery, multi-format parsing, and validation
4. **4_model_registry.py** → `src/model_registry/` - Model registry management and versioning
5. **5_type_checker.py** → `src/type_checker/` - GNN syntax validation and resource estimation
6. **6_validation.py** → `src/validation/` - Advanced validation and consistency checking
7. **7_export.py** → `src/export/` - Multi-format export (JSON, XML, GraphML, GEXF, Pickle)
8. **8_visualization.py** → `src/visualization/` - Graph and matrix visualization generation
9. **9_advanced_viz.py** → `src/advanced_visualization/` - Advanced visualization and interactive plots
10. **10_ontology.py** → `src/ontology/` - Active Inference Ontology processing and validation
11. **11_render.py** → `src/render/` - Code generation for PyMDP, RxInfer, ActiveInference.jl, JAX, Stan, PyTorch, NumPyro, DisCoPy simulation environments
12. **12_execute.py** → `src/execute/` - Execute rendered simulation scripts with result capture
13. **13_llm.py** → `src/llm/` - LLM-enhanced analysis, model interpretation, and AI assistance
14. **14_ml_integration.py** → `src/ml_integration/` - Machine learning integration and model training
15. **15_audio.py** → `src/audio/` - Audio generation (SAPF, Pedalboard, and other backends)
16. **16_analysis.py** → `src/analysis/` - Advanced analysis and statistical processing
17. **17_integration.py** → `src/integration/` - System integration and cross-module coordination
18. **18_security.py** → `src/security/` - Security validation and access control
19. **19_research.py** → `src/research/` - Research tools and experimental features
20. **20_website.py** → `src/website/` - Static HTML website generation from pipeline artifacts
21. **21_mcp.py** → `src/mcp/` - Model Context Protocol processing and tool registration
22. **22_gui.py** → `src/gui/` - Interactive GUI for constructing/editing GNN models
23. **23_report.py** → `src/report/` - Comprehensive analysis report generation
24. **24_intelligent_analysis.py** → `src/intelligent_analysis/` - AI-powered pipeline analysis and executive reports

---

## Module Status Matrix

Module-level readiness and coverage details change over time; use each module's `AGENTS.md`, `README.md`, and tests in `src/tests/` as the authoritative source.

- **[SPEC.md](SPEC.md)** — Architectural requirements and standards
- **[STEP_INDEX.md](STEP_INDEX.md)** — Complete 20-column master reference for all 25 steps
- **[README.md](../README.md)** — Project overview and documentation
- **[main.py](main.py)** — Pipeline orchestrator
- **[input/config.yaml](../input/config.yaml)** — Testing matrix configuration

---

## Quick Start

### Run Full Pipeline

```bash
python src/main.py --target-dir input/gnn_files --verbose
```

### Run Specific Steps

```bash
python src/main.py --only-steps "3,5,7,8,11,12" --verbose
```

### Run Individual Step

```bash
python src/3_gnn.py --target-dir input/gnn_files --output-dir output --verbose
```

### Framework Selection

```bash
# Execute only specific frameworks
python src/12_execute.py --frameworks "pymdp,jax" --verbose

# Use lite preset (PyMDP, JAX, DisCoPy)
python src/12_execute.py --frameworks "lite" --verbose

# All frameworks (default)
python src/12_execute.py --frameworks "all" --verbose
```

### Optional Dependencies

```bash
# Install optional groups
python src/1_setup.py --install_optional --optional_groups "pymdp,jax,viz,gui,audio,llm"

# Install specific groups
python src/1_setup.py --install_optional --optional_groups "viz,pymdp"
```

---

## Development Guidelines

### Adding New Modules

1. Create module directory: `src/new_module/`
2. Implement `__init__.py` with public API
3. Create `AGENTS.md` documentation
4. Add numbered script: `N_new_module.py`
5. Implement tests in `src/tests/`
6. Add MCP tools in `mcp.py` (if applicable)

### Code Standards

- Follow thin orchestrator pattern
- Use type hints for all public functions
- Document all classes and methods
- Maintain >80% test coverage
- Include error handling and fallbacks

---

## Testing

### Run All Tests

```bash
python src/2_tests.py --comprehensive
```

### Run Module-Specific Tests

```bash
pytest src/tests/test_[module]*.py -v
```

### Check Coverage

```bash
pytest --cov=src --cov-report=term-missing
```

---

## References

- **Main Documentation**: [README.md](../README.md)
- **GNN Documentation Index**: [doc/gnn/README.md](../doc/gnn/README.md)

---

**Last Updated**: 2026-04-12
**Pipeline Version**: 1.3.0
**Total Steps**: 25 (0-24)
**Status**: Maintained
