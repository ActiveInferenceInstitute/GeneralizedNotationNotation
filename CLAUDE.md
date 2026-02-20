# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GNN (Generalized Notation Notation) is a text-based language for standardizing Active Inference generative models. The codebase implements a 25-step processing pipeline (steps 0-24) that transforms GNN specifications into executable simulations, visualizations, and analysis reports.

## Essential Commands

```bash
# Run full pipeline
python src/main.py --target-dir input/gnn_files --verbose

# Run specific steps only (steps are 0-24)
python src/main.py --only-steps "3,5,11,12" --verbose

# Skip specific steps
python src/main.py --skip-steps "15,16" --verbose

# Run individual step directly
python src/3_gnn.py --target-dir input/gnn_files --output-dir output --verbose

# Type checking only (fast validation)
python src/main.py --only-steps 5 --strict

# Setup environment with dev dependencies
python src/main.py --only-steps 1 --dev

# Run tests
pytest src/tests/ -v
pytest src/tests/test_gnn_*.py -v  # Module-specific tests
python src/2_tests.py --comprehensive  # Full test suite via pipeline

# Check coverage
pytest --cov=src --cov-report=term-missing

# Using uv (recommended)
uv sync
uv run python src/main.py --target-dir input/gnn_files --verbose
uv run pytest
```

## Architecture

### Thin Orchestrator Pattern

All 25 pipeline steps follow a consistent pattern:

- **Numbered scripts** (`src/N_module.py`): Thin orchestrators (<150 lines) that handle CLI args, logging, and delegate to modules
- **Module directories** (`src/module/`): Contain all domain logic
  - `__init__.py`: Public API
  - `processor.py`: Core logic
  - `mcp.py`: MCP tool registration (if applicable)

### 25-Step Pipeline (0-24)

| Steps 0-9 (Core) | Steps 10-16 (Simulation) | Steps 17-24 (Output) |
|------------------|--------------------------|----------------------|
| 0: Template init | 10: Ontology | 17: Integration |
| 1: Setup | 11: Render (code gen) | 18: Security |
| 2: Tests | 12: Execute | 19: Research |
| 3: GNN parsing | 13: LLM analysis | 20: Website |
| 4: Model registry | 14: ML integration | 21: MCP |
| 5: Type checker | 15: Audio | 22: GUI |
| 6: Validation | 16: Analysis | 23: Report |
| 7: Export | | 24: Intelligent Analysis |
| 8: Visualization | | |
| 9: Advanced viz | | |

### Data Flow

Step 3 (GNN Parse) produces parsed models consumed by steps 5-8, 10-11, 13. Step 11 (Render) generates code executed by Step 12, whose results feed Step 16 (Analysis) and Step 23 (Report).

### Framework Integration

Code generation and execution support multiple backends:

- **PyMDP** (Python): `render/pymdp/`, `execute/pymdp/`
- **RxInfer.jl** (Julia): `render/rxinfer/`, `execute/rxinfer/`
- **ActiveInference.jl** (Julia): `render/activeinference_jl/`, `execute/activeinference_jl/`
- **JAX** (Python): `render/jax/`, `execute/jax/`
- **DisCoPy** (Python): `render/discopy/`, `execute/discopy/`

```bash
# Execute specific frameworks
python src/12_execute.py --frameworks "pymdp,jax" --verbose
```

## Key Locations

| Path | Purpose |
|------|---------|
| `src/main.py` | Main pipeline orchestrator |
| `src/N_*.py` | Pipeline step scripts (0-24) |
| `src/gnn/` | GNN parsing, discovery, validation |
| `src/render/` | Code generation for all frameworks |
| `src/execute/` | Simulation execution |
| `src/tests/` | Test suite (~1,083 tests across ~90 files) |
| `input/gnn_files/` | Sample GNN model files |
| `output/` | Generated outputs (25 step-specific folders) |
| `doc/gnn/gnn_syntax.md` | Complete GNN syntax specification |

## GNN File Format

GNN files are Markdown with structured sections:

```markdown
## GNNSection
ActInfPOMDP

## ModelName
My Model

## StateSpaceBlock
A[3,3,type=float]   # Likelihood matrix
B[3,3,3,type=float] # Transition matrix
s[3,1,type=float]   # Hidden state

## Connections
D>s    # D feeds into s (directed)
s-A    # s connects to A (undirected)

## InitialParameterization
A={(0.9,0.05,0.05), (0.05,0.9,0.05), (0.05,0.05,0.9)}

## ActInfOntologyAnnotation
A=LikelihoodMatrix
s=HiddenState
```

## Code Standards

- Python 3.11+ required
- Type hints for all public functions
- Exit codes: 0=success, 1=error, 2=warnings
- All modules need `__init__.py`, `processor.py`, and `AGENTS.md`
- Tests in `src/tests/test_{module}_*.py`

## Optional Dependency Groups

```bash
# Install specific optional groups using UV
uv sync --extra dev             # Development tools
uv sync --extra llm             # LLM integration (openai, anthropic, ollama)
uv sync --extra visualization   # Enhanced visualization
uv sync --extra audio           # Audio processing
uv sync --extra gui             # GUI interfaces
uv sync --all-extras            # Everything
```
