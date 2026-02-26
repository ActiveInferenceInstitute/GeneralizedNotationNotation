---
name: gnn-pipeline
description: Generalized Notation Notation (GNN) processing pipeline for Active Inference generative models. Use when working with GNN files, running the 25-step pipeline, parsing model specifications, generating simulations, or producing visualizations and reports from GNN model definitions.
---

# GNN Pipeline Skill

GNN (Generalized Notation Notation) is a text-based specification language for Active Inference generative models. This repository implements a **25-step processing pipeline** (steps 0–24) that transforms GNN specifications into executable simulations, visualizations, analysis reports, and more.

## When to Use This Skill

- Parsing or authoring `.md` GNN model files
- Running the full pipeline or individual steps
- Generating simulation code (PyMDP, RxInfer.jl, JAX, DisCoPy, ActiveInference.jl)
- Creating visualizations, exports, or reports from GNN models
- Working with Active Inference ontology annotations

## Quick Start

```bash
# Run full pipeline
python src/main.py --target-dir input/gnn_files --verbose

# Run specific steps only
python src/main.py --only-steps "3,5,11,12" --verbose

# Run a single step directly
python src/3_gnn.py --target-dir input/gnn_files --output-dir output --verbose

# Run tests
pytest src/tests/ -v

# Setup environment
uv sync && uv run python src/main.py --target-dir input/gnn_files --verbose
```

## Architecture: Thin Orchestrator Pattern

Every pipeline step follows the same pattern:

```text
src/N_module.py          → Thin orchestrator (<150 lines): CLI args, logging, delegation
src/module/              → Module directory: all domain logic
  ├── __init__.py        → Public API exports
  ├── processor.py       → Core processing logic
  ├── mcp.py             → MCP tool registration (if applicable)
  ├── AGENTS.md          → Module documentation
  ├── README.md          → Usage guide
  ├── SPEC.md            → Module specification
  └── SKILL.md           → This skill format (Claude Code activation)
```

## 25-Step Pipeline

| Phase | Steps | Purpose |
| ----- | ----- | ------- |
| **Core** (0–9) | Template, Setup, Tests, GNN Parse, Registry, Type Check, Validation, Export, Viz, Advanced Viz | Parse GNN files, validate, export, visualize |
| **Simulation** (10–16) | Ontology, Render, Execute, LLM, ML, Audio, Analysis | Generate and run simulations, analyze results |
| **Output** (17–24) | Integration, Security, Research, Website, MCP, GUI, Report, Intelligent Analysis | Produce deliverables and reports |

## GNN File Format

GNN files are Markdown documents with structured sections:

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

## Framework Selection

```bash
# Execute specific frameworks
python src/12_execute.py --frameworks "pymdp,jax" --verbose

# Lite preset (PyMDP, JAX, DisCoPy)
python src/12_execute.py --frameworks "lite" --verbose

# All frameworks (default)
python src/12_execute.py --frameworks "all" --verbose
```

## Module Skills

Each `src/module/` directory contains its own `SKILL.md` with module-specific instructions. See `src/AGENTS.md` for the complete module registry.

## Testing

```bash
# Full test suite (1,522+ tests)
pytest src/tests/ -v

# Test a specific module
pytest src/tests/test_gnn.py -v

# With coverage
pytest src/tests/ --cov=src -v
```

## References

- [CLAUDE.md](CLAUDE.md) — Claude Code project guidance
- [AGENTS.md](AGENTS.md) — Master agent scaffolding
- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture
- [SECURITY.md](SECURITY.md) — Security policy and remediation
- [SETUP_GUIDE.md](SETUP_GUIDE.md) — Environment setup guide
- [src/SPEC.md](src/SPEC.md) — Source specification
- [doc/gnn/README.md](doc/gnn/README.md) — GNN documentation index
- [doc/gnn/gnn_syntax.md](doc/gnn/gnn_syntax.md) — GNN syntax reference
- [doc/gnn/gnn_examples_doc.md](doc/gnn/gnn_examples_doc.md) — Example GNN models
