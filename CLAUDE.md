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

# Step 1 default: ``uv sync`` core dependencies (includes JAX / NumPyro / PyTorch / DisCoPy for step 12).
# ``--setup-core-only`` skips the JAX self-test during setup only.

# Run tests
pytest src/tests/ -v
pytest src/tests/test_gnn_*.py -v  # Module-specific tests
python src/2_tests.py --comprehensive  # Full test suite via pipeline

# Check coverage
pytest --cov=src --cov-report=term-missing

# Using uv (recommended)
uv sync
uv run python src/main.py --target-dir input/gnn_files --verbose
# Install dev deps (pytest) into .venv so ``uv run pytest`` uses the same interpreter as JAX/pymdp
uv sync --extra dev
uv run pytest
# If JAX/pymdp tests skip or ``ModuleNotFoundError: jax`` during tests, another venv may own ``pytest``:
# unset VIRTUAL_ENV or run ``PYTHONPATH=src .venv/bin/pytest src/tests/ …``
# Package integrity probe: ``src/utils/jax_stack_validation.py`` (also Step 1 + CI).
```

## Architecture

### Thin Orchestrator Pattern

All 25 pipeline steps follow a consistent pattern:

- **Numbered scripts** (`src/N_module.py`): Thin orchestrators (<150 lines) that handle CLI args, logging, and delegate to modules
- **Module directories** (`src/module/`): Contain all domain logic
  - `__init__.py`: Public API
  - `processor.py`: Core logic (preferred; see accepted alternatives below)
  - `mcp.py`: MCP tool registration (if applicable)

**Accepted `processor.py` alternatives** (5 modules use these patterns):
- `setup/`, `tests/`, `validation/`: Logic lives directly in `__init__.py` — functionally equivalent
- `model_registry/`: Uses `registry.py` as primary logic file — clearly named
- `website/`: Uses `renderer.py` + `generator.py`; `processor.py` is a thin shim that re-exports from renderer

**Hard imports** (steps 20, 21, 24): The website, mcp, and intelligent_analysis step scripts use direct (non-try/except) imports because these modules are pipeline-required and must always be present. All three document this with an inline `# Hard import: X is a core module` comment. All other steps use soft imports with graceful degradation fallbacks.

**`src/sapf/` module**: This is a compatibility shim, not a numbered pipeline step. It re-exports functions from `audio.sapf` so that `import sapf` works without duplicating code. The actual SAPF implementation lives in `src/audio/sapf/`.

**Research module (Step 19)**: Uses rule-based static analysis (no external LLM required). The `FEATURES = {'fallback_mode': True}` flag in `__init__.py` indicates it operates without LLM dependencies, not that it is incomplete.

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
- **PyTorch** (Python): `render/pytorch/`, `execute/pytorch/`
- **NumPyro** (Python): `render/numpyro/`, `execute/numpyro/`
- **Stan** (Stan): `render/stan/`

```bash
# Execute specific frameworks
python src/12_execute.py --frameworks "pymdp,jax" --verbose
```

### Running all execution frameworks

Step 12 (Execute) runs scripts for every framework (PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro). JAX, NumPyro, PyTorch, and DisCoPy are **core** Python dependencies: a normal ``uv sync`` installs them. If the environment is incomplete, those backends are **skipped** at step 12 (not failed) with a dependency reason. The ``execution-frameworks`` extra duplicates the same pins for explicit ``uv sync --extra execution-frameworks``. Julia backends still require a local Julia install.

## Key Locations

| Path | Purpose |
|------|---------|
| `src/main.py` | Main pipeline orchestrator |
| `src/N_*.py` | Pipeline step scripts (0-24) |
| `src/gnn/` | GNN parsing, discovery, validation |
| `src/render/` | Code generation for all frameworks |
| `src/execute/` | Simulation execution |
| `src/tests/` | Test suite (`uv sync --extra dev` then `uv run pytest src/tests/ -q --tb=no --ignore=src/tests/test_llm_ollama.py --ignore=src/tests/test_llm_ollama_integration.py`: 1,924 passed, 28 skipped, 2026-04-10; enable those files when local `ollama` is available) |
| `input/gnn_files/` | Sample GNN model files |
| `output/` | Generated outputs (25 step-specific folders) |
| `doc/gnn/reference/gnn_syntax.md` | Complete GNN syntax specification |

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
- All modules need `__init__.py`, `AGENTS.md`, and either `processor.py` or a clearly-named alternative (see Thin Orchestrator Pattern above)
- Tests in `src/tests/test_{module}_*.py`

## Optional Dependency Groups

```bash
# Install specific optional groups using UV
uv sync --extra dev                  # Development tools
uv sync --extra api                  # REST API server (FastAPI + uvicorn)
uv sync --extra llm                  # Redundant with core (kept for compatibility); base install already includes openai, ollama, dotenv, aiohttp
uv sync --extra visualization        # Same pins as core (pandas, plotly, seaborn, h5py); optional alias
uv sync --extra inference            # bnlearn only; same pin as core; optional alias
uv sync --extra audio                # Audio processing
uv sync --extra gui                  # GUI interfaces
uv sync --extra execution-frameworks # Same pins as core Step 12 stack (compatibility / explicit sync)
uv sync --all-extras                 # Everything
```

# === COGNILAYER (auto-generated, do not delete) ===

## CogniLayer v3 Active
Persistent memory is ON.
ON FIRST USER MESSAGE in this session, briefly tell the user:
  'CogniLayer v3 active — persistent memory is on. Type /cognihelp for available commands.'
Say it ONCE, keep it short, then continue with their request.

## Memory Tools
You have access to the `cognilayer` MCP server:
- memory_search(query) — search memory semantically
- memory_write(content) — save important information
- file_search(query) — search project files (PRD, docs...)
- decision_log(query) — find past decisions

When unsure about context or project history,
ALWAYS search memory first via memory_search.
When you need info from PRD or docs, use file_search
INSTEAD of reading the entire file.

## VERIFY-BEFORE-ACT — MANDATORY
When memory_search returns a fact marked with ⚠ STALE:
1. ALWAYS read the source file and verify the fact still holds
2. If the fact changed -> update it via memory_write
3. NEVER make changes based on STALE facts without verification

## PROACTIVE MEMORY — IMPORTANT
When you discover something important during work, SAVE IT IMMEDIATELY:
- Bug and fix -> memory_write(type="error_fix")
- Pitfall/danger -> memory_write(type="gotcha")
- Exact procedure -> memory_write(type="procedure")
- How components communicate -> memory_write(type="api_contract")
- Performance issue -> memory_write(type="performance")
- Important command -> memory_write(type="command")
DO NOT wait for /harvest — session may crash.

## RUNNING BRIDGE — CRITICAL
After completing each task AUTOMATICALLY update session bridge:
  session_bridge(action="save", content="Progress: ...; Open: ...")
This is Tier 1 — do it yourself, don't announce, it's part of the job.

## Safety Rules — MANDATORY
- Before ANY deploy, push, ssh, pm2, docker, db migration:
  1. ALWAYS call verify_identity(action_type="...") first
  2. If it returns BLOCKED — STOP and ask the user
  3. If it returns VERIFIED — READ the target server to the user and request confirmation

## Git Rules
- Commit often, small atomic changes. Format: "[type] what and why"
- commit = Tier 1 (do it yourself). push = Tier 3 (verify_identity).

## Project DNA: generalized-notation-notation
Stack: unknown
Style: [unknown]
Structure: .agent_rules, .benchmarks, .github, .hypothesis, .pytest_cache, .ruff_cache, doc, input
Deploy: [NOT SET]
Active: [new session]
Last: [first session]

## Last Session Bridge
[Emergency bridge — running bridge was not updated]
No changes or facts in this session.

# === END COGNILAYER ===
