# GNN Quick Reference Guide

> **Version**: 2.0 | **Last Updated**: March 2026

A concise reference for common GNN operations, syntax patterns, and frequently used commands.

---

## 🚀 Quick Start Commands

### Running the Pipeline

```bash
# Full pipeline (all 25 steps)
python src/main.py --target-dir input/gnn_files --verbose

# Run specific steps
python src/main.py --only-steps "3,5,7,8,11,12" --verbose

# Skip certain steps
python src/main.py --skip-steps "13,15" --verbose
```

### Individual Pipeline Steps

```bash
# Step 1: Setup environment
python src/1_setup.py --dev

# Step 3: Parse GNN files
python src/3_gnn.py --target-dir input/gnn_files

# Step 5: Type checking
python src/5_type_checker.py --strict

# Step 8: Visualization
python src/8_visualization.py

# Step 11: Render code
python src/11_render.py --frameworks "pymdp,jax"

# Step 12: Execute simulations
python src/12_execute.py --frameworks "pymdp"
```

---

## 📝 GNN Syntax Quick Reference

### Basic Model Structure

```markdown
# GNN Model Name
## GNNSection
ModelName

## StateSpaceBlock
# Variables
A[3,3,type=float]   # Matrix: rows x columns
B[3,3,3,type=float] # 3D tensor
C[3,type=float]     # Vector

## Connections
D>s                # D feeds into s (directed)
s-A                # s connects to A (undirected)
s>s_prime          # State transition
A-o                # Likelihood mapping
π>u                # Policy to action

## InitialParameterization
A={(0.9,0.05,0.05), (0.05,0.9,0.05), (0.05,0.05,0.9)}
C={(0.1, 0.1, 1.0)}
D={(0.33333, 0.33333, 0.33333)}

## ActInfOntologyAnnotation
A=LikelihoodMatrix
s=HiddenState
o=Observation
```

### Key Section Keywords

| Section | Purpose | Required |
|---------|---------|----------|
| `# GNN Model Name` | Model identifier | Yes |
| `## GNNSection` | Section declaration | Yes |
| `## StateSpaceBlock` | Variable definitions | Yes |
| `## Connections` | Dependency graph | Yes |
| `## InitialParameterization` | Parameter values | Recommended |
| `## ActInfOntologyAnnotation` | Semantic mapping | Optional |

---

## 🎯 Common Configuration Options

### Pipeline Options

| Flag | Description | Example |
|------|-------------|---------|
| `--target-dir` | Input GNN files | `--target-dir input/gnn_files` |
| `--output-dir` | Output location | `-o output/` |
| `--verbose` | Detailed logging | `--verbose` |
| `--strict` | Strict type checking | `--strict` |
| `--estimate-resources` | Resource estimation | `--estimate-resources` |
| `--dev` | Install dev dependencies | `--dev` |

### Framework Selection

```bash
# Specific frameworks
--frameworks "pymdp,jax,discopy"

# Presets
--frameworks "lite"     # PyMDP, JAX, DisCoPy
--frameworks "all"      # All available
--frameworks "julia"    # RxInfer, ActiveInference.jl
```

---

## 📊 Matrix Notation

### Dimension Syntax

```
variable[row,col,type=type]
variable[states,observations,type=float]
variable[states,states,actions,type=float]
```

### Connection Types

| Symbol | Meaning | Example |
|--------|---------|---------|
| `>` | Directed flow | `D>s` (D → s) |
| `-` | Bidirectional | `s-A` (s ↔ A) |
| `>` | Mapping | `A-o` (hidden → obs) |

---

## 🔧 Troubleshooting Quick Fixes

### Common Issues

| Issue | Solution |
|-------|----------|
| Import errors | Run `python src/1_setup.py --recreate-uv-env` |
| Type check failures | Use `--strict` flag for detailed errors |
| Visualization errors | Check `output/8_visualization_output/` |
| Execution timeout | Increase timeout in `input/config.yaml` |
| Memory issues | Use `--estimate-resources` to check requirements |

### Environment Reset

```bash
# Recreate environment
uv sync
uv run python src/1_setup.py --recreate-uv-env --dev

# Verify installation
uv run python -c "import gnn; print('OK')"
```

---

## 📁 Directory Reference

```
doc/                  # Documentation root
├── gnn/              # GNN language spec
│   ├── gnn_syntax.md
│   └── quickstart_tutorial.md
├── pymdp/            # PyMDP integration
├── rxinfer/          # RxInfer.jl integration
├── cognitive_phenomena/  # Example models
├── pipeline/         # Pipeline reference
└── troubleshooting/  # Issue resolution

src/                  # Source code
├── main.py           # Main orchestrator
├── 0_template.py     # Step 0
├── ...
├── 24_intelligent_analysis.py  # Step 24
└── gnn/              # GNN module

input/                # Input files
└── gnn_files/        # GNN model files

output/               # Generated outputs
├── 3_gnn_output/     # Parsed models
├── 5_type_checker_output/
├── 8_visualization_output/
└── ...
```

---

## 📖 Key Documentation Links

| Topic | Link |
|-------|------|
| Full Syntax | [doc/gnn/reference/gnn_syntax.md](../../doc/gnn/reference/gnn_syntax.md) |
| Quick Start | [doc/quickstart.md](../../doc/quickstart.md) |
| Pipeline Reference | [src/AGENTS.md](../AGENTS.md) |
| Learning Paths | [doc/learning_paths.md](../../doc/learning_paths.md) |
| API Reference | [doc/api/README.md](../../doc/api/README.md) |

---

## 🧮 Active Inference Glossary

| Term | Definition |
|------|------------|
| **GNN** | Generalized Notation Notation - standardized model language |
| **A-matrix** | Likelihood matrix P(o\|s) |
| **B-matrix** | Transition matrix P(s'\|s,a) |
| **C-vector** | Preferences log P(o) |
| **D-vector** | Initial state prior P(s) |
| **EFE** | Expected Free Energy |
| **FEP** | Free Energy Principle |

---

*For documentation index, see [doc/INDEX.md](../../doc/INDEX.md)*
*For changelog, see [CHANGELOG.md](CHANGELOG.md)*