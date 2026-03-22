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
```

---

## 📝 GNN Syntax Quick Reference

### Basic Model Structure

```markdown
# GNN Model Name
## GNNSection
ModelName

## StateSpaceBlock
A[3,3,type=float]   # Matrix
B[3,3,3,type=float] # 3D tensor
C[3,type=float]     # Vector

## Connections
D>s                # Directed flow
s-A                # Bidirectional

## InitialParameterization
A={(0.9,0.05,0.05), (0.05,0.9,0.05), (0.05,0.05,0.9)}
```

---

## 📁 Directory Reference

| Directory | Purpose |
|-----------|---------|
| `doc/gnn/` | GNN language spec |
| `doc/pymdp/` | PyMDP integration |
| `doc/rxinfer/` | RxInfer.jl integration |
| `doc/cognitive_phenomena/` | Example models |

---

## 🧮 Active Inference Glossary

| Term | Definition |
|------|------------|
| **GNN** | Generalized Notation Notation |
| **A-matrix** | Likelihood matrix P(o|s) |
| **B-matrix** | Transition matrix P(s'|s,a) |
| **C-vector** | Preferences |
| **D-vector** | Initial state prior |

---

*For comprehensive documentation, see [doc/INDEX.md](./INDEX.md)*
