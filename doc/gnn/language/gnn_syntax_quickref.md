# GNN Syntax Quick Reference

**Version**: v2.0.0  
**Status**: Cheatsheet

---

## File Structure

```gnn
## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1.1

## ModelName
Active Inference POMDP Agent

## StateSpaceBlock
A[3,3, type=float]
B[3,3,3, type=float]
s[3, type=float, default=uniform]
o[3, type=int]

## Connections
A>o
B>s
s>o:observation_mapping
D>s:prior_initialization

## InitialParameterization
A={(0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9)}
```

---

## Variable Declaration Syntax

```
NAME[dim₁, dim₂, …, type=<type>, default=<default>]
```

| Component | Example | Required? |
|-----------|---------|-----------|
| Name | `A`, `s`, `π` | ✅ |
| Dimensions | `[3,3]`, `[num_obs, num_states]` | ✅ |
| Type | `type=float`, `type=int`, `type=bool` | ✅ |
| Default | `default=uniform`, `default=zeros` | ❌ |

**Defaults**: `uniform`, `zeros`, `ones`, `eye`, `random`

---

## Connection Syntax

| Syntax | Meaning |
|--------|---------|
| `A>B` | Directed edge: A → B |
| `A-B` | Undirected edge |
| `A>B:label` | Annotated directed edge |
| `A-B:label` | Annotated undirected edge |

---

## Multi-Model Files (v1.1)

Separate models with `---` on its own line:

```gnn
## StateSpaceBlock
A[3,3, type=float]
## Connections
A>B
---
## StateSpaceBlock
C[2,2, type=float]
## Connections
C>D
```

---

## YAML Front-Matter (v1.1)

```yaml
---
author: Jane Doe
version: "1.0"
framework_targets: [pymdp, rxinfer]
tags:
  - active-inference
  - pomdp
---
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| `GNN-E001` | Missing required section |
| `GNN-E002` | Dimension mismatch |
| `GNN-E003` | Unknown variable in connection |
| `GNN-E004` | Duplicate variable declaration |
| `GNN-E005` | Unparseable connection syntax |

---

*See [GNN v1.1 Syntax Specification](../gnn_syntax.md) for the canonical reference.*
