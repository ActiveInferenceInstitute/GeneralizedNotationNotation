# GNN Language Reference Hub

The GNN language is a markdown-embedded domain-specific language for specifying Active Inference models. This hub collects all syntax and type-system documentation in one place.

**Last Updated**: February 24, 2026

## Language Documents

| Document | Description |
|----------|-------------|
| **[../gnn_syntax.md](../gnn_syntax.md)** | Core syntax: sections, variables, connections, parameters |
| **[../gnn_dsl_manual.md](../gnn_dsl_manual.md)** | Complete DSL manual with all constructs and examples |
| **[../gnn_schema.md](../gnn_schema.md)** | JSON schema for machine-readable GNN representation |
| **[../gnn_type_system.md](../gnn_type_system.md)** | Type system: continuous, categorical, binary, integer |
| **[../gnn_file_structure_doc.md](../gnn_file_structure_doc.md)** | File structure: required sections, naming conventions |
| **[../gnn_examples_doc.md](../gnn_examples_doc.md)** | Annotated examples: POMDP, T-maze, multi-factor |
| **[../gnn_standards.md](../gnn_standards.md)** | Style guide and community standards |

## Quick Syntax Reference

### Minimal Valid GNN File

```gnn
# ModelName: MyModel
# Description: A minimal Active Inference model

## StateSpaceBlock
s[2, type=hidden]   # 2 hidden states
o[2, type=observed] # 2 observations
π[2, type=policy]   # 2 actions

## Connections
A > s               # Likelihood: P(o|s)
s > B > s           # Transition: P(s'|s, π)
π > G               # Policy: EFE

## Equations
G = -E_q[ln P(o|s)] + KL[Q(s)||P(s)]

## Parameters
A = [[0.9, 0.1],
     [0.1, 0.9]]   # Identity-like likelihood
D = [0.5, 0.5]     # Uniform prior
```

### Variable Types

| Type | Keyword | Example | Use |
|------|---------|---------|-----|
| Categorical | `type=categorical` | `s[4, type=categorical]` | Discrete states |
| Continuous | `type=continuous` | `s[4, type=continuous]` | Real-valued |
| Binary | `type=binary` | `s[4, type=binary]` | Boolean/Bernoulli |
| Integer | `type=integer` | `s[4, type=integer]` | Countable |
| Hidden | `type=hidden` | `s[4, type=hidden]` | Latent variables |
| Observed | `type=observed` | `o[3, type=observed]` | Observable |
| Policy | `type=policy` | `π[2, type=policy]` | Action space |

### Connection Syntax

| Syntax | Meaning | Example |
|--------|---------|---------|
| `A > B` | Directed causal edge | `s > o` (state generates observation) |
| `A - B` | Undirected dependency | `s - s` (correlated states) |
| `A > B > C` | Composed path | `s > A > o` (via likelihood matrix) |
| `A > B[w=0.5]` | Weighted edge | `s > o[weight=0.9]` |

## Ontology Annotations

Variables can be annotated with Active Inference Ontology (ACT-O) terms:

```gnn
## Annotations
s: act:HiddenState
o: act:Observation
A: act:LikelihoodMapping
B: act:TransitionMapping
G: act:ExpectedFreeEnergy
```

See **[../ontology_system.md](../ontology_system.md)** for the full ontology term reference.

## Validation

Validate a GNN file using Step 6 (Validation) or the MCP tool:

```bash
# Pipeline step
python src/6_validation.py --target-dir input/gnn_files/ --verbose

# MCP tool
# Call: validate_gnn_file(path="input/gnn_files/my_model.md")

# Type checker
python src/5_type_checker.py --target-dir input/gnn_files/ --verbose
```

## See Also

- [Quickstart Tutorial](../quickstart_tutorial.md) — write your first GNN file
- [Advanced Modeling Patterns](../advanced_modeling_patterns.md) — hierarchical, factorial POMDPs
- [Framework Integration Guide](../framework_integration_guide.md) — render to PyMDP, RxInfer, etc.
- [Implementations](../implementations/README.md) — how GNN maps to each framework
