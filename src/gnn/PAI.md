# GNN Parser - PAI Context

## Quick Reference

**Purpose:** Parse GNN (Generalized Notation Notation) files into structured model representations.

**When to use this module:**

- Parse GNN markdown files into data structures
- Extract POMDP state spaces from GNN
- Convert between GNN and formal representations

## Common Operations

```python
from pathlib import Path
from gnn.pomdp_extractor import extract_pomdp_from_file

# POMDP matrices and dimensions from a Markdown GNN file
pomdp = extract_pomdp_from_file(Path("input/gnn_files/model.md"))
if pomdp:
    # pomdp.num_states, pomdp.num_observations, pomdp.A_matrix, pomdp.B_matrix, ...
    pass

# Multi-format parse/serialize (registry in parsers/system.py)
from gnn import GNNParsingSystem
from gnn.parsers.common import GNNFormat

system = GNNParsingSystem()
result = system.parse_file(Path("input/gnn_files/model.md"), format_hint=GNNFormat.MARKDOWN)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | input/gnn_files/ | *.md GNN files |
| **Output** | render/* | Downstream uses parsed models / POMDPStateSpace |

## Key Files

- `parsers/system.py` — `GNNParsingSystem`, `PARSER_REGISTRY`, `SERIALIZER_REGISTRY`
- `parser.py` — `validate_gnn`, `GNNFormalParser`, lightweight helpers
- `schema_validator.py` — `GNNParser` (section-level) and `GNNValidator` for strict validation
- `pomdp_extractor.py` — `POMDPExtractor`, `extract_pomdp_from_file`
- `__init__.py` — Public API exports

## Tips for AI Assistants

1. **Entry Point:** GNN parsing is Step 3 of the pipeline (`3_gnn.py`).
2. **POMDPStateSpace:** Dataclass with `A_matrix`, `B_matrix`, dimensions, etc.
3. **Format counts:** See [SPEC.md](SPEC.md) (23 enum, 22 serializers).
4. **Format:** GNN uses markdown sections for model specification.

## GNN Format Structure

```markdown
# Model Name

## StateSpaceBlock
- s[4],type=hidden  # 4 hidden states
- o[5],type=observable  # 5 observations
- u[3],type=control  # 3 control states

## Connections
s -> o  # Hidden states generate observations
u -> s  # Control states influence transitions

## InitialParameterization
### A (Likelihood)
### B (Transition)
### C (Preferences)
### D (Initial State)
```

---

**Version:** 1.1.3 | **Step:** 3 (GNN Processing)
