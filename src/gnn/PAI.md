# GNN Parser - PAI Context

## Quick Reference

**Purpose:** Parse GNN (Generalized Notation Notation) files into structured model representations.

**When to use this module:**
- Parse GNN markdown files into data structures
- Extract POMDP state spaces from GNN
- Convert between GNN and formal representations

## Common Operations

```python
# Parse GNN file
from gnn.parser import GNNParser
parser = GNNParser()
result = parser.parse_file("input/gnn_files/model.md")

# Extract POMDP
from gnn.pomdp_extractor import POMDPExtractor
extractor = POMDPExtractor()
pomdp = extractor.extract(result)
# pomdp.num_states, pomdp.num_observations, pomdp.A, pomdp.B, etc.
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | input/gnn_files/ | *.md GNN files |
| **Output** | render/* | POMDPStateSpace objects |

## Key Files

- `parser.py` - Main `GNNParser` class
- `pomdp_extractor.py` - `POMDPExtractor` for state space extraction
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Entry Point:** GNN parsing is the foundation - Step 3 of pipeline
2. **POMDPStateSpace:** Core data structure with A, B, C, D, E matrices
3. **Dimensions:** num_states, num_observations, num_actions, num_factors
4. **Format:** GNN uses markdown sections for model specification

## GNN Format Structure

```markdown
# Model Name

## StateSpace
- states: [s0, s1, s2]
- observations: [o0, o1]
- actions: [a0, a1]

## Matrices
### A (Observation)
### B (Transition)
### C (Preferences)
### D (Initial State)
```

---

**Version:** 1.1.3 | **Step:** 3 (GNN Processing)
