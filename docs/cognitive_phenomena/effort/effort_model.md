# Cognitive Effort Model - GNN Implementation

> **Status**: Production Ready | **Version**: 1.0

## Overview

Cognitive effort refers to computational resources required for perception, inference, and action selection. In Active Inference, effort is modeled as precision-weighted prediction errors.

## GNN Model

```markdown
# Cognitive Effort POMDP
## GNNSection
CognitiveEffortPOMDP

## StateSpaceBlock
s[3,1,type=float]     # Cognitive load: Low, Medium, High
o[4,1,type=float]    # Task difficulty signals

## Connections
D>s
s-A
A-o
```

## Key Concepts

- **Precision**: Inverse variance of prediction errors
- **Expected Free Energy**: Combines epistemic and pragmatic value
- **Effort minimization**: Agents prefer low computational cost

## References

- Friston, K. (2010). The free energy principle: a unified brain theory?
- Parr, T., & Friston, K. J. (2017). Working memory, attention, and salience.

---

*See also: [doc/cognitive_phenomena/executive_control/](../executive_control/) for task switching models*
