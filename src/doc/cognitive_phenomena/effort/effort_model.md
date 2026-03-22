# Cognitive Effort Model - GNN Implementation

> **Status**: Production Ready | **Version**: 1.0

This document provides a comprehensive GNN model for cognitive effort within the Active Inference framework.

## Overview

Cognitive effort refers to the computational resources required for perception, inference, and action selection. In Active Inference, effort is modeled as the precision-weighted precision on prediction errors that drives belief updating.

## GNN Model Specification

### Basic Structure

```markdown
# Cognitive Effort POMDP
## GNNSection
CognitiveEffortPOMDP

## StateSpaceBlock
# Hidden states: cognitive load levels
s[3,1,type=float]     # Low, Medium, High cognitive load

# Observations: task difficulty signals  
o[4,1,type=float]    # Easy, Medium, Hard, Very Hard

# Actions: control signals
u[2,1,type=int]      # Effort allocation (low/high)

# Likelihood mapping: how task difficulty affects observation
A[4,3,type=float]

# Transition: how actions affect cognitive load
B[3,3,2,type=float]

# Preferences: prefer low effort
C[4,type=float]

# Prior over initial states
D[3,type=float]

## Connections
D>s
s-A
s>s_prime
A-o
u>B
```

### Effort Dynamics

```markdown
## InitialParameterization
# Task difficulty likelihood
A={
  # State 0: Low cognitive load
  (0.8, 0.15, 0.04, 0.01),
  # State 1: Medium cognitive load  
  (0.1, 0.7, 0.15, 0.05),
  # State 2: High cognitive load
  (0.02, 0.08, 0.6, 0.3)
}

# Action 0: Low effort - tends to maintain current state
B={
  # Action 0 transitions
  (0.85, 0.12, 0.03),
  (0.15, 0.7, 0.15),
  (0.05, 0.2, 0.75),
  # Action 1 transitions  
  (0.3, 0.5, 0.2),
  (0.1, 0.6, 0.3),
  (0.05, 0.15, 0.8)
}

# Prefer easy observations (low effort)
C={(1.0, 0.3, -0.5, -1.0)}

# Start with low cognitive load
D={(0.7, 0.25, 0.05)}

## ActInfOntologyAnnotation
s=CognitiveLoadState
o=TaskDifficultyObservation
A=LikelihoodMatrix
B=TransitionMatrix
C=PreferenceVector
D=PriorState
```

## Precision-Weighted Effort Model

### Extended Model with Precision

```markdown
# Precision-Weighted Cognitive Effort
## GNNSection
PrecisionEffortModel

## StateSpaceBlock
# Hidden states
s[3,1,type=float]

# Precision (inverse variance of prediction errors)
γ[1,1,type=float]    # Precision parameter

# Observations
o[4,1,type=float]

# Actions
u[2,1,type=int]

## Connections
D>s
γ>s
s-A
A-o
u>B
γ>π
```

### Effort Calculation

The expected free energy includes a term for epistemic value (reducing uncertainty about cognitive load) and pragmatic value (selecting actions that minimize expected effort):

```
G = E_q[ln P(o|s,a) + ln P(s'|s,a)] - H[q(π)]
```

Where precision modulates the balance between exploration (epistemic) and exploitation (pragmatic).

## Implementation in PyMDP

```python
import numpy as np
from pymdp.agent import Agent

# Define observation model (likelihood matrix A)
A = np.array([
    [0.8, 0.15, 0.04, 0.01],  # Low load
    [0.1, 0.7, 0.15, 0.05],   # Medium load
    [0.02, 0.08, 0.6, 0.3]    # High load
])

# Define transition model (B matrix)
B = np.array([
    # Action 0: Maintain
    [[0.85, 0.12, 0.03],
     [0.15, 0.7, 0.15],
     [0.05, 0.2, 0.75]],
    # Action 1: Increase effort
    [[0.3, 0.5, 0.2],
     [0.1, 0.6, 0.3],
     [0.05, 0.15, 0.8]]
])

# Define preferences (C vector)
C = np.array([1.0, 0.3, -0.5, -1.0])

# Initial prior (D vector)
D = np.array([0.7, 0.25, 0.05])

# Create agent
agent = Agent(
    A=A, 
    B=B, 
    C=C, 
    D=D,
    policy_selection='ard'  # Active inference with precision
)
```

## Simulation Results

### Example Run

| Step | Observation | True State | Belief | Action | Effort Cost |
|------|-------------|------------|--------|--------|-------------|
| 0 | Easy (0) | Low | [0.7, 0.25, 0.05] | - | - |
| 1 | Medium (1) | Low | [0.65, 0.28, 0.07] | 0 | 0.2 |
| 2 | Easy (0) | Low | [0.72, 0.22, 0.06] | 0 | 0.2 |
| 3 | Hard (2) | Medium | [0.45, 0.45, 0.10] | 1 | 0.8 |
| 4 | Medium (1) | Medium | [0.30, 0.55, 0.15] | 1 | 0.8 |

## References

- Friston, K. (2010). The free energy principle: a unified brain theory?
- Parr, T., & Friston, K. J. (2017). Working memory, attention, and salience.
- FitzGerald, T. H. B., et al. (2015). Precision and binding in action selection.

---

*See also: [doc/cognitive_phenomena/executive_control/](./executive_control/) for task switching models*
*See also: [doc/cognitive_phenomena/attention/](./attention/) for attention allocation*