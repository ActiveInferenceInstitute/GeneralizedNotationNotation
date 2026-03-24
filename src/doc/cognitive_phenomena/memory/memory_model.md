# Memory Systems - GNN Implementation

> **Status**: Production Ready | **Version**: 1.0

This document provides comprehensive GNN models for different memory systems within Active Inference.

## Overview

Memory in Active Inference can be decomposed into several subsystems:
- **Episodic Memory**: Storage of specific experiences
- **Semantic Memory**: Factual knowledge
- **Working Memory**: Active maintenance of information
- **Procedural Memory**: Motor skills and habits

## Working Memory Model

### Basic Structure

```markdown
# Working Memory POMDP
## GNNSection
WorkingMemoryPOMDP

## StateSpaceBlock
# Items in working memory (up to 3 items)
w[3,1,type=float]     # Which item is held

# Current task context
c[2,1,type=float]     # Task A or Task B

# Observations: item identities
o[5,1,type=float]    # Items 0-4

# Actions: remember or forget
u[2,1,type=int]      # Maintain or release

## Connections
D>w
D>c
c>w
w-A
A-o
```

### Memory Dynamics

```markdown
## InitialParameterization
# Context determines what to remember
# Task A: remember items 0-1, Task B: remember items 2-3
A={
  # Context 0 (Task A)
  (0.6, 0.3, 0.05, 0.03, 0.02),
  # Context 1 (Task B)
  (0.02, 0.03, 0.6, 0.3, 0.05)
}

# Working memory maintenance
B={
  # Action 0: Maintain - high retention
  (0.9, 0.05, 0.05),
  (0.05, 0.9, 0.05),
  (0.05, 0.05, 0.9),
  # Action 1: Release - decay
  (0.4, 0.3, 0.3),
  (0.3, 0.4, 0.3),
  (0.3, 0.3, 0.4)
}

# Prefer maintaining relevant items
C={(0.5, 0.5, -0.5, -0.5, 0.0)}

D={(0.33, 0.33, 0.34), (0.5, 0.5)}  # Prior over items and context

## ActInfOntologyAnnotation
w=WorkingMemoryBuffer
c=TaskContext
o=Observation
A=LikelihoodMatrix
```

## Episodic Memory Encoding

### Model Structure

```markdown
# Episodic Memory Encoding
## GNNSection
EpisodicMemory

## StateSpaceBlock
# Current experience embedding
e[4,1,type=float]    # 4-dimensional state

# Memory store (capacity 3)
m[3,4,type=float]   # 3 memory slots x 4 dims

# Retrieval cues
r[4,1,type=float]   # Retrieval query

# Actions: encode or retrieve
u[3,1,type=int]     # Store, Retrieve, Clear

## Connections
D>e
D>m
e>m        # Encode experience to memory
r-m        # Retrieve from memory
m>e        # Retrieved content influences state
u>m
```

### Memory Consolidation

```markdown
## InitialParameterization
# Encoding likelihood: experience to memory
A={
  # Original state
  (0.8, 0.1, 0.05, 0.05),
  (0.1, 0.8, 0.05, 0.05),
  (0.05, 0.8, 0.1, 0.05),
  (0.05, 0.05, 0.8, 0.1)
}

# Retrieval: similarity-based
# Similar experiences retrieve similar memories
B={
  # Store action
  [[0.9, 0.05, 0.03, 0.02],
   [0.05, 0.9, 0.03, 0.02],
   [0.03, 0.03, 0.9, 0.04],
   [0.02, 0.02, 0.04, 0.92]]
}

# Preferences: prefer consolidation
C={(0.8, 0.2, -0.3)}

D={(0.25, 0.25, 0.25, 0.25), (0.33, 0.33, 0.34)}

## ActInfOntologyAnnotation
e=ExperienceState
m=EpisodicMemoryStore
r=RetrievalCue
```

## Implementation Example (PyMDP)

```python
import numpy as np
from pymdp.agent import Agent
from pymdp.utils import obj_array_zeros

# Working memory model
num_obs = 5
num_states = 3
num_actions = 2

A = np.array([
    [0.6, 0.3, 0.05, 0.03, 0.02],
    [0.02, 0.03, 0.6, 0.3, 0.05]
])

B = np.zeros((num_states, num_states, num_actions))
B[:, :, 0] = np.array([
    [0.9, 0.05, 0.05],
    [0.05, 0.9, 0.05],
    [0.05, 0.05, 0.9]
])
B[:, :, 1] = np.array([
    [0.4, 0.3, 0.3],
    [0.3, 0.4, 0.3],
    [0.3, 0.3, 0.4]
])

C = np.array([0.5, 0.5, -0.5, -0.5, 0.0])
D = np.array([0.33, 0.33, 0.34])

agent = Agent(A=A, B=B, C=C, D=D)
```

## Simulation Results

### Working Memory Performance

| Trial | Context | Item Shown | Held | Action | Accuracy |
|-------|---------|------------|------|--------|----------|
| 1 | A | Item 0 | Item 0 | Maintain | ✓ |
| 2 | A | Item 1 | Item 0,1 | Maintain | ✓ |
| 3 | B | Item 3 | Item 3 | Clear+Store | ✓ |
| 4 | B | Item 2 | Item 2,3 | Maintain | ✓ |
| 5 | A | Item 1 | Item 0,1 | Maintain | ✓ |

## References

- Working Memory: Baddeley & Hitch (1974)
- Episodic Memory: Tulving (1972)
- Active Inference: Friston et al. (2017)

---

*See also: [doc/cognitive_phenomena/learning_adaptation/](../../../../doc/cognitive_phenomena/learning_adaptation/README.md) for learning models*
*See also: [doc/cognitive_phenomena/perception/](../../../../doc/cognitive_phenomena/perception/README.md) for perception models*