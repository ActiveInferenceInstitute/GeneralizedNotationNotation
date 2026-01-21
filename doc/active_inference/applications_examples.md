# Active Inference Applications and Examples

> **📋 Document Metadata**  
> **Type**: Applications Reference | **Audience**: Developers, Researchers | **Complexity**: Beginner to Intermediate  
> **Cross-References**: [GNN Examples](../gnn/gnn_examples_doc.md) | [Generative Models](generative_models.md) | [PyMDP Tutorials](../pymdp/pymdp_advanced_tutorials.md)

## Overview

This document provides practical examples and applications of Active Inference in the GNN framework.

**Status**: ✅ Production Ready  
**Version**: 1.0

---

## Classic Benchmarks

### T-Maze Navigation

The T-maze is the classic Active Inference benchmark:

```
    [Cue]
      |
    [Start] --- [Junction]
                 |     |
              [Left] [Right]
              Reward?  Reward?
```

**Challenge**: Use cue to infer which arm has reward.

**Files**:
- Examples: [`examples/`](../../examples/)
- Documentation: [`doc/pymdp/pymdp_pomdp/`](../pymdp/pymdp_pomdp/)

### Grid World Exploration

Agent navigates 2D grid seeking goals:

**Skills Demonstrated**:
- Epistemic foraging (exploration)
- Goal-directed navigation
- Uncertainty reduction

---

## Application Domains

### Robotics

| Application | Active Inference Role |
|-------------|----------------------|
| Navigation | EFE-based path planning |
| Manipulation | State estimation + action selection |
| SLAM | Belief updating over maps |

### Cognitive Science

| Application | Active Inference Role |
|-------------|----------------------|
| Perception | Hierarchical inference |
| Attention | Precision optimization |
| Decision-making | Policy selection via EFE |

### AI Agents

| Application | Active Inference Role |
|-------------|----------------------|
| Game playing | Goal-directed behavior |
| Dialogue | Context inference + response |
| Planning | Multi-step policy evaluation |

---

## Example: Simple Navigation Agent

### Model Specification

```python
import numpy as np

# 4-state grid: [0,1,2,3] arranged as 2x2
# Goal: reach state 3

# A Matrix: Perfect observation
A = np.eye(4)

# B Matrix: Movement dynamics
B = np.zeros((4, 4, 4))  # [states, states, actions]

# Action 0: Up (0→2, 1→3, 2→2, 3→3)
B[:, :, 0] = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]).T

# Action 1: Down (0→0, 1→1, 2→0, 3→1)
B[:, :, 1] = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
]).T

# Action 2: Left (0→0, 1→0, 2→2, 3→2)
B[:, :, 2] = np.array([
    [1, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 0]
]).T

# Action 3: Right (0→1, 1→1, 2→3, 3→3)
B[:, :, 3] = np.array([
    [0, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 1]
]).T

# C Vector: Prefer state 3
C = np.array([0, 0, 0, 3])

# D Vector: Start at state 0
D = np.array([1, 0, 0, 0])
```

### Running the Agent

```python
from pymdp.agent import Agent

agent = Agent(A=[A], B=[B], C=[C], D=[D])

# Simulation loop
state = 0
for t in range(10):
    obs = state  # Perfect observation
    agent.infer_states([obs])
    action = agent.infer_policies()
    
    # Execute action
    next_state_probs = B[:, state, action]
    state = np.random.choice(4, p=next_state_probs)
    
    print(f"t={t}: obs={obs}, action={action}, new_state={state}")
```

---

## Example: Exploration-Exploitation

### Setup

Agent must explore to find hidden goal location.

```python
# Uncertain A matrix - agent doesn't know goal location
A_uncertain = np.ones((4, 4)) * 0.25  # Uniform

# After observing goal, becomes certain
def update_A_on_goal(A, goal_obs):
    A = A.copy()
    A[:, :] = 0.1  # Small baseline
    A[goal_obs, goal_obs] = 0.7  # High probability
    return A
```

### Epistemic Behavior

Initially: High uncertainty → Agent explores
After finding goal: Low uncertainty → Agent exploits

---

## Output Examples

### Simulation Outputs

Located in: [`output/12_execution_output/`](../../output/12_execution_output/)

| Output | Description |
|--------|-------------|
| `beliefs.json` | Belief trajectories |
| `actions.json` | Action sequences |
| `efe_values.json` | EFE per policy |

### Analysis Outputs

Located in: [`output/16_analysis_output/`](../../output/16_analysis_output/)

| Output | Description |
|--------|-------------|
| `analysis_summary.json` | Summary statistics |
| `visualizations/` | Plots and figures |

---

## Related Resources

### Documentation
- **[GNN Examples](../gnn/gnn_examples_doc.md)**
- **[PyMDP Tutorials](../pymdp/pymdp_advanced_tutorials.md)**
- **[POMDP Examples](../pymdp/pymdp_pomdp/)**

### Source Code
- **[Examples](../../examples/)**
- **[Output](../../output/)**

---

**Status**: ✅ Production Ready  
**Compliance**: GNN documentation standards
