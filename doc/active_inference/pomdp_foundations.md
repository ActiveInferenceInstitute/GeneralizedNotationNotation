# POMDP Foundations for Active Inference

> **📋 Document Metadata**  
> **Type**: Theoretical Reference | **Audience**: Developers, Researchers | **Complexity**: Intermediate  
> **Cross-References**: [Active Inference Theory](active_inference_theory.md) | [Generative Models](generative_models.md) | [POMDP Documentation](../pomdp/README.md)

## Overview

**Partially Observable Markov Decision Processes (POMDPs)** provide the mathematical framework for discrete Active Inference. This document covers POMDP fundamentals and their relationship to Active Inference.

**Status**: ✅ Production Ready  
**Version**: 1.0

---

## What is a POMDP?

A POMDP models sequential decision-making under uncertainty:

- **Partial Observability**: Agent cannot directly observe true state
- **Markov Property**: Future depends only on present state
- **Decision Process**: Agent must choose actions

### Formal Definition

A POMDP is defined by the tuple $(S, A, O, T, Z, R, \gamma)$:

| Symbol | Name | Description |
|--------|------|-------------|
| $S$ | States | Set of hidden states |
| $A$ | Actions | Set of available actions |
| $O$ | Observations | Set of possible observations |
| $T$ | Transitions | $P(s'|s, a)$ |
| $Z$ | Observation model | $P(o|s, a)$ or $P(o|s')$ |
| $R$ | Reward | $R(s, a)$ or $R(s, a, s')$ |
| $\gamma$ | Discount | Future reward discount factor |

---

## POMDP vs Active Inference

### Mapping

| POMDP Component | Active Inference | GNN Matrix |
|-----------------|------------------|------------|
| Transitions $T$ | State dynamics | **B** |
| Observations $Z$ | Likelihood | **A** |
| Reward $R$ | Log preferences | **C** |
| Initial distribution | Prior over states | **D** |
| Policy prior | Habits | **E** |
| Discount $\gamma$ | Precision | $\gamma$ |

### Key Differences

| Aspect | POMDP/RL | Active Inference |
|--------|----------|------------------|
| **Objective** | Maximize reward | Minimize free energy |
| **Exploration** | Added (ε-greedy, etc.) | Emergent (epistemic) |
| **Perception** | Separate | Integrated |
| **Model** | Optional | Required |

---

## Belief States

### Definition

A **belief state** $b(s)$ is a probability distribution over hidden states:

$$b(s) = P(s_t = s | o_{1:t}, a_{1:t-1})$$

### Belief Update

After action $a$ and observation $o$:

$$b'(s') = \eta \cdot Z(o|s') \sum_s T(s'|s,a) b(s)$$

Where $\eta$ is a normalizing constant.

In matrix form:
$$b' = \text{normalize}(A_{o,:} \odot (B_a \cdot b))$$

### Active Inference Correspondence

| POMDP | Active Inference |
|-------|------------------|
| Belief $b(s)$ | $Q(s)$ |
| Belief update | VFE minimization |
| Bayesian filter | Perception |

---

## Planning in POMDPs

### Value Function

The value of a belief state:
$$V(b) = \max_a \left[ R(b,a) + \gamma \sum_o P(o|b,a) V(b') \right]$$

### Policy

A POMDP policy maps belief states to actions:
$$\pi: b \to a$$

### Active Inference Approach

Instead of computing value functions, Active Inference:
1. Enumerates candidate policies
2. Computes Expected Free Energy for each
3. Selects policy with lowest EFE

---

## Discrete State Spaces

### State Representation

States are often discrete factors:

```python
# Example: Grid world with object
states = {
    'location': [0, 1, 2, 3],  # 4 locations
    'has_object': [True, False]  # 2 states
}
# Total: 4 × 2 = 8 states
```

### Factorized Representation

$$s = (s^{(1)}, s^{(2)}, \ldots, s^{(n)})$$

Each factor can have its own dynamics:
$$P(s'^{(i)} | s^{(i)}, a)$$ for factor $i$

---

## Time Horizon

### Finite Horizon

Agent plans for $T$ timesteps:
$$\pi = (a_1, a_2, \ldots, a_T)$$

### Policy Space

For $|A|$ actions and horizon $T$:
$$|\Pi| = |A|^T$$

This grows exponentially—pruning is essential.

### Receding Horizon

Re-plan at each timestep with sliding window.

---

## POMDP Algorithms

### Exact Methods

- **Value Iteration**: Compute optimal value function
- **Policy Iteration**: Iterate between evaluation and improvement

### Approximate Methods

- **Point-Based Value Iteration (PBVI)**
- **SARSOP**: Sampling-based
- **POMCP**: Monte Carlo tree search

### Active Inference Methods

- **Variational Message Passing**: Belief updates
- **Tree Search**: Policy evaluation via EFE
- **Amortized Inference**: Learned policies

---

## Example: Grid World

### Setup

```
+---+---+---+---+
| S |   |   | G |
+---+---+---+---+
```

- **States**: 4 locations
- **Actions**: Left, Right, Stay
- **Observations**: Location (partially observable)
- **Goal**: Reach position G

### GNN Specification

```python
# A Matrix: Noisy position observation
A = np.array([
    [0.8, 0.1, 0.05, 0.05],  # Observe pos 0
    [0.1, 0.8, 0.1, 0.0],    # Observe pos 1
    [0.05, 0.1, 0.8, 0.05],  # Observe pos 2
    [0.05, 0.0, 0.05, 0.9],  # Observe pos 3 (goal)
])

# B Matrix: Deterministic transitions
B = np.zeros((4, 4, 3))
# Action 0: Left
B[:, :, 0] = np.array([
    [1, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
]).T
# ... (similar for Right, Stay)

# C Vector: Prefer goal state
C = np.array([0, 0, 0, 3])  # Strong preference for pos 3

# D Vector: Start at position 0
D = np.array([1, 0, 0, 0])
```

---

## Connection to MDPs

### MDP as Special Case

When $Z(o|s) = \delta_{o,s}$ (identity), POMDP becomes MDP:
- Full observability
- Belief = state
- Simpler planning

### Active Inference with Full Observability

Even with full observability, Active Inference maintains:
- Epistemic value (uncertainty about future)
- Unified perception-action
- Learning dynamics

---

## Source Code References

### POMDP Documentation

| Resource | Path |
|----------|------|
| POMDP Overview | [`doc/pomdp/pomdp_overall.md`](../pomdp/pomdp_overall.md) |
| POMDP Analytics | [`doc/pomdp/pomdp_analytic.md`](../pomdp/pomdp_analytic.md) |

### Implementation

| Engine | Path |
|--------|------|
| PyMDP | [`src/execute/pymdp/`](../../src/execute/pymdp/) |
| RxInfer | [`src/execute/rxinfer/`](../../src/execute/rxinfer/) |

### GNN

| Component | Path |
|-----------|------|
| Examples | [`examples/`](../../examples/) |
| Parser | [`src/gnn/`](../../src/gnn/) |

---

## Related Documentation

- **[Active Inference Theory](active_inference_theory.md)**: Core theory
- **[Generative Models](generative_models.md)**: A, B, C, D matrices
- **[Expected Free Energy](expected_free_energy.md)**: Policy selection
- **[PyMDP Implementation](implementation_pymdp.md)**: Python POMDP
- **[POMDP Documentation](../pomdp/README.md)**: Detailed POMDP theory

---

## Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **POMDP** | Sequential decision under partial observability |
| **Belief State** | Distribution over hidden states |
| **Policy** | Mapping from beliefs to actions |
| **Horizon** | Planning depth (timesteps) |
| **Value** | Expected future reward/free energy |

---

**Status**: ✅ Production Ready  
**Compliance**: GNN documentation standards  
**Maintenance**: Regular updates with new POMDP methods
