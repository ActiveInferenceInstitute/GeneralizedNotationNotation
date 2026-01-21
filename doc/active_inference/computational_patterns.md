# Computational Patterns in Active Inference

> **📋 Document Metadata**  
> **Type**: Technical Reference | **Audience**: Developers | **Complexity**: Intermediate  
> **Cross-References**: [Variational Inference](variational_inference.md) | [Expected Free Energy](expected_free_energy.md) | [Analysis Tools](../../src/analysis/)

## Overview

This document describes common computational patterns used in Active Inference implementations, with references to GNN source code.

**Status**: ✅ Production Ready  
**Version**: 1.0

---

## Pattern 1: Belief Update Loop

### Algorithm

```python
def belief_update(prior, likelihood, observation, num_iters=16):
    """Iterative belief update via VFE minimization."""
    Q = prior.copy()
    
    for _ in range(num_iters):
        # Likelihood contribution
        ln_A = np.log(likelihood[observation, :] + 1e-10)
        
        # Prior contribution  
        ln_prior = np.log(prior + 1e-10)
        
        # Combine (in log space)
        ln_Q = ln_A + ln_prior
        
        # Normalize
        Q = softmax(ln_Q)
    
    return Q
```

### Source References

| Implementation | Path |
|----------------|------|
| PyMDP | [`src/execute/pymdp/`](../../src/execute/pymdp/) |
| Analysis | [`src/analysis/analyzer.py`](../../src/analysis/analyzer.py) |

---

## Pattern 2: Policy Evaluation

### Algorithm

```python
def evaluate_policies(policies, Q_s, A, B, C, gamma=16.0):
    """Evaluate all policies and return selection probabilities."""
    G = np.zeros(len(policies))
    
    for i, policy in enumerate(policies):
        G[i] = compute_efe(policy, Q_s, A, B, C)
    
    # Softmax selection
    P_pi = softmax(-gamma * G)
    
    return P_pi, G
```

### Key Components

| Component | Description |
|-----------|-------------|
| EFE | Expected Free Energy per policy |
| Softmax | Convert EFE to probabilities |
| Gamma | Precision (inverse temperature) |

---

## Pattern 3: Forward-Backward Inference

### Forward Pass

```python
def forward(A, B, D, observations):
    """Compute filtered beliefs."""
    T = len(observations)
    Q = [None] * T
    
    Q[0] = normalize(A[observations[0], :] * D)
    
    for t in range(1, T):
        Q_pred = B @ Q[t-1]
        Q[t] = normalize(A[observations[t], :] * Q_pred)
    
    return Q
```

### Backward Pass

```python
def backward(A, B, Q_fwd, observations):
    """Compute smoothed beliefs."""
    T = len(observations)
    Q = Q_fwd.copy()
    
    for t in range(T-2, -1, -1):
        Q_pred = B @ Q_fwd[t]
        backward_msg = B.T @ (Q[t+1] / (Q_pred + 1e-10))
        Q[t] = normalize(Q_fwd[t] * backward_msg)
    
    return Q
```

---

## Pattern 4: Learning Updates

### Dirichlet Accumulation

```python
def update_concentration(a, Q_s, observation, learning_rate=1.0):
    """Update Dirichlet concentration parameters."""
    a_new = a.copy()
    a_new[observation, :] += learning_rate * Q_s
    return a_new

def concentration_to_probability(a):
    """Convert concentration to probability matrix."""
    return a / a.sum(axis=0, keepdims=True)
```

### Precision Learning

```python
def update_precision(gamma, G_selected, G_mean, eta=0.1):
    """Update policy precision."""
    prediction_error = G_selected - G_mean
    gamma_new = gamma + eta * prediction_error
    return max(1.0, gamma_new)  # Ensure positive
```

---

## Pattern 5: Entropy Computation

### Shannon Entropy

```python
def entropy(p):
    """Compute Shannon entropy (nats)."""
    p = np.clip(p, 1e-10, 1.0)
    return -np.sum(p * np.log(p))
```

### Conditional Entropy

```python
def conditional_entropy(A, Q_s):
    """Compute H(o|s) under beliefs Q(s)."""
    H = 0.0
    for s in range(len(Q_s)):
        p_o_given_s = A[:, s]
        H += Q_s[s] * entropy(p_o_given_s)
    return H
```

---

## Pattern 6: Softmax Utilities

### Standard Softmax

```python
def softmax(x, tau=1.0):
    """Numerically stable softmax."""
    x = x / tau
    x = x - np.max(x)  # Numerical stability
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()
```

### Log Softmax

```python
def log_softmax(x, tau=1.0):
    """Numerically stable log-softmax."""
    x = x / tau
    x = x - np.max(x)
    return x - np.log(np.sum(np.exp(x)))
```

---

## Pattern 7: Matrix Utilities

### Normalize Columns

```python
def normalize_columns(M):
    """Normalize matrix columns to sum to 1."""
    col_sums = M.sum(axis=0, keepdims=True)
    return M / (col_sums + 1e-10)
```

### Safe Log

```python
def safe_log(x, eps=1e-10):
    """Logarithm with numerical safety."""
    return np.log(x + eps)
```

---

## Source Code References

### Core Utilities

| Component | Path |
|-----------|------|
| Utils | [`src/utils/`](../../src/utils/) |
| Analysis | [`src/analysis/`](../../src/analysis/) |

### Implementation-Specific

| Engine | Path |
|--------|------|
| PyMDP | [`src/execute/pymdp/`](../../src/execute/pymdp/) |
| RxInfer | [`src/execute/rxinfer/`](../../src/execute/rxinfer/) |
| ActiveInference.jl | [`src/execute/activeinference_jl/`](../../src/execute/activeinference_jl/) |

---

## Related Documentation

- **[Variational Inference](variational_inference.md)**: Theory behind patterns
- **[Expected Free Energy](expected_free_energy.md)**: EFE computation
- **[Generative Models](generative_models.md)**: Matrix specifications
- **[PyMDP Implementation](implementation_pymdp.md)**: Python examples

---

**Status**: ✅ Production Ready  
**Compliance**: GNN documentation standards
