# Cross-Model Comparison Report

**Generated:** 2026-03-17 16:54:47

**Models:** 8 | **Frameworks:** 7

## Summary Matrix

| Model | PyMDP | JAX | RxInfer | ActiveInf.jl | DisCoPy | PyTorch | NumPyro |
|---|---|---|---|---|---|---|---|
| **actinf_pomdp_agent** | ✅ 0.989 | — | ✅ 0.993 | ✅ 0.945 | — | — | — |
| **deep_planning_horizon** | ✅ 0.910 | — | ✅ 0.922 | ✅ 0.566 | — | — | — |
| **hmm_baseline** | ✅ 0.629 | — | ✅ 0.654 | — | — | — | — |
| **markov_chain** | ✅ 1.000 | — | ✅ 1.000 | — | — | — | — |
| **multi_armed_bandit** | ✅ 0.509 | — | ✅ 0.578 | ✅ 0.432 | — | — | — |
| **simple_mdp** | ✅ 1.000 | — | ✅ 1.000 | ✅ 1.000 | — | — | — |
| **tmaze_epistemic** | ✅ 1.000 | — | ✅ 0.999 | — | — | — | — |
| **two_state_bistable** | ✅ 0.810 | — | ✅ 0.810 | ✅ 0.669 | — | — | — |

> Values show validation status and mean belief confidence (max belief per timestep).

## Expected Free Energy Comparison

| Model | PyMDP | JAX | RxInfer | ActiveInf.jl | DisCoPy | PyTorch | NumPyro |
|---|---|---|---|---|---|---|---|
| **actinf_pomdp_agent** | -1.4208 | — | 0.7186 | -1.1827 | — | — | — |
| **deep_planning_horizon** | -1.7722 | — | 1.9524 | -1.4137 | — | — | — |
| **hmm_baseline** | -1.4736 | — | 1.4691 | — | — | — | — |
| **markov_chain** | -0.1627 | — | 0.1309 | — | — | — | — |
| **multi_armed_bandit** | -2.1652 | — | 2.2111 | -2.0129 | — | — | — |
| **simple_mdp** | -0.1141 | — | 0.1141 | -2.5737 | — | — | — |
| **tmaze_epistemic** | -0.2310 | — | 1.1307 | — | — | — | — |
| **two_state_bistable** | -0.7642 | — | 0.8909 | -1.1092 | — | — | — |

## Belief Entropy Comparison

Mean Shannon entropy of posterior beliefs (lower = more certain).

| Model | PyMDP | JAX | RxInfer | ActiveInf.jl | DisCoPy | PyTorch | NumPyro |
|---|---|---|---|---|---|---|---|
| **actinf_pomdp_agent** | 0.0459 | — | 0.0293 | 0.1622 | — | — | — |
| **deep_planning_horizon** | 0.3108 | — | 0.2909 | 1.1075 | — | — | — |
| **hmm_baseline** | 0.9439 | — | 0.9518 | — | — | — | — |
| **markov_chain** | 0.0000 | — | 0.0000 | — | — | — | — |
| **multi_armed_bandit** | 0.8363 | — | 0.8007 | 1.0132 | — | — | — |
| **simple_mdp** | 0.0000 | — | 0.0000 | 0.0013 | — | — | — |
| **tmaze_epistemic** | 0.0000 | — | 0.0047 | — | — | — | — |
| **two_state_bistable** | 0.4453 | — | 0.4650 | 0.6320 | — | — | — |

## Execution Time (seconds)

| Model | PyMDP | JAX | RxInfer | ActiveInf.jl | DisCoPy | PyTorch | NumPyro |
|---|---|---|---|---|---|---|---|
| **actinf_pomdp_agent** | — | — | — | — | — | — | — |
| **deep_planning_horizon** | — | — | — | — | — | — | — |
| **hmm_baseline** | — | — | — | — | — | — | — |
| **markov_chain** | — | — | — | — | — | — | — |
| **multi_armed_bandit** | — | — | — | — | — | — | — |
| **simple_mdp** | — | — | — | — | — | — | — |
| **tmaze_epistemic** | — | — | — | — | — | — | — |
| **two_state_bistable** | — | — | — | — | — | — | — |

## Per-Model Details

### actinf_pomdp_agent

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 15 | 0.9890 | -1.4208 | 0.0459 | 0.067 | ✅ |
| JAX | — | — | — | — | — | — |
| RxInfer | 15 | 0.9929 | 0.7186 | 0.0293 | 0.200 | ✅ |
| ActiveInf.jl | 15 | 0.9450 | -1.1827 | 0.1622 | 0.200 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | — | — | — | — | — | — |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** RxInfer (0.9929) | **Lowest:** ActiveInf.jl (0.9450)

### deep_planning_horizon

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 15 | 0.9099 | -1.7722 | 0.3108 | 0.133 | ✅ |
| JAX | — | — | — | — | — | — |
| RxInfer | 15 | 0.9225 | 1.9524 | 0.2909 | 0.267 | ✅ |
| ActiveInf.jl | 15 | 0.5663 | -1.4137 | 1.1075 | 0.200 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | — | — | — | — | — | — |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** RxInfer (0.9225) | **Lowest:** ActiveInf.jl (0.5663)

### hmm_baseline

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 15 | 0.6295 | -1.4736 | 0.9439 | 0.067 | ✅ |
| JAX | — | — | — | — | — | — |
| RxInfer | 15 | 0.6537 | 1.4691 | 0.9518 | 0.067 | ✅ |
| ActiveInf.jl | — | — | — | — | — | — |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | — | — | — | — | — | — |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** RxInfer (0.6537) | **Lowest:** PyMDP (0.6295)

### markov_chain

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 15 | 1.0000 | -0.1627 | 0.0000 | 0.067 | ✅ |
| JAX | — | — | — | — | — | — |
| RxInfer | 15 | 1.0000 | 0.1309 | 0.0000 | 0.067 | ✅ |
| ActiveInf.jl | — | — | — | — | — | — |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | — | — | — | — | — | — |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyMDP (1.0000) | **Lowest:** RxInfer (1.0000)

### multi_armed_bandit

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 15 | 0.5090 | -2.1652 | 0.8363 | 0.067 | ✅ |
| JAX | — | — | — | — | — | — |
| RxInfer | 15 | 0.5779 | 2.2111 | 0.8007 | 0.200 | ✅ |
| ActiveInf.jl | 15 | 0.4317 | -2.0129 | 1.0132 | 0.200 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | — | — | — | — | — | — |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** RxInfer (0.5779) | **Lowest:** ActiveInf.jl (0.4317)

### simple_mdp

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 15 | 1.0000 | -0.1141 | 0.0000 | 0.133 | ✅ |
| JAX | — | — | — | — | — | — |
| RxInfer | 15 | 1.0000 | 0.1141 | 0.0000 | 0.133 | ✅ |
| ActiveInf.jl | 15 | 0.9999 | -2.5737 | 0.0013 | 0.133 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | — | — | — | — | — | — |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyMDP (1.0000) | **Lowest:** ActiveInf.jl (0.9999)

### tmaze_epistemic

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 15 | 1.0000 | -0.2310 | 0.0000 | 0.267 | ✅ |
| JAX | — | — | — | — | — | — |
| RxInfer | 15 | 0.9993 | 1.1307 | 0.0047 | 0.267 | ✅ |
| ActiveInf.jl | — | — | — | — | — | — |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | — | — | — | — | — | — |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyMDP (1.0000) | **Lowest:** RxInfer (0.9993)

### two_state_bistable

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 15 | 0.8096 | -0.7642 | 0.4453 | 0.133 | ✅ |
| JAX | — | — | — | — | — | — |
| RxInfer | 15 | 0.8101 | 0.8909 | 0.4650 | 0.133 | ✅ |
| ActiveInf.jl | 15 | 0.6695 | -1.1092 | 0.6320 | 0.133 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | — | — | — | — | — | — |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** RxInfer (0.8101) | **Lowest:** ActiveInf.jl (0.6695)

## Cross-Model Observations

- **Highest avg. confidence:** markov_chain (1.0000)
- **Lowest avg. confidence:** multi_armed_bandit (0.5062)

---

*Generated by GNN Analysis Pipeline — 2026-03-17 16:54:47*
