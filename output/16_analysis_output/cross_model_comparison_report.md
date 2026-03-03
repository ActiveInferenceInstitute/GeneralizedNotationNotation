# Cross-Model Comparison Report

**Generated:** 2026-03-03 08:41:37

**Models:** 8 | **Frameworks:** 7

## Summary Matrix

| Model | PyMDP | JAX | RxInfer | ActiveInf.jl | DisCoPy | PyTorch | NumPyro |
|---|---|---|---|---|---|---|---|
| **actinf_pomdp_agent** | ✅ 0.993 | ✅ 0.993 | ✅ 0.993 | ✅ 0.945 | — | ✅ 1.000 | ✅ 1.000 |
| **deep_planning_horizon** | ✅ 0.917 | ✅ 0.934 | ✅ 0.922 | ✅ 0.566 | — | ✅ 1.000 | ✅ 1.000 |
| **hmm_baseline** | ✅ 0.639 | ✅ 0.808 | ✅ 0.654 | — | — | ✅ 1.000 | ✅ 1.000 |
| **markov_chain** | ✅ 1.000 | ✅ 1.000 | ✅ 1.000 | — | — | ✅ 1.000 | ✅ 1.000 |
| **multi_armed_bandit** | ✅ 0.493 | ✅ 0.511 | ✅ 0.578 | ✅ 0.432 | — | ✅ 1.000 | ✅ 1.000 |
| **simple_mdp** | ✅ 1.000 | ✅ 1.000 | ✅ 1.000 | ✅ 1.000 | — | ✅ 1.000 | ✅ 1.000 |
| **tmaze_epistemic** | ✅ 1.000 | ✅ 1.000 | ✅ 0.999 | — | — | ✅ 1.000 | ✅ 1.000 |
| **two_state_bistable** | ✅ 0.788 | ✅ 0.796 | ✅ 0.810 | ✅ 0.669 | — | ✅ 1.000 | ✅ 1.000 |

> Values show validation status and mean belief confidence (max belief per timestep).

## Expected Free Energy Comparison

| Model | PyMDP | JAX | RxInfer | ActiveInf.jl | DisCoPy | PyTorch | NumPyro |
|---|---|---|---|---|---|---|---|
| **actinf_pomdp_agent** | -0.6752 | -0.1412 | 0.7186 | -1.1827 | — | 1.4951 | 0.5951 |
| **deep_planning_horizon** | -1.9453 | 0.8795 | 1.9524 | -1.4137 | — | 2.6939 | 2.6939 |
| **hmm_baseline** | -1.4724 | 1.6440 | 1.4691 | — | — | 1.7918 | 1.7918 |
| **markov_chain** | -0.1881 | 0.0693 | 0.1309 | — | — | 1.0986 | 1.0986 |
| **multi_armed_bandit** | -1.8839 | 0.1612 | 2.2111 | -2.0129 | — | 0.1698 | 0.1698 |
| **simple_mdp** | -0.1141 | -2.2688 | 0.1141 | -2.5737 | — | 3.1392 | 3.1392 |
| **tmaze_epistemic** | -0.0924 | 0.0000 | 1.1307 | — | — | 1.3863 | 1.3863 |
| **two_state_bistable** | -0.7776 | -0.4299 | 0.8909 | -1.1092 | — | 0.1269 | 0.1269 |

## Belief Entropy Comparison

Mean Shannon entropy of posterior beliefs (lower = more certain).

| Model | PyMDP | JAX | RxInfer | ActiveInf.jl | DisCoPy | PyTorch | NumPyro |
|---|---|---|---|---|---|---|---|
| **actinf_pomdp_agent** | 0.0294 | 0.0293 | 0.0293 | 0.1622 | — | 0.0000 | 0.0000 |
| **deep_planning_horizon** | 0.3113 | 0.2016 | 0.2909 | 1.1075 | — | 0.0000 | 0.0000 |
| **hmm_baseline** | 0.9461 | 0.5118 | 0.9518 | — | — | 0.0000 | 0.0000 |
| **markov_chain** | 0.0000 | 0.0000 | 0.0000 | — | — | 0.0000 | 0.0000 |
| **multi_armed_bandit** | 0.9743 | 0.7833 | 0.8007 | 1.0132 | — | 0.0000 | 0.0000 |
| **simple_mdp** | 0.0000 | 0.0000 | 0.0000 | 0.0013 | — | 0.0000 | 0.0000 |
| **tmaze_epistemic** | 0.0000 | 0.0000 | 0.0047 | — | — | 0.0000 | 0.0000 |
| **two_state_bistable** | 0.4827 | 0.4939 | 0.4650 | 0.6320 | — | 0.0000 | 0.0000 |

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
| PyMDP | 15 | 0.9929 | -0.6752 | 0.0294 | 0.133 | ✅ |
| JAX | 15 | 0.9929 | -0.1412 | 0.0293 | 0.200 | ✅ |
| RxInfer | 15 | 0.9929 | 0.7186 | 0.0293 | 0.200 | ✅ |
| ActiveInf.jl | 15 | 0.9450 | -1.1827 | 0.1622 | 0.200 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 15 | 1.0000 | 1.4951 | 0.0000 | 0.133 | ✅ |
| NumPyro | 15 | 1.0000 | 0.5951 | 0.0000 | 0.133 | ✅ |

**Highest confidence:** NumPyro (1.0000) | **Lowest:** ActiveInf.jl (0.9450)

### deep_planning_horizon

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 15 | 0.9166 | -1.9453 | 0.3113 | 0.067 | ✅ |
| JAX | 15 | 0.9335 | 0.8795 | 0.2016 | 0.133 | ✅ |
| RxInfer | 15 | 0.9225 | 1.9524 | 0.2909 | 0.267 | ✅ |
| ActiveInf.jl | 15 | 0.5663 | -1.4137 | 1.1075 | 0.200 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 15 | 1.0000 | 2.6939 | 0.0000 | 0.133 | ✅ |
| NumPyro | 15 | 1.0000 | 2.6939 | 0.0000 | 0.133 | ✅ |

**Highest confidence:** NumPyro (1.0000) | **Lowest:** ActiveInf.jl (0.5663)

### hmm_baseline

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 15 | 0.6389 | -1.4724 | 0.9461 | 0.067 | ✅ |
| JAX | 15 | 0.8078 | 1.6440 | 0.5118 | 0.067 | ✅ |
| RxInfer | 15 | 0.6537 | 1.4691 | 0.9518 | 0.067 | ✅ |
| ActiveInf.jl | — | — | — | — | — | — |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 15 | 1.0000 | 1.7918 | 0.0000 | 0.133 | ✅ |
| NumPyro | 15 | 1.0000 | 1.7918 | 0.0000 | 0.133 | ✅ |

**Highest confidence:** NumPyro (1.0000) | **Lowest:** PyMDP (0.6389)

### markov_chain

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 15 | 1.0000 | -0.1881 | 0.0000 | 0.067 | ✅ |
| JAX | 15 | 1.0000 | 0.0693 | 0.0000 | 0.067 | ✅ |
| RxInfer | 15 | 1.0000 | 0.1309 | 0.0000 | 0.067 | ✅ |
| ActiveInf.jl | — | — | — | — | — | — |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 15 | 1.0000 | 1.0986 | 0.0000 | 0.133 | ✅ |
| NumPyro | 15 | 1.0000 | 1.0986 | 0.0000 | 0.133 | ✅ |

**Highest confidence:** PyMDP (1.0000) | **Lowest:** RxInfer (1.0000)

### multi_armed_bandit

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 15 | 0.4927 | -1.8839 | 0.9743 | 0.067 | ✅ |
| JAX | 15 | 0.5109 | 0.1612 | 0.7833 | 0.067 | ✅ |
| RxInfer | 15 | 0.5779 | 2.2111 | 0.8007 | 0.200 | ✅ |
| ActiveInf.jl | 15 | 0.4317 | -2.0129 | 1.0132 | 0.200 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 15 | 1.0000 | 0.1698 | 0.0000 | 0.133 | ✅ |
| NumPyro | 15 | 1.0000 | 0.1698 | 0.0000 | 0.133 | ✅ |

**Highest confidence:** NumPyro (1.0000) | **Lowest:** ActiveInf.jl (0.4317)

### simple_mdp

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 15 | 1.0000 | -0.1141 | 0.0000 | 0.133 | ✅ |
| JAX | 15 | 1.0000 | -2.2688 | 0.0000 | 0.200 | ✅ |
| RxInfer | 15 | 1.0000 | 0.1141 | 0.0000 | 0.133 | ✅ |
| ActiveInf.jl | 15 | 0.9999 | -2.5737 | 0.0013 | 0.133 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 15 | 1.0000 | 3.1392 | 0.0000 | 0.133 | ✅ |
| NumPyro | 15 | 1.0000 | 3.1392 | 0.0000 | 0.133 | ✅ |

**Highest confidence:** NumPyro (1.0000) | **Lowest:** ActiveInf.jl (0.9999)

### tmaze_epistemic

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 15 | 1.0000 | -0.0924 | 0.0000 | 0.267 | ✅ |
| JAX | 15 | 1.0000 | 0.0000 | 0.0000 | 0.067 | ✅ |
| RxInfer | 15 | 0.9993 | 1.1307 | 0.0047 | 0.267 | ✅ |
| ActiveInf.jl | — | — | — | — | — | — |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 15 | 1.0000 | 1.3863 | 0.0000 | 0.133 | ✅ |
| NumPyro | 15 | 1.0000 | 1.3863 | 0.0000 | 0.133 | ✅ |

**Highest confidence:** PyMDP (1.0000) | **Lowest:** RxInfer (0.9993)

### two_state_bistable

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 15 | 0.7880 | -0.7776 | 0.4827 | 0.133 | ✅ |
| JAX | 15 | 0.7960 | -0.4299 | 0.4939 | 0.133 | ✅ |
| RxInfer | 15 | 0.8101 | 0.8909 | 0.4650 | 0.133 | ✅ |
| ActiveInf.jl | 15 | 0.6695 | -1.1092 | 0.6320 | 0.133 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 15 | 1.0000 | 0.1269 | 0.0000 | 0.133 | ✅ |
| NumPyro | 15 | 1.0000 | 0.1269 | 0.0000 | 0.133 | ✅ |

**Highest confidence:** NumPyro (1.0000) | **Lowest:** ActiveInf.jl (0.6695)

## Cross-Model Observations

- **Highest avg. confidence:** markov_chain (1.0000)
- **Lowest avg. confidence:** multi_armed_bandit (0.6689)

---

*Generated by GNN Analysis Pipeline — 2026-03-03 08:41:37*
