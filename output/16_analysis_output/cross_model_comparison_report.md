# Cross-Model Comparison Report

**Generated:** 2026-05-12 07:47:00

**Models:** 10 | **Frameworks:** 7

## Summary Matrix

| Model | PyMDP | JAX | RxInfer | ActiveInf.jl | DisCoPy | PyTorch | NumPyro |
|---|---|---|---|---|---|---|---|
| **actinf_pomdp_agent** | ✅ 0.976 | ✅ 0.996 | ✅ 0.996 | ✅ 0.973 | — | ✅ 1.000 | — |
| **bnlearn_causal_model** | ✅ 0.900 | ✅ 0.909 | ✅ 0.908 | ✅ 0.772 | — | ✅ 1.000 | — |
| **deep_planning_horizon** | ✅ 0.929 | ✅ 0.936 | ✅ 0.931 | ✅ 0.574 | — | ✅ 1.000 | — |
| **hmm_baseline** | ✅ 0.602 | ✅ 0.937 | ✅ 0.653 | — | — | ✅ 1.000 | — |
| **markov_chain** | ✅ 1.000 | ✅ 1.000 | ✅ 1.000 | — | — | ✅ 1.000 | — |
| **multi_armed_bandit** | ✅ 0.563 | ✅ 0.593 | ✅ 0.577 | ✅ 0.434 | — | ✅ 1.000 | — |
| **simple_mdp** | ✅ 1.000 | ✅ 1.000 | ✅ 1.000 | ✅ 1.000 | — | ✅ 1.000 | — |
| **time_varying_dynamics** | ✅ 0.874 | ✅ 0.848 | ✅ 0.895 | ✅ 0.538 | — | ✅ 1.000 | — |
| **tmaze_epistemic** | ✅ 1.000 | ✅ 1.000 | ✅ 1.000 | — | — | ✅ 1.000 | — |
| **two_state_bistable** | ✅ 0.811 | ✅ 0.795 | ✅ 0.806 | ✅ 0.671 | — | ✅ 1.000 | — |

> Values show validation status and mean belief confidence (max belief per timestep).

## Expected Free Energy Comparison

| Model | PyMDP | JAX | RxInfer | ActiveInf.jl | DisCoPy | PyTorch | NumPyro |
|---|---|---|---|---|---|---|---|
| **actinf_pomdp_agent** | 0.8795 | -0.2734 | 0.7018 | -1.1634 | — | 1.4951 | — |
| **bnlearn_causal_model** | 0.9621 | 0.0324 | 0.4034 | -0.5487 | — | 1.3133 | — |
| **deep_planning_horizon** | 0.8929 | 0.3013 | 1.2915 | -1.4968 | — | 3.1939 | — |
| **hmm_baseline** | 0.3245 | 1.5929 | 1.4688 | — | — | 1.7918 | — |
| **markov_chain** | 0.9168 | 0.0693 | 0.1651 | — | — | 1.0986 | — |
| **multi_armed_bandit** | 1.3128 | -0.0582 | 2.1979 | -2.0181 | — | 0.1698 | — |
| **simple_mdp** | 3.0251 | -2.2688 | 0.1141 | -2.5618 | — | 3.1392 | — |
| **time_varying_dynamics** | 0.9787 | 0.4426 | 0.6283 | -0.7513 | — | 1.5514 | — |
| **tmaze_epistemic** | 1.1552 | 0.0000 | 1.3863 | — | — | 1.3863 | — |
| **two_state_bistable** | 1.3682 | -0.4291 | 0.8870 | -1.1100 | — | 2.1269 | — |

## Belief Entropy Comparison

Mean Shannon entropy of posterior beliefs (lower = more certain).

| Model | PyMDP | JAX | RxInfer | ActiveInf.jl | DisCoPy | PyTorch | NumPyro |
|---|---|---|---|---|---|---|---|
| **actinf_pomdp_agent** | 0.0564 | 0.0147 | 0.0147 | 0.0811 | — | 0.0000 | — |
| **bnlearn_causal_model** | 0.3063 | 0.2868 | 0.2897 | 0.5200 | — | 0.0000 | — |
| **deep_planning_horizon** | 0.2491 | 0.1849 | 0.2366 | 1.0848 | — | 0.0000 | — |
| **hmm_baseline** | 0.9898 | 0.1737 | 0.9346 | — | — | 0.0000 | — |
| **markov_chain** | 0.0000 | 0.0000 | 0.0000 | — | — | 0.0000 | — |
| **multi_armed_bandit** | 0.8650 | 0.7504 | 0.8147 | 0.9993 | — | 0.0000 | — |
| **simple_mdp** | 0.0000 | 0.0000 | 0.0000 | 0.0011 | — | 0.0000 | — |
| **time_varying_dynamics** | 0.3851 | 0.4158 | 0.3445 | 0.9746 | — | 0.0000 | — |
| **tmaze_epistemic** | 0.0000 | 0.0000 | 0.0000 | — | — | 0.0000 | — |
| **two_state_bistable** | 0.4446 | 0.4946 | 0.4701 | 0.6313 | — | 0.0000 | — |

## Execution Time (seconds)

| Model | PyMDP | JAX | RxInfer | ActiveInf.jl | DisCoPy | PyTorch | NumPyro |
|---|---|---|---|---|---|---|---|
| **actinf_pomdp_agent** | — | — | — | — | — | — | — |
| **bnlearn_causal_model** | — | — | — | — | — | — | — |
| **deep_planning_horizon** | — | — | — | — | — | — | — |
| **hmm_baseline** | — | — | — | — | — | — | — |
| **markov_chain** | — | — | — | — | — | — | — |
| **multi_armed_bandit** | — | — | — | — | — | — | — |
| **simple_mdp** | — | — | — | — | — | — | — |
| **time_varying_dynamics** | — | — | — | — | — | — | — |
| **tmaze_epistemic** | — | — | — | — | — | — | — |
| **two_state_bistable** | — | — | — | — | — | — | — |

## Per-Model Details

### actinf_pomdp_agent

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 30 | 0.9756 | 0.8795 | 0.0564 | 0.100 | ✅ |
| JAX | 30 | 0.9965 | -0.2734 | 0.0147 | 0.100 | ✅ |
| RxInfer | 30 | 0.9965 | 0.7018 | 0.0147 | 0.100 | ✅ |
| ActiveInf.jl | 30 | 0.9725 | -1.1634 | 0.0811 | 0.100 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 30 | 1.0000 | 1.4951 | 0.0000 | 0.067 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyTorch (1.0000) | **Lowest:** ActiveInf.jl (0.9725)

### bnlearn_causal_model

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 30 | 0.8999 | 0.9621 | 0.3063 | 0.067 | ✅ |
| JAX | 30 | 0.9090 | 0.0324 | 0.2868 | 0.067 | ✅ |
| RxInfer | 30 | 0.9075 | 0.4034 | 0.2897 | 0.067 | ✅ |
| ActiveInf.jl | 30 | 0.7717 | -0.5487 | 0.5200 | 0.067 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 30 | 1.0000 | 1.3133 | 0.0000 | 0.067 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyTorch (1.0000) | **Lowest:** ActiveInf.jl (0.7717)

### deep_planning_horizon

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 30 | 0.9293 | 0.8929 | 0.2491 | 0.067 | ✅ |
| JAX | 30 | 0.9361 | 0.3013 | 0.1849 | 0.100 | ✅ |
| RxInfer | 30 | 0.9309 | 1.2915 | 0.2366 | 0.133 | ✅ |
| ActiveInf.jl | 30 | 0.5740 | -1.4968 | 1.0848 | 0.100 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 30 | 1.0000 | 3.1939 | 0.0000 | 0.067 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyTorch (1.0000) | **Lowest:** ActiveInf.jl (0.5740)

### hmm_baseline

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 50 | 0.6023 | 0.3245 | 0.9898 | 0.020 | ✅ |
| JAX | 50 | 0.9374 | 1.5929 | 0.1737 | 0.020 | ✅ |
| RxInfer | 50 | 0.6528 | 1.4688 | 0.9346 | 0.020 | ✅ |
| ActiveInf.jl | — | — | — | — | — | — |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 50 | 1.0000 | 1.7918 | 0.0000 | 0.040 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyTorch (1.0000) | **Lowest:** PyMDP (0.6023)

### markov_chain

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 40 | 1.0000 | 0.9168 | 0.0000 | 0.025 | ✅ |
| JAX | 40 | 1.0000 | 0.0693 | 0.0000 | 0.025 | ✅ |
| RxInfer | 40 | 1.0000 | 0.1651 | 0.0000 | 0.025 | ✅ |
| ActiveInf.jl | — | — | — | — | — | — |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 40 | 1.0000 | 1.0986 | 0.0000 | 0.050 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyMDP (1.0000) | **Lowest:** RxInfer (1.0000)

### multi_armed_bandit

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 30 | 0.5630 | 1.3128 | 0.8650 | 0.033 | ✅ |
| JAX | 30 | 0.5934 | -0.0582 | 0.7504 | 0.033 | ✅ |
| RxInfer | 30 | 0.5774 | 2.1979 | 0.8147 | 0.100 | ✅ |
| ActiveInf.jl | 30 | 0.4337 | -2.0181 | 0.9993 | 0.100 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 30 | 1.0000 | 0.1698 | 0.0000 | 0.067 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyTorch (1.0000) | **Lowest:** ActiveInf.jl (0.4337)

### simple_mdp

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 25 | 1.0000 | 3.0251 | 0.0000 | 0.080 | ✅ |
| JAX | 25 | 1.0000 | -2.2688 | 0.0000 | 0.120 | ✅ |
| RxInfer | 25 | 1.0000 | 0.1141 | 0.0000 | 0.080 | ✅ |
| ActiveInf.jl | 25 | 0.9999 | -2.5618 | 0.0011 | 0.080 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 25 | 1.0000 | 3.1392 | 0.0000 | 0.080 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyMDP (1.0000) | **Lowest:** ActiveInf.jl (0.9999)

### time_varying_dynamics

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 10 | 0.8738 | 0.9787 | 0.3851 | 0.200 | ✅ |
| JAX | 10 | 0.8484 | 0.4426 | 0.4158 | 0.200 | ✅ |
| RxInfer | 10 | 0.8947 | 0.6283 | 0.3445 | 0.200 | ✅ |
| ActiveInf.jl | 10 | 0.5382 | -0.7513 | 0.9746 | 0.200 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 10 | 1.0000 | 1.5514 | 0.0000 | 0.200 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyTorch (1.0000) | **Lowest:** ActiveInf.jl (0.5382)

### tmaze_epistemic

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 3 | 1.0000 | 1.1552 | 0.0000 | 0.667 | ✅ |
| JAX | 3 | 1.0000 | 0.0000 | 0.0000 | 0.333 | ✅ |
| RxInfer | 3 | 1.0000 | 1.3863 | 0.0000 | 0.667 | ✅ |
| ActiveInf.jl | — | — | — | — | — | — |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 3 | 1.0000 | 1.3863 | 0.0000 | 0.333 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyMDP (1.0000) | **Lowest:** RxInfer (1.0000)

### two_state_bistable

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 20 | 0.8106 | 1.3682 | 0.4446 | 0.100 | ✅ |
| JAX | 20 | 0.7946 | -0.4291 | 0.4946 | 0.100 | ✅ |
| RxInfer | 20 | 0.8063 | 0.8870 | 0.4701 | 0.100 | ✅ |
| ActiveInf.jl | 20 | 0.6706 | -1.1100 | 0.6313 | 0.100 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 20 | 1.0000 | 2.1269 | 0.0000 | 0.100 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyTorch (1.0000) | **Lowest:** ActiveInf.jl (0.6706)

## Cross-Model Observations

- **Highest avg. confidence:** tmaze_epistemic (1.0000)
- **Lowest avg. confidence:** multi_armed_bandit (0.6335)

---

*Generated by GNN Analysis Pipeline — 2026-05-12 07:47:00*
