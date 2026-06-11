# Cross-Model Comparison Report

**Generated:** 2026-05-22 06:35:05

**Models:** 10 | **Frameworks:** 7

## Summary Matrix

| Model | PyMDP | JAX | RxInfer | ActiveInf.jl | DisCoPy | PyTorch | NumPyro |
|---|---|---|---|---|---|---|---|
| **actinf_pomdp_agent** | ✅ 0.976 | ✅ 0.996 | ✅ 0.996 | ✅ 0.996 | — | ✅ 0.909 | — |
| **bnlearn_causal_model** | ✅ 0.900 | ✅ 0.909 | ✅ 0.663 | ✅ 0.663 | — | ✅ 0.996 | — |
| **deep_planning_horizon** | ✅ 0.929 | ✅ 0.930 | ✅ 0.759 | ✅ 0.759 | — | ✅ 0.997 | — |
| **hmm_baseline** | ✅ 0.602 | ✅ 0.632 | ✅ 0.459 | ✅ 0.459 | — | ✅ 0.678 | — |
| **markov_chain** | ✅ 1.000 | ✅ 1.000 | ✅ 0.562 | ✅ 0.562 | — | ❌ 0.100 | — |
| **multi_armed_bandit** | ✅ 0.563 | ✅ 0.593 | ✅ 0.541 | ✅ 0.541 | — | ✅ 0.738 | — |
| **simple_mdp** | ✅ 1.000 | ✅ 1.000 | ✅ 0.900 | ✅ 0.900 | — | ❌ 0.040 | — |
| **time_varying_dynamics** | ✅ 0.874 | ✅ 0.847 | ✅ 0.685 | ✅ 0.685 | — | ✅ 0.976 | — |
| **tmaze_epistemic** | ✅ 0.667 | ✅ 0.500 | ✅ 0.708 | ✅ 0.708 | — | ❌ 0.167 | — |
| **two_state_bistable** | ✅ 0.811 | ✅ 0.787 | ✅ 0.660 | ✅ 0.660 | — | ✅ 0.838 | — |

> Values show validation status and mean belief confidence (max belief per timestep).

## Expected Free Energy Comparison

| Model | PyMDP | JAX | RxInfer | ActiveInf.jl | DisCoPy | PyTorch | NumPyro |
|---|---|---|---|---|---|---|---|
| **actinf_pomdp_agent** | 0.8795 | -0.2734 | 0.7018 | 0.7018 | — | 1.1457 | — |
| **bnlearn_causal_model** | 0.9621 | 0.0324 | 0.4034 | 0.4034 | — | 0.4744 | — |
| **deep_planning_horizon** | 0.8929 | 0.2989 | 1.2522 | 1.2522 | — | 0.4999 | — |
| **hmm_baseline** | 0.3245 | 1.7651 | 1.4688 | 1.4688 | — | 1.4862 | — |
| **markov_chain** | 0.9168 | 0.9617 | 0.1651 | 0.1651 | — | 0.0297 | — |
| **multi_armed_bandit** | 1.3128 | -0.0582 | 2.1979 | 2.1979 | — | 2.0505 | — |
| **simple_mdp** | 3.0251 | -2.2688 | 0.1141 | 0.1141 | — | 0.0046 | — |
| **time_varying_dynamics** | 0.9787 | 0.4401 | 0.6285 | 0.6285 | — | 0.8944 | — |
| **tmaze_epistemic** | 1.9185 | 1.0000 | 1.7751 | 1.7751 | — | 0.8070 | — |
| **two_state_bistable** | 1.3682 | -0.4951 | 0.7998 | 0.7998 | — | 0.8489 | — |

## Belief Entropy Comparison

Mean Shannon entropy of posterior beliefs (lower = more certain).

| Model | PyMDP | JAX | RxInfer | ActiveInf.jl | DisCoPy | PyTorch | NumPyro |
|---|---|---|---|---|---|---|---|
| **actinf_pomdp_agent** | 0.0564 | 0.0147 | 0.0147 | 0.0147 | — | 0.1979 | — |
| **bnlearn_causal_model** | 0.3063 | 0.2868 | 0.6378 | 0.6378 | — | 0.0164 | — |
| **deep_planning_horizon** | 0.2491 | 0.1939 | 0.7360 | 0.7360 | — | 0.0155 | — |
| **hmm_baseline** | 0.9898 | 0.9343 | 1.2561 | 1.2561 | — | 0.7519 | — |
| **markov_chain** | 0.0000 | 0.0000 | 0.9335 | 0.9335 | — | 0.9888 | — |
| **multi_armed_bandit** | 0.8650 | 0.7504 | 0.9109 | 0.9109 | — | 0.5882 | — |
| **simple_mdp** | 0.0000 | 0.0000 | 0.3251 | 0.3251 | — | 1.3308 | — |
| **time_varying_dynamics** | 0.3851 | 0.4166 | 0.7983 | 0.7983 | — | 0.0840 | — |
| **tmaze_epistemic** | 0.4621 | 0.6931 | 0.6931 | 0.6931 | — | 1.6173 | — |
| **two_state_bistable** | 0.4446 | 0.4815 | 0.6280 | 0.6280 | — | 0.3126 | — |

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
| ActiveInf.jl | 30 | 0.9965 | 0.7018 | 0.0147 | 0.100 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 30 | 0.9085 | 1.1457 | 0.1979 | 0.100 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** RxInfer (0.9965) | **Lowest:** PyTorch (0.9085)

### bnlearn_causal_model

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 30 | 0.8999 | 0.9621 | 0.3063 | 0.067 | ✅ |
| JAX | 30 | 0.9090 | 0.0324 | 0.2868 | 0.067 | ✅ |
| RxInfer | 30 | 0.6630 | 0.4034 | 0.6378 | 0.067 | ✅ |
| ActiveInf.jl | 30 | 0.6630 | 0.4034 | 0.6378 | 0.067 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 30 | 0.9957 | 0.4744 | 0.0164 | 0.067 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyTorch (0.9957) | **Lowest:** RxInfer (0.6630)

### deep_planning_horizon

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 30 | 0.9293 | 0.8929 | 0.2491 | 0.067 | ✅ |
| JAX | 30 | 0.9303 | 0.2989 | 0.1939 | 0.100 | ✅ |
| RxInfer | 30 | 0.7586 | 1.2522 | 0.7360 | 0.133 | ✅ |
| ActiveInf.jl | 30 | 0.7586 | 1.2522 | 0.7360 | 0.133 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 30 | 0.9965 | 0.4999 | 0.0155 | 0.133 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyTorch (0.9965) | **Lowest:** RxInfer (0.7586)

### hmm_baseline

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 50 | 0.6023 | 0.3245 | 0.9898 | 0.020 | ✅ |
| JAX | 50 | 0.6322 | 1.7651 | 0.9343 | 0.020 | ✅ |
| RxInfer | 50 | 0.4591 | 1.4688 | 1.2561 | 0.020 | ✅ |
| ActiveInf.jl | 50 | 0.4591 | 1.4688 | 1.2561 | 0.020 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 50 | 0.6784 | 1.4862 | 0.7519 | 0.020 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyTorch (0.6784) | **Lowest:** RxInfer (0.4591)

### markov_chain

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 40 | 1.0000 | 0.9168 | 0.0000 | 0.025 | ✅ |
| JAX | 40 | 1.0000 | 0.9617 | 0.0000 | 0.025 | ✅ |
| RxInfer | 40 | 0.5625 | 0.1651 | 0.9335 | 0.025 | ✅ |
| ActiveInf.jl | 40 | 0.5625 | 0.1651 | 0.9335 | 0.025 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 40 | 0.1000 | 0.0297 | 0.9888 | 0.025 | ❌ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyMDP (1.0000) | **Lowest:** PyTorch (0.1000)

### multi_armed_bandit

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 30 | 0.5630 | 1.3128 | 0.8650 | 0.033 | ✅ |
| JAX | 30 | 0.5934 | -0.0582 | 0.7504 | 0.033 | ✅ |
| RxInfer | 30 | 0.5408 | 2.1979 | 0.9109 | 0.100 | ✅ |
| ActiveInf.jl | 30 | 0.5408 | 2.1979 | 0.9109 | 0.100 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 30 | 0.7383 | 2.0505 | 0.5882 | 0.100 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyTorch (0.7383) | **Lowest:** RxInfer (0.5408)

### simple_mdp

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 25 | 1.0000 | 3.0251 | 0.0000 | 0.080 | ✅ |
| JAX | 25 | 1.0000 | -2.2688 | 0.0000 | 0.120 | ✅ |
| RxInfer | 25 | 0.9000 | 0.1141 | 0.3251 | 0.080 | ✅ |
| ActiveInf.jl | 25 | 0.9000 | 0.1141 | 0.3251 | 0.080 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 25 | 0.0400 | 0.0046 | 1.3308 | 0.160 | ❌ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyMDP (1.0000) | **Lowest:** PyTorch (0.0400)

### time_varying_dynamics

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 10 | 0.8738 | 0.9787 | 0.3851 | 0.200 | ✅ |
| JAX | 10 | 0.8472 | 0.4401 | 0.4166 | 0.200 | ✅ |
| RxInfer | 10 | 0.6852 | 0.6285 | 0.7983 | 0.200 | ✅ |
| ActiveInf.jl | 10 | 0.6852 | 0.6285 | 0.7983 | 0.200 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 10 | 0.9762 | 0.8944 | 0.0840 | 0.200 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyTorch (0.9762) | **Lowest:** RxInfer (0.6852)

### tmaze_epistemic

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 3 | 0.6667 | 1.9185 | 0.4621 | 0.667 | ✅ |
| JAX | 3 | 0.5000 | 1.0000 | 0.6931 | 0.333 | ✅ |
| RxInfer | 3 | 0.7083 | 1.7751 | 0.6931 | 0.667 | ✅ |
| ActiveInf.jl | 3 | 0.7083 | 1.7751 | 0.6931 | 0.667 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 3 | 0.1667 | 0.8070 | 1.6173 | 0.667 | ❌ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** RxInfer (0.7083) | **Lowest:** PyTorch (0.1667)

### two_state_bistable

| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |
|-----------|-------|------------|------------|---------|------------------|------------|
| PyMDP | 20 | 0.8106 | 1.3682 | 0.4446 | 0.100 | ✅ |
| JAX | 20 | 0.7873 | -0.4951 | 0.4815 | 0.100 | ✅ |
| RxInfer | 20 | 0.6603 | 0.7998 | 0.6280 | 0.100 | ✅ |
| ActiveInf.jl | 20 | 0.6603 | 0.7998 | 0.6280 | 0.100 | ✅ |
| DisCoPy | — | — | — | — | — | — |
| PyTorch | 20 | 0.8384 | 0.8489 | 0.3126 | 0.100 | ✅ |
| NumPyro | — | — | — | — | — | — |

**Highest confidence:** PyTorch (0.8384) | **Lowest:** RxInfer (0.6603)

## Cross-Model Observations

- **Highest avg. confidence:** actinf_pomdp_agent (0.9747)
- **Lowest avg. confidence:** tmaze_epistemic (0.5500)

---

*Generated by GNN Analysis Pipeline — 2026-05-22 06:35:05*
