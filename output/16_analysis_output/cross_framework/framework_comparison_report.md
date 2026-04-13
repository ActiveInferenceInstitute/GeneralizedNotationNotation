# Framework Execution Comparison Report

Generated: 2026-04-12T17:34:53.492008

## Summary

- Total Frameworks: 8
- Total Executions: 72
- Successful Executions: 72
- Overall Success Rate: 100.00%

## Framework Details

### NUMPYRO

- Success Rate: 100.0% (9/9)
- Execution Time: 0.90s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, beliefs_sum_to_one=✅, actions_in_range=✅, all_valid=✅)
- Data Source: `output/12_execute_output/simple_mdp/numpyro/simulation_data/simulation_results.json`

### PYMDP

- Success Rate: 100.0% (9/9)
- Execution Time: 2.68s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅, pymdp_version_ge_1_0_0=✅)
- Data Source: `output/12_execute_output/simple_mdp/pymdp/simulation_data/simulation_results.json`

### PYTORCH

- Success Rate: 100.0% (9/9)
- Execution Time: 0.49s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, beliefs_sum_to_one=✅, actions_in_range=✅, all_valid=✅)
- Data Source: `output/12_execute_output/simple_mdp/pytorch/simulation_data/simulation_results.json`

### JAX

- Success Rate: 100.0% (9/9)
- Execution Time: 0.72s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/simple_mdp/jax/simulation_data/simulation_results.json`

### DISCOPY

- Success Rate: 100.0% (9/9)
- Execution Time: 0.24s
- Data Source: `output/12_execute_output/simple_mdp/discopy/execution_logs/Simple MDP Agent_discopy.py_results.json`

### BNLEARN

- Success Rate: 100.0% (9/9)
- Execution Time: 1.87s
- Data Source: `output/12_execute_output/simple_mdp/bnlearn/execution_logs/Simple MDP Agent_bnlearn.py_results.json`

### RXINFER

- Success Rate: 100.0% (9/9)
- Execution Time: 16.71s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, actions_in_range=✅, beliefs_sum_to_one=✅)
- Data Source: `output/12_execute_output/simple_mdp/rxinfer/simulation_data/simulation_results.json`

### ACTIVEINFERENCE_JL

- Success Rate: 100.0% (9/9)
- Execution Time: 16.40s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, actions_in_range=✅, all_valid=✅, beliefs_sum_to_one=✅)
- Data Source: `output/12_execute_output/simple_mdp/activeinference_jl/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| numpyro | 15 | 1.0000 | 3.1392 | 0.0000 |
| pymdp | 15 | 1.0000 | 1.0751 | 1.1325 |
| pytorch | 15 | 1.0000 | 0.1392 | 0.0000 |
| jax | 15 | 1.0000 | -0.3188 | 1.1325 |
| rxinfer | 15 | 1.0000 | 0.1141 | 0.0000 |
| activeinference_jl | 15 | 0.9999 | -2.0638 | 1.1323 |

## Data Coverage

| Framework | Beliefs | Actions | Observations | Free Energy | Validation |
|-----------|---------|---------|--------------|-------------|------------|
| numpyro | ✅ | ✅ | ✅ | ✅ | ✅ |
| pymdp | ✅ | ✅ | ✅ | ✅ | ✅ |
| pytorch | ✅ | ✅ | ✅ | ✅ | ✅ |
| jax | ✅ | ✅ | ✅ | ✅ | ✅ |
| discopy | ❌ | ❌ | ❌ | ❌ | — |
| bnlearn | ❌ | ❌ | ❌ | ❌ | — |
| rxinfer | ✅ | ✅ | ✅ | ✅ | ✅ |
| activeinference_jl | ✅ | ✅ | ✅ | ✅ | ✅ |

## Cross-Framework Metric Agreement

- **numpyro_vs_pymdp**: confidence correlation = 0.0000
- **numpyro_vs_pytorch**: confidence correlation = 0.0000
- **numpyro_vs_jax**: confidence correlation = 0.0000
- **numpyro_vs_rxinfer**: confidence correlation = 0.0000
- **numpyro_vs_activeinference_jl**: confidence correlation = 0.0000
- **pymdp_vs_pytorch**: confidence correlation = 0.0000
- **pymdp_vs_jax**: confidence correlation = 0.0000
- **pymdp_vs_rxinfer**: confidence correlation = 0.0000
- **pymdp_vs_activeinference_jl**: confidence correlation = 0.0000
- **pytorch_vs_jax**: confidence correlation = -0.0690
- **pytorch_vs_rxinfer**: confidence correlation = 0.2380
- **pytorch_vs_activeinference_jl**: confidence correlation = 0.1288
- **jax_vs_rxinfer**: confidence correlation = -0.0920
- **jax_vs_activeinference_jl**: confidence correlation = -0.1215
- **rxinfer_vs_activeinference_jl**: confidence correlation = -0.0759

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| numpyro | 0.757 | 0.086 | 0.634 | 0.897 |
| pymdp | 2.708 | 0.106 | 2.626 | 2.995 |
| pytorch | 0.470 | 0.012 | 0.460 | 0.497 |
| jax | 0.714 | 0.053 | 0.659 | 0.831 |
| discopy | 0.241 | 0.003 | 0.238 | 0.250 |
| bnlearn | 1.886 | 0.023 | 1.864 | 1.934 |
| rxinfer | 14.182 | 0.904 | 13.752 | 16.707 |
| activeinference_jl | 13.661 | 1.761 | 10.608 | 16.397 |
