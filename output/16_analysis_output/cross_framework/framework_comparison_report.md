# Framework Execution Comparison Report

Generated: 2026-04-14T11:10:46.943836

## Summary

- Total Frameworks: 8
- Total Executions: 72
- Successful Executions: 72
- Overall Success Rate: 100.00%

## Framework Details

### NUMPYRO

- Success Rate: 100.0% (9/9)
- Execution Time: 0.74s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, beliefs_sum_to_one=✅, actions_in_range=✅, all_valid=✅)
- Data Source: `output/12_execute_output/simple_mdp/numpyro/simulation_data/simulation_results.json`

### PYMDP

- Success Rate: 100.0% (9/9)
- Execution Time: 2.74s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅, pymdp_version_ge_1_0_0=✅)
- Data Source: `output/12_execute_output/simple_mdp/pymdp/simulation_data/simulation_results.json`

### PYTORCH

- Success Rate: 100.0% (9/9)
- Execution Time: 0.56s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, beliefs_sum_to_one=✅, actions_in_range=✅, all_valid=✅)
- Data Source: `output/12_execute_output/simple_mdp/pytorch/simulation_data/simulation_results.json`

### JAX

- Success Rate: 100.0% (9/9)
- Execution Time: 0.67s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/simple_mdp/jax/simulation_data/simulation_results.json`

### DISCOPY

- Success Rate: 100.0% (9/9)
- Execution Time: 0.25s
- Data Source: `output/12_execute_output/simple_mdp/discopy/execution_logs/Simple MDP Agent_discopy.py_results.json`

### BNLEARN

- Success Rate: 100.0% (9/9)
- Execution Time: 2.07s
- Data Source: `output/12_execute_output/simple_mdp/bnlearn/execution_logs/Simple MDP Agent_bnlearn.py_results.json`

### RXINFER

- Success Rate: 100.0% (9/9)
- Execution Time: 17.14s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, actions_in_range=✅, beliefs_sum_to_one=✅)
- Data Source: `output/12_execute_output/simple_mdp/rxinfer/simulation_data/simulation_results.json`

### ACTIVEINFERENCE_JL

- Success Rate: 100.0% (9/9)
- Execution Time: 29.12s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, actions_in_range=✅, all_valid=✅, beliefs_sum_to_one=✅)
- Data Source: `output/12_execute_output/simple_mdp/activeinference_jl/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| numpyro | 15 | 1.0000 | 3.1392 | 0.0000 |
| pymdp | 15 | 1.0000 | 1.0751 | 1.1325 |
| pytorch | 15 | 1.0000 | 3.1392 | 0.0000 |
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
| numpyro | 1.137 | 1.162 | 0.621 | 4.419 |
| pymdp | 2.812 | 0.151 | 2.599 | 3.122 |
| pytorch | 0.482 | 0.029 | 0.460 | 0.559 |
| jax | 0.705 | 0.056 | 0.648 | 0.825 |
| discopy | 0.247 | 0.004 | 0.242 | 0.254 |
| bnlearn | 2.062 | 0.110 | 1.965 | 2.290 |
| rxinfer | 14.146 | 1.076 | 13.543 | 17.136 |
| activeinference_jl | 16.135 | 4.619 | 13.992 | 29.115 |
