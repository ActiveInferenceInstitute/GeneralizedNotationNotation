# Framework Execution Comparison Report

Generated: 2026-04-14T12:04:00.745416

## Summary

- Total Frameworks: 8
- Total Executions: 72
- Successful Executions: 54
- Overall Success Rate: 75.00%

## Framework Details

### NUMPYRO

- Success Rate: 100.0% (9/9)
- Execution Time: 1.19s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, beliefs_sum_to_one=✅, actions_in_range=✅, all_valid=✅)
- Data Source: `output/12_execute_output/simple_mdp/numpyro/simulation_data/simulation_results.json`

### PYMDP

- Success Rate: 100.0% (9/9)
- Execution Time: 3.07s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅, pymdp_version_ge_1_0_0=✅)
- Data Source: `output/12_execute_output/simple_mdp/pymdp/simulation_data/simulation_results.json`

### PYTORCH

- Success Rate: 100.0% (9/9)
- Execution Time: 0.50s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, beliefs_sum_to_one=✅, actions_in_range=✅, all_valid=✅)
- Data Source: `output/12_execute_output/simple_mdp/pytorch/simulation_data/simulation_results.json`

### JAX

- Success Rate: 100.0% (9/9)
- Execution Time: 0.70s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/simple_mdp/jax/simulation_data/simulation_results.json`

### DISCOPY

- Success Rate: 100.0% (9/9)
- Execution Time: 0.26s
- Data Source: `output/12_execute_output/simple_mdp/discopy/execution_logs/Simple MDP Agent_discopy.py_results.json`

### BNLEARN

- Success Rate: 100.0% (9/9)
- Execution Time: 2.29s
- Data Source: `output/12_execute_output/simple_mdp/bnlearn/execution_logs/Simple MDP Agent_bnlearn.py_results.json`

### RXINFER

- Success Rate: 0.0% (0/9)
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/markov_chain/jax/simulation_data/simulation_results.json`

### ACTIVEINFERENCE_JL

- Success Rate: 0.0% (0/9)
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/markov_chain/jax/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| numpyro | 15 | 1.0000 | 3.1392 | 0.0000 |
| pymdp | 15 | 1.0000 | 1.0751 | 1.1325 |
| pytorch | 15 | 1.0000 | 3.1392 | 0.0000 |
| jax | 15 | 1.0000 | -0.3188 | 1.1325 |
| rxinfer | 15 | 1.0000 | 0.0693 | 0.0000 |
| activeinference_jl | 15 | 1.0000 | 0.0693 | 0.0000 |

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
- **numpyro_vs_rxinfer**: different dimensions ([15, 4] vs [15, 3])
- **numpyro_vs_activeinference_jl**: different dimensions ([15, 4] vs [15, 3])
- **pymdp_vs_pytorch**: confidence correlation = 0.0000
- **pymdp_vs_jax**: confidence correlation = 0.0000
- **pymdp_vs_rxinfer**: different dimensions ([15, 4] vs [15, 3])
- **pymdp_vs_activeinference_jl**: different dimensions ([15, 4] vs [15, 3])
- **pytorch_vs_jax**: confidence correlation = -0.0690
- **pytorch_vs_rxinfer**: different dimensions ([15, 4] vs [15, 3])
- **pytorch_vs_activeinference_jl**: different dimensions ([15, 4] vs [15, 3])
- **jax_vs_rxinfer**: different dimensions ([15, 4] vs [15, 3])
- **jax_vs_activeinference_jl**: different dimensions ([15, 4] vs [15, 3])
- **rxinfer_vs_activeinference_jl**: confidence correlation = 0.0000

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| numpyro | 0.776 | 0.160 | 0.635 | 1.191 |
| pymdp | 2.914 | 0.140 | 2.771 | 3.234 |
| pytorch | 0.482 | 0.015 | 0.464 | 0.506 |
| jax | 0.706 | 0.041 | 0.656 | 0.799 |
| discopy | 0.247 | 0.004 | 0.243 | 0.257 |
| bnlearn | 2.039 | 0.093 | 1.972 | 2.288 |
