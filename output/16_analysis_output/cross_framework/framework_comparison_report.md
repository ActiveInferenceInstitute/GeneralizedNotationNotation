# Framework Execution Comparison Report

Generated: 2026-05-12T07:46:45.100217

## Summary

- Total Frameworks: 8
- Total Executions: 80
- Successful Executions: 70
- Overall Success Rate: 87.50%

## Framework Details

### NUMPYRO

- Success Rate: 0.0% (0/10)
- Timesteps: 40
- Data: beliefs=40, actions=40, observations=40, free_energy=40
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/markov_chain/jax/simulation_data/simulation_results.json`

### PYMDP

- Success Rate: 100.0% (10/10)
- Execution Time: 3.76s
- Timesteps: 25
- Data: beliefs=25, actions=25, observations=25, free_energy=25
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅, pymdp_version_ge_1_0_0=✅)
- Data Source: `output/12_execute_output/simple_mdp/pymdp/simulation_data/simulation_results.json`

### PYTORCH

- Success Rate: 100.0% (10/10)
- Execution Time: 0.53s
- Timesteps: 25
- Data: beliefs=25, actions=25, observations=25, free_energy=25
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, beliefs_sum_to_one=✅, actions_in_range=✅, all_valid=✅)
- Data Source: `output/12_execute_output/simple_mdp/pytorch/simulation_data/simulation_results.json`

### JAX

- Success Rate: 100.0% (10/10)
- Execution Time: 0.82s
- Timesteps: 25
- Data: beliefs=25, actions=25, observations=25, free_energy=25
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/simple_mdp/jax/simulation_data/simulation_results.json`

### DISCOPY

- Success Rate: 100.0% (10/10)
- Execution Time: 0.23s
- Data Source: `output/12_execute_output/simple_mdp/discopy/execution_logs/Simple MDP Agent_discopy.py_results.json`

### BNLEARN

- Success Rate: 100.0% (10/10)
- Execution Time: 1.94s
- Data Source: `output/12_execute_output/simple_mdp/bnlearn/execution_logs/Simple MDP Agent_bnlearn.py_results.json`

### RXINFER

- Success Rate: 100.0% (10/10)
- Execution Time: 19.12s
- Timesteps: 25
- Data: beliefs=25, actions=25, observations=25, free_energy=25
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, actions_in_range=✅, beliefs_sum_to_one=✅)
- Data Source: `output/12_execute_output/simple_mdp/rxinfer/simulation_data/simulation_results.json`

### ACTIVEINFERENCE_JL

- Success Rate: 100.0% (10/10)
- Execution Time: 14.65s
- Timesteps: 25
- Data: beliefs=25, actions=25, observations=25, free_energy=25
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, actions_in_range=✅, all_valid=✅, beliefs_sum_to_one=✅)
- Data Source: `output/12_execute_output/simple_mdp/activeinference_jl/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| numpyro | 40 | 1.0000 | 0.0693 | 0.0000 |
| pymdp | 25 | 1.0000 | 1.0751 | 1.1325 |
| pytorch | 25 | 1.0000 | 3.1392 | 0.0000 |
| jax | 25 | 1.0000 | -0.3188 | 1.1325 |
| rxinfer | 25 | 1.0000 | 0.1141 | 0.0000 |
| activeinference_jl | 25 | 0.9999 | -2.0638 | 1.1323 |

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

- **numpyro_vs_pymdp**: different dimensions ([40, 3] vs [25, 4])
- **numpyro_vs_pytorch**: different dimensions ([40, 3] vs [25, 4])
- **numpyro_vs_jax**: different dimensions ([40, 3] vs [25, 4])
- **numpyro_vs_rxinfer**: different dimensions ([40, 3] vs [25, 4])
- **numpyro_vs_activeinference_jl**: different dimensions ([40, 3] vs [25, 4])
- **pymdp_vs_pytorch**: confidence correlation = 0.0000
- **pymdp_vs_jax**: confidence correlation = 0.0000
- **pymdp_vs_rxinfer**: confidence correlation = 0.0000
- **pymdp_vs_activeinference_jl**: confidence correlation = 0.0000
- **pytorch_vs_jax**: confidence correlation = -0.0408
- **pytorch_vs_rxinfer**: confidence correlation = 0.2673
- **pytorch_vs_activeinference_jl**: confidence correlation = 0.1178
- **jax_vs_rxinfer**: confidence correlation = -0.0532
- **jax_vs_activeinference_jl**: confidence correlation = -0.0832
- **rxinfer_vs_activeinference_jl**: confidence correlation = -0.0425

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| pymdp | 3.243 | 0.704 | 2.153 | 4.918 |
| pytorch | 0.481 | 0.023 | 0.459 | 0.530 |
| jax | 0.798 | 0.062 | 0.718 | 0.948 |
| discopy | 0.239 | 0.012 | 0.233 | 0.273 |
| bnlearn | 1.902 | 0.023 | 1.877 | 1.940 |
| rxinfer | 13.910 | 1.738 | 13.252 | 19.119 |
| activeinference_jl | 13.012 | 1.559 | 9.889 | 14.651 |
