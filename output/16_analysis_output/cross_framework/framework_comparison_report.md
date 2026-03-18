# Framework Execution Comparison Report

Generated: 2026-03-17T16:54:41.663505

## Summary

- Total Frameworks: 7
- Total Executions: 56
- Successful Executions: 24
- Overall Success Rate: 42.86%

## Framework Details

### NUMPYRO

- Success Rate: 0.0% (0/8)
- Execution Time: 0.09s

### PYMDP

- Success Rate: 100.0% (8/8)
- Execution Time: 6.23s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/markov_chain/pymdp/simulation_data/simulation_results.json`

### PYTORCH

- Success Rate: 0.0% (0/8)
- Execution Time: 0.26s

### JAX

- Success Rate: 0.0% (0/8)
- Execution Time: 0.05s

### DISCOPY

- Success Rate: 0.0% (0/8)
- Execution Time: 0.14s

### RXINFER

- Success Rate: 100.0% (8/8)
- Execution Time: 22.15s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, actions_in_range=✅, beliefs_sum_to_one=✅)
- Data Source: `output/12_execute_output/markov_chain/rxinfer/simulation_data/simulation_results.json`

### ACTIVEINFERENCE_JL

- Success Rate: 100.0% (8/8)
- Execution Time: 19.23s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, actions_in_range=✅, all_valid=✅, beliefs_sum_to_one=✅)
- Data Source: `output/12_execute_output/deep_planning_horizon/activeinference_jl/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| pymdp | 15 | 1.0000 | -0.1627 | 0.1298 |
| rxinfer | 15 | 1.0000 | 0.1309 | 0.1329 |
| activeinference_jl | 15 | 0.5663 | -1.3731 | 0.3075 |

## Data Coverage

| Framework | Beliefs | Actions | Observations | Free Energy | Validation |
|-----------|---------|---------|--------------|-------------|------------|
| numpyro | ❌ | ❌ | ❌ | ❌ | — |
| pymdp | ✅ | ✅ | ✅ | ✅ | ✅ |
| pytorch | ❌ | ❌ | ❌ | ❌ | — |
| jax | ❌ | ❌ | ❌ | ❌ | — |
| discopy | ❌ | ❌ | ❌ | ❌ | — |
| rxinfer | ✅ | ✅ | ✅ | ✅ | ✅ |
| activeinference_jl | ✅ | ✅ | ✅ | ✅ | ✅ |

## Cross-Framework Metric Agreement

- **pymdp_vs_rxinfer**: confidence correlation = -0.0034
- **pymdp_vs_activeinference_jl**: different dimensions ([15, 3] vs [15, 4])
- **rxinfer_vs_activeinference_jl**: different dimensions ([15, 3] vs [15, 4])

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| numpyro | 0.048 | 0.028 | 0.027 | 0.100 |
| pymdp | 2.767 | 2.306 | 1.147 | 7.022 |
| pytorch | 0.066 | 0.075 | 0.027 | 0.259 |
| jax | 0.031 | 0.015 | 0.020 | 0.063 |
| discopy | 0.082 | 0.047 | 0.048 | 0.181 |
| rxinfer | 18.810 | 5.518 | 15.375 | 32.350 |
| activeinference_jl | 16.749 | 3.702 | 11.430 | 24.447 |
