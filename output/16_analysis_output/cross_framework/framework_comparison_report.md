# Framework Execution Comparison Report

Generated: 2026-03-03T08:41:30.531422

## Summary

- Total Frameworks: 7
- Total Executions: 56
- Successful Executions: 56
- Overall Success Rate: 100.00%

## Framework Details

### NUMPYRO

- Success Rate: 100.0% (8/8)
- Execution Time: 1.67s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, beliefs_sum_to_one=✅, actions_in_range=✅, all_valid=✅)
- Data Source: `output/12_execute_output/markov_chain/numpyro/simulation_data/simulation_results.json`

### PYMDP

- Success Rate: 100.0% (8/8)
- Execution Time: 2.80s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/markov_chain/pymdp/simulation_data/simulation_results.json`

### PYTORCH

- Success Rate: 100.0% (8/8)
- Execution Time: 1.16s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, beliefs_sum_to_one=✅, actions_in_range=✅, all_valid=✅)
- Data Source: `output/12_execute_output/markov_chain/pytorch/simulation_data/simulation_results.json`

### JAX

- Success Rate: 100.0% (8/8)
- Execution Time: 0.99s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/markov_chain/jax/simulation_data/simulation_results.json`

### DISCOPY

- Success Rate: 100.0% (8/8)
- Execution Time: 0.39s
- Data Source: `output/12_execute_output/simple_mdp/discopy/simulation_data/circuit_info.json`

### RXINFER

- Success Rate: 100.0% (8/8)
- Execution Time: 18.88s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, actions_in_range=✅, beliefs_sum_to_one=✅)
- Data Source: `output/12_execute_output/markov_chain/rxinfer/simulation_data/simulation_results.json`

### ACTIVEINFERENCE_JL

- Success Rate: 100.0% (8/8)
- Execution Time: 17.47s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, actions_in_range=✅, all_valid=✅, beliefs_sum_to_one=✅)
- Data Source: `output/12_execute_output/deep_planning_horizon/activeinference_jl/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| numpyro | 15 | 1.0000 | 1.0986 | 0.0000 |
| pymdp | 15 | 1.0000 | -0.2585 | 0.0976 |
| pytorch | 15 | 1.0000 | 1.0986 | 0.0000 |
| jax | 15 | 1.0000 | 0.0693 | 0.0000 |
| rxinfer | 15 | 1.0000 | 0.1309 | 0.1329 |
| activeinference_jl | 15 | 0.5663 | -1.3731 | 0.3075 |

## Data Coverage

| Framework | Beliefs | Actions | Observations | Free Energy | Validation |
|-----------|---------|---------|--------------|-------------|------------|
| numpyro | ✅ | ✅ | ✅ | ✅ | ✅ |
| pymdp | ✅ | ✅ | ✅ | ✅ | ✅ |
| pytorch | ✅ | ✅ | ✅ | ✅ | ✅ |
| jax | ✅ | ✅ | ✅ | ✅ | ✅ |
| discopy | ❌ | ❌ | ❌ | ❌ | — |
| rxinfer | ✅ | ✅ | ✅ | ✅ | ✅ |
| activeinference_jl | ✅ | ✅ | ✅ | ✅ | ✅ |

## Cross-Framework Metric Agreement

- **numpyro_vs_pymdp**: confidence correlation = 0.0000
- **numpyro_vs_pytorch**: confidence correlation = 0.0000
- **numpyro_vs_jax**: confidence correlation = 0.0000
- **numpyro_vs_rxinfer**: confidence correlation = 0.0000
- **numpyro_vs_activeinference_jl**: different dimensions ([15, 3] vs [15, 4])
- **pymdp_vs_pytorch**: confidence correlation = 0.0000
- **pymdp_vs_jax**: confidence correlation = 0.0000
- **pymdp_vs_rxinfer**: confidence correlation = -0.3305
- **pymdp_vs_activeinference_jl**: different dimensions ([15, 3] vs [15, 4])
- **pytorch_vs_jax**: confidence correlation = 0.0000
- **pytorch_vs_rxinfer**: confidence correlation = 0.1148
- **pytorch_vs_activeinference_jl**: different dimensions ([15, 3] vs [15, 4])
- **jax_vs_rxinfer**: confidence correlation = 0.0000
- **jax_vs_activeinference_jl**: different dimensions ([15, 3] vs [15, 4])
- **rxinfer_vs_activeinference_jl**: different dimensions ([15, 3] vs [15, 4])

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| numpyro | 1.075 | 0.233 | 0.885 | 1.667 |
| pymdp | 1.773 | 0.395 | 1.549 | 2.803 |
| pytorch | 0.716 | 0.171 | 0.622 | 1.161 |
| jax | 0.945 | 0.079 | 0.864 | 1.097 |
| discopy | 0.348 | 0.017 | 0.328 | 0.385 |
| rxinfer | 17.974 | 1.326 | 16.026 | 20.085 |
| activeinference_jl | 16.415 | 2.246 | 12.184 | 18.768 |
