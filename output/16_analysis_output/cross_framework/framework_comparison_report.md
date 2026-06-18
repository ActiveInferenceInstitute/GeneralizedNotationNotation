# Framework Execution Comparison Report

Generated: 2026-06-18T09:10:41.867729

## Summary

- Total Frameworks: 6
- Total Executions: 6
- Successful Executions: 6
- Overall Success Rate: 100.00%

## Framework Details

### NUMPYRO

- Success Rate: 100.0% (1/1)
- Execution Time: 1.08s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, beliefs_sum_to_one=✅, actions_in_range=✅, all_valid=✅)
- Data Source: `/Users/4d/Documents/GitHub/GeneralizedNotationNotation/output/12_execute_output/pomdp_gridworld_3x3/numpyro/simulation_data/simulation_results.json`

### PYMDP

- Success Rate: 100.0% (1/1)
- Execution Time: 2.70s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅, pymdp_version_ge_1_0_0=✅, all_valid=✅)
- Data Source: `/Users/4d/Documents/GitHub/GeneralizedNotationNotation/output/12_execute_output/pomdp_gridworld_3x3/pymdp/simulation_data/simulation_results.json`

### JAX

- Success Rate: 100.0% (1/1)
- Execution Time: 0.99s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `/Users/4d/Documents/GitHub/GeneralizedNotationNotation/output/12_execute_output/pomdp_gridworld_3x3/jax/simulation_data/simulation_results.json`

### DISCOPY

- Success Rate: 100.0% (1/1)
- Execution Time: 0.30s
- Data Source: `/Users/4d/Documents/GitHub/GeneralizedNotationNotation/output/12_execute_output/pomdp_gridworld_3x3/discopy/execution_logs/POMDP GridWorld 3x3_discopy.py_results.json`

### RXINFER

- Success Rate: 100.0% (1/1)
- Execution Time: 6.35s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, actions_in_range=✅, all_valid=✅, beliefs_sum_to_one=✅)
- Data Source: `/Users/4d/Documents/GitHub/GeneralizedNotationNotation/output/12_execute_output/pomdp_gridworld_3x3/rxinfer/simulation_data/simulation_results.json`

### ACTIVEINFERENCE_JL

- Success Rate: 100.0% (1/1)
- Execution Time: 9.35s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, actions_in_range=✅, all_valid=✅, beliefs_sum_to_one=✅)
- Data Source: `/Users/4d/Documents/GitHub/GeneralizedNotationNotation/output/12_execute_output/pomdp_gridworld_3x3/activeinference_jl/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| numpyro | 15 | 1.0000 | 3.3218 | 0.0407 |
| pymdp | 15 | 1.0000 | 1.4964 | 1.0264 |
| jax | 15 | 1.0000 | 1.3295 | 0.8617 |
| rxinfer | 15 | 1.0000 | 2.4407 | 1.1292 |
| activeinference_jl | 15 | 1.0000 | 2.4407 | 1.1292 |

## Data Coverage

| Framework | Beliefs | Actions | Observations | Free Energy | Validation |
|-----------|---------|---------|--------------|-------------|------------|
| numpyro | ✅ | ✅ | ✅ | ✅ | ✅ |
| pymdp | ✅ | ✅ | ✅ | ✅ | ✅ |
| jax | ✅ | ✅ | ✅ | ✅ | ✅ |
| discopy | ❌ | ❌ | ❌ | ❌ | — |
| rxinfer | ✅ | ✅ | ✅ | ✅ | ✅ |
| activeinference_jl | ✅ | ✅ | ✅ | ✅ | ✅ |

## Cross-Framework Metric Agreement

- **numpyro_vs_pymdp**: confidence correlation = 0.0000
- **numpyro_vs_jax**: confidence correlation = 0.0000
- **numpyro_vs_rxinfer**: confidence correlation = 0.0000
- **numpyro_vs_activeinference_jl**: confidence correlation = 0.0000
- **pymdp_vs_jax**: confidence correlation = 0.0000
- **pymdp_vs_rxinfer**: confidence correlation = 0.0000
- **pymdp_vs_activeinference_jl**: confidence correlation = 0.0000
- **jax_vs_rxinfer**: confidence correlation = 0.0000
- **jax_vs_activeinference_jl**: confidence correlation = 0.0000
- **rxinfer_vs_activeinference_jl**: confidence correlation = 0.0000

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| numpyro | 1.077 | 0.000 | 1.077 | 1.077 |
| pymdp | 2.698 | 0.000 | 2.698 | 2.698 |
| jax | 0.993 | 0.000 | 0.993 | 0.993 |
| discopy | 0.298 | 0.000 | 0.298 | 0.298 |
| rxinfer | 6.349 | 0.000 | 6.349 | 6.349 |
| activeinference_jl | 9.351 | 0.000 | 9.351 | 9.351 |
