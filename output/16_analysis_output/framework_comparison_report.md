# Framework Execution Comparison Report

Generated: 2026-02-15T14:32:42.627444

## Summary

- Total Frameworks: 5
- Total Executions: 5
- Successful Executions: 5
- Overall Success Rate: 100.00%

## Framework Details

### JAX

- Success Rate: 100.0% (1/1)
- Execution Time: 1.26s
- Timesteps: 30
- Data: beliefs=30, actions=30, observations=0, free_energy=30
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/actinf_pomdp_agent/jax/simulation_data/simulation_results.json`

### DISCOPY

- Success Rate: 100.0% (1/1)
- Execution Time: 0.59s

### PYMDP_GEN

- Success Rate: 100.0% (1/1)
- Execution Time: 1.39s
- Timesteps: 30
- Data: beliefs=30, actions=30, observations=30, free_energy=30
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/actinf_pomdp_agent/pymdp_gen/simulation_data/simulation_results.json`

### RXINFER

- Success Rate: 100.0% (1/1)
- Execution Time: 24.50s
- Timesteps: 30
- Data: beliefs=30, actions=0, observations=30, free_energy=0
- Data Source: `output/12_execute_output/actinf_pomdp_agent/rxinfer/simulation_data/simulation_results.json`

### ACTIVEINFERENCE_JL

- Success Rate: 100.0% (1/1)
- Execution Time: 28.53s

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| jax | 30 | 0.7770 | -0.0021 | 0.2995 |
| pymdp_gen | 30 | 0.9776 | -0.6956 | 0.0967 |
| rxinfer | 30 | 0.9474 | N/A | N/A |

## Data Coverage

| Framework | Beliefs | Actions | Observations | Free Energy | Validation |
|-----------|---------|---------|--------------|-------------|------------|
| jax | ✅ | ✅ | ❌ | ✅ | ✅ |
| discopy | ❌ | ❌ | ❌ | ❌ | — |
| pymdp_gen | ✅ | ✅ | ✅ | ✅ | ✅ |
| rxinfer | ✅ | ❌ | ✅ | ❌ | — |
| activeinference_jl | ❌ | ❌ | ❌ | ❌ | — |

## Cross-Framework Metric Agreement

- **jax_vs_pymdp_gen**: confidence correlation = 0.2202
- **jax_vs_rxinfer**: confidence correlation = -0.0091
- **pymdp_gen_vs_rxinfer**: confidence correlation = -0.4377

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| jax | 1.258 | 0.000 | 1.258 | 1.258 |
| discopy | 0.590 | 0.000 | 0.590 | 0.590 |
| pymdp_gen | 1.388 | 0.000 | 1.388 | 1.388 |
| rxinfer | 24.499 | 0.000 | 24.499 | 24.499 |
| activeinference_jl | 28.530 | 0.000 | 28.530 | 28.530 |
