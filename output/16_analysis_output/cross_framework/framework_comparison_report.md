# Framework Execution Comparison Report

Generated: 2026-02-23T07:07:47.657552

## Summary

- Total Frameworks: 5
- Total Executions: 5
- Successful Executions: 5
- Overall Success Rate: 100.00%

## Framework Details

### PYMDP

- Success Rate: 100.0% (1/1)
- Execution Time: 1.74s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/actinf_pomdp_agent/pymdp/simulation_data/simulation_results.json`

### JAX

- Success Rate: 100.0% (1/1)
- Execution Time: 1.15s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/actinf_pomdp_agent/jax/simulation_data/simulation_results.json`

### DISCOPY

- Success Rate: 100.0% (1/1)
- Execution Time: 0.37s
- Data Source: `output/12_execute_output/actinf_pomdp_agent/discopy/simulation_data/circuit_info.json`

### RXINFER

- Success Rate: 100.0% (1/1)
- Execution Time: 19.40s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, actions_in_range=✅, beliefs_sum_to_one=✅)
- Data Source: `output/12_execute_output/actinf_pomdp_agent/rxinfer/simulation_data/simulation_results.json`

### ACTIVEINFERENCE_JL

- Success Rate: 100.0% (1/1)
- Execution Time: 19.10s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (beliefs_in_range=✅, actions_in_range=✅, all_valid=✅, beliefs_sum_to_one=✅)
- Data Source: `output/12_execute_output/actinf_pomdp_agent/activeinference_jl/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| pymdp | 15 | 0.9929 | -1.3471 | 0.2422 |
| jax | 15 | 0.9929 | 0.0458 | 0.3874 |
| rxinfer | 15 | 0.9929 | 0.7186 | 0.1284 |
| activeinference_jl | 15 | 0.9450 | -0.9795 | 0.3414 |

## Data Coverage

| Framework | Beliefs | Actions | Observations | Free Energy | Validation |
|-----------|---------|---------|--------------|-------------|------------|
| pymdp | ✅ | ✅ | ✅ | ✅ | ✅ |
| jax | ✅ | ✅ | ✅ | ✅ | ✅ |
| discopy | ❌ | ❌ | ❌ | ❌ | — |
| rxinfer | ✅ | ✅ | ✅ | ✅ | ✅ |
| activeinference_jl | ✅ | ✅ | ✅ | ✅ | ✅ |

## Cross-Framework Metric Agreement

- **pymdp_vs_jax**: confidence correlation = 1.0000
- **pymdp_vs_rxinfer**: confidence correlation = 1.0000
- **pymdp_vs_activeinference_jl**: confidence correlation = 0.8910
- **jax_vs_rxinfer**: confidence correlation = 1.0000
- **jax_vs_activeinference_jl**: confidence correlation = 0.8910
- **rxinfer_vs_activeinference_jl**: confidence correlation = 0.8910

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| pymdp | 1.741 | 0.000 | 1.741 | 1.741 |
| jax | 1.154 | 0.000 | 1.154 | 1.154 |
| discopy | 0.374 | 0.000 | 0.374 | 0.374 |
| rxinfer | 19.398 | 0.000 | 19.398 | 19.398 |
| activeinference_jl | 19.104 | 0.000 | 19.104 | 19.104 |
