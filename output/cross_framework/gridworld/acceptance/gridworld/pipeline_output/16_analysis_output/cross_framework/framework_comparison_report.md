# Framework Execution Comparison Report

Generated: 2026-09-06T12:20:03.485185

## Summary

- Total Frameworks: 3
- Total Executions: 3
- Successful Executions: 3
- Overall Success Rate: 100.00%

## Framework Details

### ACTIVEINFERENCE_JL

- Success Rate: 100.0% (1/1)
- Execution Time: 38.53s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (actions_in_range=✅, all_beliefs_valid=✅, all_valid=✅, beliefs_sum_to_one=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/cross_framework/gridworld/acceptance/gridworld/pipeline_output/12_execute_output/pomdp_gridworld_3x3/activeinference_jl/simulation_data/simulation_results.json`

### PYMDP

- Success Rate: 100.0% (1/1)
- Execution Time: 11.85s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅, pymdp_version_ge_1_0_0=✅, all_valid=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/cross_framework/gridworld/acceptance/gridworld/pipeline_output/12_execute_output/pomdp_gridworld_3x3/pymdp/simulation_data/simulation_results.json`

### RXINFER

- Success Rate: 100.0% (1/1)
- Execution Time: 56.18s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (actions_in_range=✅, all_beliefs_valid=✅, all_valid=✅, belief_accuracy=✅, belief_accuracy_ok=✅, belief_entropy_max=✅, belief_entropy_mean=✅, belief_entropy_min=✅, belief_entropy_ok=✅, beliefs_sum_to_one=✅, inference_converged=✅, vfe_present=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/cross_framework/gridworld/acceptance/gridworld/pipeline_output/12_execute_output/pomdp_gridworld_3x3/rxinfer/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| activeinference_jl | 15 | 1.0000 | 2.4407 | 1.1292 |
| pymdp | 15 | 1.0000 | 1.4964 | 1.0264 |
| rxinfer | 15 | 1.0000 | 2.4407 | 1.1292 |

## Data Coverage

| Framework | Beliefs | Actions | Observations | Free Energy | Validation |
|-----------|---------|---------|--------------|-------------|------------|
| activeinference_jl | ✅ | ✅ | ✅ | ✅ | ✅ |
| pymdp | ✅ | ✅ | ✅ | ✅ | ✅ |
| rxinfer | ✅ | ✅ | ✅ | ✅ | ✅ |

## Cross-Framework Metric Agreement

- **activeinference_jl_vs_pymdp**: confidence correlation = 0.0000
- **activeinference_jl_vs_rxinfer**: confidence correlation = 0.0000
- **pymdp_vs_rxinfer**: confidence correlation = 0.0000

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| activeinference_jl | 38.526 | 0.000 | 38.526 | 38.526 |
| pymdp | 11.846 | 0.000 | 11.846 | 11.846 |
| rxinfer | 56.176 | 0.000 | 56.176 | 56.176 |
