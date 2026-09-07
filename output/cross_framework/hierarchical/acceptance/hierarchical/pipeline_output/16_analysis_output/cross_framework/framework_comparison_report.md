# Framework Execution Comparison Report

Generated: 2026-09-06T12:05:51.454221

## Summary

- Total Frameworks: 3
- Total Executions: 6
- Successful Executions: 6
- Overall Success Rate: 100.00%

## Framework Details

### JAX

- Success Rate: 100.0% (2/2)
- Execution Time: 4.62s
- Timesteps: 100
- Data: beliefs=100, actions=100, observations=100, free_energy=100
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/cross_framework/hierarchical/acceptance/hierarchical/pipeline_output/12_execute_output/temporal_hierarchy/jax/simulation_data/simulation_results.json`

### PYMDP

- Success Rate: 100.0% (2/2)
- Execution Time: 12.90s
- Timesteps: 100
- Data: beliefs=100, actions=100, observations=100, free_energy=100
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅, pymdp_version_ge_1_0_0=✅, all_valid=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/cross_framework/hierarchical/acceptance/hierarchical/pipeline_output/12_execute_output/temporal_hierarchy/pymdp/simulation_data/simulation_results.json`

### RXINFER

- Success Rate: 100.0% (2/2)
- Execution Time: 44.17s
- Timesteps: 100
- Data: beliefs=100, actions=100, observations=100, free_energy=100
- Validation: ✅ ALL PASSED (actions_in_range=✅, all_beliefs_valid=✅, all_valid=✅, belief_accuracy=✅, belief_accuracy_ok=✅, belief_entropy_max=✅, belief_entropy_mean=✅, belief_entropy_min=✅, belief_entropy_ok=✅, beliefs_sum_to_one=✅, inference_converged=✅, vfe_present=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/cross_framework/hierarchical/acceptance/hierarchical/pipeline_output/12_execute_output/temporal_hierarchy/rxinfer/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| jax | 100 | 0.6764 | 2.1842 | 0.9871 |
| pymdp | 100 | 0.6322 | 2.6182 | 0.6001 |
| rxinfer | 100 | 0.9103 | 3.6053 | 1.0762 |

## Data Coverage

| Framework | Beliefs | Actions | Observations | Free Energy | Validation |
|-----------|---------|---------|--------------|-------------|------------|
| jax | ✅ | ✅ | ✅ | ✅ | ✅ |
| pymdp | ✅ | ✅ | ✅ | ✅ | ✅ |
| rxinfer | ✅ | ✅ | ✅ | ✅ | ✅ |

## Cross-Framework Metric Agreement

- **jax_vs_pymdp**: confidence correlation = 0.0963
- **jax_vs_rxinfer**: confidence correlation = -0.1287
- **pymdp_vs_rxinfer**: confidence correlation = -0.0225

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| jax | 5.447 | 0.825 | 4.622 | 6.271 |
| pymdp | 27.636 | 14.731 | 12.905 | 42.367 |
| rxinfer | 44.921 | 0.751 | 44.170 | 45.671 |
