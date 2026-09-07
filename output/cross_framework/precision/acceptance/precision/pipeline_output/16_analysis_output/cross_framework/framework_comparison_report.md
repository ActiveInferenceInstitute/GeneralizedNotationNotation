# Framework Execution Comparison Report

Generated: 2026-09-06T12:14:57.335355

## Summary

- Total Frameworks: 1
- Total Executions: 2
- Successful Executions: 2
- Overall Success Rate: 100.00%

## Framework Details

### PYMDP

- Success Rate: 100.0% (2/2)
- Execution Time: 28.68s
- Timesteps: 30
- Data: beliefs=30, actions=30, observations=30, free_energy=30
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅, pymdp_version_ge_1_0_0=✅, all_valid=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/cross_framework/precision/acceptance/precision/pipeline_output/12_execute_output/precision_weighted/pymdp/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| pymdp | 30 | 0.9756 | 0.6344 | 0.3552 |

## Data Coverage

| Framework | Beliefs | Actions | Observations | Free Energy | Validation |
|-----------|---------|---------|--------------|-------------|------------|
| pymdp | ✅ | ✅ | ✅ | ✅ | ✅ |

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| pymdp | 28.728 | 0.045 | 28.683 | 28.773 |
