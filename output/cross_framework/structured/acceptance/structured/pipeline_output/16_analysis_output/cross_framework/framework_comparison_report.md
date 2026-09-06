# Framework Execution Comparison Report

Generated: 2026-09-06T12:16:28.200027

## Summary

- Total Frameworks: 1
- Total Executions: 1
- Successful Executions: 1
- Overall Success Rate: 100.00%

## Framework Details

### PYMDP

- Success Rate: 100.0% (1/1)
- Execution Time: 9.84s
- Timesteps: 15
- Data: beliefs=15, actions=15, observations=15, free_energy=15
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅, pymdp_version_ge_1_0_0=✅, all_valid=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/cross_framework/structured/acceptance/structured/pipeline_output/12_execute_output/factorized_posterior/pymdp/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| pymdp | 15 | 0.7479 | 1.1809 | 0.2404 |

## Data Coverage

| Framework | Beliefs | Actions | Observations | Free Energy | Validation |
|-----------|---------|---------|--------------|-------------|------------|
| pymdp | ✅ | ✅ | ✅ | ✅ | ✅ |

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| pymdp | 9.838 | 0.000 | 9.838 | 9.838 |
