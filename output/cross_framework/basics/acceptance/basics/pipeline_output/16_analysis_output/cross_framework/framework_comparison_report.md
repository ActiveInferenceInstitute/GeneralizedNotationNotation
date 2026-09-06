# Framework Execution Comparison Report

Generated: 2026-09-06T11:46:22.481151

## Summary

- Total Frameworks: 1
- Total Executions: 1
- Successful Executions: 1
- Overall Success Rate: 100.00%

## Framework Details

### PYMDP

- Success Rate: 100.0% (1/1)
- Execution Time: 27.01s
- Timesteps: 10
- Data: beliefs=10, actions=10, observations=10, free_energy=10
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅, pymdp_version_ge_1_0_0=✅, all_valid=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/cross_framework/basics/acceptance/basics/pipeline_output/12_execute_output/dynamic_perception/pymdp/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| pymdp | 10 | 0.8261 | 0.2618 | 0.0073 |

## Data Coverage

| Framework | Beliefs | Actions | Observations | Free Energy | Validation |
|-----------|---------|---------|--------------|-------------|------------|
| pymdp | ✅ | ✅ | ✅ | ✅ | ✅ |

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| pymdp | 27.012 | 0.000 | 27.012 | 27.012 |
