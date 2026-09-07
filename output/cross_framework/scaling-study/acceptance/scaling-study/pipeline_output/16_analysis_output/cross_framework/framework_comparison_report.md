# Framework Execution Comparison Report

Generated: 2026-09-06T12:23:31.797327

## Summary

- Total Frameworks: 1
- Total Executions: 2
- Successful Executions: 2
- Overall Success Rate: 100.00%

## Framework Details

### PYMDP

- Success Rate: 100.0% (2/2)
- Execution Time: 42.56s
- Timesteps: 100
- Data: beliefs=100, actions=100, observations=100, free_energy=100
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅, pymdp_version_ge_1_0_0=✅, all_valid=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/cross_framework/scaling-study/acceptance/scaling-study/pipeline_output/12_execute_output/pymdp_scaling_N8_T100/pymdp/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| pymdp | 100 | 0.9134 | 0.9415 | 0.7144 |

## Data Coverage

| Framework | Beliefs | Actions | Observations | Free Energy | Validation |
|-----------|---------|---------|--------------|-------------|------------|
| pymdp | ✅ | ✅ | ✅ | ✅ | ✅ |

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| pymdp | 44.071 | 1.509 | 42.563 | 45.580 |
