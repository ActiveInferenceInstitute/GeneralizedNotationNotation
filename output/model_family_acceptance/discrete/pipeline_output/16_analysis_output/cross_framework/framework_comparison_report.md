# Framework Execution Comparison Report

Generated: 2026-09-06T11:09:29.744317

## Summary

- Total Frameworks: 1
- Total Executions: 2
- Successful Executions: 2
- Overall Success Rate: 100.00%

## Framework Details

### PYMDP

- Success Rate: 100.0% (2/2)
- Execution Time: 15.47s
- Timesteps: 25
- Data: beliefs=25, actions=25, observations=25, free_energy=25
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅, pymdp_version_ge_1_0_0=✅, all_valid=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/model_family_acceptance/discrete/pipeline_output/12_execute_output/simple_mdp/pymdp/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| pymdp | 25 | 1.0000 | 1.0751 | 1.1325 |

## Data Coverage

| Framework | Beliefs | Actions | Observations | Free Energy | Validation |
|-----------|---------|---------|--------------|-------------|------------|
| pymdp | ✅ | ✅ | ✅ | ✅ | ✅ |

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| pymdp | 14.387 | 1.084 | 13.303 | 15.471 |
