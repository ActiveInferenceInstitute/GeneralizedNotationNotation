# Framework Execution Comparison Report

Generated: 2026-09-06T11:57:22.339434

## Summary

- Total Frameworks: 4
- Total Executions: 8
- Successful Executions: 8
- Overall Success Rate: 100.00%

## Framework Details

### JAX

- Success Rate: 100.0% (2/2)
- Execution Time: 4.83s
- Timesteps: 15
- Data: beliefs=15, actions=0, observations=0, free_energy=0
- Validation: ✅ ALL PASSED (means_finite=✅, posterior_cov_psd=✅, rmse_finite=✅, controls_finite=✅, all_valid=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/cross_framework/continuous/acceptance/continuous/pipeline_output/12_execute_output/stochastic_dynamics/jax/simulation_data/simulation_results.json`

### NUMPYRO

- Success Rate: 100.0% (2/2)
- Execution Time: 16.59s
- Timesteps: 15
- Data: beliefs=15, actions=0, observations=0, free_energy=0
- Validation: ✅ ALL PASSED (means_finite=✅, posterior_cov_psd=✅, rmse_finite=✅, controls_finite=✅, mcmc_finite=✅, all_valid=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/cross_framework/continuous/acceptance/continuous/pipeline_output/12_execute_output/stochastic_dynamics/numpyro/simulation_data/simulation_results.json`

### RXINFER

- Success Rate: 100.0% (2/2)
- Execution Time: 70.72s
- Timesteps: 15
- Data: beliefs=15, actions=0, observations=0, free_energy=0
- Validation: ✅ ALL PASSED (all_valid=✅, inference_converged=✅, means_finite=✅, posterior_cov_psd=✅, rmse_finite=✅, rmse_vs_true=✅, vfe_finite=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/cross_framework/continuous/acceptance/continuous/pipeline_output/12_execute_output/stochastic_dynamics/rxinfer/simulation_data/simulation_results.json`

### STAN

- Success Rate: 100.0% (2/2)
- Execution Time: 52.15s
- Timesteps: 15
- Data: beliefs=15, actions=0, observations=0, free_energy=0
- Validation: ✅ ALL PASSED (means_finite=✅, posterior_cov_psd=✅, rmse_finite=✅, controls_finite=✅, rhat_ok=✅, all_valid=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/cross_framework/continuous/acceptance/continuous/pipeline_output/12_execute_output/stochastic_dynamics/stan/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| jax | 15 | 1.0727 | N/A | N/A |
| numpyro | 15 | 1.0727 | N/A | N/A |
| rxinfer | 15 | 0.0775 | N/A | N/A |
| stan | 15 | -0.2411 | N/A | N/A |

## Data Coverage

| Framework | Beliefs | Actions | Observations | Free Energy | Validation |
|-----------|---------|---------|--------------|-------------|------------|
| jax | ✅ | ❌ | ❌ | ❌ | ✅ |
| numpyro | ✅ | ❌ | ❌ | ❌ | ✅ |
| rxinfer | ✅ | ❌ | ❌ | ❌ | ✅ |
| stan | ✅ | ❌ | ❌ | ❌ | ✅ |

## Cross-Framework Metric Agreement

- **jax_vs_numpyro**: confidence correlation = 1.0000
- **jax_vs_rxinfer**: confidence correlation = -0.4758
- **jax_vs_stan**: confidence correlation = 0.1295
- **numpyro_vs_rxinfer**: confidence correlation = -0.4758
- **numpyro_vs_stan**: confidence correlation = 0.1295
- **rxinfer_vs_stan**: confidence correlation = -0.3153

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| jax | 3.954 | 0.873 | 3.081 | 4.827 |
| numpyro | 13.488 | 3.104 | 10.384 | 16.592 |
| rxinfer | 58.954 | 11.763 | 47.191 | 70.717 |
| stan | 52.796 | 0.647 | 52.149 | 53.443 |
