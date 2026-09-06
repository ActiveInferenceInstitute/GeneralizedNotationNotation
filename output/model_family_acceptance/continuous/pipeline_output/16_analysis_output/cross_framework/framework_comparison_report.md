# Framework Execution Comparison Report

Generated: 2026-09-06T11:17:09.863958

## Summary

- Total Frameworks: 4
- Total Executions: 8
- Successful Executions: 8
- Overall Success Rate: 100.00%

## Framework Details

### JAX

- Success Rate: 100.0% (2/2)
- Execution Time: 3.41s
- Timesteps: 15
- Data: beliefs=15, actions=0, observations=0, free_energy=0
- Validation: ✅ ALL PASSED (means_finite=✅, posterior_cov_psd=✅, rmse_finite=✅, controls_finite=✅, all_valid=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/model_family_acceptance/continuous/pipeline_output/12_execute_output/stochastic_dynamics/jax/simulation_data/simulation_results.json`

### NUMPYRO

- Success Rate: 100.0% (2/2)
- Execution Time: 12.07s
- Timesteps: 15
- Data: beliefs=15, actions=0, observations=0, free_energy=0
- Validation: ✅ ALL PASSED (means_finite=✅, posterior_cov_psd=✅, rmse_finite=✅, controls_finite=✅, mcmc_finite=✅, all_valid=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/model_family_acceptance/continuous/pipeline_output/12_execute_output/stochastic_dynamics/numpyro/simulation_data/simulation_results.json`

### RXINFER

- Success Rate: 100.0% (2/2)
- Execution Time: 62.37s
- Timesteps: 15
- Data: beliefs=15, actions=0, observations=0, free_energy=0
- Validation: ✅ ALL PASSED (all_valid=✅, inference_converged=✅, means_finite=✅, posterior_cov_psd=✅, rmse_finite=✅, rmse_vs_true=✅, vfe_finite=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/model_family_acceptance/continuous/pipeline_output/12_execute_output/stochastic_dynamics/rxinfer/simulation_data/simulation_results.json`

### STAN

- Success Rate: 100.0% (2/2)
- Execution Time: 60.16s
- Timesteps: 15
- Data: beliefs=15, actions=0, observations=0, free_energy=0
- Validation: ✅ ALL PASSED (means_finite=✅, posterior_cov_psd=✅, rmse_finite=✅, controls_finite=✅, rhat_ok=✅, all_valid=✅)
- Data Source: `/private/tmp/claude-501/-Users-hum-Documents-GitHub-HumOS/61e99712-2be6-42f2-9983-d73d6ffca46e/scratchpad/wt/GeneralizedNotationNotation/output/model_family_acceptance/continuous/pipeline_output/12_execute_output/stochastic_dynamics/stan/simulation_data/simulation_results.json`

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
| jax | 4.797 | 1.391 | 3.406 | 6.188 |
| numpyro | 13.479 | 1.412 | 12.067 | 14.891 |
| rxinfer | 79.614 | 17.245 | 62.368 | 96.859 |
| stan | 59.112 | 1.049 | 58.063 | 60.161 |
