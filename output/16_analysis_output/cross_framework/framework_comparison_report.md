# Framework Execution Comparison Report

Generated: 2026-05-22T06:34:44.799793

## Summary

- Total Frameworks: 8
- Total Executions: 80
- Successful Executions: 70
- Overall Success Rate: 87.50%

## Framework Details

### NUMPYRO

- Success Rate: 0.0% (0/10)
- Timesteps: 40
- Data: beliefs=40, actions=40, observations=40, free_energy=40
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/markov_chain/jax/simulation_data/simulation_results.json`

### PYMDP

- Success Rate: 100.0% (10/10)
- Execution Time: 3.35s
- Timesteps: 25
- Data: beliefs=25, actions=25, observations=25, free_energy=25
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅, pymdp_version_ge_1_0_0=✅, all_valid=✅)
- Data Source: `output/12_execute_output/simple_mdp/pymdp/simulation_data/simulation_results.json`

### PYTORCH

- Success Rate: 100.0% (10/10)
- Execution Time: 0.50s
- Timesteps: 25
- Data: beliefs=25, actions=25, observations=25, free_energy=25
- Validation: ⚠️ ISSUES (beliefs_in_range=✅, beliefs_sum_to_one=❌, actions_in_range=✅, all_valid=❌)
- Data Source: `output/12_execute_output/simple_mdp/pytorch/simulation_data/simulation_results.json`

### JAX

- Success Rate: 100.0% (10/10)
- Execution Time: 0.87s
- Timesteps: 25
- Data: beliefs=25, actions=25, observations=25, free_energy=25
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, beliefs_sum_to_one=✅, actions_in_range=✅)
- Data Source: `output/12_execute_output/simple_mdp/jax/simulation_data/simulation_results.json`

### DISCOPY

- Success Rate: 100.0% (10/10)
- Execution Time: 0.24s
- Data Source: `output/12_execute_output/simple_mdp/discopy/execution_logs/Simple MDP Agent_discopy.py_results.json`

### BNLEARN

- Success Rate: 100.0% (10/10)
- Execution Time: 2.04s
- Data Source: `output/12_execute_output/simple_mdp/bnlearn/execution_logs/Simple MDP Agent_bnlearn.py_results.json`

### RXINFER

- Success Rate: 100.0% (10/10)
- Execution Time: 5.54s
- Timesteps: 25
- Data: beliefs=25, actions=25, observations=25, free_energy=25
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, actions_in_range=✅, all_valid=✅, beliefs_sum_to_one=✅)
- Data Source: `output/12_execute_output/simple_mdp/rxinfer/simulation_data/simulation_results.json`

### ACTIVEINFERENCE_JL

- Success Rate: 100.0% (10/10)
- Execution Time: 7.84s
- Timesteps: 25
- Data: beliefs=25, actions=25, observations=25, free_energy=25
- Validation: ✅ ALL PASSED (all_beliefs_valid=✅, actions_in_range=✅, all_valid=✅, beliefs_sum_to_one=✅)
- Data Source: `output/12_execute_output/simple_mdp/activeinference_jl/simulation_data/simulation_results.json`

## Simulation Data Comparison

| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |
|-----------|-----------|-----------------|----------|---------|
| numpyro | 40 | 1.0000 | 0.9617 | 0.1037 |
| pymdp | 25 | 1.0000 | 1.0751 | 1.1325 |
| pytorch | 25 | 0.0400 | 0.0826 | 0.4636 |
| jax | 25 | 1.0000 | -0.3188 | 1.1325 |
| rxinfer | 25 | 0.9000 | 0.1141 | 0.0000 |
| activeinference_jl | 25 | 0.9000 | 0.1141 | 0.0000 |

## Data Coverage

| Framework | Beliefs | Actions | Observations | Free Energy | Validation |
|-----------|---------|---------|--------------|-------------|------------|
| numpyro | ✅ | ✅ | ✅ | ✅ | ✅ |
| pymdp | ✅ | ✅ | ✅ | ✅ | ✅ |
| pytorch | ✅ | ✅ | ✅ | ✅ | ❌ |
| jax | ✅ | ✅ | ✅ | ✅ | ✅ |
| discopy | ❌ | ❌ | ❌ | ❌ | — |
| bnlearn | ❌ | ❌ | ❌ | ❌ | — |
| rxinfer | ✅ | ✅ | ✅ | ✅ | ✅ |
| activeinference_jl | ✅ | ✅ | ✅ | ✅ | ✅ |

## Cross-Framework Metric Agreement

- **numpyro_vs_pymdp**: different dimensions ([40, 3] vs [25, 4])
- **numpyro_vs_pytorch**: different dimensions ([40, 3] vs [25, 4])
- **numpyro_vs_jax**: different dimensions ([40, 3] vs [25, 4])
- **numpyro_vs_rxinfer**: different dimensions ([40, 3] vs [25, 4])
- **numpyro_vs_activeinference_jl**: different dimensions ([40, 3] vs [25, 4])
- **pymdp_vs_pytorch**: confidence correlation = 0.0000
- **pymdp_vs_jax**: confidence correlation = 0.0000
- **pymdp_vs_rxinfer**: confidence correlation = 0.0000
- **pymdp_vs_activeinference_jl**: confidence correlation = 0.0000
- **pytorch_vs_jax**: confidence correlation = 0.0417
- **pytorch_vs_rxinfer**: confidence correlation = 0.0000
- **pytorch_vs_activeinference_jl**: confidence correlation = 0.0000
- **jax_vs_rxinfer**: confidence correlation = 0.0000
- **jax_vs_activeinference_jl**: confidence correlation = 0.0000
- **rxinfer_vs_activeinference_jl**: confidence correlation = 0.0000

## Performance Comparison

| Framework | Mean Time (s) | Std Dev | Min | Max |
|-----------|---------------|---------|-----|-----|
| pymdp | 2.710 | 0.714 | 1.704 | 4.336 |
| pytorch | 0.512 | 0.037 | 0.476 | 0.598 |
| jax | 0.861 | 0.073 | 0.785 | 1.004 |
| discopy | 0.242 | 0.004 | 0.233 | 0.249 |
| bnlearn | 2.065 | 0.047 | 2.029 | 2.198 |
| rxinfer | 5.442 | 0.035 | 5.417 | 5.540 |
| activeinference_jl | 7.846 | 0.019 | 7.819 | 7.879 |
