# GNN Script Execution Report

**Generated:** 2026-06-18T09:04:52.080655
**Target Directory:** input/gnn_files/pomdp_gridworld
**Output Directory:** output/12_execute_output

## Summary

- **Total Scripts Found:** 8
- **Successful Executions:** 6
- **Failed Executions:** 0
- **Skipped (dependency not installed):** 2

## Execution Details

### POMDP GridWorld 3x3_numpyro.py - ✅ SUCCESS

- **Framework:** numpyro
- **Executor:** /Users/4d/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python
- **Path:** `output/11_render_output/pomdp_gridworld_3x3/numpyro/POMDP GridWorld 3x3_numpyro.py`
- **Return Code:** 0
- **Execution Time:** 1.08 seconds
- **Detailed Output:** output/12_execute_output/pomdp_gridworld_3x3/numpyro/execution_logs/POMDP GridWorld 3x3_numpyro.py_execution.log

### POMDP GridWorld 3x3_pymdp.py - ✅ SUCCESS

- **Framework:** pymdp
- **Executor:** /Users/4d/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python
- **Path:** `output/11_render_output/pomdp_gridworld_3x3/pymdp/POMDP GridWorld 3x3_pymdp.py`
- **Return Code:** 0
- **Execution Time:** 2.70 seconds
- **Detailed Output:** output/12_execute_output/pomdp_gridworld_3x3/pymdp/execution_logs/POMDP GridWorld 3x3_pymdp.py_execution.log

### POMDP GridWorld 3x3_pytorch.py - ⏭️ SKIPPED

- **Framework:** pytorch
- **Executor:** /Users/4d/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python
- **Path:** `output/11_render_output/pomdp_gridworld_3x3/pytorch/POMDP GridWorld 3x3_pytorch.py`
- **Reason:** Dependency not installed: torch

### POMDP GridWorld 3x3_jax.py - ✅ SUCCESS

- **Framework:** jax
- **Executor:** /Users/4d/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python
- **Path:** `output/11_render_output/pomdp_gridworld_3x3/jax/POMDP GridWorld 3x3_jax.py`
- **Return Code:** 0
- **Execution Time:** 0.99 seconds
- **Detailed Output:** output/12_execute_output/pomdp_gridworld_3x3/jax/execution_logs/POMDP GridWorld 3x3_jax.py_execution.log

### POMDP GridWorld 3x3_discopy.py - ✅ SUCCESS

- **Framework:** discopy
- **Executor:** /Users/4d/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python
- **Path:** `output/11_render_output/pomdp_gridworld_3x3/discopy/POMDP GridWorld 3x3_discopy.py`
- **Return Code:** 0
- **Execution Time:** 0.30 seconds
- **Detailed Output:** output/12_execute_output/pomdp_gridworld_3x3/discopy/execution_logs/POMDP GridWorld 3x3_discopy.py_execution.log

### POMDP GridWorld 3x3_bnlearn.py - ⏭️ SKIPPED

- **Framework:** bnlearn
- **Executor:** /Users/4d/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python
- **Path:** `output/11_render_output/pomdp_gridworld_3x3/bnlearn/POMDP GridWorld 3x3_bnlearn.py`
- **Reason:** Dependency not installed: bnlearn

### POMDP GridWorld 3x3_rxinfer.jl - ✅ SUCCESS

- **Framework:** rxinfer
- **Executor:** julia
- **Path:** `output/11_render_output/pomdp_gridworld_3x3/rxinfer/POMDP GridWorld 3x3_rxinfer.jl`
- **Return Code:** 0
- **Execution Time:** 6.35 seconds
- **Detailed Output:** output/12_execute_output/pomdp_gridworld_3x3/rxinfer/execution_logs/POMDP GridWorld 3x3_rxinfer.jl_execution.log

### POMDP GridWorld 3x3_activeinference.jl - ✅ SUCCESS

- **Framework:** activeinference_jl
- **Executor:** julia
- **Path:** `output/11_render_output/pomdp_gridworld_3x3/activeinference_jl/POMDP GridWorld 3x3_activeinference.jl`
- **Return Code:** 0
- **Execution Time:** 9.35 seconds
- **Detailed Output:** output/12_execute_output/pomdp_gridworld_3x3/activeinference_jl/execution_logs/POMDP GridWorld 3x3_activeinference.jl_execution.log

## Next Steps

Skipped scripts are due to missing optional dependencies or unavailable system runtimes. Run `uv sync` for core Python backends; add `uv sync --extra ml-ai --extra graphs` for optional Python extension groups, and install Julia/D2 system tools as needed.

