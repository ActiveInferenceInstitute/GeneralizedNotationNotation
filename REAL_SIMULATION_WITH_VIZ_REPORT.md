# REAL Simulation WITH VISUALIZATION Execution Report

**Generated:** 2025-08-29T13:47:00.192652
**Completion:** 2025-08-29T13:47:59.947739

## Summary

- **Total Frameworks:** 4
- **Successful Simulations:** 3
- **Successful Visualizations:** 2
- **Total PNG Files Generated:** 5
- **Simulation Success Rate:** 75.0%
- **Visualization Success Rate:** 50.0%

## Framework Results

### PyMDP - ✅ SUCCESS

- **Script:** `output/11_render_output/11_render_output/actinf_pomdp_agent/pymdp/actinf_pomdp_agent_pymdp_WITH_VIZ.py`
- **Executor:** python3
- **Return Code:** 0
- **Visualizations:** ❌ No visualizations

### DisCoPy - ✅ SUCCESS

- **Script:** `output/11_render_output/11_render_output/actinf_pomdp_agent/discopy/actinf_pomdp_agent_discopy_WITH_VIZ.py`
- **Executor:** python3
- **Return Code:** 0
- **Visualizations:** 🎨 2 PNG files
- **PNG Files:**
  - `REAL_categorical_analysis.png`
  - `REAL_analysis_progress.png`

### ActiveInference.jl - ✅ SUCCESS

- **Script:** `output/11_render_output/11_render_output/actinf_pomdp_agent/activeinference_jl/actinf_pomdp_agent_activeinference_jl_WITH_VIZ.jl`
- **Executor:** julia
- **Return Code:** 0
- **Visualizations:** 🎨 3 PNG files
- **PNG Files:**
  - `REAL_activeinference_analysis.png`
  - `REAL_belief_evolution.png`
  - `REAL_free_energy_evolution.png`

### RxInfer.jl - ❌ FAILED

- **Script:** `output/11_render_output/11_render_output/actinf_pomdp_agent/rxinfer/actinf_pomdp_agent_rxinfer_WITH_VIZ.jl`
- **Executor:** julia
- **Return Code:** 1
- **Visualizations:** ❌ No visualizations


## Failure Details

### RxInfer.jl
```
┌ Warning: Model specification language does not support keyword arguments. Ignoring 1 keyword arguments.
└ @ GraphPPL ~/.julia/packages/GraphPPL/xPNyo/src/model_macro.jl:801
ERROR: LoadError: UndefVarError: `now` not defined in `Main`
Suggestion: check for spelling errors or missing imports.
Hint: a global variable of this name may be made accessible by importing Dates in the current active module Main
Stacktrace:
 [1] log_simulation_failure(monitor::VizRxInferMonitor, sim_name::String, error::
```

