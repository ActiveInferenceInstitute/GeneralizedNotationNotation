# RxInfer Analysis — Technical Specification

**Version**: 1.6.0

## Input

- `simulation_results.json` from RxInfer (Julia) execution step (schema `rxinfer_simulation_v1`)
- Genuine variational message-passing inference results (`@model` + `infer()`, `free_energy = true`)

## Output

- Belief trajectory plots (PNG)
- Message flow analysis (PNG)
- Convergence diagnostics (JSON) — based on the genuine `variational_free_energy` trace (`inference_converged`, `vfe_present`)

## Framework

- Julia RxInfer genuine variational message-passing results (`variational_free_energy` populated with real values; previously `Float64[]`)
- Matplotlib visualization

## Error Handling

- Missing Julia results → graceful skip
- Non-convergent inference → diagnostic warning
