# RxInfer Analysis — Technical Specification

**Version**: 1.6.0

## Input

- `simulation_results.json` from RxInfer (Julia) execution step
- Message-passing inference results

## Output

- Belief trajectory plots (PNG)
- Message flow analysis (PNG)
- Convergence diagnostics (JSON)

## Framework

- Julia RxInfer reactive inference results
- Matplotlib visualization

## Error Handling

- Missing Julia results → graceful skip
- Non-convergent inference → diagnostic warning
