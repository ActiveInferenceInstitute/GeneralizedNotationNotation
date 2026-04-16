# ActiveInference.jl Analysis — Technical Specification

**Version**: 1.6.0

## Input

- `simulation_results.json` from ActiveInference.jl execution step

## Output

- Belief trajectory plots (PNG)
- Action distribution analysis (PNG)
- EFE decomposition plots (PNG)
- `analysis_summary.json`

## Framework

- Julia-based Active Inference results parsed from JSON
- Matplotlib visualization with numpy processing

## Error Handling

- Missing Julia results → warning + empty analysis
- Graceful degradation when matplotlib unavailable
