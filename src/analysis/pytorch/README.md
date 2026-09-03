# PyTorch Analysis

Framework-specific analysis module for PyTorch simulation outputs.

## Usage

```python
from analysis.pytorch.analyzer import generate_analysis_from_logs

generate_analysis_from_logs(
    results_dir="output/12_execute_output", output_dir="output/16_analysis_output"
)
```

## Outputs

- Belief trajectory plot: `belief_trajectory.png`
- Action distribution histogram: `action_distribution.png`
- EFE history plot: `efe_history.png`
- JSON summary: `pytorch_analysis.json`

## Dependencies

- `numpy`, `matplotlib` (required)
- `torch` (optional, for advanced result interpretation)

## See Also

- [Parent: analysis/README.md](../README.md)
- [AGENTS.md](AGENTS.md) — Architecture documentation
- [SPEC.md](SPEC.md) — Technical specification
