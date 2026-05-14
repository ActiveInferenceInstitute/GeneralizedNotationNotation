# PyMDP Analysis — Technical Specification

**Version**: 1.6.0

## Input

- `simulation_results.json` from Step 12 with
  `"schema_version": "pymdp_simulation_v1"`
- Observations by modality, hidden states by factor, actions by control factor,
  beliefs by factor, expected/variational free energy, policy posterior,
  validation, matrix provenance, and runtime metadata

## Output

- Belief trajectory plots
- Action sequence and action distribution plots
- Expected and variational free-energy plots
- Observation sequence plots
- Preference accumulation plots when preference metrics are present

## Rules

- Reject older PyMDP result shapes for PyMDP-specific analysis.
- Treat missing results as no-output analysis with a warning.
- Keep execution concerns in `src/execute/pymdp/`; this module only reads
  completed execution artifacts.

## Verification

```bash
uv run pytest \
    src/tests/analysis/test_analysis_post_simulation.py \
    src/tests/analysis/test_analysis_overall.py \
    -q --tb=short
```
