# Analysis Module Specification

## Overview
Post-simulation analysis and metrics for GNN model execution.

## Components

### Core
- `processor.py` - Analysis processor
- `analyzer.py` - Main analyzer
- `post_simulation.py` - Post-simulation analysis (1337 lines)

## Analysis Types
- Belief trajectory analysis
- Action analysis
- Free energy plots
- Observation analysis
- GridWorld GIF animation and manifest generation for current PyMDP,
  RxInfer.jl, and ActiveInference.jl schemas

## Contracts
- `generate_animations` is the canonical Step 16 flag. The CLI enables
  animations by default and `--no-animations` sets `generate_animations=False`.
- Legacy `no_animations` is accepted only as an inverse compatibility key when
  `generate_animations` is absent. Supplying conflicting values is a hard
  argument error.
- Empty or missing input directories return exit code `2` so the pipeline can
  report `SUCCESS_WITH_WARNINGS` instead of a false success or hard crash.

## Key Exports
```python
from analysis import process_analysis, PostSimulationAnalyzer
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
