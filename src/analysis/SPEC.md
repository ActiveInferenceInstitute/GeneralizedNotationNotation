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

## Key Exports
```python
from analysis import process_analysis, PostSimulationAnalyzer
```
