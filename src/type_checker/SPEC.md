# Type Checker Module Specification

## Overview
GNN model type checking and resource estimation.

## Components

### Core
- `checker.py` - Type checker (1344 lines)
- `resource_estimator.py` - Resource estimation (1742 lines)

### Utilities
- `analysis_utils.py` - Analysis utilities
- `output_utils.py` - Output formatting
- `cli.py` - Command-line interface

## Features
- Variable type validation
- Dimension checking
- Memory/FLOPS estimation
- Inference time prediction

## Key Exports
```python
from type_checker import GNNTypeChecker, GNNResourceEstimator
```
