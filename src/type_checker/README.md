# Type Checker

This module provides functionality for type checking and validating GNN files against the GNN specification.

## Features

- Validates GNN file structure and required sections
- Checks variable types and dimensions
- Verifies consistency between variables and connections
- Validates time specifications
- Generates detailed reports with errors and warnings

## Usage

### Command Line Interface

The module provides a command-line interface for type checking GNN files:

```bash
# Check a single file
python -m type_checker path/to/gnn_file.md

# Check all files in a directory
python -m type_checker input/gnn_files/

# Recursively check all files in a directory and its subdirectories
python -m type_checker path/to/directory --recursive

# Enable strict type checking mode
python -m type_checker path/to/gnn_file.md --strict

# Save report to a file
python -m type_checker path/to/gnn_file.md -o report.md
```

### Programmatic Usage

The module can also be used programmatically:

```python
from type_checker import GNNTypeChecker

# Create a type checker
checker = GNNTypeChecker(strict_mode=True)

# Check a single file
is_valid, errors, warnings = checker.check_file("path/to/gnn_file.md")

# Check all files in a directory
results = checker.check_directory("path/to/directory", recursive=True)

# Generate a report
report = checker.generate_report(results, output_file="report.md")
```

## Validation Rules

The type checker enforces the following rules:

1. All required sections must be present
2. Variables must have valid dimensions and types
3. Connections must refer to defined variables
4. Time specifications must be valid
5. Equations should use defined variables 