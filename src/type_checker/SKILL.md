---
name: gnn-type-checker
description: GNN structural type checking and resource estimation. Use when validating GNN dimensions, syntax, type consistency, or generated type-checker reports.
---

# Core Skill: `type_check`

**Function**: Evaluates GNN structural schemas assigning validation integers and load capacities producing automated Abstract Model Cards.

## Example Flow
```python
# The main dispatcher triggers the core checker tracking directory integrity
from type_checker.checking import GNNTypeChecker
from pathlib import Path

# Spin up analyzer mapping
checker = GNNTypeChecker()
success = checker.validate_gnn_files(Path("models/"), Path("output/5_type_checker"))

# This cleanly generates Type Validity Mosaics and Baseball Cards intrinsically evaluated
print("Completed successfully:", success)

# Pure content validation (no file on disk) with strict-mode B-orientation checking
from type_checker import GNNTypeChecker as _TC
result = _TC(strict_mode=True).validate_content(spec_text, source_name="model.gnn")
print("valid:", result["valid"], "errors:", len(result["errors"]))

# Typed summary of a directory run
from type_checker import summarize_type_check_results
print(summarize_type_check_results({"validation_results": [result]}))
```

## Features
- **Visual Synthesis**: Constructs isolated Model Baseball Cards tracking matrices, parameters, FLOPS and validity using Matplotlib abstractions directly embedded into the generated output markdown.
- **Resource Analytics**: Estimates accurate floating point requirements linking straight into `estimation_strategies.py` natively.
- **Strict Parsing Evaluation**: Prevents ambiguous mathematical strings from triggering false type checks.


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `validate_gnn_files`
- `validate_single_gnn_file`
