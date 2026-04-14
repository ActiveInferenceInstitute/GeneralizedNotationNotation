# Core Skill: `type_check`

**Function**: Evaluates GNN structural schemas assigning validation integers and load capacities producing automated Abstract Model Cards.

## Example Flow
```python
# The main dispatcher triggers `processor.py` internally tracking directory integrity
from type_checker.processor import GNNTypeChecker
from pathlib import Path

# Spin up analyzer mapping
checker = GNNTypeChecker()
success = checker.validate_gnn_files(Path("models/"), Path("output/5_type_checker"))

# This cleanly generates Type Validity Mosaics and Baseball Cards intrinsically evaluated
print("Completed successfully:", success)
```

## Features
- **Visual Synthesis**: Constructs isolated Model Baseball Cards tracking matrices, parameters, FLOPS and validity using Matplotlib abstractions directly embedded into the generated output markdown.
- **Resource Analytics**: Estimates accurate floating point requirements linking straight into `estimation_strategies.py` natively.
- **Strict Parsing Evaluation**: Prevents ambiguous mathematical strings from triggering false type checks.
