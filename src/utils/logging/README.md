# Logging Utilities

Centralized logging infrastructure for the GNN pipeline.

## Files

- `logging_utils.py` (1070 lines) — Full logging implementation with step-level logging, visual progress, color-coded output, correlation IDs, and screen reader support.

## Quick Start

```python
from utils.logging.logging_utils import setup_step_logging, log_step_start, log_step_success

setup_step_logging("step_3_gnn", verbose=True)
log_step_start(3, "GNN file processing")
# ... do work ...
log_step_success(3, "Processed 9 models")
```

## See Also

- [Parent: utils/README.md](../README.md)
