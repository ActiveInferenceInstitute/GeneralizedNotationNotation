# Logging utilities

`src/utils/logging/` centralizes logging helpers for the GNN pipeline so that:

- numbered step scripts emit consistent console output,
- log files are written in predictable locations,
- modules can share conventions (levels, formats, correlation IDs where supported).

## Primary entry points

See `logging_utils.py` for the concrete functions used across the repo.

