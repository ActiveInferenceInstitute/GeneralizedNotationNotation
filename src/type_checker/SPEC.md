# Type Checker — Technical Specification

**Version**: 1.6.0

## Purpose

Step 5 — Static type analysis and resource estimation for parsed GNN models.

## Architecture

```
type_checker/
├── __init__.py              # Package exports
├── processor.py             # Core type checking logic (571 lines)
├── resource_estimator.py    # Resource estimation engine (1068 lines)
├── type_rules.py            # Type inference rules
└── mcp.py                   # MCP tool registration
```

## Type Checking Rules

1. **Variable type consistency** — All variables must have declared types matching usage
2. **Matrix dimension agreement** — Transition/observation matrix dimensions must match state/observation counts
3. **Probability normalization** — Stochastic matrices must have rows summing to 1.0 (within tolerance)
4. **Prior compatibility** — Prior distributions must match model structure

## Resource Estimation

- Memory requirements per model (estimated from matrix dimensions)
- Computational complexity classification (O(n²), O(n³))
- Recommended framework based on model scale

## Input

- Parsed GNN models from Step 3

## Output

- `type_check_results.json` — Type errors, warnings, and resource estimates
- Exit code: 0 (clean), 1 (errors), 2 (warnings only)
