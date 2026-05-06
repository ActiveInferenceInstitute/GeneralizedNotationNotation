# Type Checker — Technical Specification

**Version**: 1.6.0

## Purpose

Step 5 — Static type analysis and resource estimation for parsed GNN models.

## Architecture

```
type_checker/
├── __init__.py              # Package exports
├── checking/                # Core validation subpackage
│   ├── core.py              # Orchestrator
│   ├── dimensions.py        # Shape analysis
│   └── rules.py             # Type rule engine
├── estimation/              # Resource estimation subpackage
│   ├── estimator.py         # Hardware projections
│   ├── strategies.py        # Math utilities
│   ├── report_html.py       # HTML reporting
│   └── report_markdown.py   # Text reporting
├── processor.py             # Legacy compat orchestrator (thin delegation)
├── resource_estimator.py    # Legacy compat estimator (thin delegation)
├── estimation_strategies.py # Legacy compat strategies (thin delegation)
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
