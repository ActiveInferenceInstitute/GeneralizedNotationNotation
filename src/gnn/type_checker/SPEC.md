# Type Checker — Technical Specification

**Version**: 1.7.0

## Purpose

Step 5 — Static type analysis and resource estimation for parsed GNN models.

## Architecture

```
type_checker/
├── __init__.py              # Package exports (GNNTypeChecker, estimate_file_resources,
│                            #   extract_gnn_dimensions, validate_dimension_compatibility,
│                            #   summarize_type_check_results, ValidationSummary)
├── checking/                # Core validation subpackage
│   ├── core.py              # GNNTypeChecker orchestrator (validate_content/_file/_files)
│   ├── sections.py          # Section-scoped content extraction (shared w/ estimation)
│   ├── summary.py           # ValidationSummary TypedDict + summarize_type_check_results
│   ├── dimensions.py        # Shape analysis + B-orientation verdicts
│   └── rules.py             # Type rule engine
├── estimation/              # Resource estimation subpackage
│   ├── estimator.py         # GNNResourceEstimator (## Time classifier)
│   ├── strategies.py        # Math utilities
│   ├── report_html.py       # HTML reporting
│   └── report_markdown.py   # Text reporting
├── processor.py             # Public checking facade
├── resource_estimator.py    # Resource estimator CLI facade
├── estimation_strategies.py # Public strategy exports
├── analysis_utils.py        # Variable/connection/complexity analysis helpers
├── output_utils.py          # Per-file + cross-file report renderers
├── visualizer.py            # Mosaics, pies, radars, baseball cards
├── cli.py                   # `python -m type_checker.cli` entry point
└── mcp.py                   # MCP tool registration
```

## Type Checking Rules

1. **Variable type consistency** — All variables must have declared types matching usage
2. **Matrix dimension agreement** — Transition/observation matrix dimensions must match state/observation counts
3. **Probability normalization** — Stochastic matrices must have rows summing to 1.0 (within tolerance)
4. **Prior compatibility** — Prior distributions must match model structure
5. **B-orientation contradictions** — `[GNN-E002]` flags comment-vs-comment axis-order contradictions and row-stochastic-only slices; warnings by default, errors in `strict_mode`

## Resource Estimation

- Memory requirements per model (estimated from matrix dimensions)
- Computational complexity classification (O(n²), O(n³))
- `## Time` section classified as Static / Dynamic / Hierarchical
- Recommended framework based on model scale

## Input

- Parsed GNN models from Step 3

## Output

- `type_check_results.json` — Type errors, warnings, and resource estimates
- `type_check_summary.md` — Markdown summary with inline card images
- `type_check_summary.json` — Typed validation summary (counts, complexity tiers, totals)
- `visualizations/cards/<model_name>_card.png` — Per-model trading cards
- `resource_estimates/` — `resource_data.json` + `resource_report.md` (when `--estimate-resources`)
- Exit code: 0 (clean), 1 (errors), 2 (warnings only)
