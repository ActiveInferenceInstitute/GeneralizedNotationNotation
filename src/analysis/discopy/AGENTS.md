# DisCoPy Analysis - Agent Scaffolding

## Overview

Framework-specific analyzer for DisCoPy categorical diagram results. Part of the Analysis module (Step 16).

## Module Structure

```
analysis/discopy/
├── __init__.py    # Public API
├── analyzer.py    # DisCoPyAnalyzer class
├── README.md      # Human documentation
└── AGENTS.md      # This file
```

## Key Functions

### analyzer.py

- `generate_analysis_from_logs(execution_dir, output_dir, verbose)` - Main entry point
- `_parse_diagram_outputs(filepath)` - Parse DisCoPy outputs
- `_analyze_structure(diagram)` - Analyze diagram topology
- `_analyze_composition(diagram)` - Composition analysis
- `_generate_report(metrics)` - Report generation

## Integration Points

**Upstream:** Execute module (Step 12) produces DisCoPy diagram results
**Downstream:** Report module (Step 23) consumes analysis outputs

## Dependencies

- discopy (optional): Native diagram analysis
- numpy: Numerical operations
- matplotlib (optional): Visualization

---

**Version:** 1.1.3
**Last Updated:** 2026-01-23
