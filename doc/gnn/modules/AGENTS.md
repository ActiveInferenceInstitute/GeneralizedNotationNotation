# Documentation Module Agent Sync Maps

## Purpose
This directory serves as a dynamically synchronized documentation index for the 25 distinct `Modules` existing in the backend architecture.

**Status**: Synced to Python Source

## Reconciled Agent Maps
Every document located inside this folder (`00_template.md` through `24_intelligent_analysis.md`) is accurately aligned with its respective `src/module/AGENTS.md` counterpart ensuring LLM Context layers never learn out-of-date or "Mock" parameters.

## Contained Indices

| # | Document | Module path | One-line purpose |
|---|----------|-------------|-------------------|
| 0 | [00_template.md](00_template.md) | `src/template/` | Scaffold for new GNN models from templates |
| 1 | [01_setup.md](01_setup.md) | `src/setup/` | Environment setup: uv sync, dep validation |
| 2 | [02_tests.md](02_tests.md) | `src/tests/` | Test orchestration (delegates to pytest) |
| 3 | [03_gnn.md](03_gnn.md) | `src/gnn/` | GNN parsing and discovery |
| 4 | [04_model_registry.md](04_model_registry.md) | `src/model_registry/` | Model metadata registration |
| 5 | [05_type_checker.md](05_type_checker.md) | `src/type_checker/` | Static validation + resource estimation |
| 6 | [06_validation.md](06_validation.md) | `src/validation/` | Deep consistency checking |
| 7 | [07_export.md](07_export.md) | `src/export/` | Multi-format export (JSON, XML, GraphML, etc.) |
| 8 | [08_visualization.md](08_visualization.md) | `src/visualization/` | Diagrams, connectivity plots |
| 9 | [09_advanced_viz.md](09_advanced_viz.md) | `src/advanced_visualization/` | Interactive + 3D visualizations |
| 10 | [10_ontology.md](10_ontology.md) | `src/ontology/` | Active Inference ontology mapping |
| 11 | [11_render.md](11_render.md) | `src/render/` | Code generation for all frameworks |
| 12 | [12_execute.md](12_execute.md) | `src/execute/` | Simulation execution |
| 13 | [13_llm.md](13_llm.md) | `src/llm/` | LLM analysis & enrichment |
| 14 | [14_ml_integration.md](14_ml_integration.md) | `src/ml_integration/` | PyTorch/JAX array export |
| 15 | [15_audio.md](15_audio.md) | `src/audio/` | Audio rendering (SAPF, pedalboard) |
| 16 | [16_analysis.md](16_analysis.md) | `src/analysis/` | Post-simulation statistical analysis |
| 17 | [17_integration.md](17_integration.md) | `src/integration/` | Cross-system integration |
| 18 | [18_security.md](18_security.md) | `src/security/` | Security scan + provenance checks |
| 19 | [19_research.md](19_research.md) | `src/research/` | Rule-based research artifact generation |
| 20 | [20_website.md](20_website.md) | `src/website/` | HTML dashboard generation (hard import) |
| 21 | [21_mcp.md](21_mcp.md) | `src/mcp/` | Model Context Protocol server (hard import) |
| 22 | [22_gui.md](22_gui.md) | `src/gui/` | Interactive GUI |
| 23 | [23_report.md](23_report.md) | `src/report/` | Consolidated pipeline report |
| 24 | [24_intelligent_analysis.md](24_intelligent_analysis.md) | `src/intelligent_analysis/` | LLM-assisted remediation (hard import) |

## External Usage
Model Context Protocol servers routing to this directory instantly pull the entire zero-mock capability dictionary natively defined inside the core testing frameworks.
