# Documentation Module Agent Sync Maps

## Purpose
This directory serves as a dynamically synchronized documentation index for the 25 distinct `Modules` existing in the backend architecture.

**Status**: Synced to Python Source

## Reconciled Agent Maps
Every document located inside this folder (`00_template.md` through `24_intelligent_analysis.md`) is aligned with its respective `src/<module>/AGENTS.md` counterpart (e.g. `00_template.md` ↔ `src/gnn/template/AGENTS.md`), ensuring LLM Context layers learn current module parameters.

## Contained Indices

| # | Document | Module path | One-line purpose |
|---|----------|-------------|-------------------|
| 0 | [00_template.md](00_template.md) | `src/gnn/template/` | Scaffold for new GNN models from templates |
| 1 | [01_setup.md](01_setup.md) | `src/gnn/setup/` | Environment setup: uv sync, dep validation |
| 2 | [02_tests.md](02_tests.md) | `tests/` | Test orchestration (delegates to pytest) |
| 3 | [03_gnn.md](03_gnn.md) | `src/gnn/` | GNN parsing and discovery |
| 4 | [04_model_registry.md](04_model_registry.md) | `src/gnn/model_registry/` | Model metadata registration |
| 5 | [05_type_checker.md](05_type_checker.md) | `src/gnn/type_checker/` | Static validation + resource estimation |
| 6 | [06_validation.md](06_validation.md) | `src/gnn/validation/` | Deep consistency checking |
| 7 | [07_export.md](07_export.md) | `src/gnn/export/` | Multi-format export (JSON, XML, GraphML, etc.) |
| 8 | [08_visualization.md](08_visualization.md) | `src/gnn/visualization/` | Diagrams, connectivity plots |
| 9 | [09_advanced_viz.md](09_advanced_viz.md) | `src/gnn/advanced_visualization/` | Interactive + 3D visualizations |
| 10 | [10_ontology.md](10_ontology.md) | `src/gnn/ontology/` | Active Inference ontology mapping |
| 11 | [11_render.md](11_render.md) | `src/gnn/render/` | Code generation for all frameworks |
| 12 | [12_execute.md](12_execute.md) | `src/gnn/execute/` | Simulation execution |
| 13 | [13_llm.md](13_llm.md) | `src/gnn/llm/` | LLM analysis & enrichment |
| 14 | [14_ml_integration.md](14_ml_integration.md) | `src/gnn/ml_integration/` | PyTorch/JAX array export |
| 15 | [15_audio.md](15_audio.md) | `src/gnn/audio/` | Audio rendering (SAPF, pedalboard) |
| 16 | [16_analysis.md](16_analysis.md) | `src/gnn/analysis/` | Post-simulation statistical analysis |
| 17 | [17_integration.md](17_integration.md) | `src/gnn/integration/` | Cross-system integration |
| 18 | [18_security.md](18_security.md) | `src/gnn/security/` | Security scan + provenance checks |
| 19 | [19_research.md](19_research.md) | `src/gnn/research/` | Rule-based research artifact generation |
| 20 | [20_website.md](20_website.md) | `src/gnn/website/` | HTML dashboard generation (hard import) |
| 21 | [21_mcp.md](21_mcp.md) | `src/gnn/mcp/` | Model Context Protocol server (hard import) |
| 22 | [22_gui.md](22_gui.md) | `src/gnn/gui/` | Interactive GUI |
| 23 | [23_report.md](23_report.md) | `src/gnn/report/` | Consolidated pipeline report |
| 24 | [24_intelligent_analysis.md](24_intelligent_analysis.md) | `src/gnn/intelligent_analysis/` | LLM-assisted remediation (hard import) |

## External Usage
Model Context Protocol servers routing to this directory pull the real-implementation capability dictionary defined inside the core testing frameworks.
