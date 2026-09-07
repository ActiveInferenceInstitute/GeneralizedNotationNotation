# Pipeline scripts (thin orchestrators)

## Overview

All **25** numbered entrypoints **`src/N_*.py`** (steps **0–24**) follow the **thin orchestrator** pattern: parse CLI args, configure logging and output dirs, delegate to **`src/<module>/`**, return standard exit codes (0=success, 1=error, 2=success with warnings/skipped). Domain logic lives in modules, not in the numbered scripts.

**Authoritative step matrix** (timeouts, dependencies, recovery): [`src/gnn/STEP_INDEX.md`](../src/gnn/STEP_INDEX.md). **Canonical step registry** (single source of truth for all 25 steps): [`src/gnn/pipeline/step_registry.py`](../src/gnn/pipeline/step_registry.py). **Commands and test notes**: [`CLAUDE.md`](../CLAUDE.md).

Run from **repository root** with:

```bash
uv run python src/gnn/main.py --target-dir input/gnn_files --verbose
```

## Master step table (0–24)

| Step | Script | Module directory | Purpose |
|:----:|--------|------------------|---------|
| 0 | [`0_template.py`](../src/gnn/0_template.py) | [`template/`](../src/gnn/template/) | Pipeline template and initialization |
| 1 | [`1_setup.py`](../src/gnn/1_setup.py) | [`setup/`](../src/gnn/setup/) | Environment setup and dependencies |
| 2 | [`2_tests.py`](../src/gnn/2_tests.py) | [`tests/`](../tests/) | Test suite execution |
| 3 | [`3_gnn.py`](../src/gnn/3_gnn.py) | [`gnn/`](../src/gnn/) | GNN discovery, parsing, multi-format serialization |
| 4 | [`4_model_registry.py`](../src/gnn/4_model_registry.py) | [`model_registry/`](../src/gnn/model_registry/) | Model registry and versioning |
| 5 | [`5_type_checker.py`](../src/gnn/5_type_checker.py) | [`type_checker/`](../src/gnn/type_checker/) | Type checking and resource estimation |
| 6 | [`6_validation.py`](../src/gnn/6_validation.py) | [`validation/`](../src/gnn/validation/) | Semantic validation |
| 7 | [`7_export.py`](../src/gnn/7_export.py) | [`export/`](../src/gnn/export/) | Multi-format export |
| 8 | [`8_visualization.py`](../src/gnn/8_visualization.py) | [`visualization/`](../src/gnn/visualization/) | Graph and matrix visualization |
| 9 | [`9_advanced_viz.py`](../src/gnn/9_advanced_viz.py) | [`advanced_visualization/`](../src/gnn/advanced_visualization/) | Advanced / interactive visualization |
| 10 | [`10_ontology.py`](../src/gnn/10_ontology.py) | [`ontology/`](../src/gnn/ontology/) | Ontology processing |
| 11 | [`11_render.py`](../src/gnn/11_render.py) | [`render/`](../src/gnn/render/) | Simulation code generation |
| 12 | [`12_execute.py`](../src/gnn/12_execute.py) | [`execute/`](../src/gnn/execute/) | Execute rendered simulations |
| 13 | [`13_llm.py`](../src/gnn/13_llm.py) | [`llm/`](../src/gnn/llm/) | LLM-enhanced analysis |
| 14 | [`14_ml_integration.py`](../src/gnn/14_ml_integration.py) | [`ml_integration/`](../src/gnn/ml_integration/) | ML integration |
| 15 | [`15_audio.py`](../src/gnn/15_audio.py) | [`audio/`](../src/gnn/audio/) | Audio / sonification |
| 16 | [`16_analysis.py`](../src/gnn/16_analysis.py) | [`analysis/`](../src/gnn/analysis/) | Statistical analysis |
| 17 | [`17_integration.py`](../src/gnn/17_integration.py) | [`integration/`](../src/gnn/integration/) | Cross-module integration |
| 18 | [`18_security.py`](../src/gnn/18_security.py) | [`security/`](../src/gnn/security/) | Security validation |
| 19 | [`19_research.py`](../src/gnn/19_research.py) | [`research/`](../src/gnn/research/) | Research tools |
| 20 | [`20_website.py`](../src/gnn/20_website.py) | [`website/`](../src/gnn/website/) | Static site generation |
| 21 | [`21_mcp.py`](../src/gnn/21_mcp.py) | [`mcp/`](../src/gnn/mcp/) | MCP processing and tool registration |
| 22 | [`22_gui.py`](../src/gnn/22_gui.py) | [`gui/`](../src/gnn/gui/) | Interactive GUI |
| 23 | [`23_report.py`](../src/gnn/23_report.py) | [`report/`](../src/gnn/report/) | Report generation |
| 24 | [`24_intelligent_analysis.py`](../src/gnn/24_intelligent_analysis.py) | [`intelligent_analysis/`](../src/gnn/intelligent_analysis/) | Intelligent / executive analysis |

## Typical usage

### Full pipeline

```bash
uv run python src/gnn/main.py --target-dir input/gnn_files --output-dir output --verbose
```

### Subset of steps

```bash
uv run python src/gnn/main.py --only-steps "3,5,11,12" --target-dir input/gnn_files --verbose
```

### Single step (example: analysis)

```bash
uv run python src/gnn/16_analysis.py --target-dir input/gnn_files --output-dir output --verbose
```

### Skip steps

```bash
uv run python src/gnn/main.py --skip-steps "14,18" --target-dir input/gnn_files --verbose
```

## Module pattern

Numbered scripts typically wrap a module entrypoint such as `process_<module>(target_dir, output_dir, logger, ...)` registered via `utils.pipeline_template.create_standardized_pipeline_script`. See any `src/N_*.py` and the matching [`src/<module>/AGENTS.md`](../src/gnn/AGENTS.md) for the public API.

## Related documentation

- [`src/gnn/main.py`](../src/gnn/main.py) — orchestrator
- [`docs/gnn/operations/gnn_tools.md`](gnn/operations/gnn_tools.md) — tooling and pipeline narrative
