# Pipeline scripts (thin orchestrators)

## Overview

All **25** numbered entrypoints **`src/N_*.py`** (steps **0–24**) follow the **thin orchestrator** pattern: parse CLI args, configure logging and output dirs, delegate to **`src/<module>/`**, return standard exit codes (0 success, 1 error, 2 warnings). Domain logic lives in modules, not in the numbered scripts.

**Authoritative step matrix** (timeouts, dependencies, recovery): [`src/STEP_INDEX.md`](../src/STEP_INDEX.md). **Commands and test notes**: [`CLAUDE.md`](../CLAUDE.md).

Run from **repository root** with:

```bash
uv run python src/main.py --target-dir input/gnn_files --verbose
```

## Master step table (0–24)

| Step | Script | Module directory | Purpose |
|:----:|--------|------------------|---------|
| 0 | [`0_template.py`](../src/0_template.py) | [`template/`](../src/template/) | Pipeline template and initialization |
| 1 | [`1_setup.py`](../src/1_setup.py) | [`setup/`](../src/setup/) | Environment setup and dependencies |
| 2 | [`2_tests.py`](../src/2_tests.py) | [`tests/`](../src/tests/) | Test suite execution |
| 3 | [`3_gnn.py`](../src/3_gnn.py) | [`gnn/`](../src/gnn/) | GNN discovery, parsing, multi-format serialization |
| 4 | [`4_model_registry.py`](../src/4_model_registry.py) | [`model_registry/`](../src/model_registry/) | Model registry and versioning |
| 5 | [`5_type_checker.py`](../src/5_type_checker.py) | [`type_checker/`](../src/type_checker/) | Type checking and resource estimation |
| 6 | [`6_validation.py`](../src/6_validation.py) | [`validation/`](../src/validation/) | Semantic validation |
| 7 | [`7_export.py`](../src/7_export.py) | [`export/`](../src/export/) | Multi-format export |
| 8 | [`8_visualization.py`](../src/8_visualization.py) | [`visualization/`](../src/visualization/) | Graph and matrix visualization |
| 9 | [`9_advanced_viz.py`](../src/9_advanced_viz.py) | [`advanced_visualization/`](../src/advanced_visualization/) | Advanced / interactive visualization |
| 10 | [`10_ontology.py`](../src/10_ontology.py) | [`ontology/`](../src/ontology/) | Ontology processing |
| 11 | [`11_render.py`](../src/11_render.py) | [`render/`](../src/render/) | Simulation code generation |
| 12 | [`12_execute.py`](../src/12_execute.py) | [`execute/`](../src/execute/) | Execute rendered simulations |
| 13 | [`13_llm.py`](../src/13_llm.py) | [`llm/`](../src/llm/) | LLM-enhanced analysis |
| 14 | [`14_ml_integration.py`](../src/14_ml_integration.py) | [`ml_integration/`](../src/ml_integration/) | ML integration |
| 15 | [`15_audio.py`](../src/15_audio.py) | [`audio/`](../src/audio/) | Audio / sonification |
| 16 | [`16_analysis.py`](../src/16_analysis.py) | [`analysis/`](../src/analysis/) | Statistical analysis |
| 17 | [`17_integration.py`](../src/17_integration.py) | [`integration/`](../src/integration/) | Cross-module integration |
| 18 | [`18_security.py`](../src/18_security.py) | [`security/`](../src/security/) | Security validation |
| 19 | [`19_research.py`](../src/19_research.py) | [`research/`](../src/research/) | Research tools |
| 20 | [`20_website.py`](../src/20_website.py) | [`website/`](../src/website/) | Static site generation |
| 21 | [`21_mcp.py`](../src/21_mcp.py) | [`mcp/`](../src/mcp/) | MCP processing and tool registration |
| 22 | [`22_gui.py`](../src/22_gui.py) | [`gui/`](../src/gui/) | Interactive GUI |
| 23 | [`23_report.py`](../src/23_report.py) | [`report/`](../src/report/) | Report generation |
| 24 | [`24_intelligent_analysis.py`](../src/24_intelligent_analysis.py) | [`intelligent_analysis/`](../src/intelligent_analysis/) | Intelligent / executive analysis |

## Typical usage

### Full pipeline

```bash
uv run python src/main.py --target-dir input/gnn_files --output-dir output --verbose
```

### Subset of steps

```bash
uv run python src/main.py --only-steps "3,5,11,12" --target-dir input/gnn_files --verbose
```

### Single step (example: analysis)

```bash
uv run python src/16_analysis.py --target-dir input/gnn_files --output-dir output --verbose
```

### Skip steps

```bash
uv run python src/main.py --skip-steps "14,18" --target-dir input/gnn_files --verbose
```

## Module pattern

Numbered scripts typically wrap a module entrypoint such as `process_<module>(target_dir, output_dir, logger, ...)` registered via `utils.pipeline_template.create_standardized_pipeline_script`. See any `src/N_*.py` and the matching [`src/<module>/AGENTS.md`](../src/AGENTS.md) for the public API.

## Related documentation

- [`src/main.py`](../src/main.py) — orchestrator
- [`doc/gnn/operations/gnn_tools.md`](gnn/operations/gnn_tools.md) — tooling and pipeline narrative
