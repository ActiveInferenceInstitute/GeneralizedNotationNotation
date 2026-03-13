# Step 21: MCP — Model Context Protocol Server

## Overview

Orchestrates Model Context Protocol (MCP) processing. Discovers all pipeline modules, registers every module's domain-specific tools, and serves them as MCP-compatible tools accessible to AI agents and IDE extensions.

**Last Updated**: March 6, 2026  
**Status**: ✅ All tools real (no placeholders), 0 skips, 1,522+ tests passing

## Usage

```bash
python src/21_mcp.py --target-dir input/gnn_files --output-dir output --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/21_mcp.py` (63 lines) |
| Core MCP engine | `src/mcp/` |
| Module function | `process_mcp()` |
| Server | `src/mcp/server.py` |
| Audit script | `src/mcp/validate_tools.py` |

## MCP Tool Registry (131 real tools)

All tools are real named callable functions — no placeholders, no lambdas, no generic wrappers. Each tool has a non-empty description and its `register_tools()` calls `logger.info`.

### GNN Core (gnn module)

| Tool | Description |
|------|-------------|
| `parse_gnn_content` | Parse a GNN specification string |
| `validate_gnn_content` | Validate a GNN specification string |
| `get_gnn_module_info` | Return GNN module version and capabilities |

### Analysis (analysis module)

| Tool | Description |
|------|-------------|
| `process_analysis` | Run statistical analysis on GNN files |
| `get_analysis_results` | Retrieve cached analysis results |
| `compute_complexity_metrics` | Compute complexity scores for a GNN model |
| `list_analysis_tools` | List available analysis capabilities |

### Audio (audio module)

| Tool | Description |
|------|-------------|
| `process_audio` | Run GNN audio processing pipeline |
| `check_audio_backends` | Check which audio backends are available |
| `get_audio_generation_options` | List configurable audio generation options |
| `analyze_audio_characteristics` | Analyse characteristics of a GNN model for sonification |
| `validate_audio_content` | Validate audio content from GNN specs |
| `get_audio_module_info` | Return audio module version and capabilities |

### Advanced Visualization (advanced_visualization module)

| Tool | Description |
|------|-------------|
| `process_advanced_visualization` | Process advanced visualization for GNN files (D2 diagrams, dashboards) |
| `check_visualization_capabilities` | Check which advanced visualization libraries are available |
| `list_d2_visualization_types` | List supported D2 diagram types |
| `get_advanced_visualization_module_info` | Return module info and capabilities |

### Execute (execute module)

| Tool | Description |
|------|-------------|
| `process_execute` | Run GNN execution pipeline across frameworks |
| `execute_gnn_model` | Execute a single GNN model file |
| `execute_pymdp_simulation` | Run a PyMDP simulation from rendered code |
| `check_execute_dependencies` | Check execution framework dependencies |
| `get_execute_module_info` | Return execute module version and capabilities |

### Export (export module)

| Tool | Description |
|------|-------------|
| `process_export` | Run multi-format export pipeline |
| `list_export_formats` | List all supported export formats |
| `validate_export_format` | Validate an export format specifier |

### GUI (gui module)

| Tool | Description |
|------|-------------|
| `process_gui` | Process GNN files for GUI generation |
| `list_available_guis` | List available GUI types |
| `get_gui_module_info` | Return GUI module version and feature flags |

### Integration (integration module)

| Tool | Description |
|------|-------------|
| `process_integration` | Run system integration pipeline |
| `list_supported_integrations` | List all supported integration targets |
| `check_integration_dependencies` | Check integration dependency availability |

### Intelligent Analysis (intelligent_analysis module)

| Tool | Description |
|------|-------------|
| `process_intelligent_analysis` | Run intelligent analysis on pipeline results |
| `get_analysis_capabilities` | List intelligent analysis capabilities |
| `get_intelligent_analysis_module_info` | Return module version and capabilities |

### LLM (llm module)

| Tool | Description |
|------|-------------|
| `process_llm` | Run LLM analysis pipeline for all GNN files |
| `analyze_gnn_with_llm` | Analyse a single GNN file with the configured LLM |
| `generate_llm_documentation` | Generate LLM-powered documentation for a GNN model |
| `get_llm_providers` | List available LLM providers and their status |
| `get_llm_module_info` | Return LLM module version and capabilities |

### ML Integration (ml_integration module)

| Tool | Description |
|------|-------------|
| `process_ml_integration` | Process ML integration for GNN files |
| `check_ml_frameworks` | Check which ML frameworks are available |
| `list_ml_integration_targets` | List ML integration targets |
| `get_ml_module_info` | Return ML integration module version and capabilities |

### Ontology (ontology module)

| Tool | Description |
|------|-------------|
| `process_ontology` | Run Active Inference ontology processing |
| `validate_ontology_terms` | Validate ontology annotation terms |
| `extract_ontology_annotations` | Extract ontology annotations from a GNN file |
| `list_standard_ontology_terms` | List all standard Active Inference ontology terms |

### Pipeline (pipeline module)

| Tool | Description |
|------|-------------|
| `get_pipeline_steps` | List all 25 pipeline steps (0-24) with metadata |
| `get_pipeline_status` | Get current pipeline execution status |

### Render (render module)

| Tool | Description |
|------|-------------|
| `process_render` | Run code generation for all GNN files |
| `render_gnn_to_format` | Render a GNN file to a specific framework format |
| `list_render_frameworks` | List supported rendering frameworks |
| `get_render_module_info` | Return render module version and capabilities |

### Report (report module)

| Tool | Description |
|------|-------------|
| `generate_report` | Generate a comprehensive pipeline report |
| `list_report_formats` | List available report formats |
| `read_report` | Read a previously generated report |
| `get_report_module_info` | Return report module version and capabilities |

### Research (research module)

| Tool | Description |
|------|-------------|
| `process_research` | Run research tools pipeline |
| `list_research_topics` | List active research topics and experimental features |

### SAPF (sapf module)

| Tool | Description |
|------|-------------|
| `process_sapf` | Run SAPF audio pipeline |
| `list_audio_artifacts` | List generated audio artifacts |
| `get_sapf_module_info` | Return SAPF module version and capabilities |

### Security (security module)

| Tool | Description |
|------|-------------|
| `process_security` | Run security validation pipeline |
| `scan_gnn_file` | Scan a GNN file for security issues |
| `list_security_checks` | List all security checks performed |

### Utils (utils module)

| Tool | Description |
|------|-------------|
| `get_utils_info` | Return utils module version and available utilities |

### Validation (validation module)

| Tool | Description |
|------|-------------|
| `process_validation` | Run GNN validation pipeline |
| `validate_gnn_file` | Validate a single GNN file |
| `get_validation_report` | Get the latest validation report |
| `check_schema_compliance` | Check schema compliance for a GNN file |

### Visualization (visualization module)

| Tool | Description |
|------|-------------|
| `process_visualization` | Run graph and matrix visualization pipeline |
| `get_visualization_options` | List configurable visualization options |
| `list_visualization_artifacts` | List generated visualization artifacts |

### Website (website module)

| Tool | Description |
|------|-------------|
| `process_website` | Generate static HTML website from pipeline output |
| `build_website_from_pipeline_output` | Build website from pipeline artifacts |
| `get_website_status` | Get current website generation status |
| `list_generated_website_pages` | List all generated website pages |
| `get_website_module_info` | Return website module version and capabilities |

## Tool Quality Audit

All tools verified by `src/tests/test_mcp_audit.py` (1,522+ tests, 0 failures, 0 skips):

- ✅ Every tool has a callable named function (no lambdas, no `None`)
- ✅ Every tool has a non-empty description
- ✅ Every `register_tools()` calls `logger.info` with tool count
- ✅ Zero generic placeholders (`list_functions` / `call_function` removed)
- ✅ Zero async polling timing issues (fixture polls for stabilisation)

Run the audit:

```bash
PYTHONPATH=src python -m pytest src/tests/test_mcp_audit.py -v
```

## Output

- **Directory**: `output/21_mcp_output/`
- MCP server configuration, audit report (`mcp_audit_report.json`), tool registrations

## Source

- **Script**: [src/21_mcp.py](#placeholder)
- **Audit Tests**: [src/tests/test_mcp_audit.py](#placeholder)
