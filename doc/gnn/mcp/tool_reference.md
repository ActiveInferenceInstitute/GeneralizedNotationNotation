# GNN MCP Tool Quick Reference

All 131 real tools registered by the GNN MCP server (v1.6.0 Engine; reference refreshed 2026-04-20).  
Sorted alphabetically by domain. For full per-domain documentation see **[../modules/21_mcp.md](../modules/21_mcp.md)**.

## Full Tool Table

| Domain | Tool | Description |
|--------|------|-------------|
| analysis | `compute_complexity_metrics` | Compute complexity scores for a GNN model |
| analysis | `get_analysis_results` | Retrieve cached analysis results |
| analysis | `list_analysis_tools` | List available analysis capabilities |
| analysis | `process_analysis` | Run statistical analysis on GNN files |
| audio | `analyze_audio_characteristics` | Analyse characteristics of a GNN model for sonification |
| audio | `check_audio_backends` | Check which audio backends are available (SAPF, Pedalboard, soundfile) |
| audio | `get_audio_generation_options` | List configurable audio generation options |
| audio | `get_audio_module_info` | Return audio module version and capabilities |
| audio | `process_audio` | Run GNN audio processing pipeline |
| audio | `validate_audio_content` | Validate audio content from GNN specs |
| advanced_visualization | `check_visualization_capabilities` | Check which advanced visualization libraries are available |
| advanced_visualization | `get_advanced_visualization_module_info` | Return module info and capabilities |
| advanced_visualization | `list_d2_visualization_types` | List supported D2 diagram types |
| advanced_visualization | `process_advanced_visualization` | Process advanced visualization (D2, dashboards) |
| execute | `check_execute_dependencies` | Check execution framework dependency availability |
| execute | `execute_gnn_model` | Execute a single GNN model file and capture results |
| execute | `execute_pymdp_simulation` | Run a PyMDP simulation from a rendered Python script |
| execute | `get_execute_module_info` | Return execute module version and capabilities |
| execute | `process_execute` | Run GNN execution pipeline across all frameworks |
| export | `list_export_formats` | List all supported export formats |
| export | `process_export` | Run multi-format export pipeline |
| export | `validate_export_format` | Validate an export format specifier |
| gnn | `get_gnn_module_info` | Return GNN module version and capabilities |
| gnn | `parse_gnn_content` | Parse a GNN specification string |
| gnn | `validate_gnn_content` | Validate a GNN specification string |
| gui | `get_gui_module_info` | Return GUI module version and feature flags |
| gui | `list_available_guis` | List available GUI types |
| gui | `process_gui` | Process GNN files for GUI generation |
| integration | `check_integration_dependencies` | Check integration dependency availability |
| integration | `list_supported_integrations` | List all supported integration targets |
| integration | `process_integration` | Run system integration pipeline |
| intelligent_analysis | `get_analysis_capabilities` | List intelligent analysis capabilities |
| intelligent_analysis | `get_intelligent_analysis_module_info` | Return module version and capabilities |
| intelligent_analysis | `process_intelligent_analysis` | Run intelligent analysis on pipeline results |
| llm | `analyze_gnn_with_llm` | Analyse a single GNN file with the configured LLM provider |
| llm | `generate_llm_documentation` | Generate LLM-powered documentation for a GNN model |
| llm | `get_llm_module_info` | Return LLM module version and capabilities |
| llm | `get_llm_providers` | List available LLM providers and their status |
| llm | `process_llm` | Run LLM analysis pipeline for all GNN files |
| ml_integration | `check_ml_frameworks` | Check which ML frameworks are available |
| ml_integration | `get_ml_module_info` | Return ML integration module version and capabilities |
| ml_integration | `list_ml_integration_targets` | List ML integration targets |
| ml_integration | `process_ml_integration` | Process ML integration for GNN files |
| ontology | `extract_ontology_annotations` | Extract ontology annotations from a GNN file |
| ontology | `list_standard_ontology_terms` | List all standard Active Inference ontology terms |
| ontology | `process_ontology` | Run Active Inference ontology processing |
| ontology | `validate_ontology_terms` | Validate ontology annotation terms |
| pipeline | `get_pipeline_status` | Get current pipeline execution status |
| pipeline | `get_pipeline_steps` | List all 25 pipeline steps (0-24) with metadata |
| render | `get_render_module_info` | Return render module version and capabilities |
| render | `list_render_frameworks` | List supported rendering frameworks |
| render | `process_render` | Run code generation for all GNN files |
| render | `render_gnn_to_format` | Render a GNN file to a specific framework format |
| report | `generate_report` | Generate a comprehensive pipeline report |
| report | `get_report_module_info` | Return report module version and capabilities |
| report | `list_report_formats` | List available report formats |
| report | `read_report` | Read a previously generated report |
| research | `list_research_topics` | List active research topics and experimental features |
| research | `process_research` | Run research tools pipeline |
| sapf | `get_sapf_module_info` | Return SAPF module version and capabilities |
| sapf | `list_audio_artifacts` | List generated audio artifacts |
| sapf | `process_sapf` | Run SAPF audio pipeline |
| security | `list_security_checks` | List all security checks performed |
| security | `process_security` | Run security validation pipeline |
| security | `scan_gnn_file` | Scan a GNN file for security issues |
| utils | `get_utils_info` | Return utils module version and available utilities |
| validation | `check_schema_compliance` | Check schema compliance for a GNN file |
| validation | `get_validation_report` | Get the latest validation report |
| validation | `process_validation` | Run GNN validation pipeline |
| validation | `validate_gnn_file` | Validate a single GNN file |
| visualization | `get_visualization_options` | List configurable visualization options |
| visualization | `list_visualization_artifacts` | List generated visualization artifacts |
| visualization | `process_visualization` | Run graph and matrix visualization pipeline |
| website | `build_website_from_pipeline_output` | Build website from pipeline artifacts |
| website | `get_website_module_info` | Return website module version and capabilities |
| website | `get_website_status` | Get current website generation status |
| website | `list_generated_website_pages` | List all generated website pages |
| website | `process_website` | Generate static HTML website from pipeline output |

**Total: 131 tools across 38+ domains**  
Verified by `src/tests/test_mcp_audit.py` as part of the full `src/tests/` suite (current pass/skip counts: repository [README.md](../../../README.md)).
