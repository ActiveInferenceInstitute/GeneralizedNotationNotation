# GNN Pipeline Documentation

> **📋 Document Metadata**  
> **Type**: Technical Guide | **Audience**: Developers, Users | **Complexity**: Intermediate  
> **Last Updated**: July 2025 | **Status**: Production-Ready  
> **Cross-References**: [Main Documentation](../README.md) | [API Reference](../api/README.md)

## Overview
The GNN Processing Pipeline is a comprehensive 24-step system for processing Generalized Notation Notation files from initialization through execution, analysis, GUI construction, and comprehensive reporting.

## Pipeline Architecture: Thin Orchestrator Pattern

The pipeline follows a **thin orchestrator pattern** for maintainability, modularity, and testability:

### 🏗️ Architectural Pattern

- **Numbered Scripts** (e.g., `11_render.py`, `10_ontology.py`): Thin orchestrators that handle pipeline orchestration, argument parsing, logging, and result aggregation
- **Module `__init__.py`**: Imports and exposes functions from modular files within the module folder  
- **Modular Files** (e.g., `src/render/renderer.py`, `src/ontology/processor.py`): Contain the actual implementation of core methods
- **Tests**: All methods are tested in `src/tests/` with comprehensive test coverage

### 📁 File Organization Example

```
src/
├── 11_render.py                    # Thin orchestrator - imports from render/
├── render/
│   ├── __init__.py                 # Imports from renderer.py, pymdp/, etc.
│   ├── renderer.py                 # Core rendering functions
│   ├── pymdp/                      # PyMDP-specific rendering
│   ├── rxinfer/                    # RxInfer.jl-specific rendering
│   └── discopy/                    # DisCoPy-specific rendering
├── 10_ontology.py                  # Thin orchestrator - imports from ontology/
├── ontology/
│   ├── __init__.py                 # Imports from processor.py
│   └── processor.py                # Core ontology processing functions
└── tests/
    ├── test_render_integration.py  # Tests for render module
    └── test_ontology_integration.py # Tests for ontology module
```

### ✅ Correct Pattern Examples

- `11_render.py` imports from `src/render/` and calls `generate_pymdp_code()`, `generate_rxinfer_code()`, etc.
- `10_ontology.py` imports from `src/ontology/` and calls `process_ontology_file()`, `extract_ontology_terms()`, etc.
- Scripts contain only orchestration logic, not domain-specific processing code

### ❌ Incorrect Pattern Examples

- Defining `generate_pymdp_code()` directly in `11_render.py`
- Defining `process_ontology_file()` directly in `10_ontology.py`
- Any long method definitions (>20 lines) in numbered scripts

## Complete Pipeline Steps (0-23)

### Foundation and Setup

#### Step 0: Template Initialization (`0_template.py`)
- **Purpose**: Pipeline template and initialization
- **Input**: Project configuration
- **Output**: `template/template_results.json`
- **Key Features**: Pipeline structure setup, initialization logging

#### Step 1: Environment Setup (`1_setup.py`) **[CRITICAL]**
- **Purpose**: Virtual environment setup and dependency installation
- **Input**: Project requirements
- **Output**: `1_setup_output/` (setup artifacts and logs)
- **Key Features**: Virtual environment creation, dependency validation, package listing
- **Criticality**: Pipeline halts if this step fails

#### Step 2: Test Execution (`2_tests.py`)
- **Purpose**: Run comprehensive test suites
- **Input**: Test files in `src/tests/`
- **Output**: `2_tests_output/test_results/`
- **Key Features**: Unit tests, integration tests, coverage reports

### Core Processing Chain

#### Step 3: GNN File Discovery (`3_gnn.py`) **[CRITICAL]**
- **Purpose**: Discover and perform comprehensive parsing of GNN (.md) files
- **Input**: Target directory containing GNN files  
- **Output**: `3_gnn_output/`
- **Key Features**: Multi-format parsing (21 formats), round-trip validation, semantic preservation

#### Step 4: Model Registry (`4_model_registry.py`)
- **Purpose**: Model registry management and versioning
- **Input**: GNN files from step 3
- **Output**: `4_model_registry_output/`
- **Key Features**: Model registration, version tracking, metadata management

#### Step 5: Type Checking (`5_type_checker.py`)
- **Purpose**: GNN syntax validation and computational resource estimation
- **Input**: GNN files and model registry
- **Output**: `5_type_checker_output/`
- **Key Features**: Syntax validation, dimension checking, resource estimation

#### Step 6: Validation (`6_validation.py`)
- **Purpose**: Advanced validation and consistency checking
- **Input**: Type-checked GNN files
- **Output**: `6_validation_output/`
- **Key Features**: Cross-format validation, semantic consistency checks, performance profiling

#### Step 7: Export Processing (`7_export.py`)
- **Purpose**: Export GNN models to multiple formats
- **Input**: Validated GNN data
- **Output**: `7_export_output/` (contains `gnn_exports/` files)
- **Key Features**: Multi-format export (JSON, XML, GraphML, GEXF, Pickle)

### Visualization and Analysis

#### Step 8: Visualization (`8_visualization.py`)
- **Purpose**: Generate standard graph and matrix visualizations with safe-to-fail patterns
- **Input**: Exported GNN data
- **Output**: `8_visualization_output/` (contains visualization PNG/HTML and summaries)
- **Key Features**: Network graphs, matrix heatmaps, statistical plots, comprehensive fallback systems

#### Step 9: Advanced Visualization (`9_advanced_viz.py`)
- **Purpose**: Advanced visualization and interactive plots with comprehensive safety patterns
- **Input**: Exported GNN data
- **Output**: `9_advanced_viz_output/`
- **Key Features**: 3D visualizations, interactive dashboards, dynamic plots, robust error recovery

#### Step 10: Ontology Processing (`10_ontology.py`)
- **Purpose**: Active Inference Ontology processing and validation
- **Input**: Exported GNN data
- **Output**: `10_ontology_output/`
- **Key Features**: Ontology mapping, validation, semantic analysis

### Code Generation and Execution

#### Step 11: Code Rendering (`11_render.py`)
- **Purpose**: Code generation for PyMDP, RxInfer, ActiveInference.jl simulation environments
- **Input**: Validated GNN data
- **Output**: `11_render_output/`
- **Key Features**: Multi-platform code generation (PyMDP, RxInfer.jl, ActiveInference.jl, DisCoPy)

#### Step 12: Execution (`12_execute.py`)
- **Purpose**: Execute rendered simulation scripts with comprehensive safety patterns and result capture
- **Input**: Generated simulation code
- **Output**: `12_execute_output/` (contains `execution_results/`)
- **Key Features**: Multi-environment execution, result capture, performance monitoring, safe-to-fail execution

### AI and Machine Learning Integration

#### Step 13: LLM Processing (`13_llm.py`)
- **Purpose**: LLM-enhanced analysis, model interpretation, and AI assistance
- **Input**: GNN files and analysis results
- **Output**: `13_llm_output/`
- **Key Features**: AI-powered analysis, natural language interpretation, insights generation

#### Step 14: ML Integration (`14_ml_integration.py`)
- **Purpose**: Machine learning integration and model training
- **Input**: GNN data and execution results
- **Output**: `14_ml_integration_output/`
- **Key Features**: Model training, ML pipeline integration, performance analysis

### Specialized Processing

#### Step 15: Audio Processing (`15_audio.py`)
- **Purpose**: Audio generation (SAPF, Pedalboard, and other backends)
- **Input**: GNN data for sonification
- **Output**: `15_audio_output/` (contains audio files)
- **Key Features**: SAPF sonification, Pedalboard effects, multi-format audio output

#### Step 16: Analysis (`16_analysis.py`)
- **Purpose**: Advanced analysis and statistical processing
- **Input**: Visualization, ontology, and audio processing results
- **Output**: `16_analysis_output/`
- **Key Features**: Statistical analysis, performance metrics, comprehensive insights

#### Step 17: Integration (`17_integration.py`)
- **Purpose**: System integration and cross-module coordination
- **Input**: Analysis results
- **Output**: `17_integration_output/`
- **Key Features**: Cross-module coordination, system integration, unified reporting

#### Step 18: Security (`18_security.py`)
- **Purpose**: Security validation and access control
- **Input**: Integration results
- **Output**: `18_security_output/`
- **Key Features**: Security scanning, validation, access control

#### Step 19: Research (`19_research.py`)
- **Purpose**: Research tools and experimental features
- **Input**: Security-validated results
- **Output**: `19_research_output/`
- **Key Features**: Experimental analysis, research tools, advanced metrics

### Final Output Generation

#### Step 20: Website Generation (`20_website.py`)
- **Purpose**: Static HTML website generation from pipeline artifacts
- **Input**: Visualizations, reports, and analysis
- **Output**: `20_website_output/website/` (static HTML site)
- **Key Features**: Static site generation, documentation compilation, web interface

#### Step 21: MCP (`21_mcp.py`)
- **Purpose**: Model Context Protocol processing and tool registration
- **Input**: GNN files and available tools
- **Output**: `21_mcp_output/`
- **Key Features**: Tool discovery, registration, protocol validation

#### Step 22: GUI (`22_gui.py`)
- **Purpose**: Interactive GUI for constructing/editing GNN models
- **Input**: GNN models and templates
- **Output**: `22_gui_output/`
- **Key Features**: Two-pane UI (controls + markdown editor), headless artifact mode

#### Step 23: Report Generation (`23_report.py`)
- **Purpose**: Comprehensive analysis report generation
- **Input**: All pipeline artifacts
- **Output**: `23_report_output/`
- **Key Features**: Comprehensive reporting, summary generation, final documentation

## Pipeline Configuration

### Step Control
```bash
# Run specific steps
python main.py --only-steps 1,4,5

# Skip optional steps  
python main.py --skip-steps 3,13

# Run with full verbosity
python main.py --verbose
```

### Advanced Options
```bash
# LLM configuration
python main.py --llm-tasks summarize,explain --llm-timeout 300

# Resource estimation
python main.py --estimate-resources --strict

# Target specific directory
python main.py --target-dir /path/to/gnn/files --output-dir /custom/output
```

## Complete Output Structure (24 Steps)
```
output/
├── 0_template_output/
├── 1_setup_output/
├── 2_tests_output/
├── 3_gnn_output/
├── 4_model_registry_output/
├── 5_type_checker_output/
├── 6_validation_output/
├── 7_export_output/
├── 8_visualization_output/
├── 9_advanced_viz_output/
├── 10_ontology_output/
├── 11_render_output/
├── 12_execute_output/
├── 13_llm_output/
├── 14_ml_integration_output/
├── 15_audio_output/
├── 16_analysis_output/
├── 17_integration_output/
├── 18_security_output/
├── 19_research_output/
├── 20_website_output/
├── 21_mcp_output/
├── 22_gui_output/
├── 23_report_output/
└── pipeline_execution_summary.json
```

## Dependencies & Requirements

### Core Dependencies
- Python 3.10+
- NumPy, SciPy, Matplotlib
- NetworkX for graph processing
- PyMDP for Active Inference

### Optional Dependencies  
- JAX/JAXlib for high-performance computing
- DisCoPy for category theory
- OpenAI API for LLM features
- Julia for RxInfer.jl execution
- SAPF/Pedalboard for audio processing

### Development Dependencies
- pytest for testing
- mypy for type checking
- black, isort for code formatting

## Error Handling and Safety Features

### Critical Failures
- **Step 1 (setup) failure**: Halts entire pipeline
- **Step 3 (gnn) failure**: Halts entire pipeline
- **Missing dependencies**: Graceful degradation where possible

### Safe-to-Fail Implementation
- **Steps 8, 9, 12**: Comprehensive safe-to-fail patterns
- **Continuation Policy**: Standard exit codes (0=success, 1=critical, 2=warnings); continuation is config-driven (fail-fast vs continue)
- **Error Recovery**: Multiple fallback levels with detailed diagnostics
- **Output Generation**: Steps strive to produce useful outputs even under partial failure

### Non-Critical Failures
- Most steps log errors but allow pipeline continuation
- Comprehensive error capture in execution summary
- Detailed error classification and recovery suggestions

## Extension Guidelines

### Adding New Steps
1. Create `N_description.py` in `src/`
2. Follow thin orchestrator pattern (delegate to modules)
3. Add to `PIPELINE_STEP_CONFIGURATION` in `src/pipeline/config.py`
4. Define timeout in `STEP_TIMEOUTS`
5. Add argument support in `SCRIPT_ARG_SUPPORT`
6. Update documentation

### Adding New Export Formats
1. Extend `src/export/format_exporters.py`
2. Add to `AVAILABLE_EXPORT_FUNCTIONS`
3. Update format documentation

## Troubleshooting

### Common Issues
1. **Virtual environment problems**: Check Step 1 logs
2. **Import errors**: Verify dependencies installed
3. **Memory issues**: Monitor resource estimation output
4. **Timeout errors**: Adjust timeouts in config
5. **Visualization failures**: Check Step 8/9 fallback outputs

### Debug Mode
```bash
python main.py --verbose --only-steps 1,3,7 --target-dir test_files/
```

### Safety Verification
```bash
# Verify safe-to-fail patterns
python src/8_visualization.py --verbose  # Should always produce outputs
python src/9_advanced_viz.py --verbose   # Should create fallback HTML
python src/12_execute.py --verbose       # Should continue pipeline regardless
```

For detailed troubleshooting, see:
- [Common Errors](../troubleshooting/common_errors.md)
- [Performance Guide](../troubleshooting/performance.md)
- [Pipeline Flow Details](PIPELINE_FLOW.md)
- [Safety Patterns Documentation](../../src/README.md) 