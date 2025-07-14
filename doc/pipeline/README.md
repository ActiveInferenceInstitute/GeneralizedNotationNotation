# GNN Pipeline Documentation

> **📋 Document Metadata**  
> **Type**: Technical Guide | **Audience**: Developers, Users | **Complexity**: Intermediate  
> **Last Updated**: June 2025 | **Status**: Production-Ready  
> **Cross-References**: [Main Documentation](../README.md) | [API Reference](../api/README.md)

## Overview
The GNN Processing Pipeline is a comprehensive 14-step system for processing Generalized Notation Notation files from parsing through execution and analysis.

## Complete Pipeline Steps

### Core Processing Chain

#### Step 1: GNN File Discovery (`1_gnn.py`)
- **Purpose**: Discover and perform basic parsing of GNN (.md) files
- **Input**: Target directory containing GNN files  
- **Output**: `gnn_processing_step/1_gnn_discovery_report.md`
- **Key Features**: ModelName extraction, StateSpaceBlock detection, Connections identification

#### Step 2: Environment Setup (`2_setup.py`) **[CRITICAL]**
- **Purpose**: Virtual environment setup and dependency installation
- **Input**: Project requirements
- **Output**: Configured environment, package reports
- **Key Features**: Virtual environment creation, dependency validation, package listing
- **Criticality**: Pipeline halts if this step fails

#### Step 3: Test Execution (`3_tests.py`)
- **Purpose**: Run comprehensive test suites
- **Input**: Test files in `src/tests/`
- **Output**: `test_reports/pytest_report.xml`
- **Key Features**: Unit tests, integration tests, JUnit XML reports

#### Step 4: Type Checking (`4_type_checker.py`)
- **Purpose**: GNN syntax validation and computational resource estimation
- **Input**: GNN files from target directory
- **Output**: `gnn_type_check/type_check_report.md`, resource estimation HTML
- **Key Features**: Syntax validation, dimension checking, resource estimation

#### Step 5: Export Processing (`5_export.py`)
- **Purpose**: Export GNN models to multiple formats
- **Input**: GNN files
- **Output**: `gnn_exports/` with JSON, XML, GraphML, etc.
- **Key Features**: Multi-format export, summary generation, format validation

#### Step 6: Visualization (`6_visualization.py`)
- **Purpose**: Generate graphical representations of GNN models
- **Input**: GNN files
- **Output**: `visualization/` with PNG diagrams and reports
- **Key Features**: Factor graphs, model structure diagrams, interactive visualizations

#### Step 7: MCP Integration (`7_mcp.py`)
- **Purpose**: Model Context Protocol integration analysis
- **Input**: Project MCP modules
- **Output**: `mcp_processing_step/7_mcp_integration_report.md`
- **Key Features**: MCP tool discovery, API documentation, integration status

#### Step 8: Ontology Processing (`8_ontology.py`)
- **Purpose**: Active Inference ontology validation and mapping
- **Input**: GNN files, ontology terms JSON
- **Output**: `ontology_processing/ontology_processing_report.md`
- **Key Features**: Ontology term validation, annotation checking, concept mapping

### Execution Chain

#### Step 9: Code Rendering (`9_render.py`)
- **Purpose**: Generate executable code from GNN specifications
- **Input**: Exported GNN models (JSON)
- **Output**: `gnn_rendered_simulators/` with PyMDP and RxInfer code
- **Key Features**: PyMDP Python generation, RxInfer.jl TOML configuration

#### Step 10: Simulator Execution (`10_execute.py`)
- **Purpose**: Execute rendered simulator code
- **Input**: Rendered code from Step 9
- **Output**: Execution results, simulation outputs
- **Key Features**: PyMDP script execution, RxInfer.jl execution, Julia detection

### Advanced Analysis Chain

#### Step 11: LLM Integration (`11_llm.py`)
- **Purpose**: AI-enhanced analysis and documentation generation
- **Input**: GNN files, previous pipeline outputs
- **Output**: `llm_processing_step/` with AI-generated analysis
- **Key Features**: Model explanation, structure analysis, natural language summaries
- **Configuration**: Supports multiple LLM providers, configurable timeout

#### Step 12: DisCoPy Translation (`12_discopy.py`)
- **Purpose**: Category theory diagram generation using DisCoPy
- **Input**: GNN files
- **Output**: `discopy_gnn/` with categorical diagrams
- **Key Features**: String diagram generation, categorical model representation

#### Step 13: JAX Evaluation (`13_discopy_jax_eval.py`) **[EXPERIMENTAL]**
- **Purpose**: High-performance numerical evaluation using JAX
- **Input**: GNN files with tensor definitions
- **Output**: JAX evaluation results, performance visualizations
- **Key Features**: JAX-compiled evaluation, tensor visualization, performance analysis
- **Status**: Disabled by default, experimental

#### Step 14: Site Generation (`14_site.py`)
- **Purpose**: Comprehensive HTML summary website generation
- **Input**: All previous pipeline outputs
- **Output**: `gnn_pipeline_summary_site.html`
- **Key Features**: Interactive dashboard, result aggregation, artifact navigation

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

# DisCoPy options
python main.py --discopy-jax-seed 42

# Resource estimation
python main.py --estimate-resources --strict
```

## Output Structure
```
output/
├── gnn_processing_step/          # Step 1: Discovery reports
├── gnn_type_check/              # Step 4: Validation results  
├── gnn_exports/                 # Step 5: Multi-format exports
├── visualization/               # Step 6: Graphical diagrams
├── mcp_processing_step/         # Step 7: MCP integration
├── ontology_processing/         # Step 8: Ontology analysis
├── gnn_rendered_simulators/     # Step 9: Generated code
├── llm_processing_step/         # Step 11: AI analysis
├── discopy_gnn/                # Step 12: Category diagrams
├── test_reports/               # Step 3: Test results
├── logs/                       # Pipeline execution logs
├── pipeline_execution_summary.json
└── gnn_pipeline_summary_site.html
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

### Development Dependencies
- pytest for testing
- mypy for type checking
- black, isort for code formatting

## Error Handling

### Critical Failures
- **Step 2 failure**: Halts entire pipeline
- **Missing dependencies**: Graceful degradation where possible

### Non-Critical Failures
- Most steps log errors but allow pipeline continuation
- Comprehensive error capture in execution summary

## Extension Guidelines

### Adding New Steps
1. Create `N_description.py` in `src/`
2. Add to `PIPELINE_STEP_CONFIGURATION` in `src/pipeline/config.py`
3. Define timeout in `STEP_TIMEOUTS`
4. Add argument support in `SCRIPT_ARG_SUPPORT`
5. Update documentation

### Adding New Export Formats
1. Extend `src/export/format_exporters.py`
2. Add to `AVAILABLE_EXPORT_FUNCTIONS`
3. Update format documentation

## Troubleshooting

### Common Issues
1. **Virtual environment problems**: Check Step 2 logs
2. **Import errors**: Verify dependencies installed
3. **Memory issues**: Monitor resource estimation output
4. **Timeout errors**: Adjust timeouts in config

### Debug Mode
```bash
python main.py --verbose --only-steps 1,4 --target-dir test_files/
```

For detailed troubleshooting, see:
- [Common Errors](../troubleshooting/common_errors.md)
- [Performance Guide](../troubleshooting/performance.md)
- [FAQ](../troubleshooting/faq.md) 