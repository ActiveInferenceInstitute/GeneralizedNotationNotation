# GNN Pipeline Documentation

> **📋 Document Metadata**  
> **Type**: Technical Guide | **Audience**: Developers, Users | **Complexity**: Intermediate  
> **Last Updated**: June 2025 | **Status**: Production-Ready  
> **Cross-References**: [Main Documentation](../README.md) | [API Reference](../api/README.md)

## Overview
The GNN Processing Pipeline is a comprehensive 22-step system for processing Generalized Notation Notation files from parsing through execution and analysis.

## Complete Pipeline Steps

### Core Processing Chain

#### Step 0: Template Initialization (`0_template.py`)
- **Purpose**: Pipeline template and initialization
- **Input**: Project configuration
- **Output**: Template initialization artifacts
- **Key Features**: Pipeline structure setup, initialization logging

#### Step 1: Environment Setup (`1_setup.py`) **[CRITICAL]**
- **Purpose**: Virtual environment setup and dependency installation
- **Input**: Project requirements
- **Output**: Configured environment, package reports
- **Key Features**: Virtual environment creation, dependency validation, package listing
- **Criticality**: Pipeline halts if this step fails

#### Step 2: Test Execution (`2_tests.py`)
- **Purpose**: Run comprehensive test suites
- **Input**: Test files in `src/tests/`
- **Output**: `test_reports/pytest_report.xml`
- **Key Features**: Unit tests, integration tests, JUnit XML reports

#### Step 3: GNN File Discovery (`3_gnn.py`)
- **Purpose**: Discover and perform basic parsing of GNN (.md) files
- **Input**: Target directory containing GNN files  
- **Output**: `gnn_processing_step/gnn_processing_report.md`
- **Key Features**: ModelName extraction, StateSpaceBlock detection, Connections identification

#### Step 4: Model Registry (`4_model_registry.py`)
- **Purpose**: Model registry management and versioning
- **Input**: GNN files
- **Output**: `model_registry/model_registry.json`
- **Key Features**: Model registration, version tracking, metadata management

#### Step 5: Type Checking (`5_type_checker.py`)
- **Purpose**: GNN syntax validation and computational resource estimation
- **Input**: GNN files from target directory
- **Output**: `type_check/type_check_report.md`, resource estimation HTML
- **Key Features**: Syntax validation, dimension checking, resource estimation

#### Step 6: Validation (`6_validation.py`)
- **Purpose**: Advanced validation and consistency checking
- **Input**: GNN files
- **Output**: `validation/validation_report.md`
- **Key Features**: Cross-format validation, semantic consistency checks

#### Step 7: Export Processing (`7_export.py`)
- **Purpose**: Export GNN models to multiple formats
- **Input**: GNN files
- **Output**: `gnn_exports/` with JSON, XML, GraphML, etc.
- **Key Features**: Multi-format export (JSON, XML, GraphML, GEXF, Pickle)

#### Step 8: Visualization (`8_visualization.py`)
- **Purpose**: Generate standard graph and matrix visualizations
- **Input**: Exported GNN data
- **Output**: `visualization/` with PNG, SVG, HTML plots
- **Key Features**: Network graphs, matrix heatmaps, statistical plots

#### Step 9: Advanced Visualization (`9_advanced_viz.py`)
- **Purpose**: Advanced visualization and interactive plots
- **Input**: Exported GNN data
- **Output**: `advanced_visualization/` with interactive HTML
- **Key Features**: 3D visualizations, interactive dashboards, dynamic plots

#### Step 10: Ontology Processing (`10_ontology.py`)
- **Purpose**: Active Inference Ontology processing and validation
- **Input**: GNN files
- **Output**: `ontology_processing/ontology_results.json`
- **Key Features**: Ontology mapping, validation, semantic analysis

#### Step 11: Code Rendering (`11_render.py`)
- **Purpose**: Code generation for PyMDP, RxInfer, ActiveInference.jl simulation environments
- **Input**: Validated GNN data
- **Output**: `gnn_rendered_simulators/` with executable code
- **Key Features**: Multi-platform code generation, simulation templates

#### Step 12: Execution (`12_execute.py`)
- **Purpose**: Execute rendered simulation scripts with result capture
- **Input**: Generated simulation code
- **Output**: `execution_results/` with simulation outputs
- **Key Features**: Multi-environment execution, result capture, performance monitoring

#### Step 13: LLM Processing (`13_llm.py`)
- **Purpose**: LLM-enhanced analysis, model interpretation, and AI assistance
- **Input**: GNN files and analysis results
- **Output**: `llm_processing_step/llm_results.md`
- **Key Features**: AI-powered analysis, natural language interpretation, insights generation

#### Step 14: ML Integration (`14_ml_integration.py`)
- **Purpose**: Machine learning integration and model training
- **Input**: GNN data and execution results
- **Output**: `ml_integration/ml_integration_results.json`
- **Key Features**: Model training, ML pipeline integration, performance analysis

#### Step 15: Audio Processing (`15_audio.py`)
- **Purpose**: Audio generation (SAPF, Pedalboard, and other backends)
- **Input**: GNN data for sonification
- **Output**: `audio_processing_step/audio_results/` with WAV files
- **Key Features**: SAPF sonification, Pedalboard effects, multi-format audio output

#### Step 16: Analysis (`16_analysis.py`)
- **Purpose**: Advanced analysis and statistical processing
- **Input**: All pipeline artifacts
- **Output**: `analysis/analysis_results.json`
- **Key Features**: Statistical analysis, performance metrics, comprehensive insights

#### Step 17: Integration (`17_integration.py`)
- **Purpose**: System integration and cross-module coordination
- **Input**: All pipeline outputs
- **Output**: `integration/integration_results.json`
- **Key Features**: Cross-module coordination, system integration, unified reporting

#### Step 18: Security (`18_security.py`)
- **Purpose**: Security validation and access control
- **Input**: Pipeline artifacts
- **Output**: `security/security_results.json`
- **Key Features**: Security scanning, validation, access control

#### Step 19: Research (`19_research.py`)
- **Purpose**: Research tools and experimental features
- **Input**: Analysis results
- **Output**: `research/research_results.json`
- **Key Features**: Experimental analysis, research tools, advanced metrics

#### Step 20: Website Generation (`20_website.py`)
- **Purpose**: Static HTML website generation from pipeline artifacts
- **Input**: Visualizations, reports, and analysis
- **Output**: `website/` with static HTML site
- **Key Features**: Static site generation, documentation compilation, web interface

#### Step 21: Report Generation (`21_report.py`)
- **Purpose**: Comprehensive analysis report generation
- **Input**: All pipeline artifacts
- **Output**: `report_processing_step/report_results.json`
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

# DisCoPy options
python main.py --discopy-jax-seed 42

# Resource estimation
python main.py --estimate-resources --strict
```

## Output Structure
```
output/
├── setup_artifacts/             # Step 1: Environment setup
├── gnn_processing_step/         # Step 2: Discovery reports
├── test_reports/               # Step 3: Test results
├── type_check/                 # Step 4: Validation results  
├── gnn_exports/                # Step 5: Multi-format exports
├── visualization/              # Step 6: Graphical diagrams
├── mcp_processing_step/        # Step 7: MCP integration
├── ontology_processing/        # Step 8: Ontology analysis
├── gnn_rendered_simulators/    # Step 9: Generated code
├── execution_results/          # Step 10: Simulation results
├── llm_processing_step/        # Step 11: AI analysis
├── audio_processing_step/      # Step 12: Audio generation
├── website/                    # Step 13: Static website
├── report_processing_step/     # Step 14: Comprehensive reports
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
- **Step 1 (setup) failure**: Halts entire pipeline
- **Step 2 (gnn) failure**: Halts entire pipeline
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