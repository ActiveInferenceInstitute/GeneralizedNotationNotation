# GNN Pipeline Documentation

> **📋 Document Metadata**  
> **Type**: Technical Guide | **Audience**: Developers, Users | **Complexity**: Intermediate  
> **Last Updated**: July 2025 | **Status**: Production-Ready  
> **Cross-References**: [Main Documentation](../README.md) | [API Reference](../api/README.md)

## Overview
The GNN Processing Pipeline is a comprehensive 22-step system for processing Generalized Notation Notation files from initialization through execution, analysis, and comprehensive reporting.

## Complete Pipeline Steps (0-21)

### Foundation and Setup

#### Step 0: Template Initialization (`0_template.py`)
- **Purpose**: Pipeline template and initialization
- **Input**: Project configuration
- **Output**: `template/template_results.json`
- **Key Features**: Pipeline structure setup, initialization logging

#### Step 1: Environment Setup (`1_setup.py`) **[CRITICAL]**
- **Purpose**: Virtual environment setup and dependency installation
- **Input**: Project requirements
- **Output**: `setup_artifacts/setup_results.json`
- **Key Features**: Virtual environment creation, dependency validation, package listing
- **Criticality**: Pipeline halts if this step fails

#### Step 2: Test Execution (`2_tests.py`)
- **Purpose**: Run comprehensive test suites
- **Input**: Test files in `src/tests/`
- **Output**: `test_reports/test_results.json`
- **Key Features**: Unit tests, integration tests, coverage reports

### Core Processing Chain

#### Step 3: GNN File Discovery (`3_gnn.py`) **[CRITICAL]**
- **Purpose**: Discover and perform comprehensive parsing of GNN (.md) files
- **Input**: Target directory containing GNN files  
- **Output**: `gnn_processing_step/gnn_processing_results.json`
- **Key Features**: Multi-format parsing (21 formats), round-trip validation, semantic preservation

#### Step 4: Model Registry (`4_model_registry.py`)
- **Purpose**: Model registry management and versioning
- **Input**: GNN files from step 3
- **Output**: `model_registry/model_registry_summary.json`
- **Key Features**: Model registration, version tracking, metadata management

#### Step 5: Type Checking (`5_type_checker.py`)
- **Purpose**: GNN syntax validation and computational resource estimation
- **Input**: GNN files and model registry
- **Output**: `type_check/type_check_results.json`
- **Key Features**: Syntax validation, dimension checking, resource estimation

#### Step 6: Validation (`6_validation.py`)
- **Purpose**: Advanced validation and consistency checking
- **Input**: Type-checked GNN files
- **Output**: `validation/validation_summary.json`
- **Key Features**: Cross-format validation, semantic consistency checks, performance profiling

#### Step 7: Export Processing (`7_export.py`)
- **Purpose**: Export GNN models to multiple formats
- **Input**: Validated GNN data
- **Output**: `gnn_exports/` with JSON, XML, GraphML, GEXF, Pickle files
- **Key Features**: Multi-format export (JSON, XML, GraphML, GEXF, Pickle)

### Visualization and Analysis

#### Step 8: Visualization (`8_visualization.py`)
- **Purpose**: Generate standard graph and matrix visualizations with safe-to-fail patterns
- **Input**: Exported GNN data
- **Output**: `visualization/visualization_results.json` with PNG, SVG, HTML plots
- **Key Features**: Network graphs, matrix heatmaps, statistical plots, comprehensive fallback systems

#### Step 9: Advanced Visualization (`9_advanced_viz.py`)
- **Purpose**: Advanced visualization and interactive plots with comprehensive safety patterns
- **Input**: Exported GNN data
- **Output**: `advanced_visualization/` with interactive HTML dashboards
- **Key Features**: 3D visualizations, interactive dashboards, dynamic plots, robust error recovery

#### Step 10: Ontology Processing (`10_ontology.py`)
- **Purpose**: Active Inference Ontology processing and validation
- **Input**: Exported GNN data
- **Output**: `ontology_processing/ontology_results.json`
- **Key Features**: Ontology mapping, validation, semantic analysis

### Code Generation and Execution

#### Step 11: Code Rendering (`11_render.py`)
- **Purpose**: Code generation for PyMDP, RxInfer, ActiveInference.jl simulation environments
- **Input**: Validated GNN data
- **Output**: `gnn_rendered_simulators/` with executable code
- **Key Features**: Multi-platform code generation (PyMDP, RxInfer.jl, ActiveInference.jl, DisCoPy)

#### Step 12: Execution (`12_execute.py`)
- **Purpose**: Execute rendered simulation scripts with comprehensive safety patterns and result capture
- **Input**: Generated simulation code
- **Output**: `execution_results/execution_results.json` with simulation outputs
- **Key Features**: Multi-environment execution, result capture, performance monitoring, safe-to-fail execution

### AI and Machine Learning Integration

#### Step 13: LLM Processing (`13_llm.py`)
- **Purpose**: LLM-enhanced analysis, model interpretation, and AI assistance
- **Input**: GNN files and analysis results
- **Output**: `llm_processing_step/llm_results.json`
- **Key Features**: AI-powered analysis, natural language interpretation, insights generation

#### Step 14: ML Integration (`14_ml_integration.py`)
- **Purpose**: Machine learning integration and model training
- **Input**: GNN data and execution results
- **Output**: `ml_integration/ml_integration_results.json`
- **Key Features**: Model training, ML pipeline integration, performance analysis

### Specialized Processing

#### Step 15: Audio Processing (`15_audio.py`)
- **Purpose**: Audio generation (SAPF, Pedalboard, and other backends)
- **Input**: GNN data for sonification
- **Output**: `audio_processing_step/` with WAV files and audio analysis
- **Key Features**: SAPF sonification, Pedalboard effects, multi-format audio output

#### Step 16: Analysis (`16_analysis.py`)
- **Purpose**: Advanced analysis and statistical processing
- **Input**: Visualization, ontology, and audio processing results
- **Output**: `analysis/analysis_results.json`
- **Key Features**: Statistical analysis, performance metrics, comprehensive insights

#### Step 17: Integration (`17_integration.py`)
- **Purpose**: System integration and cross-module coordination
- **Input**: Analysis results
- **Output**: `integration/integration_results.json`
- **Key Features**: Cross-module coordination, system integration, unified reporting

#### Step 18: Security (`18_security.py`)
- **Purpose**: Security validation and access control
- **Input**: Integration results
- **Output**: `security/security_results.json`
- **Key Features**: Security scanning, validation, access control

#### Step 19: Research (`19_research.py`)
- **Purpose**: Research tools and experimental features
- **Input**: Security-validated results
- **Output**: `research/research_results.json`
- **Key Features**: Experimental analysis, research tools, advanced metrics

### Final Output Generation

#### Step 20: Website Generation (`20_website.py`)
- **Purpose**: Static HTML website generation from pipeline artifacts
- **Input**: Visualizations, reports, and analysis
- **Output**: `website/` with static HTML site
- **Key Features**: Static site generation, documentation compilation, web interface

#### Step 21: Report Generation (`21_report.py`)
- **Purpose**: Comprehensive analysis report generation
- **Input**: All pipeline artifacts
- **Output**: `report_processing_step/` with comprehensive reports
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

## Complete Output Structure (22 Steps)
```
output/
├── template/                        # Step 0: Template initialization
├── setup_artifacts/                 # Step 1: Environment setup
├── test_reports/                    # Step 2: Test results
├── gnn_processing_step/             # Step 3: GNN discovery and parsing
├── model_registry/                  # Step 4: Model registry
├── type_check/                      # Step 5: Type checking results
├── validation/                      # Step 6: Validation results
├── gnn_exports/                     # Step 7: Multi-format exports
├── visualization/                   # Step 8: Standard visualizations
├── advanced_visualization/          # Step 9: Advanced visualizations
├── ontology_processing/             # Step 10: Ontology analysis
├── gnn_rendered_simulators/         # Step 11: Generated simulation code
├── execution_results/               # Step 12: Simulation execution results
├── llm_processing_step/             # Step 13: LLM analysis
├── ml_integration/                  # Step 14: ML integration
├── audio_processing_step/           # Step 15: Audio generation
├── analysis/                        # Step 16: Advanced analysis
├── integration/                     # Step 17: System integration
├── security/                        # Step 18: Security validation
├── research/                        # Step 19: Research tools
├── website/                         # Step 20: Static website
├── report_processing_step/          # Step 21: Comprehensive reports
├── logs/                           # Pipeline execution logs
├── pipeline_execution_summary.json # Overall pipeline results
└── gnn_pipeline_summary_site.html  # Pipeline summary website
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
- **All Steps**: Guaranteed pipeline continuation (return 0)
- **Error Recovery**: Multiple fallback levels with detailed diagnostics
- **Output Generation**: All steps produce outputs regardless of internal success/failure

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