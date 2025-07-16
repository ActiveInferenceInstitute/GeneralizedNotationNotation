# GNN Processing Pipeline - Comprehensive Documentation

## Overview

The GNN (Generalized Notation Notation) Processing Pipeline is a sophisticated, modular system for processing Active Inference generative models. This pipeline consists of 13 numbered steps that transform GNN specifications into various formats and provide comprehensive analysis, including cutting-edge audio representations through SAPF (Sound As Pure Form).

### Pipeline Flow

The pipeline processes GNN models through a systematic workflow:

1. **Discovery & Parsing** (Steps 1-4): Find and validate GNN files
2. **Export & Visualization** (Steps 5-6): Generate multiple output formats and visualizations  
3. **Integration & Analysis** (Steps 7-8): MCP tools and ontology processing
4. **Execution & Enhancement** (Steps 9-11): Code generation, simulation, and LLM analysis
5. **Advanced Representations** (Steps 12-13): Site generation and audio sonification

## Pipeline Architecture

### Core Components

- **Main Orchestrator**: `main.py` - Discovers and executes numbered pipeline scripts (1-13)
- **Centralized Utilities**: `utils/` package providing logging, argument parsing, and validation
- **Pipeline Configuration**: `pipeline/config.py` - Centralized configuration management
- **Pipeline Validation**: `pipeline_validation.py` - Validates consistency and functionality

### 13-Step Pipeline

| Step | Script | Purpose | Status | Output Directory |
|------|--------|---------|--------|------------------|
| 1 | `1_gnn.py` | GNN file discovery and parsing | ✅ WORKING | `gnn_processing_step/` |
| 2 | `2_setup.py` | Environment setup and dependencies | ✅ WORKING | `setup_artifacts/` |
| 3 | `3_tests.py` | Test execution and validation | ✅ WORKING | `test_reports/` |
| 4 | `4_type_checker.py` | Type checking and validation | ✅ WORKING | `type_check/` |
| 5 | `5_export.py` | Multi-format export (JSON, XML, etc.) | ✅ WORKING | `gnn_exports/` |
| 6 | `6_visualization.py` | Graph and statistical visualizations | ✅ WORKING | `visualization/` |
| 7 | `7_mcp.py` | Model Context Protocol operations | ⚠️ PARTIAL | `mcp_processing_step/` |
| 8 | `8_ontology.py` | Ontology processing and validation | ✅ WORKING | `ontology_processing/` |
| 9 | `9_render.py` | Code generation (PyMDP, RxInfer, ActiveInference.jl) | ✅ WORKING | `gnn_rendered_simulators/` |
| 10 | `10_execute.py` | Execute rendered simulators | ⚠️ NEEDS_DEPS | `execution_results/` |
| 11 | `11_llm.py` | LLM-enhanced analysis | ✅ WORKING | `llm_processing_step/` |
| 12 | `12_website.py` | HTML website generation | ⚠️ PARTIAL | `website/` |
| 13 | `13_sapf.py` | SAPF audio generation | ✅ WORKING | `sapf_processing_step/` |

**Note**: The pipeline is designed to be fully extensible, with each step building upon previous outputs while remaining independently executable for targeted processing.

## Functional Status Analysis

### ✅ Fully Functional (10/13 steps)
Scripts 1-6, 8-9, 11, and 13 are fully operational with proper logging, error handling, and output generation.

### ⚠️ Partially Functional (3/13 steps)
- **Step 7 (MCP)**: Core functionality works but may need MCP system initialization
- **Step 10 (Execute)**: PyMDP/RxInfer execution depends on availability of dependencies
- **Step 12 (Website)**: Basic HTML website generation works, full generator may need additional dependencies

## Code Quality Assessment

### Strengths

1. **Consistent Logging**: All scripts use centralized `utils` package with correlation IDs
2. **Modular Design**: Each step is independent and can run standalone
3. **Error Handling**: Comprehensive try/catch blocks with graceful failures
4. **Type Hints**: Extensive use of Python type annotations
5. **Documentation**: Well-documented functions and classes
6. **Flexible Arguments**: Support for both pipeline and standalone execution
7. **Output Management**: Structured output directories with clear naming
8. **Centralized Configuration**: Unified configuration management via `pipeline/config.py`

### Areas for Improvement

1. **Dependency Management**: Some steps need optional dependency handling
2. **Configuration**: Centralized configuration is implemented but could be enhanced
3. **Parallel Processing**: Some steps could run in parallel
4. **Caching**: Intermediate results could be cached for re-runs

## Technical Implementation Details

### Centralized Utilities (`utils/`)

The pipeline uses a sophisticated utility system:

```python
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_warning,
    log_step_error,
    validate_output_directory,
    UTILS_AVAILABLE
)
```

Features:
- **Correlation IDs**: Every step gets a unique correlation ID for tracing
- **Structured Logging**: JSON-structured log data for analysis
- **Fallback Support**: Graceful degradation if utilities unavailable
- **Performance Tracking**: Built-in performance monitoring

### Pipeline Configuration (`pipeline/`)

Centralized configuration management:

```python
from pipeline import (
    get_pipeline_config,
    get_output_dir_for_script,
    STEP_METADATA
)
```

Features:
- **Step Metadata**: Centralized metadata for all pipeline steps
- **Dependency Management**: Automatic dependency resolution
- **Output Directory Management**: Standardized output directory structure
- **Environment Overrides**: Support for environment variable configuration

### Output Structure

All pipeline outputs are organized under a main output directory:

```
output/
├── gnn_processing_step/           # Step 1: GNN discovery results
├── setup_artifacts/               # Step 2: Environment setup logs
├── test_reports/                  # Step 3: Test execution results
├── type_check/                   # Step 4: Type checking reports
├── gnn_exports/                  # Step 5: Multi-format exports
├── visualization/                # Step 6: Generated visualizations
├── mcp_processing_step/          # Step 7: MCP integration reports
├── ontology_processing/          # Step 8: Ontology analysis
├── gnn_rendered_simulators/      # Step 9: Generated code
├── execution_results/            # Step 10: Simulation results
├── llm_processing_step/          # Step 11: LLM analysis
├── website/                      # Step 12: HTML documentation
├── sapf_processing_step/         # Step 13: SAPF audio generation
└── logs/                         # Pipeline execution logs
```

### Validation System

The `pipeline_validation.py` script provides comprehensive validation:

- **Import Consistency**: Validates all scripts use centralized utilities
- **Output Verification**: Checks expected outputs are generated
- **Logging Patterns**: Ensures consistent logging usage
- **Error Detection**: Identifies common integration issues
- **Configuration Validation**: Validates pipeline configuration consistency

## Usage

### Basic Pipeline Execution

```bash
python3 src/main.py --target-dir input/gnn_files --output-dir output

python3 src/main.py --target-dir input/gnn_files --output-dir output --verbose

python3 src/main.py --only-steps 1,2,3 --target-dir input/gnn_files --output-dir output

python3 src/main.py --skip-steps 10,12,13 --target-dir input/gnn_files --output-dir output
```

### Individual Step Execution

Each step can be run independently:

```bash
python3 src/1_gnn.py --target-dir input/gnn_files --output-dir output --verbose

python3 src/4_type_checker.py --target-dir input/gnn_files --output-dir output --strict

python3 src/5_export.py --target-dir input/gnn_files --output-dir output

python3 src/13_sapf.py --target-dir input/gnn_files --output-dir output --duration 30
```

### Validation

```bash
python3 src/main.py --target-dir input/gnn_files --estimate-resources --verbose
```

## Dependencies

### Core Dependencies (Required)
- Python 3.8+
- pathlib, argparse, json, datetime (standard library)

### Step-Specific Dependencies
- **Step 2**: Virtual environment tools
- **Step 3**: pytest
- **Step 5**: networkx (optional, for graph exports)
- **Step 6**: matplotlib, graphviz (for visualizations)
- **Step 9**: PyMDP, RxInfer.jl, ActiveInference.jl (optional, for code generation)
- **Step 10**: PyMDP, Julia/RxInfer.jl, ActiveInference.jl (optional, for simulation execution)
- **Step 11**: OpenAI API or similar LLM access
- **Step 12**: Jinja2 or similar templating (for advanced website generation)
- **Step 13**: SAPF binary (optional), numpy, wave (for audio generation)

## Error Handling and Recovery

The pipeline is designed with graceful failure modes:

1. **Non-Critical Failures**: Steps that fail don't stop the entire pipeline
2. **Dependency Checks**: Scripts check for required dependencies before execution
3. **Fallback Modes**: Many steps have fallback implementations
4. **Detailed Logging**: All failures are logged with context and suggested fixes
5. **Critical Step Protection**: Only critical steps (like setup) halt the pipeline

## Performance Characteristics

Based on current implementation:

- **Small GNN files** (< 1MB): Full pipeline in 30-60 seconds
- **Medium GNN files** (1-10MB): Full pipeline in 2-5 minutes  
- **Large GNN files** (10MB+): Full pipeline in 5-15 minutes

Bottlenecks typically occur in:
- Step 11 (LLM API calls)
- Step 9 (Code generation when enabled)
- Step 10 (Simulation execution)
- Step 13 (Audio generation for large models)

## Future Enhancements

1. **Parallel Execution**: Run independent steps concurrently
2. **Caching System**: Cache intermediate results for faster re-runs
3. **Configuration Management**: Enhanced YAML/TOML configuration files
4. **Web Interface**: Browser-based pipeline management
5. **Cloud Integration**: Support for cloud-based execution
6. **Plugin System**: Allow custom steps via plugins
7. **Monitoring Dashboard**: Real-time pipeline status monitoring

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed in the correct environment
2. **Permission Errors**: Check write permissions for output directory
3. **Memory Issues**: Large GNN files may require more RAM for processing
4. **Dependency Conflicts**: Use virtual environments to isolate dependencies

### Debug Mode

Enable debug mode for detailed troubleshooting:

```bash
python3 src/main.py --verbose --target-dir src/gnn/examples --output-dir output 2>&1 | tee debug.log
```

This captures all debug output including correlation IDs for tracing issues across the pipeline.

## Contributing

When adding new pipeline steps:

1. Follow the numbered naming convention (`N_new_step.py` where N is 14 or higher)
2. Use centralized utilities from `utils/` package
3. Implement proper error handling and logging
4. Add step configuration to `pipeline/config.py`
5. Add output validation to `pipeline_validation.py`
6. Update this documentation and main.py
7. Include unit tests
8. Update step dependency mapping in pipeline configuration

The pipeline is designed to be extensible while maintaining consistency and reliability across all components. 

### Standardization Improvements
All pipeline scripts have been standardized using the template from utils/pipeline_template.py for consistent argument handling, logging, and execution. 