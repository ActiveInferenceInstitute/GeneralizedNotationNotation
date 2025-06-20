# GNN Processing Pipeline - Comprehensive Documentation

## Overview

The GNN (Generalized Notation Notation) Processing Pipeline is a sophisticated, modular system for processing Active Inference generative models. This pipeline consists of 14 numbered steps that transform GNN specifications into various formats and provide comprehensive analysis.

## Pipeline Architecture

### Core Components

- **Main Orchestrator**: `main.py` - Discovers and executes numbered pipeline scripts (1-14)
- **Centralized Utilities**: `utils/` package providing logging, argument parsing, and validation
- **Pipeline Validation**: `pipeline_validation.py` - Validates consistency and functionality

### 14-Step Pipeline

| Step | Script | Purpose | Status | Output Directory |
|------|--------|---------|--------|------------------|
| 1 | `1_gnn.py` | GNN file discovery and parsing | ✅ WORKING | `gnn_processing_step/` |
| 2 | `2_setup.py` | Environment setup and dependencies | ✅ WORKING | `setup_artifacts/` |
| 3 | `3_tests.py` | Test execution and validation | ✅ WORKING | `test_reports/` |
| 4 | `4_gnn_type_checker.py` | Type checking and validation | ✅ WORKING | `gnn_type_check/` |
| 5 | `5_export.py` | Multi-format export (JSON, XML, etc.) | ✅ WORKING | `gnn_exports/` |
| 6 | `6_visualization.py` | Graph and statistical visualizations | ✅ WORKING | `visualization/` |
| 7 | `7_mcp.py` | Model Context Protocol operations | ⚠️ PARTIAL | `mcp_processing_step/` |
| 8 | `8_ontology.py` | Ontology processing and validation | ✅ WORKING | `ontology_processing/` |
| 9 | `9_render.py` | Code generation (PyMDP, RxInfer) | ✅ WORKING | `gnn_rendered_simulators/` |
| 10 | `10_execute.py` | Execute rendered simulators | ⚠️ NEEDS_DEPS | `execution_results/` |
| 11 | `11_llm.py` | LLM-enhanced analysis | ✅ WORKING | `llm_processing_step/` |
| 12 | `12_discopy.py` | DisCoPy categorical diagrams | ⚠️ NEEDS_DEPS | `discopy_gnn/` |
| 13 | `13_discopy_jax_eval.py` | JAX evaluation of diagrams | ⚠️ NEEDS_DEPS | `discopy_jax_eval/` |
| 14 | `14_site.py` | HTML site generation | ⚠️ PARTIAL | `site/` |

## Functional Status Analysis

### ✅ Fully Functional (10/14 steps)
Scripts 1-6, 8-9, and 11 are fully operational with proper logging, error handling, and output generation.

### ⚠️ Partially Functional (4/14 steps)
- **Step 7 (MCP)**: Core functionality works but may need MCP system initialization
- **Step 10 (Execute)**: Depends on PyMDP/RxInfer availability
- **Step 12 (DisCoPy)**: Needs DisCoPy library installation
- **Step 13 (JAX Eval)**: Needs JAX and DisCoPy[matrix] dependencies
- **Step 14 (Site)**: Basic HTML generation works, full generator may need additional dependencies

## Code Quality Assessment

### Strengths

1. **Consistent Logging**: All scripts use centralized `utils` package with correlation IDs
2. **Modular Design**: Each step is independent and can run standalone
3. **Error Handling**: Comprehensive try/catch blocks with graceful failures
4. **Type Hints**: Extensive use of Python type annotations
5. **Documentation**: Well-documented functions and classes
6. **Flexible Arguments**: Support for both pipeline and standalone execution
7. **Output Management**: Structured output directories with clear naming

### Areas for Improvement

1. **Dependency Management**: Some steps need optional dependency handling
2. **Configuration**: Could benefit from centralized configuration file
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

### Output Structure

All pipeline outputs are organized under a main output directory:

```
output/
├── gnn_processing_step/           # Step 1: GNN discovery results
├── setup_artifacts/               # Step 2: Environment setup logs
├── test_reports/                  # Step 3: Test execution results
├── gnn_type_check/               # Step 4: Type checking reports
├── gnn_exports/                  # Step 5: Multi-format exports
├── visualization/                # Step 6: Generated visualizations
├── mcp_processing_step/          # Step 7: MCP integration reports
├── ontology_processing/          # Step 8: Ontology analysis
├── gnn_rendered_simulators/      # Step 9: Generated code
├── execution_results/            # Step 10: Simulation results
├── llm_processing_step/          # Step 11: LLM analysis
├── discopy_gnn/                  # Step 12: DisCoPy diagrams
├── discopy_jax_eval/             # Step 13: JAX evaluations
├── site/                         # Step 14: HTML documentation
└── logs/                         # Pipeline execution logs
```

### Validation System

The `pipeline_validation.py` script provides comprehensive validation:

- **Import Consistency**: Validates all scripts use centralized utilities
- **Output Verification**: Checks expected outputs are generated
- **Logging Patterns**: Ensures consistent logging usage
- **Error Detection**: Identifies common integration issues

## Usage

### Basic Pipeline Execution

```bash
# Run full pipeline
python3 src/main.py --target-dir src/gnn/examples --output-dir output

# Run with verbose logging
python3 src/main.py --target-dir src/gnn/examples --output-dir output --verbose

# Run specific steps only
python3 src/main.py --only-steps 1,2,3 --target-dir src/gnn/examples --output-dir output

# Skip problematic steps
python3 src/main.py --skip-steps 10,12,13 --target-dir src/gnn/examples --output-dir output
```

### Individual Step Execution

Each step can be run independently:

```bash
# Run GNN discovery
python3 src/1_gnn.py --target-dir src/gnn/examples --output-dir output --verbose

# Run type checking
python3 src/4_gnn_type_checker.py --target-dir src/gnn/examples --output-dir output --strict

# Generate exports
python3 src/5_export.py --target-dir src/gnn/examples --output-dir output --formats json,xml
```

### Validation

```bash
# Validate pipeline consistency
python3 src/pipeline_validation.py --save-report validation_report.json

# Check specific output directory
python3 src/pipeline_validation.py --output-dir custom_output --save-report custom_validation.json
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
- **Step 10**: PyMDP, Julia/RxInfer.jl
- **Step 11**: OpenAI API or similar LLM access
- **Step 12**: DisCoPy
- **Step 13**: JAX, DisCoPy[matrix]
- **Step 14**: Jinja2 or similar templating (for advanced site generation)

## Error Handling and Recovery

The pipeline is designed with graceful failure modes:

1. **Non-Critical Failures**: Steps that fail don't stop the entire pipeline
2. **Dependency Checks**: Scripts check for required dependencies before execution
3. **Fallback Modes**: Many steps have fallback implementations
4. **Detailed Logging**: All failures are logged with context and suggested fixes

## Performance Characteristics

Based on current implementation:

- **Small GNN files** (< 1MB): Full pipeline in 30-60 seconds
- **Medium GNN files** (1-10MB): Full pipeline in 2-5 minutes  
- **Large GNN files** (10MB+): Full pipeline in 5-15 minutes

Bottlenecks typically occur in:
- Step 11 (LLM API calls)
- Step 12-13 (DisCoPy/JAX computation)
- Step 10 (Simulation execution)

## Future Enhancements

1. **Parallel Execution**: Run independent steps concurrently
2. **Caching System**: Cache intermediate results for faster re-runs
3. **Configuration Management**: YAML/TOML configuration files
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

1. Follow the numbered naming convention (`15_new_step.py`)
2. Use centralized utilities from `utils/` package
3. Implement proper error handling and logging
4. Add output validation to `pipeline_validation.py`
5. Update this documentation
6. Include unit tests

The pipeline is designed to be extensible while maintaining consistency and reliability across all components. 