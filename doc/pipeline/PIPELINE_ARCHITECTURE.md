# GNN Pipeline Architecture

## Overview

The GNN Processing Pipeline is a comprehensive, modular system for processing Generalized Notation Notation (GNN) files through a series of well-defined stages. Each stage performs specific operations and outputs results that can be consumed by subsequent stages or used independently.

## Pipeline Flow

```mermaid
graph TD
    A[1_gnn.py] --> B[2_setup.py]
    B --> C[4_type_checker.py]
    C --> D[5_export.py]
    D --> E[6_visualization.py]
    E --> F[7_mcp.py]
    F --> G[8_ontology.py]
    G --> H[9_render.py]
    H --> I[10_execute.py]
    I --> J[11_llm.py]
    J --> K[12_website.py]
    K --> L[13_sapf.py]
    
    B -.-> M[3_tests.py]
    
    style B fill:#ffcccc
    style M fill:#e6f3ff
```

## Stage Descriptions

### Core Processing Stages

| Stage | Script | Purpose | Timeout | Critical |
|-------|--------|---------|---------|----------|
| 1 | `1_gnn.py` | GNN file discovery and basic parsing | 2 min | No |
| 2 | `2_setup.py` | Environment setup and dependency management | **20 min** | **Yes** |
| 3 | `3_tests.py` | Test execution (optional) | 5 min | No |
| 4 | `4_type_checker.py` | Type checking and resource estimation | 2 min | No |
| 5 | `5_export.py` | Export to multiple formats (JSON, XML, etc.) | 2 min | No |
| 6 | `6_visualization.py` | Generate visual representations | 5 min | No |
| 7 | `7_mcp.py` | MCP integration analysis | 2 min | No |
| 8 | `8_ontology.py` | Ontology validation and mapping | 2 min | No |
| 9 | `9_render.py` | Code generation for simulators | 2 min | No |
| 10 | `10_execute.py` | Execute rendered simulator code | 5 min | No |
| 11 | `11_llm.py` | LLM-powered analysis and documentation | Configurable | No |
| 12 | `12_website.py` | Website and HTML summary generation | 2 min | No |
| 13 | `13_sapf.py` | SAPF audio generation and processing | 5 min | No |

### Step Dependencies

- **Step 2 (setup.py)** is critical - pipeline halts on failure
- **Step 3 (tests.py)** is optional and disabled by default
- **Step 13 (discopy_jax_eval.py)** is experimental and disabled by default
- Most steps depend on the output of **Step 1 (gnn.py)** for GNN file discovery
- **Step 13 (sapf.py)** completes processing with audio generation

## Configuration Management

### Step Configuration

Steps can be enabled/disabled via `PIPELINE_STEP_CONFIGURATION` in `src/pipeline/config.py`:

```python
PIPELINE_STEP_CONFIGURATION = {
    "1_gnn.py": True,
    "2_setup.py": True,
    "3_tests.py": False,  # Disabled by default
    # ... etc
}
```

### Timeout Configuration

Each step has a configured timeout in `STEP_TIMEOUTS`:

- **Short steps** (2 minutes): Basic processing, exports
- **Medium steps** (5 minutes): Visualization, execution, testing
- **Long steps** (20 minutes): Environment setup
- **Configurable**: LLM processing (via `--llm-timeout`)

### Critical Steps

Only **Step 2 (setup.py)** is marked as critical. If it fails, the entire pipeline halts to prevent cascading failures from missing dependencies.

## Output Structure

```
output/
├── gnn_processing_step/           # Step 1 output
├── gnn_type_check/               # Step 4 output
├── gnn_exports/                  # Step 5 output
├── visualization/                # Step 6 output
├── mcp_processing_step/          # Step 7 output
├── ontology_processing/          # Step 8 output
├── gnn_rendered_simulators/      # Step 9 output
├── llm_processing_step/          # Step 11 output
├── discopy_gnn/                  # Step 12 output
├── logs/                         # Pipeline logs
├── pipeline_execution_summary.json
└── gnn_pipeline_summary_site.html
```

## Error Handling

### Error Types

1. **SUCCESS**: Step completed without issues
2. **FAILED_NONZERO_EXIT**: Process exited with error code
3. **FAILED_TIMEOUT**: Step exceeded configured timeout
4. **ERROR_UNHANDLED_EXCEPTION**: Unexpected error during execution

### Recovery Strategies

- **Non-critical steps**: Pipeline continues on failure
- **Critical steps**: Pipeline halts immediately
- **Timeout handling**: Graceful termination with output capture
- **Detailed logging**: Full stdout/stderr capture for debugging

## Performance Monitoring

The pipeline tracks:

- **Memory usage**: Peak memory consumption per step
- **Execution time**: Duration of each step
- **System resources**: CPU, disk space, etc.
- **Step dependencies**: Which steps were skipped/failed

## Command Line Interface

### Basic Usage

```bash
python main.py                          # Run with defaults
python main.py --verbose               # Enable detailed logging
python main.py --skip-steps 3,13       # Skip optional steps
python main.py --only-steps 1,4,5      # Run only specific steps
```

### Advanced Options

```bash
python main.py \
  --target-dir /path/to/gnn/files \
  --output-dir /custom/output \
  --llm-timeout 300 \
  --strict \
  --no-estimate-resources
```

## Extension Points

### Adding New Steps

1. Create `N_name.py` script in `src/`
2. Add entry to `PIPELINE_STEP_CONFIGURATION`
3. Define timeout in `STEP_TIMEOUTS`
4. Add argument support in `SCRIPT_ARG_SUPPORT`
5. Update documentation

### Custom Configuration

- Override defaults via command line arguments
- Use environment variables for sensitive data (API keys)
- Modify configuration files for persistent changes

## Best Practices

### Development

- **Test individual steps**: Each script can run standalone
- **Use verbose mode**: `--verbose` for detailed debugging
- **Check logs**: Pipeline and step-specific log files
- **Monitor resources**: Use built-in performance tracking

### Production

- **Set appropriate timeouts**: Based on expected data volumes
- **Enable critical steps only**: Disable experimental features
- **Monitor disk space**: Outputs can be large for complex models
- **Regular cleanup**: Archive old pipeline runs

## Troubleshooting

### Common Issues

1. **Step 2 (setup) fails**: Check internet connection, disk space
2. **Timeout errors**: Increase timeouts for slow systems
3. **Memory issues**: Reduce concurrent processing or increase system RAM
4. **Missing dependencies**: Ensure virtual environment is properly set up

### Debug Mode

Run with maximum verbosity:

```bash
python main.py --verbose --only-steps 1,4 --target-dir small_test_set/
```

### Log Analysis

Check specific log files:

- `output/logs/pipeline.log`: Main pipeline log
- `output/pipeline_execution_summary.json`: Structured execution data
- Step-specific outputs in respective directories 