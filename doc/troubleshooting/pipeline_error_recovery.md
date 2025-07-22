# Pipeline Error Recovery Guide

> **📋 Document Metadata**  
> **Type**: Troubleshooting Guide | **Audience**: Developers & Operators | **Complexity**: Intermediate  
> **Last Updated**: July 2025 | **Status**: Active  
> **Cross-References**: [Pipeline Architecture](../pipeline/README.md) | [Testing Guide](../testing/README.md)

## Overview

This guide provides comprehensive error recovery procedures for the GNN pipeline, focusing on common failure modes and their resolutions.

## Critical Error Patterns

### 1. NumPy Recursion Error (Step 9 - Render)

#### Symptoms
- Error message: `RecursionError: maximum recursion depth exceeded`
- Occurs during type imports in NumPy
- Typically in `numpy._typing`

#### Resolution
```python
import sys
sys.setrecursionlimit(3000)  # Increase from default 1000

# If using in code:
try:
    import numpy as np
except RecursionError:
    import sys
    sys.setrecursionlimit(3000)
    import numpy as np
```

#### Prevention
- Add recursion limit configuration to `1_setup.py`
- Monitor memory usage during type checking
- Consider using lightweight type checking for large models

### 2. Async/Await Issues (Step 11 - LLM)

#### Symptoms
- Warning: `RuntimeWarning: coroutine 'OpenAIProvider.generate_response' was never awaited`
- Incomplete LLM analysis results
- Hanging API calls

#### Resolution
```python
# Correct implementation:
async def analyze_gnn_file(file_path: Path) -> Dict[str, Any]:
    async with OpenAIProvider() as provider:
        response = await provider.analyze(file_path.read_text())
        return {"analysis": response.strip()}

# In synchronous code:
import asyncio
result = asyncio.run(analyze_gnn_file(file_path))
```

#### Prevention
- Use async/await consistently throughout LLM module
- Implement proper cleanup in async context managers
- Add timeout handling for API calls

### 3. Missing GNN Lightweight Processing

#### Symptoms
- Error: `cannot import name 'process_gnn_directory_lightweight'`
- GNN processing fallback fails
- Step 2 warnings

#### Resolution
```python
# In src/gnn/__init__.py:
def process_gnn_directory_lightweight(directory: Path) -> Dict[str, Any]:
    """Lightweight GNN directory processing fallback."""
    results = {}
    for file in directory.glob("**/*.md"):
        try:
            results[str(file)] = {
                "status": "processed",
                "format": "markdown",
                "size": file.stat().st_size
            }
        except Exception as e:
            results[str(file)] = {"status": "error", "error": str(e)}
    return results
```

#### Prevention
- Implement lightweight processing in all critical modules
- Add feature detection during setup
- Document fallback mechanisms

### 4. JAX/TPU Initialization (Step 10 - Execute)

#### Symptoms
- Error: `INTERNAL: Failed to open libtpu.so`
- JAX device initialization failures
- Missing accelerator support

#### Resolution
```python
# In src/execute/jax_runner.py:
def initialize_jax_devices():
    """Initialize JAX with graceful fallback."""
    try:
        import jax
        devices = jax.devices()
    except:
        # Fallback to CPU-only
        import os
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        import jax
        devices = jax.devices()
    return devices
```

#### Prevention
- Check hardware capabilities during setup
- Provide CPU fallback configurations
- Document platform-specific requirements

## Step-Specific Recovery Procedures

### Step 9 (Render)
1. Check Python recursion limit
2. Verify NumPy installation
3. Monitor memory usage
4. Use incremental rendering for large models

### Step 10 (Execute)
1. Verify framework availability
2. Check hardware requirements
3. Validate simulation configurations
4. Monitor resource usage

### Step 11 (LLM)
1. Verify API credentials
2. Check network connectivity
3. Monitor rate limits
4. Implement retry mechanisms

## General Recovery Guidelines

### 1. Logging and Diagnostics
- Enable verbose logging: `--verbose`
- Check step-specific logs in `output/logs/`
- Monitor system resources
- Review pipeline execution summary

### 2. Resource Management
- Monitor memory usage
- Track disk space
- Check CPU utilization
- Manage network connections

### 3. Data Integrity
- Validate input files
- Check output consistency
- Verify file permissions
- Monitor file system operations

### 4. Error Reporting
- Collect error details
- Generate diagnostic reports
- Track error patterns
- Update error documentation

## Automated Recovery Tools

### 1. Pipeline Validator
```bash
python src/pipeline/validate_pipeline.py --fix-issues
```

### 2. Environment Checker
```bash
python src/1_setup.py --validate-only
```

### 3. Resource Monitor
```bash
python src/utils/monitor_resources.py
```

## Contributing to Error Recovery

When encountering new error patterns:

1. Document the error details
2. Identify root causes
3. Develop recovery procedures
4. Update this guide
5. Add regression tests

## References

- [Pipeline Architecture](../pipeline/README.md)
- [Testing Guide](../testing/README.md)
- [Configuration Guide](../configuration/README.md)
- [Development Guide](../development/README.md) 