# Troubleshooting Guide

## Overview
This guide helps resolve common issues encountered when working with GeneralizedNotationNotation (GNN).

## Common Error Categories

### 1. GNN Syntax Errors

#### Missing Required Sections
**Error**: `Missing required section: StateSpaceBlock`
**Cause**: GNN file doesn't contain mandatory sections
**Solution**:
```markdown
# Your GNN file must include:
## StateSpaceBlock
s_f0[2,1,type=categorical]
o_m0[3,1,type=categorical]

## Connections
s_f0 > o_m0
```

#### Invalid Variable Naming
**Error**: `Invalid variable name: 'state_1'`
**Cause**: Variable names don't follow GNN conventions
**Solution**: Use GNN naming: `s_f0`, `o_m0`, `u_c0`, `π_c0`

#### Dimension Mismatch
**Error**: `Matrix A dimensions [3,2] don't match state space [2,3]`
**Cause**: Matrix dimensions incompatible with variable definitions
**Solution**: Ensure matrix dims match: `A_m0[obs_dims, state_dims]`

### 2. Pipeline Execution Errors

#### Step 2 Setup Failure
**Error**: `Setup step failed - pipeline halted`
**Cause**: Virtual environment or dependency issues
**Solutions**:
```bash
# Clean reinstall
rm -rf src/.venv
python src/main.py --only-steps 2 --force-reinstall

# Manual dependency check
pip list | grep -E "(numpy|scipy|matplotlib)"

# Python version check (requires 3.8+)
python --version
```

#### Permission Errors
**Error**: `Permission denied: output/`
**Cause**: Insufficient write permissions
**Solutions**:
```bash
# Fix permissions
chmod -R 755 output/

# Use different output directory
python src/main.py --output-dir /tmp/gnn_output
```

#### Memory Issues
**Error**: `MemoryError: Unable to allocate array`
**Cause**: Large model exceeds available RAM
**Solutions**:
- Use `--conservative-memory` flag
- Reduce model complexity
- Process models individually: `--target-dir single_model/`

### 3. Rendering/Simulation Errors

#### PyMDP Import Errors
**Error**: `ModuleNotFoundError: No module named 'pymdp'`
**Solutions**:
```bash
# Install PyMDP
pip install pymdp

# Alternative: use conda
conda install -c conda-forge pymdp
```

#### RxInfer.jl Setup Issues
**Error**: `Julia not found or RxInfer.jl not installed`
**Solutions**:
```bash
# Install Julia
wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.0-linux-x86_64.tar.gz
# Follow Julia installation guide

# Install RxInfer.jl
julia -e 'using Pkg; Pkg.add("RxInfer")'
```

#### JAX Compilation Errors
**Error**: `XLA compilation failed`
**Solutions**:
- Update JAX: `pip install --upgrade jax jaxlib`
- Use CPU-only mode: `export JAX_PLATFORM_NAME=cpu`
- Simplify model complexity

### 4. Visualization Errors

#### Graphviz Missing
**Error**: `Graphviz not found`
**Solutions**:
```bash
# Ubuntu/Debian
sudo apt-get install graphviz graphviz-dev

# macOS
brew install graphviz

# Windows
# Download from https://graphviz.org/download/
```

#### Large Graph Rendering
**Error**: `Graph too large to render`
**Solutions**:
- Use `--max-nodes 50` to limit graph size
- Enable hierarchical layout: `--layout hierarchical`
- Generate SVG instead of PDF: `--format svg`

### 5. LLM Integration Issues

#### API Key Errors
**Error**: `Invalid API key for OpenAI/Anthropic`
**Solutions**:
```bash
# Set environment variables
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Or use config file
echo "OPENAI_API_KEY=your-key" > .env
```

#### Token Limit Exceeded
**Error**: `Context length exceeded`
**Solutions**:
- Use smaller model: `--llm-model gpt-3.5-turbo`
- Process models in chunks: `--chunk-size 1000`
- Simplify model before LLM analysis

## Diagnostic Commands

### System Health Check
```bash
# Run comprehensive diagnostics
python src/main.py --diagnostics

# Check specific components
python src/4_type_checker.py --validate-only
python src/2_setup.py --check-deps
```

### Debug Mode
```bash
# Enable detailed logging
python src/main.py --debug --target-dir examples/

# Verbose output for specific steps
python src/main.py --only-steps 4 --verbose
```

### Environment Information
```bash
# System info
python -c "import sys; print(f'Python: {sys.version}')"
pip list | head -20

# GNN-specific info
python -c "import src.gnn as gnn; print(gnn.__version__)"
```

## Performance Optimization

### Speed Issues
**Problem**: Pipeline runs slowly
**Solutions**:
- Use `--parallel` for multi-core processing
- Skip expensive steps: `--skip 6,11,14`
- Use conservative mode: `--conservative`

### Memory Optimization
**Problem**: High memory usage
**Solutions**:
- Process models sequentially: `--sequential`
- Reduce visualization quality: `--viz-quality low`
- Clean up intermediate files: `--cleanup`

## File-Specific Issues

### Invalid GNN File Structure
**Problem**: File doesn't parse correctly
**Debugging Steps**:
1. Check file encoding (must be UTF-8)
2. Verify all required sections present
3. Validate syntax with type checker
4. Compare against working examples

### Export Format Issues
**Problem**: Export fails or produces invalid output
**Solutions**:
- Check export format support: `python src/5_export.py --list-formats`
- Use alternative format: `--export-format json`
- Validate output: `--validate-export`

## Getting Additional Help

### Log Files
Key log locations:
- Main pipeline: `output/logs/pipeline.log`
- Step-specific: `output/logs/step_XX.log`
- Error details: `output/logs/errors.log`

### Debug Information
Include this info when reporting issues:
```bash
# Generate debug package
python src/main.py --debug-package

# This creates: output/debug_package_TIMESTAMP.zip
# Contains: logs, system info, example files, config
```

### Community Support
- Check existing GitHub issues
- Search documentation
- Join Active Inference Institute community
- Provide minimal reproducible example

## Prevention Best Practices

### Model Development
- Start with simple examples
- Validate syntax early and often
- Use version control for model files
- Test across different backends

### Pipeline Usage
- Run setup step first
- Check dependencies regularly
- Use appropriate hardware for large models
- Monitor resource usage

### Maintenance
- Update dependencies monthly
- Clean output directories regularly
- Backup important model files
- Document custom configurations

## Quick Reference

### Most Common Fixes
```bash
# Reset everything
rm -rf output/ src/.venv/
python src/main.py --only-steps 2

# Fix permissions
find . -name "*.py" -exec chmod +x {} \;

# Clean install
pip uninstall -y $(pip list --format=freeze | cut -d= -f1)
pip install -r requirements.txt

# Minimal test
python src/main.py --target-dir src/gnn/examples/basic/ --only-steps 1,4
```

### Emergency Recovery
If pipeline is completely broken:
```bash
# Nuclear option - complete reset
git checkout HEAD -- .
rm -rf output/ src/.venv/ __pycache__/
python src/2_setup.py --clean-install
```

This troubleshooting guide covers the most common issues. For persistent problems, please file a GitHub issue with debug information. 