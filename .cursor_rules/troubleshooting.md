# Troubleshooting Guide

> **Environment Note**: Most issues can be resolved by ensuring you're using the `uv` environment correctly. Run `uv pip list` to verify installed packages.

## Overview

This guide covers common issues encountered when working with the GNN pipeline, their causes, and solutions.

---

## Quick Diagnostics

### Pipeline Health Check

```bash
# Run full pipeline to check health
python src/main.py --verbose

# Run quick test suite
python src/2_tests.py --fast-only

# Check environment
python src/1_setup.py --verbose
```

### Expected Healthy Output

```
✅ Step 0 completed with SUCCESS
✅ Step 1 completed with SUCCESS (Environment setup)
✅ Step 2 completed with SUCCESS (Tests: 579/638 passed)
✅ Step 3 completed with SUCCESS (GNN: 22 formats)
...
✅ All 24 steps completed successfully
Total time: ~2-3 minutes
Peak memory: <50MB
```

---

## Common Issues by Category

### 1. Import Errors

#### `ModuleNotFoundError: No module named 'pymdp'`

**Cause**: PyMDP is an optional dependency not installed.

**Solution**:
```bash
pip install pymdp
```

**Note**: This is expected behavior. The pipeline continues without PyMDP using fallback mode. The warning message is informational:
```
WARNING: PyMDP not available - this is normal if not installed.
To enable PyMDP simulations, install with: pip install pymdp.
Alternatively, use other frameworks: RxInfer.jl, ActiveInference.jl, or JAX.
```

#### `ModuleNotFoundError: No module named 'flax'`

**Cause**: Old JAX code required Flax, but this dependency has been removed.

**Solution**:
- Re-run the render step to regenerate JAX code:
  ```bash
  python src/11_render.py --target-dir input/gnn_files
  ```
- The new JAX code uses pure JAX without Flax

**Note**: As of December 2025, the JAX renderer generates pure JAX code that does NOT require Flax.

#### `ImportError: cannot import name 'X' from 'module'`

**Cause**: Version mismatch or outdated package.

**Solution**:
```bash
# Update all packages
uv pip install -U -r requirements.txt

# Or update specific package
uv pip install -U package_name
```

---

### 2. Julia/RxInfer Issues

#### `Package RxInfer not found in current path`

**Cause**: Julia packages not installed.

**Solution**:
```bash
# Install Julia packages
julia -e 'using Pkg; Pkg.add("RxInfer"); Pkg.add("Distributions")'

# Or use the setup script
julia src/execute/rxinfer/setup_environment.jl --verbose
```

#### `Julia not found`

**Cause**: Julia not installed or not in PATH.

**Solution**:
1. Install Julia from https://julialang.org
2. Add to PATH:
   ```bash
   # macOS/Linux
   export PATH="$PATH:/Applications/Julia-1.10.app/Contents/Resources/julia/bin"
   
   # Or add to ~/.zshrc / ~/.bashrc
   ```

#### Julia Package Compilation Slow

**Cause**: First-time Julia package compilation.

**Solution**: This is normal. First run takes longer (~2-5 minutes). Subsequent runs are fast.

---

### 3. LLM/Ollama Issues

#### `Ollama connection refused`

**Cause**: Ollama server not running.

**Solution**:
```bash
# Start Ollama
ollama serve

# In another terminal, verify
ollama list
```

#### `No models available`

**Cause**: No Ollama models installed.

**Solution**:
```bash
# Install a small model
ollama pull smollm2:135m-instruct-q4_K_S

# Or a larger model
ollama pull tinyllama
```

#### LLM Processing Slow

**Cause**: Large model or limited resources.

**Solution**:
```bash
# Use smaller model
export OLLAMA_MODEL=smollm2:135m-instruct-q4_K_S

# Or reduce tokens
export OLLAMA_MAX_TOKENS=256
```

---

### 4. Test Failures

#### Tests Skipped

**Cause**: Optional dependencies not available.

**Expected**: Some tests skip when dependencies unavailable. This is normal:
```
Tests run: 638
Passed: 579
Skipped: 59  <- This is OK
Failed: 0
```

#### `RecursionError` in Tests

**Cause**: Pytest plugin conflicts.

**Solution**:
```bash
# Run with minimal plugins
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest src/tests/ -v
```

#### Test Timeout

**Cause**: Tests taking too long.

**Solution**:
```bash
# Increase timeout
export FAST_TESTS_TIMEOUT=300
python src/2_tests.py
```

---

### 5. MCP Issues

#### `MCP module loading slow`

**Cause**: Sequential module loading.

**Current Status**: MCP modules load in ~10 seconds. This is optimized for the number of modules (21+).

#### `Tool not found`

**Cause**: Module not loaded or registration failed.

**Solution**:
```bash
# Check registered tools
python -c "from mcp import MCPServer; s = MCPServer(); print(len(s.tools))"
```

---

### 6. Visualization Issues

#### `No plots generated`

**Cause**: Missing visualization dependencies.

**Solution**:
```bash
pip install matplotlib seaborn plotly networkx
```

#### D2 Diagrams Not Generated

**Cause**: D2 not installed.

**Solution**:
```bash
# macOS
brew install d2

# Other platforms
# See https://d2lang.com/tour/install
```

---

### 7. Performance Issues

#### Pipeline Too Slow

**Expected**: Full pipeline should complete in 2-5 minutes.

**Diagnosis**:
```bash
# Check which step is slow
python src/main.py --verbose 2>&1 | grep "completed"
```

**Common Slow Steps**:
- Step 2 (Tests): ~90 seconds (normal)
- Step 9 (Advanced Viz): ~8 seconds (normal)
- Step 12 (Execute): ~30 seconds (normal)
- Step 21 (MCP): ~10 seconds (normal)

#### High Memory Usage

**Expected**: <500MB typical, <2GB peak.

**Diagnosis**:
```bash
# Monitor during run
while true; do ps aux | grep python | head -1; sleep 1; done
```

**Solution**: If memory exceeds 2GB, check for memory leaks in custom code.

---

### 8. Path Issues

#### `FileNotFoundError: input/gnn_files`

**Cause**: Wrong working directory.

**Solution**:
```bash
# Run from project root
cd /path/to/generalizednotationnotation
python src/main.py
```

#### Outputs in Wrong Location

**Cause**: Custom output directory not used.

**Solution**:
```bash
# Specify output directory
python src/main.py --output-dir /custom/output
```

---

## Diagnostic Commands

### Environment Check

```bash
# Python version
python --version  # Should be 3.10+

# Installed packages
uv pip list | grep -E "numpy|scipy|matplotlib|jax|pytest"

# Julia version
julia --version

# Ollama status
ollama list
```

### File System Check

```bash
# Check input files
ls -la input/gnn_files/

# Check output structure
ls -la output/

# Check specific step output
ls -la output/11_render_output/
```

### Log Analysis

```bash
# View recent pipeline output
cat output/pipeline_execution_summary.json | jq '.overall_status'

# Check step-specific logs
grep -r "ERROR" output/*/

# View test output
cat output/2_tests_output/pytest_comprehensive_output.txt | tail -50
```

---

## Error Message Reference

### Common Error Messages and Meanings

| Error Message | Meaning | Action |
|--------------|---------|--------|
| `PyMDP not available` | Optional dep missing | Install or ignore |
| `Flax not found` | Outdated code | Re-run render step |
| `Julia not found` | Julia not installed | Install Julia |
| `Ollama connection refused` | Server not running | Start ollama serve |
| `Circuit breaker open` | Repeated failures | Check underlying issue |
| `Timeout expired` | Step took too long | Increase timeout |

---

## Recovery Procedures

### Reset Pipeline State

```bash
# Clean output directory
rm -rf output/*

# Re-run full pipeline
python src/main.py --verbose
```

### Rebuild Environment

```bash
# Remove and recreate virtual environment
rm -rf .venv
uv venv
uv pip install -e .
```

### Resume from Checkpoint

```bash
# Skip completed steps
python src/main.py --skip-steps "0,1,2,3"

# Run specific steps only
python src/main.py --only-steps "11,12,13"
```

---

## Getting Help

### Debug Information to Collect

When reporting issues, include:

1. **Environment**:
   ```bash
   python --version
   uv pip freeze > environment.txt
   ```

2. **Error Output**:
   ```bash
   python src/main.py --verbose 2>&1 | tee debug.log
   ```

3. **Pipeline Summary**:
   ```bash
   cat output/pipeline_execution_summary.json
   ```

### Common Questions

**Q: Why are some tests skipped?**
A: Tests skip when optional dependencies are unavailable. This is expected behavior.

**Q: Why does step 12 show failures?**
A: Step 12 (Execute) may show partial failures if PyMDP or Julia aren't installed. The pipeline continues with available frameworks.

**Q: How do I run just one step?**
A: Use `python src/N_step.py --target-dir input/gnn_files --output-dir output`

---

**Last Updated**: December 2025  
**Status**: Production Standard


