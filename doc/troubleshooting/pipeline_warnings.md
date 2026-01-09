# Pipeline Warning Troubleshooting Guide

> **📋 Document Metadata**  
> **Type**: Troubleshooting Guide | **Audience**: All Users | **Complexity**: Intermediate  
> **Cross-References**: [Setup Guide](../SETUP.md) | [Pipeline Assessment](../pipeline/pipeline_warning_assessment.md)

This guide helps you diagnose and resolve common warnings in the GNN pipeline execution.

## Overview

The GNN pipeline executes 24 steps sequentially. Some steps may complete with warnings while still producing valid output. This guide explains each warning type and how to resolve them.

---

## Common Warning Scenarios

### Step 10: Advanced Visualization Warnings

**Symptom**: Step shows `SUCCESS_WITH_WARNINGS` status  
**Impact**: ✅ None - visualizations generated successfully  
**Severity**: 🟢 Low (False Positive)

#### Root Cause
Matplotlib backend warnings appear in stderr output, triggering warning detection:
```
UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
```

#### Verification
Check that visualization files were created:
```bash
ls output/9_advanced_viz_output/*.png
ls output/9_advanced_viz_output/d2_diagrams/
```

#### Action Required
**None** - This is a false positive. The module works correctly.

#### Optional Fix
To eliminate the warning message, you can suppress matplotlib warnings:
```python
# Add to advanced_visualization/d2_visualizer.py
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
```

---

### Step 13: Execution Framework Warnings

**Symptom**: Some frameworks fail during execution  
**Impact**: ⚠️ Reduced framework coverage (2/5 frameworks execute)  
**Severity**: 🟡 Medium (Missing Optional Dependencies)

#### Common Failures

##### PyMDP Failure
**Error Message**: `ModuleNotFoundError: No module named 'pymdp.agent'`

**Root Cause**: Wrong PyMDP package installed or package not installed

**Solution**:
```bash
# Install correct PyMDP package (inferactively-pymdp) with uv
uv pip install inferactively-pymdp

# Or install from source
uv pip install git+https://github.com/infer-actively/pymdp.git

# Verify installation using modern API
python3 -c "from pymdp import Agent; print('✅ PyMDP OK')"
```

##### JAX/Flax Failure
**Error Message**: `ModuleNotFoundError: No module named 'flax'`

**Root Cause**: JAX installed without Flax neural network library

**Solution**:
```bash
# Install JAX with Flax using uv
uv pip install jax[cpu] flax optax

# For GPU support (NVIDIA only)
uv pip install jax[cuda12_pip] flax optax

# Verify installation
python3 -c "import jax; import flax.linen; print('✅ JAX + Flax OK')"
python3 -c "import jax; print(f'Devices: {jax.devices()}')"
```

##### RxInfer.jl Failure
**Error Message**: `Half-edge has been found: s_prev_1`

**Root Cause**: Model generation bug in older RxInfer template

**Solution**: Template has been fixed in latest version. Regenerate code:
```bash
# Regenerate RxInfer code with fixed template
python src/11_render.py --target-dir input/gnn_files

# Re-run execution
python src/12_execute.py --frameworks "rxinfer"
```

#### Quick Fix - Install All Frameworks
```bash
# Install lite preset (recommended)
python src/1_setup.py --install_optional --optional_groups "pymdp,jax"

# Or install everything with uv
uv pip install inferactively-pymdp jax[cpu] flax optax
```

#### Framework Availability Check
```bash
# Check which frameworks are available
python src/12_execute.py --frameworks "all" --dry-run
```

---

### Step 14: LLM Processing Warnings

**Symptom**: Some LLM prompts timeout or return low-quality responses  
**Impact**: ⚠️ Reduced analysis quality (6/9 prompts succeed)  
**Severity**: 🟡 Medium (LLM Configuration)

#### Common Issues

##### Ollama Timeouts
**Error Message**: `Prompt execution timed out after 60 seconds`

**Root Cause**: Small Ollama model (`smollm2:135m-instruct`) too slow for complex prompts

**Solution Option 1 - Use Larger Ollama Model**:
```bash
# Install more capable model
ollama pull llama2:7b

# Or even better
ollama pull mixtral:8x7b

# Verify model
ollama list
```

**Solution Option 2 - Use OpenAI**:
```bash
# Set OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"

# Run LLM step (will use OpenAI as fallback)
python src/13_llm.py --target-dir input/gnn_files
```

**Solution Option 3 - Skip LLM Step**:
```bash
# Skip LLM analysis if not needed
python src/main.py --skip-steps "13"
```

##### Low-Quality Responses
**Symptom**: LLM returns code instead of prose, or nonsensical output

**Root Cause**: Small model lacks reasoning capability

**Solution**: Use larger model (see Option 1 above) or disable LLM step

#### Configuration
Increase timeouts for complex prompts by editing `src/llm/processor.py`:
```python
PROMPT_TIMEOUTS = {
    'practical_applications': 120,  # Increase from 60
    'technical_description': 120,   # Increase from 60
    # ... other prompts
}
```

---

### Step 3: Test Suite Timeout

**Symptom**: Tests timeout after 15 minutes  
**Impact**: ⚠️ Pipeline blocked, cannot complete  
**Severity**: 🟠 High (Execution Blocker)

#### Root Cause
Some tests (especially integration/slow tests) exceed timeout limits

#### Solution 1 - Skip Slow Tests (Recommended for Pipeline)
Tests now automatically skip slow tests in pipeline mode:
```bash
# Runs only fast tests (default in pipeline)
python src/main.py

# The test step now includes:
# --timeout=120 --timeout-method=thread -m "not slow"
```

#### Solution 2 - Skip Tests Entirely
```bash
# Set environment variable to skip tests
export SKIP_TESTS_IN_PIPELINE=1
python src/main.py
```

#### Solution 3 - Run Tests Separately
```bash
# Run pipeline without tests
python src/main.py --skip-steps "2"

# Run comprehensive tests separately (may take 15+ minutes)
python src/2_tests.py --comprehensive
```

#### Verification
Check test output:
```bash
cat output/2_tests_output/pytest_fast_output.txt
```

---

## Framework-Specific Troubleshooting

### DisCoPy Issues

**Symptom**: Import errors or diagram generation failures

**Solution**:
```bash
# Reinstall DisCoPy with uv
uv pip uninstall discopy -y
uv pip install "discopy[matrix]>=1.0.0"

# Verify
python3 -c "import discopy; print(discopy.__version__)"
```

### Julia Framework Issues

**Symptom**: Julia packages not found or version conflicts

**Solution**:
```bash
# Update all Julia packages
julia -e 'using Pkg; Pkg.update()'

# Clean package cache
julia -e 'using Pkg; Pkg.gc()'

# Reinstall specific packages
julia -e 'using Pkg; Pkg.add("ActiveInference"); Pkg.add("RxInfer")'

# Check installed packages
julia -e 'using Pkg; Pkg.status()'
```

---

## Quick Diagnosis Checklist

Run through this checklist to identify issues:

### 1. Check Pipeline Execution Summary
```bash
cat output/pipeline_execution_summary.json | python3 -m json.tool
```

Look for:
- `overall_status`: Should be `SUCCESS` or `SUCCESS_WITH_WARNINGS`
- `failed_steps`: Should be `0`
- `successful_steps`: Should be `23` or `24`

### 2. Check Individual Step Output
```bash
# Check specific step output
ls output/[STEP_NUMBER]_*_output/

# Read step summary
cat output/[STEP_NUMBER]_*_output/*.json | python3 -m json.tool
```

### 3. Check Framework Availability
```bash
# Test each framework
python3 -c "import discopy; print('✅ DisCoPy')"
python3 -c "from pymdp import Agent; print('✅ PyMDP')"
python3 -c "import jax; import flax.linen; print('✅ JAX')"
julia -e 'using ActiveInference; println("✅ ActiveInference.jl")'
julia -e 'using RxInfer; println("✅ RxInfer.jl")'
```

### 4. Check Python Environment
```bash
# Verify Python version
python3 --version  # Should be 3.9+

# Check installed packages with uv
uv pip list | grep -E "pymdp|jax|flax|discopy"

# Check virtual environment
which python3
```

### 5. Check Disk Space
```bash
# Check available space
df -h .

# Check output directory size
du -sh output/
```

---

## Prevention Strategies

### For Clean Pipeline Execution

1. **Install Recommended Frameworks**:
   ```bash
   python src/1_setup.py --install_optional --optional_groups "pymdp,jax"
   ```

2. **Use Fast Test Mode** (default):
   ```bash
   # Fast tests only (completes in ~5 minutes)
   python src/main.py
   ```

3. **Skip Optional Steps** if not needed:
   ```bash
   # Skip LLM and slow steps
   python src/main.py --skip-steps "13,14"
   ```

4. **Monitor Resource Usage**:
   ```bash
   # Run with monitoring
   time python src/main.py
   ```

### For Development

1. **Use Comprehensive Tests** separately:
   ```bash
   # Run full test suite separately
   python src/2_tests.py --comprehensive
   ```

2. **Test Individual Frameworks**:
   ```bash
   # Test one framework at a time
   python src/12_execute.py --frameworks "pymdp"
   python src/12_execute.py --frameworks "jax"
   ```

3. **Use Verbose Logging**:
   ```bash
   # Get detailed output
   python src/main.py --verbose
   ```

---

## Getting Help

### Self-Service Resources

1. **Check Logs**: All steps log to `output/[step]_output/`
2. **Read Documentation**: See [Setup Guide](../SETUP.md) for installation help
3. **Review Assessment**: See [Pipeline Assessment](../pipeline/pipeline_warning_assessment.md)

### Community Support

1. **GitHub Issues**: [Create an issue](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/issues)
2. **Discussions**: [Ask in discussions](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/discussions)
3. **Documentation**: [Main README](../../README.md)

### Reporting Issues

When reporting warnings/errors, include:

1. **System Information**:
   ```bash
   python3 --version
   pip list > packages.txt
   uname -a
   ```

2. **Pipeline Summary**:
   ```bash
   cat output/pipeline_execution_summary.json
   ```

3. **Error Logs**:
   ```bash
   # Include relevant step output
   cat output/[failing_step]_output/*.log
   ```

4. **Reproduction Steps**:
   - Command used
   - Input files
   - Configuration

---

## Status Interpretation

### Success States
- ✅ `SUCCESS`: Step completed perfectly
- ✅ `SUCCESS_WITH_WARNINGS`: Step completed with minor issues (output still valid)

### Warning States
- ⚠️ `PARTIAL_SUCCESS`: Some outputs generated, some failed
- ⚠️ `SUCCESS_WITH_WARNINGS`: Non-critical warnings present

### Failure States
- ❌ `FAILED`: Step failed completely
- ❌ `TIMEOUT`: Step exceeded time limit
- ❌ `ERROR`: Unexpected error occurred

---

**Guide Version**: 1.0  
**Compatible Pipeline Version**: 1.1.1+  

