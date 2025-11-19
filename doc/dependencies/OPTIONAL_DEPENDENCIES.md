# Optional Dependencies Guide

This document provides a comprehensive guide to optional dependencies in the GNN Pipeline. Understanding which dependencies are optional helps you install only what you need for your specific use case.

## Quick Reference

| Framework | Status | Purpose | Install Command | Pipeline Step |
|-----------|--------|---------|-----------------|---|
| PyMDP | Optional | POMDP agent simulation | `pip install pymdp` | 12 (Execute) |
| Flax | Optional | JAX neural networks | `pip install flax` | 12 (Execute) |
| RxInfer.jl | Optional | Julia probabilistic inference | `julia -e 'import Pkg; Pkg.add("RxInfer")'` | 12 (Execute) |
| Plotly | Optional | Interactive visualizations | `pip install plotly` | 8-9 (Visualization) |
| GraphViz | Optional | Advanced graph layouts | `apt-get install graphviz` | 8-9 (Visualization) |

---

## Detailed Framework Information

### PyMDP - Optional Simulation Framework

**Status**: ❌ Not installed by default (OPTIONAL)

**What it does**: Enables execution of rendered Python code for POMDP simulations using the PyMDP library

**Used in**: Step 12 (Execute) - Simulation execution

**Installation**:
```bash
pip install pymdp
```

**Error if missing**:
```
ERROR:src.execute.pymdp.executor:PyMDP import failed: No module named 'pymdp.agent'
```

**Impact on pipeline**:
- ✅ Pipeline continues successfully
- ✅ DisCoPy and ActiveInference.jl still work
- ⚠️ PyMDP simulations are skipped
- Result: 2/5 frameworks working (40% execution success rate)

**When to install**:
- You need PyMDP-based POMDP simulations
- You're testing all supported frameworks
- You want comprehensive framework coverage

**When you don't need it**:
- You only need Julia (ActiveInference.jl, RxInfer) simulations
- You only need Python-based simulations (JAX, DisCoPy)
- You want a minimal installation

### Flax - Optional JAX Dependency

**Status**: ❌ Not installed by default (OPTIONAL)

**What it does**: Enables execution of JAX-based neural network code for Active Inference models

**Used in**: Step 12 (Execute) - JAX simulation execution

**Installation**:
```bash
pip install flax
```

**Error if missing**:
```
ModuleNotFoundError: No module named 'flax'
```

**Impact on pipeline**:
- ✅ Pipeline continues successfully
- ✅ JAX rendering still works (code generation succeeds)
- ⚠️ JAX simulations fail to execute
- Result: 2/5 frameworks working (40% execution success rate)

**When to install**:
- You need JAX neural network simulations
- You're using JAX-based Active Inference implementations
- You want to execute JAX-rendered code

**When you don't need it**:
- You don't use JAX for simulations
- You prefer Julia or Python alternatives
- You want a lighter installation

### RxInfer.jl - Optional Julia Framework

**Status**: ❌ Not installed by default (OPTIONAL)

**What it does**: Enables execution of Julia code for probabilistic inference simulations using RxInfer

**Used in**: Step 12 (Execute) - RxInfer execution

**Installation**:
```bash
julia -e 'import Pkg; Pkg.add("RxInfer")'
```

**Prerequisites**: Julia must be installed (`julia --version`)

**Error if missing**:
```
ERROR: LoadError: ArgumentError: Package RxInfer not found in current path.
Run `import Pkg; Pkg.add("RxInfer")` to install the RxInfer package.
```

**Impact on pipeline**:
- ✅ Pipeline continues successfully
- ✅ RxInfer code rendering still works
- ⚠️ RxInfer simulations fail to execute
- Result: 2/5 frameworks working (40% execution success rate)

**When to install**:
- You need Julia-based probabilistic inference
- You're testing RxInfer simulations
- You work primarily with Julia

**When you don't need it**:
- You don't use Julia
- You prefer Python frameworks (PyMDP, JAX)
- You want quick prototyping without Julia overhead

---

## Visualization Dependencies

### Plotly - Optional Interactive Visualizations

**Status**: ❌ Not installed by default (OPTIONAL)

**What it does**: Enables interactive 3D visualizations and web-based charts

**Used in**: Step 9 (Advanced Visualization) - Advanced visualization features

**Installation**:
```bash
pip install plotly
```

**Impact on pipeline**:
- ✅ Static visualizations still work (matplotlib-based)
- ⚠️ Interactive visualizations are skipped
- Result: Basic visualization features work, advanced features disabled

**When to install**:
- You need interactive 3D plots
- You want web-based visualization dashboards
- You're sharing visualizations as HTML files

### GraphViz - Optional Graph Layout Engine

**Status**: ❌ Not installed by default (OPTIONAL)

**What it does**: Provides advanced graph layout algorithms for network visualizations

**Used in**: Step 8-9 (Visualization) - Graph layout optimization

**Installation**:

**macOS**:
```bash
brew install graphviz
```

**Linux (Ubuntu/Debian)**:
```bash
apt-get install graphviz
```

**Windows**:
Download from: https://graphviz.org/download/

**Impact on pipeline**:
- ✅ Basic visualizations work (using networkx layouts)
- ⚠️ Advanced graph layouts fall back to simpler algorithms
- Result: Visualizations still generated, possibly less optimal

**When to install**:
- You need Graphviz-specific layout algorithms
- Your graphs have complex structures
- You want publication-quality layouts

---

## Installation Strategies

### Strategy 1: Minimal Installation (Fastest)

**Best for**: Quick testing, CI/CD pipelines, resource-constrained environments

```bash
# Core-only installation - no optional dependencies
pip install -r requirements.txt
```

**Result**:
- ✅ Full pipeline works
- ✅ All rendering (code generation) works
- ⚠️ Execution (Step 12) has limited framework support
- ✅ Test suite runs (90% pass rate)

**Framework support**: DisCoPy ✅, ActiveInference.jl ✅

### Strategy 2: Standard Installation (Balanced)

**Best for**: Most users, development, standard workloads

```bash
# Install core + common optional dependencies
pip install -r requirements.txt
pip install pymdp flax  # Optional frameworks
```

**Result**:
- ✅ Full pipeline works
- ✅ Most frameworks available
- ✅ Execution mostly works

**Framework support**: PyMDP ✅, JAX+Flax ✅, DisCoPy ✅, ActiveInference.jl ✅, RxInfer ❌

### Strategy 3: Complete Installation (All Features)

**Best for**: Comprehensive testing, research, all frameworks

```bash
# Install everything
pip install -r requirements.txt
pip install pymdp flax plotly  # Python optional deps

# Then separately install Julia packages
julia -e 'import Pkg; Pkg.add(["RxInfer"])'

# And system dependencies
# macOS:
brew install graphviz
# Linux:
sudo apt-get install graphviz
```

**Result**:
- ✅ Full pipeline with all features
- ✅ All frameworks available
- ✅ All visualization features enabled

**Framework support**: All ✅

---

## Checking What's Installed

### Check installed packages

```bash
# List installed Python packages
pip list | grep -E "pymdp|flax|plotly"

# Check for system dependencies
which julia  # Julia installed?
which dot    # GraphViz installed?
```

### Check Framework Availability at Runtime

```bash
# Run the setup step with verbose mode
python src/1_setup.py --verbose

# Or check within Python
python -c "
try:
    import pymdp
    print('PyMDP: Available')
except:
    print('PyMDP: Not installed')
    
try:
    import flax
    print('Flax: Available')
except:
    print('Flax: Not installed')
"
```

### Check MCP Tool Registration

```bash
# See which modules successfully loaded
python src/21_mcp.py --verbose

# Grep output for successfully loaded modules
grep -i "successfully loaded" output/21_mcp_output/*.log
```

---

## Troubleshooting

### Issue: "PyMDP not available" warning during execution

**Cause**: PyMDP not installed

**Solutions**:
1. Install PyMDP: `pip install pymdp`
2. Or use other frameworks: Execution continues with available frameworks
3. Check installation: `python -c "import pymdp; print(pymdp.__version__)"`

### Issue: JAX simulations fail but rendering works

**Cause**: Flax not installed (JAX is installed but Flax is missing)

**Solutions**:
1. Install Flax: `pip install flax`
2. Or skip JAX simulations in Step 12 settings
3. Check installation: `python -c "import flax; print(flax.__version__)"`

### Issue: RxInfer simulations skipped

**Cause**: RxInfer.jl package not installed in Julia

**Solutions**:
1. Install RxInfer: `julia -e 'import Pkg; Pkg.add("RxInfer")'`
2. Verify Julia: `julia --version`
3. Check installation: `julia -e 'using RxInfer'`

### Issue: Advanced visualizations not generating

**Cause**: GraphViz or Plotly not installed

**Solutions**:
1. Install GraphViz (system): See installation steps above
2. Install Plotly: `pip install plotly`
3. Run Step 9 with verbose: `python src/9_advanced_viz.py --verbose`

---

## Recommendations by Use Case

### For Research & Publication
```bash
# Install everything for maximum features
pip install -r requirements.txt
pip install pymdp flax plotly
# Plus Julia packages
julia -e 'import Pkg; Pkg.add(["RxInfer"])'
# Plus system dependencies
brew install graphviz  # or apt-get on Linux
```

### For Development/Testing
```bash
# Install core + testing frameworks
pip install -r requirements.txt
pip install pymdp flax  # Most common frameworks
```

### For Quick Prototyping
```bash
# Install only core
pip install -r requirements.txt
# Add optional deps as needed
```

### For CI/CD Pipelines
```bash
# Install core - skip optional for speed
pip install -r requirements.txt
# CI will test that pipeline works without optional deps
```

### For Production Deployment
```bash
# Install known-working set
pip install -r requirements.txt
pip install pymdp flax  # Tested combination
# Skip rarely-used optional deps
```

---

## FAQ

**Q: Do I need to install all optional dependencies?**  
A: No. The pipeline works without them. Rendering always works; execution uses whatever frameworks are installed.

**Q: Will the pipeline fail if I don't install optional dependencies?**  
A: No. The pipeline completes successfully with status SUCCESS_WITH_WARNINGS. Missing frameworks are skipped gracefully.

**Q: How many frameworks do I need for success?**  
A: Minimum 1. The success criteria: "60% or at least 1 framework succeeds" - so DisCoPy alone is enough.

**Q: Can I install dependencies later?**  
A: Yes. The pipeline detects available frameworks at execution time. Install a new framework and re-run Step 12.

**Q: What's the best framework to start with?**  
A: DisCoPy or ActiveInference.jl - they're already installed. Add PyMDP + Flax for Python framework coverage.

---

## Performance Impact

### Installation Size
- **Core only**: ~500MB (+ dependencies)
- **Core + PyMDP + Flax**: ~1.2GB
- **Full installation**: ~3-5GB (with Julia)

### Installation Time
- **Core only**: 2-5 minutes
- **Core + PyMDP + Flax**: 5-10 minutes
- **Full installation**: 15-30 minutes (includes Julia)

### Runtime Performance
- **Optional deps loading**: <100ms per framework
- **Missing framework skip**: ~10ms per framework
- **Overall impact**: Negligible - pipeline still completes in 3-4 minutes

---

## Contributing

If you encounter issues with optional dependencies:

1. Check version compatibility: Some frameworks require specific versions
2. Try fresh installation: `pip install --upgrade --force-reinstall pymdp`
3. Report issues with: Python version, OS, and error messages
4. See [TROUBLESHOOTING.md](../troubleshooting/README.md) for more help

---

**Last Updated**: November 19, 2025  
**Status**: ✅ Current for Pipeline v2.1.0

