# Framework Availability Guide

This document explains how to check framework availability before execution and how the pipeline handles missing frameworks.

## Quick Check

```bash
# Check which frameworks are available
python -c "
import sys
sys.path.insert(0, 'src')
from execute import get_execution_health_status
status = get_execution_health_status()
for fw, info in status.items():
    print(f'{fw}: {\"✅ Available\" if info[\"available\"] else \"❌ Not available\"} - {info.get(\"reason\", \"\")}')
"
```

## Framework Availability at Runtime

### PyMDP
- **Python module**: `pymdp`
- **Status file**: `output/12_execute_output/framework_status.json`
- **Check command**: `python -c "import pymdp; print(pymdp.__version__)"`
- **Install**: `pip install pymdp`

### JAX + Flax
- **Python modules**: `jax`, `flax`
- **Check command**: `python -c "import jax; import flax; print(f'JAX: {jax.__version__}, Flax: {flax.__version__}')"`
- **Install**: `pip install jax flax`

### RxInfer.jl
- **Julia package**: `RxInfer`
- **Check command**: `julia -e "using RxInfer; println(\"RxInfer available\")"`
- **Install**: `julia -e 'import Pkg; Pkg.add("RxInfer")'`

### ActiveInference.jl
- **Julia package**: `ActiveInference`
- **Check command**: `julia -e "using ActiveInference; println(\"ActiveInference available\")"`
- **Install**: `julia -e 'import Pkg; Pkg.add("ActiveInference")'`

### DisCoPy
- **Python module**: `discopy`
- **Check command**: `python -c "import discopy; print(discopy.__version__)"`
- **Install**: Usually pre-installed as core dependency

## Framework Status During Execution

### Before Execution (Step 12 Start)
The execute module automatically detects available frameworks and logs them:

```
2025-11-19 11:07:11 [execute] INFO - Checking framework availability...
2025-11-19 11:07:11 [execute] INFO - ✅ DisCoPy available
2025-11-19 11:07:11 [execute] INFO - ✅ ActiveInference.jl available
2025-11-19 11:07:11 [execute] INFO - ❌ PyMDP not available (optional - install with: pip install pymdp)
2025-11-19 11:07:11 [execute] INFO - ❌ Flax not available (JAX requires Flax - install with: pip install flax)
2025-11-19 11:07:11 [execute] INFO - ❌ RxInfer not available (optional - install Julia first, then: julia -e 'import Pkg; Pkg.add("RxInfer")')
```

### During Execution
For each framework:

**Available**:
```
2025-11-19 11:07:13 [execute] INFO - ✅ Successfully executed model_name_discopy.py
```

**Missing**:
```
2025-11-19 11:07:13 [execute] WARNING - ❌ model_name_pymdp.py failed
2025-11-19 11:07:13 [execute] WARNING - Error: PyMDP not available - install with: pip install pymdp
```

### After Execution
The execution report shows framework statistics:

```json
{
  "frameworks": {
    "total": 5,
    "available": 2,
    "executed": 2,
    "succeeded": 2,
    "failed": 3
  },
  "framework_details": {
    "pymdp": {
      "status": "not_available",
      "reason": "Module pymdp not found",
      "install_command": "pip install pymdp"
    },
    "jax": {
      "status": "not_available",
      "reason": "Module flax not found (required by JAX)",
      "install_command": "pip install flax"
    },
    "discopy": {
      "status": "success",
      "scripts_executed": 1,
      "scripts_failed": 0
    }
  }
}
```

## Framework Dependencies

### Full Dependency Tree

```
PyMDP (Python)
├── pymdp package
├── numpy
└── scipy

JAX (Python)
├── jax package
├── flax package (for neural networks)
├── jaxlib
└── numpy

DisCoPy (Python)
├── discopy package
└── numpy

RxInfer.jl (Julia)
├── Julia runtime
└── RxInfer.jl package

ActiveInference.jl (Julia)
├── Julia runtime
└── ActiveInference.jl package
```

## Determining What You Need

### Minimum for Basic Pipeline
```bash
# Just core dependencies
pip install -r requirements.txt
# Result: Only DisCoPy works, but pipeline completes successfully
```

### Minimum for Most Use Cases
```bash
pip install -r requirements.txt
pip install pymdp flax
# Result: PyMDP, JAX, DisCoPy work (3/5 frameworks)
```

### For Complete Coverage
```bash
pip install -r requirements.txt
pip install pymdp flax plotly
julia -e 'import Pkg; Pkg.add(["RxInfer", "ActiveInference"])'
# Result: All 5 frameworks work
```

## Troubleshooting

### Framework Not Detected But Installed

**Problem**: Framework shows as "not available" but you installed it

**Solutions**:
1. Check installation: `pip list | grep pymdp`
2. Verify Python path: `which python`
3. Try direct import: `python -c "import pymdp"`
4. Reinstall: `pip install --force-reinstall pymdp`

### Julia Packages Not Found

**Problem**: Julia shows available but packages not found

**Solutions**:
1. Check Julia version: `julia --version`
2. Verify packages: `julia -e "import Pkg; Pkg.status()"`
3. Add missing: `julia -e 'import Pkg; Pkg.add("RxInfer")'`
4. Update: `julia -e 'import Pkg; Pkg.update()'`

### Mixed Python/Julia Errors

**Problem**: Some frameworks work, others don't

**Solutions**:
1. Verify environments separately:
   ```bash
   python -c "import pymdp; print('✅ PyMDP')" || echo "❌ PyMDP"
   julia -e "using RxInfer; println(\"✅ RxInfer\")" || echo "❌ RxInfer"
   ```
2. Check PATH: `echo $PATH` (should include both python and julia)
3. Use full paths if needed:
   ```bash
   /usr/bin/python3 -c "import pymdp"
   /usr/local/bin/julia -e "using RxInfer"
   ```

## Viewing Framework Status

### In Real-time During Execution
```bash
python src/12_execute.py --verbose --target-dir input/gnn_files --output-dir output
```

### After Execution
```bash
# View summary
cat output/12_execute_output/execution_results.json | jq .framework_details

# View detailed status
cat output/12_execute_output/framework_status.json | jq .
```

### Via Python API
```python
from execute import get_execution_health_status

status = get_execution_health_status()
for framework, details in status.items():
    if details['available']:
        print(f"✅ {framework}: {details.get('version', 'unknown')}")
    else:
        print(f"❌ {framework}: {details.get('reason', 'unknown')}")
        print(f"   Install: {details.get('install_command', 'unknown')}")
```

## Performance Impact of Missing Frameworks

The pipeline gracefully handles missing frameworks:

- **Pipeline completion**: Not affected (still SUCCESS or SUCCESS_WITH_WARNINGS)
- **Execution time**: Slightly faster (skips unavailable frameworks)
- **Memory usage**: No change
- **Test results**: 90%+ pass rate maintained

## Framework Selection During Execution

You can control which frameworks to use:

```bash
# Use only specific frameworks
python src/12_execute.py --frameworks "discopy,activeinference_jl" ...

# Use preset combinations
python src/12_execute.py --frameworks "lite" ...  # Fast: pymdp, jax, discopy
python src/12_execute.py --frameworks "all" ...   # All 5 frameworks
```

## Next Steps

1. **Check your environment**: `python -c "from execute import get_execution_health_status; print(get_execution_health_status())"`
2. **Install needed frameworks**: See [OPTIONAL_DEPENDENCIES.md](../dependencies/OPTIONAL_DEPENDENCIES.md)
3. **Run execution**: `python src/12_execute.py --verbose`
4. **Check results**: `cat output/12_execute_output/execution_results.json | jq`

---

**Last Updated**: November 19, 2025  
**Compatible with**: Pipeline v2.1.0+

