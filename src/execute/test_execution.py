#!/usr/bin/env python3
"""
Comprehensive Execution Test Script

This script tests all execution components to ensure they work properly
with the enhanced error handling and dependency checking.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from execute.executor import execute_rendered_simulators
from execute.pymdp.pymdp_runner import validate_and_clean_pymdp_script

def find_pymdp_scripts(rendered_simulators_dir):
    from pathlib import Path
    base = Path(rendered_simulators_dir)
    pymdp_dir = base / "pymdp"
    if not pymdp_dir.exists():
        return []
    return [p for p in pymdp_dir.rglob("*.py") if not any(part.startswith('__') for part in p.parts)]
from execute.jax.jax_runner import is_jax_available
from execute.activeinference_jl.activeinference_runner import is_julia_available
try:
    from utils import setup_step_logging, log_step_start, log_step_success, log_step_warning, log_step_error
except Exception:
    import logging as _logging
    def setup_step_logging(name: str, verbose: bool = False):
        logger = _logging.getLogger(name)
        handler = _logging.StreamHandler(sys.stdout)
        logger.handlers = [handler]
        logger.setLevel(_logging.DEBUG if verbose else _logging.INFO)
        return logger
    def log_step_start(logger, msg): logger.info(f"🚀 {msg}")
    def log_step_success(logger, msg): logger.info(f"✅ {msg}")
    def log_step_warning(logger, msg): logger.warning(f"⚠️ {msg}")
    def log_step_error(logger, msg): logger.error(f"❌ {msg}")

def test_dependency_checking():
    """Test dependency checking for all execution environments."""
    print("🔍 Testing dependency checking...")
    
    # Test Python dependencies
    python_deps = ["numpy", "pymdp", "flax", "jax", "optax"]
    missing_python_deps = []
    
    for dep in python_deps:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            missing_python_deps.append(dep)
            print(f"  ❌ {dep} (missing)")
    
    # Test Julia availability
    julia_available = is_julia_available()
    print(f"  {'✅' if julia_available else '❌'} Julia")
    
    # Test JAX availability
    jax_available = is_jax_available()
    print(f"  {'✅' if jax_available else '❌'} JAX")
    
    return {
        "python_deps": python_deps,
        "missing_python_deps": missing_python_deps,
        "julia_available": julia_available,
        "jax_available": jax_available
    }

def test_script_validation():
    """Test script validation and cleanup."""
    print("\n🔍 Testing script validation...")
    
    # Test PyMDP script validation
    pymdp_dir = Path("../output/gnn_rendered_simulators/gnn_rendered_simulators/pymdp")
    if pymdp_dir.exists():
        pymdp_scripts = list(pymdp_dir.glob("*.py"))
        for script in pymdp_scripts:
            print(f"  Testing PyMDP script: {script.name}")
            is_valid = validate_and_clean_pymdp_script(script)
            print(f"    {'✅' if is_valid else '❌'} Valid")
    else:
        print("  ⚠️ No PyMDP scripts found")

def test_execution_components():
    """Test individual execution components."""
    print("\n🚀 Testing execution components...")
    
    # Test PyMDP script discovery
    print("  Testing PyMDP script discovery...")
    pymdp_scripts = find_pymdp_scripts("../output")
    print(f"    Found {len(pymdp_scripts)} PyMDP scripts")
    
    # Test JAX script discovery
    print("  Testing JAX script discovery...")
    from execute.jax.jax_runner import find_jax_scripts
    jax_scripts = find_jax_scripts("../output")
    print(f"    Found {len(jax_scripts)} JAX scripts")
    
    # Test ActiveInference.jl script discovery
    print("  Testing ActiveInference.jl script discovery...")
    from execute.activeinference_jl.activeinference_runner import find_activeinference_scripts
    activeinference_scripts = find_activeinference_scripts("../output")
    print(f"    Found {len(activeinference_scripts)} ActiveInference.jl scripts")

def test_full_execution():
    """Test the full execution pipeline."""
    print("\n🚀 Testing full execution pipeline...")
    
    # Set up logging
    logger = setup_step_logging("test_execution", verbose=True)
    
    # Test the main execution function
    success = execute_rendered_simulators(
        target_dir=Path("../output"),
        output_dir=Path("../output/execute_logs"),
        logger=logger,
        recursive=True,
        verbose=True
    )
    
    print(f"  Full execution {'✅ succeeded' if success else '❌ failed'}")
    return success

def main():
    """Main test function."""
    print("🧪 Comprehensive Execution Test Suite")
    print("=" * 50)
    
    # Test dependency checking
    deps = test_dependency_checking()
    
    # Test script validation
    test_script_validation()
    
    # Test execution components
    test_execution_components()
    
    # Test full execution
    success = test_full_execution()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    print(f"Python Dependencies: {len(deps['python_deps']) - len(deps['missing_python_deps'])}/{len(deps['python_deps'])} available")
    print(f"Julia: {'✅ Available' if deps['julia_available'] else '❌ Not Available'}")
    print(f"JAX: {'✅ Available' if deps['jax_available'] else '❌ Not Available'}")
    print(f"Full Execution: {'✅ Success' if success else '❌ Failed'}")
    
    if deps['missing_python_deps']:
        print(f"\n💡 Missing Python dependencies: {', '.join(deps['missing_python_deps'])}")
        print("   Install with: pip install " + " ".join(deps['missing_python_deps']))
    
    if not deps['julia_available']:
        print("\n💡 Julia not available. Install from: https://julialang.org/downloads/")
    
    if not deps['jax_available']:
        print("\n💡 JAX not available. Install with: pip install jax jaxlib")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 