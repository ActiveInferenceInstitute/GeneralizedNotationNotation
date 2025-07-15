#!/usr/bin/env python3
"""
Dependency Installation Script for GNN Execution System

This script installs missing dependencies for all execution environments.
"""

import subprocess
import sys
from pathlib import Path

def install_python_dependencies():
    """Install missing Python dependencies."""
    print("🐍 Installing Python dependencies...")
    
    # Core dependencies
    core_deps = [
        "numpy",
        "pymdp",
        "flax",
        "jax",
        "jaxlib",
        "optax",
        "discopy",
        "networkx",
        "matplotlib"
    ]
    
    for dep in core_deps:
        print(f"  Installing {dep}...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", dep],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                print(f"    ✅ {dep} installed successfully")
            else:
                print(f"    ❌ Failed to install {dep}: {result.stderr}")
        except Exception as e:
            print(f"    ❌ Error installing {dep}: {e}")

def install_julia_dependencies():
    """Install Julia dependencies for ActiveInference.jl."""
    print("\n🔬 Installing Julia dependencies...")
    
    # Julia packages needed for ActiveInference.jl
    julia_packages = [
        "DelimitedFiles",
        "CSV",
        "DataFrames",
        "Plots",
        "Statistics",
        "LinearAlgebra",
        "Random",
        "Distributions"
    ]
    
    for pkg in julia_packages:
        print(f"  Installing Julia package {pkg}...")
        try:
            result = subprocess.run(
                ["julia", "-e", f'using Pkg; Pkg.add("{pkg}")'],
                capture_output=True,
                text=True,
                check=False,
                timeout=60
            )
            if result.returncode == 0:
                print(f"    ✅ {pkg} installed successfully")
            else:
                print(f"    ❌ Failed to install {pkg}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"    ⚠️ Timeout installing {pkg}")
        except Exception as e:
            print(f"    ❌ Error installing {pkg}: {e}")

def verify_installations():
    """Verify that all dependencies are properly installed."""
    print("\n🔍 Verifying installations...")
    
    # Test Python imports
    python_deps = ["numpy", "pymdp", "flax", "jax", "optax"]
    for dep in python_deps:
        try:
            __import__(dep)
            print(f"  ✅ {dep} (Python)")
        except ImportError:
            print(f"  ❌ {dep} (Python) - not available")
    
    # Test Julia availability
    try:
        result = subprocess.run(
            ["julia", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print(f"  ✅ Julia: {result.stdout.strip()}")
        else:
            print("  ❌ Julia - not available")
    except FileNotFoundError:
        print("  ❌ Julia - not found in PATH")

def main():
    """Main installation function."""
    print("📦 GNN Execution System Dependency Installer")
    print("=" * 50)
    
    # Install Python dependencies
    install_python_dependencies()
    
    # Install Julia dependencies
    install_julia_dependencies()
    
    # Verify installations
    verify_installations()
    
    print("\n🎉 Installation complete!")
    print("\n💡 If any dependencies failed to install, you may need to:")
    print("   - Install them manually: pip install <package_name>")
    print("   - Install Julia from: https://julialang.org/downloads/")
    print("   - Check your Python environment and virtual environment")

if __name__ == "__main__":
    main() 