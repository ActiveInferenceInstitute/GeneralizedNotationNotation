#!/usr/bin/env python3
"""
Dependency Installation Script for GNN Execution System

This script installs missing dependencies for all execution environments.
"""

import subprocess
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def install_python_dependencies():
    """Install missing Python dependencies."""
    logger.info("Installing Python dependencies...")
    
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
        logger.info(f"  Installing {dep}...")
        try:
            result = subprocess.run(
                ["uv", "pip", "install", dep],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                logger.info(f"    ✅ {dep} installed successfully")
            else:
                logger.error(f"    ❌ Failed to install {dep}: {result.stderr}")
        except Exception as e:
            logger.error(f"    ❌ Error installing {dep}: {e}")

def install_julia_dependencies():
    """Install Julia dependencies for ActiveInference.jl."""
    logger.info("Installing Julia dependencies...")
    
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
        logger.info(f"  Installing Julia package {pkg}...")
        try:
            result = subprocess.run(
                ["julia", "-e", f'using Pkg; Pkg.add("{pkg}")'],
                capture_output=True,
                text=True,
                check=False,
                timeout=60
            )
            if result.returncode == 0:
                logger.info(f"    ✅ {pkg} installed successfully")
            else:
                logger.error(f"    ❌ Failed to install {pkg}: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning(f"    ⚠️ Timeout installing {pkg}")
        except Exception as e:
            logger.error(f"    ❌ Error installing {pkg}: {e}")

def verify_installations():
    """Verify that all dependencies are properly installed."""
    logger.info("Verifying installations...")
    
    # Test Python imports
    python_deps = ["numpy", "pymdp", "flax", "jax", "optax"]
    for dep in python_deps:
        try:
            __import__(dep)
            logger.info(f"  ✅ {dep} (Python)")
        except ImportError:
            logger.warning(f"  ❌ {dep} (Python) - not available")
    
    # Test Julia availability
    try:
        result = subprocess.run(
            ["julia", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            logger.info(f"  ✅ Julia: {result.stdout.strip()}")
        else:
            logger.warning("  ❌ Julia - not available")
    except FileNotFoundError:
        logger.warning("  ❌ Julia - not found in PATH")

def main():
    """Main installation function."""
    logger.info("GNN Execution System Dependency Installer")
    logger.info("=" * 50)
    
    # Install Python dependencies
    install_python_dependencies()
    
    # Install Julia dependencies
    install_julia_dependencies()
    
    # Verify installations
    verify_installations()
    
    logger.info("Installation complete!")
    logger.info("If any dependencies failed to install, you may need to:")
    logger.info("   - Install them manually: uv pip install <package_name>")
    logger.info("   - Install Julia from: https://julialang.org/downloads/")
    logger.info("   - Check your Python environment and virtual environment")

if __name__ == "__main__":
    main() 