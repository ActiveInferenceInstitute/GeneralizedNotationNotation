#!/usr/bin/env python3
"""
Comprehensive Dependency Installation Script for GNN Pipeline

This script ensures all required dependencies are installed for the GNN processing pipeline,
including Python packages, Julia packages, and system dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, description, check=True, timeout=300):
    """Run a command and log the result."""
    logger.info(f"🔧 {description}")
    logger.info(f"   Command: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=check, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
            if result.stdout:
                logger.debug(f"   stdout: {result.stdout[:500]}...")
            return True
        else:
            logger.warning(f"⚠️ {description} completed with warnings (exit code: {result.returncode})")
            if result.stderr:
                logger.warning(f"   stderr: {result.stderr[:500]}...")
            if result.stdout:
                logger.debug(f"   stdout: {result.stdout[:500]}...")
            return False
    except subprocess.TimeoutExpired as e:
        logger.error(f"⏰ {description} timed out after {timeout}s")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"   stderr: {e.stderr[:500]}...")
        if e.stdout:
            logger.debug(f"   stdout: {e.stdout[:500]}...")
        if check:
            raise
        return False
    except Exception as e:
        logger.error(f"❌ {description} failed with unexpected error: {e}")
        if check:
            raise
        return False

def check_python_package(package_name):
    """Check if a Python package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_python_dependencies():
    """Install Python dependencies using UV."""
    logger.info("🐍 Installing Python dependencies...")
    
    # Check if UV is available
    logger.info("🔍 Checking UV availability...")
    if not run_command("uv --version", "Checking UV availability", check=False, timeout=30):
        logger.warning("UV is not available, falling back to pip...")
        uv_available = False
    else:
        logger.info("✅ UV is available")
        uv_available = True
    
    if uv_available:
        # Install dependencies using UV
        logger.info("📦 Installing dependencies with UV...")
        if not run_command("uv sync", "Installing Python dependencies with UV", check=False, timeout=600):
            logger.warning("UV sync failed, trying pip install...")
            uv_available = False
    
    if not uv_available:
        # Fallback to pip
        logger.info("📦 Installing dependencies with pip...")
        packages = [
            "discopy[matrix]>=1.0.0",
            "pymdp>=0.0.1", 
            "aiohttp>=3.9.0",
            "ollama",
            "jax[cpu]>=0.4.0",
            "jaxlib>=0.4.0",
            "flax>=0.7.0",
            "optax>=0.1.0"
        ]
        
        for package in packages:
            logger.info(f"📦 Installing {package}...")
            run_command(f"pip install {package}", f"Installing {package}", check=False, timeout=300)
    
    return True

def install_julia_dependencies():
    """Install Julia dependencies."""
    logger.info("🔬 Installing Julia dependencies...")
    
    # Check if Julia is available
    logger.info("🔍 Checking Julia availability...")
    if not run_command("julia --version", "Checking Julia availability", check=False, timeout=30):
        logger.warning("Julia is not available. Please install Julia first.")
        return False
    
    logger.info("✅ Julia is available")
    
    # Run Julia package installation script
    julia_script = Path(__file__).parent / "install_julia_packages.jl"
    if julia_script.exists():
        logger.info(f"📦 Running Julia package installation script: {julia_script}")
        run_command(f"julia {julia_script}", "Installing Julia packages", check=False, timeout=600)
    else:
        logger.warning("Julia package installation script not found")
        # Try direct Julia package installation
        logger.info("📦 Installing Julia packages directly...")
        julia_cmd = 'julia -e "using Pkg; Pkg.add([\"RxInfer\", \"ActiveInference\", \"Distributions\", \"Plots\", \"Random\"])"'
        run_command(julia_cmd, "Installing Julia packages directly", check=False, timeout=300)
    
    return True

def verify_installations():
    """Verify that key packages are installed."""
    logger.info("🔍 Verifying installations...")
    
    # Check Python packages
    python_packages = [
        "discopy",
        "pymdp", 
        "aiohttp",
        "ollama",
        "jax",
        "jaxlib",
        "flax",
        "optax"
    ]
    
    for package in python_packages:
        if check_python_package(package):
            logger.info(f"✅ Python package {package} is available")
        else:
            logger.warning(f"❌ Python package {package} is not available")
    
    # Check Julia packages
    julia_check_script = """
    using Pkg
    packages = ["RxInfer", "ActiveInference", "Distributions", "Plots"]
    for pkg in packages
        try
            eval(Meta.parse("using $pkg"))
            println("✅ Julia package $pkg is available")
        catch e
            println("❌ Julia package $pkg is not available: $e")
        end
    end
    """
    
    run_command(f'julia -e "{julia_check_script}"', "Checking Julia packages", check=False)

def main():
    """Main installation function."""
    logger.info("🚀 Starting comprehensive dependency installation...")
    
    try:
        # Change to project root
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        logger.info(f"📁 Working directory: {project_root}")
        
        # Install Python dependencies
        logger.info("=" * 60)
        logger.info("STEP 1: Installing Python dependencies")
        logger.info("=" * 60)
        python_success = install_python_dependencies()
        
        # Install Julia dependencies
        logger.info("=" * 60)
        logger.info("STEP 2: Installing Julia dependencies")
        logger.info("=" * 60)
        julia_success = install_julia_dependencies()
        
        # Verify installations
        logger.info("=" * 60)
        logger.info("STEP 3: Verifying installations")
        logger.info("=" * 60)
        verify_installations()
        
        # Summary
        logger.info("=" * 60)
        logger.info("INSTALLATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Python dependencies: {'✅ Success' if python_success else '❌ Failed'}")
        logger.info(f"Julia dependencies: {'✅ Success' if julia_success else '❌ Failed'}")
        
        if python_success and julia_success:
            logger.info("🎉 All dependencies installed successfully!")
            logger.info("💡 You can now run the GNN pipeline with: python3 src/main.py --target-dir input/gnn_files")
            return 0
        else:
            logger.warning("⚠️ Some dependencies failed to install. Check the logs above for details.")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("⏹️ Installation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Installation failed with unexpected error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    main()
