#!/usr/bin/env python3
"""
Step 1: Project Setup and Environment Validation

This step handles project initialization, virtual environment setup,
dependency installation, and environment validation.
"""

import sys
import subprocess
import venv
import logging
import platform
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

def log_system_info(logger: logging.Logger) -> Dict[str, Any]:
    """Log comprehensive system information."""
    try:
        log_step_start(logger, "Logging system information")
        
        system_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "node": platform.node()
        }
        
        logger.info(f"System Platform: {system_info['platform']}")
        logger.info(f"Python Version: {system_info['python_version']}")
        logger.info(f"Python Executable: {system_info['python_executable']}")
        logger.info(f"Architecture: {system_info['architecture']}")
        logger.info(f"Processor: {system_info['processor']}")
        logger.info(f"Machine: {system_info['machine']}")
        logger.info(f"Node: {system_info['node']}")
        
        log_step_success(logger, "System information logged")
        return system_info
        
    except Exception as e:
        log_step_error(logger, f"Failed to log system information: {e}")
        return {}

def setup_virtual_environment(venv_path: Path, logger: logging.Logger) -> bool:
    """Create and configure virtual environment with comprehensive logging."""
    try:
        log_step_start(logger, "Creating virtual environment")
        
        logger.info(f"🎯 Target virtual environment path: {venv_path.absolute()}")
        logger.info(f"📁 Current working directory: {Path.cwd()}")
        
        if venv_path.exists():
            log_step_warning(logger, f"Virtual environment already exists at {venv_path.absolute()}")
            logger.info(f"📊 Existing venv size: {sum(f.stat().st_size for f in venv_path.rglob('*') if f.is_file()) / 1024 / 1024:.2f} MB")
            logger.info("🔄 Using existing virtual environment")
            # Validate the existing virtual environment
            python_path = venv_path / "bin" / "python"
            if sys.platform == "win32":
                python_path = venv_path / "Scripts" / "python.exe"
            
            if python_path.exists():
                logger.info(f"✅ Existing virtual environment is valid: {python_path}")
                return True
            else:
                log_step_error(logger, f"Existing virtual environment is invalid: {python_path} not found")
                return False
        
        # Create virtual environment
        logger.info("🆕 Creating new virtual environment...")
        logger.info(f"📍 Creating at: {venv_path.absolute()}")
        venv.create(venv_path, with_pip=True)
        logger.info("✅ Virtual environment created successfully")
        
        # Log venv structure
        logger.info("Virtual environment structure:")
        for item in sorted(venv_path.iterdir()):
            if item.is_dir():
                logger.info(f"  📁 {item.name}/")
            else:
                logger.info(f"  📄 {item.name}")
        
        # Check Python executable
        python_path = venv_path / "bin" / "python"
        if sys.platform == "win32":
            python_path = venv_path / "Scripts" / "python.exe"
        
        if python_path.exists():
            logger.info(f"Virtual environment Python: {python_path}")
            
            # Test Python version
            result = subprocess.run(
                [str(python_path), "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"Virtual environment Python version: {result.stdout.strip()}")
        else:
            log_step_error(logger, f"Python executable not found at {python_path}")
            return False
        
        log_step_success(logger, "Virtual environment created successfully")
        return True
        
    except Exception as e:
        log_step_error(logger, f"Failed to create virtual environment: {e}")
        return False

def install_dependencies(venv_path: Path, requirements_file: Path, logger: logging.Logger) -> bool:
    """Install project dependencies with detailed logging."""
    try:
        log_step_start(logger, "Installing dependencies")
        
        logger.info(f"Requirements file: {requirements_file}")
        logger.info(f"Requirements file exists: {requirements_file.exists()}")
        
        if not requirements_file.exists():
            log_step_warning(logger, f"Requirements file not found at {requirements_file}")
            return True
        
        # Read and log requirements
        try:
            with open(requirements_file, 'r') as f:
                requirements = f.read()
            logger.info(f"Requirements file contents ({len(requirements.splitlines())} lines):")
            for line in requirements.splitlines():
                if line.strip() and not line.startswith('#'):
                    logger.info(f"  📦 {line.strip()}")
        except Exception as e:
            logger.warning(f"Could not read requirements file: {e}")
        
        # Get pip path
        pip_path = venv_path / "bin" / "pip"
        if sys.platform == "win32":
            pip_path = venv_path / "Scripts" / "pip.exe"
        
        logger.info(f"Using pip: {pip_path}")
        logger.info(f"Pip exists: {pip_path.exists()}")
        
        # Upgrade pip first
        logger.info("Upgrading pip...")
        upgrade_result = subprocess.run(
            [str(pip_path), "install", "--upgrade", "pip"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if upgrade_result.returncode == 0:
            logger.info("Pip upgraded successfully")
        else:
            logger.warning(f"Pip upgrade failed: {upgrade_result.stderr}")
        
        # Install dependencies
        logger.info("Installing project dependencies...")
        result = subprocess.run(
            [str(pip_path), "install", "-r", str(requirements_file)],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            logger.info("Dependencies installed successfully")
            
            # List installed packages
            list_result = subprocess.run(
                [str(pip_path), "list"],
                capture_output=True,
                text=True
            )
            
            if list_result.returncode == 0:
                packages = list_result.stdout.strip().split('\n')[2:]  # Skip header
                logger.info(f"Installed packages ({len(packages)}):")
                for package in packages[:10]:  # Show first 10
                    if package.strip():
                        logger.info(f"  📦 {package.strip()}")
                if len(packages) > 10:
                    logger.info(f"  ... and {len(packages) - 10} more packages")
            
            return True
        else:
            log_step_error(logger, f"Failed to install dependencies: {result.stderr}")
            logger.error(f"Pip install stdout: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        log_step_error(logger, "Dependency installation timed out after 5 minutes")
        return False
    except Exception as e:
        log_step_error(logger, f"Failed to install dependencies: {e}")
        return False

def validate_environment(venv_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Validate the environment setup with comprehensive checks."""
    try:
        log_step_start(logger, "Validating environment")
        
        validation_results = {
            "venv_exists": venv_path.exists(),
            "python_version": None,
            "pip_available": False,
            "key_packages": {},
            "package_versions": {},
            "system_info": {}
        }
        
        logger.info(f"Virtual environment exists: {validation_results['venv_exists']}")
        
        # Check Python version
        python_path = venv_path / "bin" / "python"
        if sys.platform == "win32":
            python_path = venv_path / "Scripts" / "python.exe"
        
        if python_path.exists():
            logger.info(f"Python executable: {python_path}")
            
            result = subprocess.run(
                [str(python_path), "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                validation_results["python_version"] = result.stdout.strip()
                logger.info(f"Python version: {validation_results['python_version']}")
            
            # Check pip availability
            pip_path = venv_path / "bin" / "pip"
            if sys.platform == "win32":
                pip_path = venv_path / "Scripts" / "pip.exe"
            
            validation_results["pip_available"] = pip_path.exists()
            logger.info(f"Pip available: {validation_results['pip_available']}")
            
            # Check key packages with versions
            key_packages = ["numpy", "matplotlib", "networkx", "pandas", "pyyaml", "scipy", "scikit-learn"]
            logger.info("Validating key packages:")
            
            for package in key_packages:
                try:
                    # Check import
                    import_result = subprocess.run(
                        [str(python_path), "-c", f"import {package}; print({package}.__version__)"],
                        capture_output=True,
                        text=True
                    )
                    
                    if import_result.returncode == 0:
                        version = import_result.stdout.strip()
                        validation_results["key_packages"][package] = True
                        validation_results["package_versions"][package] = version
                        logger.info(f"  ✅ {package}: {version}")
                    else:
                        validation_results["key_packages"][package] = False
                        validation_results["package_versions"][package] = None
                        logger.warning(f"  ❌ {package}: Not available")
                        
                except Exception as e:
                    validation_results["key_packages"][package] = False
                    validation_results["package_versions"][package] = None
                    logger.warning(f"  ❌ {package}: Error checking - {e}")
            
            # Check system resources
            try:
                import psutil
                validation_results["system_info"] = {
                    "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                    "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                    "disk_usage_percent": psutil.disk_usage('/').percent,
                    "cpu_count": psutil.cpu_count()
                }
                logger.info(f"System memory: {validation_results['system_info']['memory_available_gb']:.1f}GB available of {validation_results['system_info']['memory_total_gb']:.1f}GB total")
                logger.info(f"CPU cores: {validation_results['system_info']['cpu_count']}")
                logger.info(f"Disk usage: {validation_results['system_info']['disk_usage_percent']:.1f}%")
            except ImportError:
                logger.warning("psutil not available - system resource information limited")
                validation_results["system_info"] = {"error": "psutil_not_available"}
        
        else:
            log_step_error(logger, f"Python executable not found at {python_path}")
        
        log_step_success(logger, "Environment validation completed")
        return validation_results
        
    except Exception as e:
        log_step_error(logger, f"Environment validation failed: {e}")
        return {}

def create_project_structure(output_dir: Path, logger: logging.Logger) -> bool:
    """Create necessary project directories with detailed logging."""
    try:
        log_step_start(logger, "Creating project structure")
        
        directories = [
            "input/gnn_files",
            "output",
            "logs",
            "temp",
            "docs"
        ]
        
        logger.info(f"Creating directories relative to: {output_dir.parent}")
        
        for dir_name in directories:
            dir_path = output_dir.parent / dir_name
            logger.info(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Verify creation
            if dir_path.exists():
                logger.info(f"  ✅ {dir_name}/ created successfully")
            else:
                log_step_error(logger, f"Failed to create directory: {dir_name}")
                return False
        
        log_step_success(logger, "Project structure created successfully")
        return True
        
    except Exception as e:
        log_step_error(logger, f"Failed to create project structure: {e}")
        return False

def main():
    """Main setup function."""
    args = EnhancedArgumentParser.parse_step_arguments("1_setup")
    
    # Setup logging
    logger = setup_step_logging("setup", args)
    
    try:
        # Log system information
        system_info = log_system_info(logger)
        
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("1_setup.py", Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {output_dir}")
        
        # Setup virtual environment at project root
        venv_path = Path(__file__).parent.parent / ".venv"  # Use .venv at project root
        logger.info(f"🚀 Setting up virtual environment at: {venv_path.absolute()}")
        if not setup_virtual_environment(venv_path, logger):
            return 1
        
        # Install dependencies
        requirements_file = Path(__file__).parent.parent / "requirements.txt"
        if not requirements_file.exists():
            log_step_warning(logger, f"Requirements file not found at {requirements_file}, skipping dependency installation")
        else:
            if not install_dependencies(venv_path, requirements_file, logger):
                return 1
        
        # Validate environment
        validation_results = validate_environment(venv_path, logger)
        if not validation_results:
            return 1
        
        # Create project structure
        if not create_project_structure(output_dir, logger):
            return 1
        
        # Save comprehensive setup results
        setup_results = {
            "venv_path": str(venv_path.absolute()),
            "validation_results": validation_results,
            "output_dir": str(output_dir),
            "system_info": system_info,
            "setup_timestamp": str(Path(__file__).stat().st_mtime),
            "requirements_file": str(requirements_file) if requirements_file.exists() else None
        }
        
        results_file = output_dir / "setup_results.json"
        with open(results_file, 'w') as f:
            json.dump(setup_results, f, indent=2)
        
        logger.info(f"Setup results saved to: {results_file}")
        
        # Log summary
        logger.info("=== SETUP SUMMARY ===")
        logger.info(f"🎯 Virtual Environment Location: {venv_path.absolute()}")
        logger.info(f"🐍 Python Version: {validation_results.get('python_version', 'Unknown')}")
        logger.info(f"📦 Key Packages Available: {sum(validation_results.get('key_packages', {}).values())}/{len(validation_results.get('key_packages', {}))}")
        logger.info(f"📁 Output Directory: {output_dir}")
        logger.info(f"✅ Virtual Environment Status: {'EXISTS' if venv_path.exists() else 'MISSING'}")
        
        log_step_success(logger, "Setup completed successfully")
        return 0
        
    except Exception as e:
        log_step_error(logger, f"Setup failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 