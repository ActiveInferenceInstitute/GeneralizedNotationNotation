#!/usr/bin/env python3
"""
Step 1: Project Setup and Environment Validation with UV

This step handles project initialization, UV environment setup,
dependency installation, and environment validation using modern
Python packaging standards.
"""

import sys
import subprocess
import logging
import platform
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import time

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

def check_uv_availability(logger: logging.Logger) -> bool:
    """Check if UV is available and properly installed."""
    try:
        log_step_start(logger, "Checking UV availability")
        
        # Check if uv is installed
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info(f"✅ UV is available: {version}")
            
            # Check if uv is up to date
            try:
                update_result = subprocess.run(
                    ["uv", "self", "update", "--check"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if update_result.returncode == 0:
                    logger.info("✅ UV is up to date")
                else:
                    logger.warning("⚠️ UV update available")
            except Exception as e:
                logger.warning(f"⚠️ Could not check UV update status: {e}")
            
            log_step_success(logger, "UV availability check passed")
            return True
        else:
            log_step_error(logger, f"UV not found or not working: {result.stderr}")
            return False
        
    except FileNotFoundError:
        log_step_error(logger, "UV not found. Please install UV first:")
        logger.error("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        logger.error("  or visit: https://docs.astral.sh/uv/getting-started/installation/")
        return False
    except Exception as e:
        log_step_error(logger, f"Failed to check UV availability: {e}")
        return False

def setup_uv_environment(project_root: Path, logger: logging.Logger) -> bool:
    """Initialize UV environment and sync dependencies."""
    try:
        log_step_start(logger, "Setting up UV environment")
        
        logger.info(f"🎯 Project root: {project_root.absolute()}")
        logger.info(f"📁 Current working directory: {Path.cwd()}")
        
        # Check if pyproject.toml exists
        pyproject_path = project_root / "pyproject.toml"
        if not pyproject_path.exists():
            log_step_error(logger, f"pyproject.toml not found at {pyproject_path}")
            return False
        
        logger.info(f"✅ Found pyproject.toml at {pyproject_path}")
        
        # Initialize UV environment (creates .venv and uv.lock)
        logger.info("🆕 Initializing UV environment...")
        init_result = subprocess.run(
            ["uv", "init", "--python", "3.12"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if init_result.returncode != 0:
            logger.warning(f"UV init warning (may already be initialized): {init_result.stderr}")
        
        # Sync dependencies (install from pyproject.toml)
        logger.info("📦 Syncing dependencies with UV...")
        logger.info("⏱️ This may take several minutes for first-time setup...")
        
        start_time = time.time()
        sync_result = subprocess.run(
            ["uv", "sync"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes
        )
        
        sync_time = time.time() - start_time
        
        if sync_result.returncode == 0:
            logger.info("✅ Dependencies synced successfully")
            logger.info(f"⏱️ Sync time: {sync_time:.1f} seconds")
            
            # Log what was installed
            if sync_result.stdout:
                logger.info("📋 Sync output:")
                for line in sync_result.stdout.split('\n')[-10:]:  # Last 10 lines
                    if line.strip():
                        logger.info(f"  {line}")
            
            return True
        else:
            log_step_error(logger, f"Failed to sync dependencies: {sync_result.stderr}")
            logger.error(f"⏱️ Sync failed after {sync_time:.1f} seconds")
            return False
            
    except subprocess.TimeoutExpired:
        log_step_error(logger, "UV sync timed out after 30 minutes")
        return False
    except Exception as e:
        log_step_error(logger, f"Failed to setup UV environment: {e}")
        return False

def install_optional_dependencies(project_root: Path, logger: logging.Logger, 
                                package_groups: List[str] = None) -> bool:
    """Install optional dependency groups using UV."""
    try:
        log_step_start(logger, "Installing optional dependencies")
        
        # Define available optional groups
        available_groups = {
            "dev": "Development dependencies (testing, linting, docs)",
            "ml-ai": "Machine Learning & AI (torch, transformers)",
            "llm": "LLM Integration (openai, anthropic)",
            "visualization": "Advanced Visualization (plotly, seaborn, bokeh)",
            "audio": "Audio Processing (librosa, soundfile, pedalboard)",
            "graphs": "Graph Visualization (igraph, graphviz)",
            "research": "Research Tools (jupyter, ipywidgets)",
            "all": "All optional dependencies"
        }
        
        if package_groups is None:
            logger.info("📦 Available optional dependency groups:")
            for group, description in available_groups.items():
                logger.info(f"  🔸 {group}: {description}")
            logger.info("💡 Use --install-optional <group1,group2> to install specific groups")
            return True
        
        total_installed = 0
        total_groups = len(package_groups)
        
        for group in package_groups:
            if group not in available_groups:
                logger.warning(f"⚠️ Unknown dependency group: {group}")
                continue
            
            logger.info(f"📦 Installing {group} dependencies: {available_groups[group]}")
            
            try:
                # Install the optional dependency group
                    result = subprocess.run(
                    ["uv", "sync", f"--extra={group}"],
                    cwd=project_root,
                        capture_output=True,
                        text=True,
                    timeout=600  # 10 minutes per group
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"  ✅ {group} dependencies installed successfully")
                        total_installed += 1
                    else:
                        logger.warning(f"  ❌ {group} failed: {result.stderr}")
                        
            except subprocess.TimeoutExpired:
                logger.error(f"  ⏰ {group} installation timed out")
            except Exception as e:
                logger.error(f"  ❌ {group} error: {e}")
        
        logger.info(f"📊 Optional dependencies: {total_installed}/{total_groups} groups installed")
        
        if total_installed >= total_groups * 0.8:  # 80% success rate
            log_step_success(logger, f"Optional dependencies installed successfully ({total_installed}/{total_groups})")
            return True
        else:
            log_step_warning(logger, f"Some optional dependencies failed to install ({total_installed}/{total_groups})")
            return True  # Continue with partial installation
            
    except Exception as e:
        log_step_error(logger, f"Failed to install optional dependencies: {e}")
        return False

def validate_uv_environment(project_root: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Validate the UV environment setup with comprehensive checks."""
    try:
        log_step_start(logger, "Validating UV environment")
        
        validation_results = {
            "uv_available": False,
            "pyproject_exists": False,
            "lock_file_exists": False,
            "venv_exists": False,
            "python_version": None,
            "key_packages": {},
            "package_versions": {},
            "system_info": {}
        }
        
        # Check UV availability
        validation_results["uv_available"] = check_uv_availability(logger)
        
        # Check pyproject.toml
        pyproject_path = project_root / "pyproject.toml"
        validation_results["pyproject_exists"] = pyproject_path.exists()
        logger.info(f"pyproject.toml exists: {validation_results['pyproject_exists']}")
        
        # Check uv.lock
        lock_path = project_root / "uv.lock"
        validation_results["lock_file_exists"] = lock_path.exists()
        logger.info(f"uv.lock exists: {validation_results['lock_file_exists']}")
        
        # Check .venv
        venv_path = project_root / ".venv"
        validation_results["venv_exists"] = venv_path.exists()
        logger.info(f"Virtual environment exists: {validation_results['venv_exists']}")
        
        if venv_path.exists():
            # Check Python version in venv
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
                
                # Check key packages with versions
                key_packages = ["numpy", "matplotlib", "networkx", "pandas", "yaml", "scipy", "sklearn"]
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
        
        else:
            log_step_error(logger, f"Virtual environment not found at {venv_path}")
        
        log_step_success(logger, "UV environment validation completed")
        return validation_results
        
    except Exception as e:
        log_step_error(logger, f"UV environment validation failed: {e}")
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
    """Main setup function with UV-based environment management."""
    args = EnhancedArgumentParser.parse_step_arguments("1_setup")
    
    # Setup logging first
    logger = setup_step_logging("setup", args)
    
    try:
        # Log system information
        system_info = log_system_info(logger)
        
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("1_setup.py", Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {output_dir}")
        
        # Get project root (parent of src directory)
        project_root = Path(__file__).parent.parent
        logger.info(f"🚀 Project root: {project_root.absolute()}")
        
        # Add UV-specific arguments
        if hasattr(args, 'install_optional') and args.install_optional:
            package_groups = [group.strip() for group in args.install_optional.split(',')]
            logger.info(f"Installing optional dependency groups: {package_groups}")
            install_optional_dependencies(project_root, logger, package_groups)
        
        # Check UV availability
        if not check_uv_availability(logger):
            return 1
        
        # Setup UV environment
        if not setup_uv_environment(project_root, logger):
                return 1
        
        # Install optional dependencies if requested
        if hasattr(args, 'install_optional') and args.install_optional:
            package_groups = [group.strip() for group in args.install_optional.split(',')]
            install_optional_dependencies(project_root, logger, package_groups)
        
        # Validate environment
        validation_results = validate_uv_environment(project_root, logger)
        if not validation_results:
            return 1
        
        # Create project structure
        if not create_project_structure(output_dir, logger):
            return 1
        
        # Save comprehensive setup results
        setup_results = {
            "project_root": str(project_root.absolute()),
            "validation_results": validation_results,
            "output_dir": str(output_dir),
            "system_info": system_info,
            "setup_timestamp": str(Path(__file__).stat().st_mtime),
            "pyproject_toml": str(project_root / "pyproject.toml"),
            "uv_lock": str(project_root / "uv.lock"),
            "optional_packages_installed": getattr(args, 'install_optional', None)
        }
        
        results_file = output_dir / "setup_results.json"
        with open(results_file, 'w') as f:
            json.dump(setup_results, f, indent=2)
        
        logger.info(f"Setup results saved to: {results_file}")
        
        # Enhanced setup summary
        logger.info("=" * 60)
        logger.info("🎯 UV SETUP SUMMARY")
        logger.info("=" * 60)
        logger.info(f"🔧 Project Root: {project_root.absolute()}")
        logger.info(f"🐍 Python Version: {validation_results.get('python_version', 'Unknown')}")
        logger.info(f"📦 Core Packages Available: {sum(validation_results.get('key_packages', {}).values())}/{len(validation_results.get('key_packages', {}))}")
        logger.info(f"📁 Output Directory: {output_dir}")
        logger.info(f"✅ UV Environment Status: {'READY' if validation_results.get('venv_exists') else 'MISSING'}")
        logger.info(f"🔒 Lock File: {'EXISTS' if validation_results.get('lock_file_exists') else 'MISSING'}")
        
        # Optional packages status
        if hasattr(args, 'install_optional') and args.install_optional:
            logger.info(f"🔸 Optional Packages: {args.install_optional}")
        else:
            logger.info("🔸 Optional Packages: Not installed (use --install-optional to install)")
        
        # System resources
        if 'system_info' in validation_results and 'memory_total_gb' in validation_results['system_info']:
            mem_info = validation_results['system_info']
            logger.info(f"💾 System Memory: {mem_info.get('memory_available_gb', 0):.1f}GB available of {mem_info.get('memory_total_gb', 0):.1f}GB total")
            logger.info(f"🖥️ CPU Cores: {mem_info.get('cpu_count', 0)}")
        
        logger.info("=" * 60)
        logger.info("💡 Next steps:")
        logger.info("  uv run python src/main.py --help")
        logger.info("  uv run pytest src/tests/")
        logger.info("  uv add <package-name>  # Add new dependencies")
        
        log_step_success(logger, "UV setup completed successfully")
        return 0
        
    except Exception as e:
        log_step_error(logger, f"UV setup failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 