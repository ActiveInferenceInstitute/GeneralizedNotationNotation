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
from typing import Dict, Any, Optional, List
import time # Added for dependency installation progress tracking

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
    """Install project dependencies with comprehensive logging and progress tracking."""
    try:
        log_step_start(logger, "Installing dependencies")
        
        logger.info(f"Requirements file: {requirements_file}")
        logger.info(f"Requirements file exists: {requirements_file.exists()}")
        
        if not requirements_file.exists():
            log_step_warning(logger, f"Requirements file not found at {requirements_file}")
            return True
        
        # Read and categorize requirements
        try:
            with open(requirements_file, 'r') as f:
                requirements_content = f.read()
            
            # Parse requirements into categories
            requirements_lines = requirements_content.splitlines()
            core_packages = []
            optional_packages = []
            current_section = "core"
            
            logger.info("📋 Analyzing requirements file structure:")
            for line in requirements_lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    if "OPTIONAL" in line.upper() or "HEAVY" in line.upper():
                        current_section = "optional"
                        logger.info(f"  📦 Found optional section: {line}")
                    elif "CORE" in line.upper() or "ESSENTIAL" in line.upper():
                        current_section = "core"
                        logger.info(f"  📦 Found core section: {line}")
                    continue
                
                if current_section == "core":
                    core_packages.append(line)
                elif current_section == "optional":
                    optional_packages.append(line)
            
            logger.info(f"  ✅ Core packages: {len(core_packages)}")
            logger.info(f"  ⚠️ Optional packages: {len(optional_packages)}")
            
            # Log core packages with details
            logger.info("📦 Core packages to install:")
            for package in core_packages:
                logger.info(f"  🔹 {package}")
            
            if optional_packages:
                logger.info("📦 Optional packages (commented out):")
                for package in optional_packages:
                    logger.info(f"  🔸 {package}")
            
        except Exception as e:
            logger.warning(f"Could not parse requirements file: {e}")
            # Fallback to reading all non-comment lines
            core_packages = [line.strip() for line in requirements_lines 
                           if line.strip() and not line.startswith('#')]
        
        # Get pip path
        pip_path = venv_path / "bin" / "pip"
        if sys.platform == "win32":
            pip_path = venv_path / "Scripts" / "pip.exe"
        
        logger.info(f"🔧 Using pip: {pip_path}")
        logger.info(f"🔧 Pip exists: {pip_path.exists()}")
        
        # Upgrade pip first with detailed logging
        logger.info("🔄 Upgrading pip...")
        upgrade_result = subprocess.run(
            [str(pip_path), "install", "--upgrade", "pip"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if upgrade_result.returncode == 0:
            logger.info("✅ Pip upgraded successfully")
        else:
            logger.warning(f"⚠️ Pip upgrade failed: {upgrade_result.stderr}")
        
        # Install core dependencies with enhanced logging
        logger.info("🚀 Installing core dependencies...")
        logger.info(f"⏱️ Timeout set to: 30 minutes")
        logger.info(f"📦 Packages to install: {len(core_packages)}")
        
        # Use optimized pip settings
        install_cmd = [
            str(pip_path), "install", "-r", str(requirements_file),
            "--timeout", "300",  # Per package timeout
            "--retries", "3",    # Retry failed downloads
            "--no-cache-dir",    # Don't use cache to avoid corruption
            "--prefer-binary"    # Prefer binary wheels over source
        ]
        
        logger.info(f"🔧 Install command: {' '.join(install_cmd)}")
        
        # Track installation progress
        start_time = time.time()
        result = subprocess.run(
            install_cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes
        )
        
        installation_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info("✅ Core dependencies installed successfully")
            logger.info(f"⏱️ Installation time: {installation_time:.1f} seconds")
            
            # List installed packages with categorization
            list_result = subprocess.run(
                [str(pip_path), "list"],
                capture_output=True,
                text=True
            )
            
            if list_result.returncode == 0:
                packages = list_result.stdout.strip().split('\n')[2:]  # Skip header
                logger.info(f"📦 Installed packages ({len(packages)}):")
                
                # Categorize installed packages
                core_installed = []
                optional_installed = []
                other_installed = []
                
                for package in packages:
                    if package.strip():
                        pkg_name = package.split()[0].lower()
                        if any(core_pkg.split('>=')[0].split('==')[0].lower() in pkg_name 
                               for core_pkg in core_packages):
                            core_installed.append(package.strip())
                        elif any(opt_pkg.split('>=')[0].split('==')[0].lower() in pkg_name 
                                for opt_pkg in optional_packages):
                            optional_installed.append(package.strip())
                        else:
                            other_installed.append(package.strip())
                
                logger.info(f"  🔹 Core packages: {len(core_installed)}")
                for pkg in core_installed[:5]:  # Show first 5
                    logger.info(f"    ✅ {pkg}")
                if len(core_installed) > 5:
                    logger.info(f"    ... and {len(core_installed) - 5} more core packages")
                
                if optional_installed:
                    logger.info(f"  🔸 Optional packages: {len(optional_installed)}")
                    for pkg in optional_installed[:3]:
                        logger.info(f"    ⚠️ {pkg}")
                
                logger.info(f"  📚 Other packages: {len(other_installed)}")
            
            return True
        else:
            log_step_error(logger, f"Failed to install core dependencies: {result.stderr}")
            logger.error(f"🔧 Pip install stdout: {result.stdout}")
            logger.error(f"⏱️ Installation failed after {installation_time:.1f} seconds")
            
            # Try installing essential packages only
            logger.info("🔄 Attempting to install essential packages only...")
            essential_packages = [
                "numpy>=1.21.0", "matplotlib>=3.5.0", "networkx>=2.6.0", 
                "pandas>=1.3.0", "pyyaml>=6.0", "scipy>=1.7.0", 
                "pytest>=6.0.0", "pytest-cov>=3.0.0", "pytest-xdist>=2.5.0"
            ]
            
            essential_success = 0
            for package in essential_packages:
                try:
                    logger.info(f"📦 Installing essential package: {package}")
                    pkg_result = subprocess.run(
                        [str(pip_path), "install", package],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    if pkg_result.returncode == 0:
                        logger.info(f"  ✅ {package} installed successfully")
                        essential_success += 1
                    else:
                        logger.warning(f"  ❌ {package} failed: {pkg_result.stderr}")
                except Exception as e:
                    logger.error(f"  ❌ {package} error: {e}")
            
            logger.info(f"📊 Essential packages installed: {essential_success}/{len(essential_packages)}")
            if essential_success >= len(essential_packages) * 0.8:  # 80% success rate
                logger.warning("⚠️ Essential packages installed. Some optional dependencies may be missing.")
                return True
            else:
                logger.error("❌ Too many essential packages failed to install")
                return False
            
    except subprocess.TimeoutExpired:
        log_step_error(logger, "Dependency installation timed out after 30 minutes")
        logger.error("💡 Consider installing heavy packages separately:")
        logger.error("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        logger.error("  pip install transformers jax jaxlib")
        logger.error("  pip install plotly seaborn bokeh holoviews")
        
        # Try installing essential packages only
        logger.info("🔄 Attempting to install essential packages only...")
        essential_packages = [
            "numpy>=1.21.0", "matplotlib>=3.5.0", "networkx>=2.6.0", 
            "pandas>=1.3.0", "pyyaml>=6.0", "scipy>=1.7.0", 
            "pytest>=6.0.0", "pytest-cov>=3.0.0", "pytest-xdist>=2.5.0"
        ]
        
        try:
            for package in essential_packages:
                logger.info(f"📦 Installing essential package: {package}")
                result = subprocess.run(
                    [str(pip_path), "install", package],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode != 0:
                    logger.warning(f"Failed to install {package}: {result.stderr}")
            
            logger.warning("⚠️ Essential packages installed. Some optional dependencies may be missing.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to install essential packages: {e}")
            return False
    except Exception as e:
        log_step_error(logger, f"Failed to install dependencies: {e}")
        return False

def install_optional_packages(venv_path: Path, logger: logging.Logger, package_groups: List[str] = None) -> bool:
    """Install optional heavy packages in separate groups."""
    try:
        log_step_start(logger, "Installing optional packages")
        
        # Define package groups with size estimates
        package_groups_definitions = {
            "ml_ai": {
                "name": "Machine Learning & AI",
                "packages": [
                    "transformers>=4.20.0",  # ~2GB
                    "torch>=1.12.0",         # ~1.5GB
                    "jax>=0.4.0",            # ~500MB
                    "jaxlib>=0.4.0"          # ~300MB
                ],
                "estimated_size": "4.3GB"
            },
            "llm": {
                "name": "LLM Integration",
                "packages": [
                    "openai>=1.0.0",         # ~50MB
                    "anthropic>=0.5.0"       # ~30MB
                ],
                "estimated_size": "80MB"
            },
            "visualization": {
                "name": "Advanced Visualization",
                "packages": [
                    "plotly>=5.0.0",         # ~200MB
                    "seaborn>=0.11.0",       # ~100MB
                    "bokeh>=2.4.0",          # ~150MB
                    "holoviews>=1.14.0",     # ~100MB
                    "panel>=0.13.0"          # ~80MB
                ],
                "estimated_size": "630MB"
            },
            "audio": {
                "name": "Audio Processing",
                "packages": [
                    "librosa>=0.9.0",        # ~300MB
                    "soundfile>=0.10.0",     # ~50MB
                    "pedalboard>=0.5.0"      # ~200MB
                ],
                "estimated_size": "550MB"
            },
            "graphs": {
                "name": "Graph Visualization",
                "packages": [
                    "igraph>=0.9.0",         # ~100MB
                    "graphviz>=0.19.0"       # ~50MB
                ],
                "estimated_size": "150MB"
            },
            "research": {
                "name": "Research Tools",
                "packages": [
                    "jupyter>=1.0.0",        # ~500MB
                    "ipywidgets>=7.6.0"      # ~100MB
                ],
                "estimated_size": "600MB"
            },
            "active_inference": {
                "name": "Active Inference",
                "packages": [
                    "pymdp>=0.0.1"           # ~50MB
                ],
                "estimated_size": "50MB"
            }
        }
        
        if package_groups is None:
            # Show available groups
            logger.info("📦 Available optional package groups:")
            for group_id, group_info in package_groups_definitions.items():
                logger.info(f"  🔸 {group_id}: {group_info['name']} (~{group_info['estimated_size']})")
            logger.info("💡 Use --install-optional <group1,group2> to install specific groups")
            return True
        
        # Get pip path
        pip_path = venv_path / "bin" / "pip"
        if sys.platform == "win32":
            pip_path = venv_path / "Scripts" / "pip.exe"
        
        total_installed = 0
        total_packages = 0
        
        for group_id in package_groups:
            if group_id not in package_groups_definitions:
                logger.warning(f"⚠️ Unknown package group: {group_id}")
                continue
            
            group_info = package_groups_definitions[group_id]
            packages = group_info["packages"]
            total_packages += len(packages)
            
            logger.info(f"📦 Installing {group_info['name']} packages (~{group_info['estimated_size']}):")
            for package in packages:
                logger.info(f"  🔹 {package}")
            
            # Install packages in this group
            group_success = 0
            for package in packages:
                try:
                    logger.info(f"📦 Installing {package}...")
                    result = subprocess.run(
                        [str(pip_path), "install", package],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minutes per package
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"  ✅ {package} installed successfully")
                        group_success += 1
                        total_installed += 1
                    else:
                        logger.warning(f"  ❌ {package} failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"  ⏰ {package} installation timed out")
                except Exception as e:
                    logger.error(f"  ❌ {package} error: {e}")
            
            logger.info(f"📊 {group_info['name']}: {group_success}/{len(packages)} packages installed")
        
        logger.info(f"📊 Total optional packages: {total_installed}/{total_packages} installed")
        
        if total_installed >= total_packages * 0.8:  # 80% success rate
            log_step_success(logger, f"Optional packages installed successfully ({total_installed}/{total_packages})")
            return True
        else:
            log_step_warning(logger, f"Some optional packages failed to install ({total_installed}/{total_packages})")
            return True  # Continue with partial installation
            
    except Exception as e:
        log_step_error(logger, f"Failed to install optional packages: {e}")
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
    """Main setup function with enhanced logging and optional package support."""
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
        
        # Install optional packages if requested
        if hasattr(args, 'install_optional') and args.install_optional:
            package_groups = [group.strip() for group in args.install_optional.split(',')]
            install_optional_packages(venv_path, logger, package_groups)
        
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
            "requirements_file": str(requirements_file) if requirements_file.exists() else None,
            "optional_packages_installed": getattr(args, 'install_optional', None)
        }
        
        results_file = output_dir / "setup_results.json"
        with open(results_file, 'w') as f:
            json.dump(setup_results, f, indent=2)
        
        logger.info(f"Setup results saved to: {results_file}")
        
        # Enhanced setup summary
        logger.info("=" * 60)
        logger.info("🎯 SETUP SUMMARY")
        logger.info("=" * 60)
        logger.info(f"🔧 Virtual Environment: {venv_path.absolute()}")
        logger.info(f"🐍 Python Version: {validation_results.get('python_version', 'Unknown')}")
        logger.info(f"📦 Core Packages Available: {sum(validation_results.get('key_packages', {}).values())}/{len(validation_results.get('key_packages', {}))}")
        logger.info(f"📁 Output Directory: {output_dir}")
        logger.info(f"✅ Virtual Environment Status: {'EXISTS' if venv_path.exists() else 'MISSING'}")
        
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
        
        log_step_success(logger, "Setup completed successfully")
        return 0
        
    except Exception as e:
        log_step_error(logger, f"Setup failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 