"""
UV Environment Validator for GNN Pipeline

This module provides comprehensive validation of UV environment setup
and health checks for the GNN pipeline.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

def validate_uv_installation() -> Dict[str, Any]:
    """
    Validate UV installation and version.
    
    Returns:
        Dictionary with UV installation validation results
    """
    validation = {
        "uv_installed": False,
        "uv_version": None,
        "uv_working": False,
        "error_message": None
    }
    
    try:
        # Check if UV is installed
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            validation["uv_installed"] = True
            validation["uv_version"] = result.stdout.strip()
            validation["uv_working"] = True
        else:
            validation["error_message"] = f"UV command failed: {result.stderr}"
            
    except FileNotFoundError:
        validation["error_message"] = "UV not found in PATH"
    except subprocess.TimeoutExpired:
        validation["error_message"] = "UV command timed out"
    except Exception as e:
        validation["error_message"] = f"Unexpected error: {e}"
    
    return validation

def validate_uv_environment(project_root: Path) -> Dict[str, Any]:
    """
    Validate UV environment setup.
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        Dictionary with environment validation results
    """
    validation = {
        "environment_exists": False,
        "python_executable": None,
        "python_version": None,
        "pip_available": False,
        "packages_installed": 0,
        "core_packages": {},
        "error_message": None
    }
    
    try:
        # Check if .venv directory exists
        venv_path = project_root / ".venv"
        validation["environment_exists"] = venv_path.exists()
        
        if not validation["environment_exists"]:
            validation["error_message"] = "UV environment not found"
            return validation
        
        # Check Python executable
        if sys.platform == "win32":
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        
        if python_exe.exists():
            validation["python_executable"] = str(python_exe)
            
            # Check Python version
            try:
                result = subprocess.run(
                    [str(python_exe), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    validation["python_version"] = result.stdout.strip()
            except Exception as e:
                validation["error_message"] = f"Failed to get Python version: {e}"
                return validation
        else:
            validation["error_message"] = "Python executable not found in UV environment"
            return validation
        
        # Check pip availability
        try:
            result = subprocess.run(
                [str(python_exe), "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            validation["pip_available"] = result.returncode == 0
        except Exception:
            validation["pip_available"] = False
        
        # Check installed packages
        try:
            result = subprocess.run(
                ["uv", "pip", "list", "--format=json"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                validation["packages_installed"] = len(packages)
                
                # Check core packages
                core_packages = ["numpy", "matplotlib", "pytest", "pandas", "scipy"]
                for pkg in core_packages:
                    validation["core_packages"][pkg] = any(
                        p["name"] == pkg for p in packages
                    )
            else:
                validation["error_message"] = f"Failed to list packages: {result.stderr}"
                
        except Exception as e:
            validation["error_message"] = f"Failed to check packages: {e}"
    
    except Exception as e:
        validation["error_message"] = f"Environment validation failed: {e}"
    
    return validation

def validate_uv_dependencies(project_root: Path) -> Dict[str, Any]:
    """
    Validate UV dependencies and lock file.
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        Dictionary with dependency validation results
    """
    validation = {
        "pyproject_toml_exists": False,
        "uv_lock_exists": False,
        "dependencies_synced": False,
        "lock_file_valid": False,
        "missing_dependencies": [],
        "error_message": None
    }
    
    try:
        # Check pyproject.toml
        pyproject_path = project_root / "pyproject.toml"
        validation["pyproject_toml_exists"] = pyproject_path.exists()
        
        if not validation["pyproject_toml_exists"]:
            validation["error_message"] = "pyproject.toml not found"
            return validation
        
        # Check uv.lock
        lock_path = project_root / "uv.lock"
        validation["uv_lock_exists"] = lock_path.exists()
        
        if validation["uv_lock_exists"]:
            # Validate lock file
            try:
                result = subprocess.run(
                    ["uv", "lock", "--check"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                validation["lock_file_valid"] = result.returncode == 0
            except Exception as e:
                validation["error_message"] = f"Lock file validation failed: {e}"
        
        # Check if dependencies are synced
        try:
            result = subprocess.run(
                ["uv", "sync", "--dry-run"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            validation["dependencies_synced"] = result.returncode == 0
            
            if result.returncode != 0:
                validation["error_message"] = f"Dependencies not synced: {result.stderr}"
                
        except Exception as e:
            validation["error_message"] = f"Failed to check dependency sync: {e}"
    
    except Exception as e:
        validation["error_message"] = f"Dependency validation failed: {e}"
    
    return validation

def validate_uv_scripts(project_root: Path) -> Dict[str, Any]:
    """
    Validate UV scripts defined in pyproject.toml.
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        Dictionary with script validation results
    """
    validation = {
        "scripts_available": False,
        "test_script_works": False,
        "setup_script_works": False,
        "pipeline_script_works": False,
        "error_message": None
    }
    
    try:
        # Check if UV scripts are available
        result = subprocess.run(
            ["uv", "run", "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        validation["scripts_available"] = result.returncode == 0
        
        if not validation["scripts_available"]:
            validation["error_message"] = "UV run command not available"
            return validation
        
        # Test specific scripts
        scripts_to_test = [
            ("test", "uv run test --help"),
            ("setup", "uv run setup --help"),
            ("pipeline", "uv run pipeline --help")
        ]
        
        for script_name, command in scripts_to_test:
            try:
                result = subprocess.run(
                    command.split(),
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if script_name == "test":
                    validation["test_script_works"] = result.returncode == 0
                elif script_name == "setup":
                    validation["setup_script_works"] = result.returncode == 0
                elif script_name == "pipeline":
                    validation["pipeline_script_works"] = result.returncode == 0
                    
            except Exception as e:
                validation["error_message"] = f"Failed to test {script_name} script: {e}"
    
    except Exception as e:
        validation["error_message"] = f"Script validation failed: {e}"
    
    return validation

def comprehensive_uv_validation(project_root: Path) -> Dict[str, Any]:
    """
    Perform comprehensive UV validation.
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        Dictionary with comprehensive validation results
    """
    logger.info("Starting comprehensive UV validation...")
    
    validation = {
        "timestamp": time.time(),
        "project_root": str(project_root),
        "uv_installation": validate_uv_installation(),
        "environment": validate_uv_environment(project_root),
        "dependencies": validate_uv_dependencies(project_root),
        "scripts": validate_uv_scripts(project_root),
        "overall_status": "UNKNOWN",
        "recommendations": []
    }
    
    # Determine overall status
    all_checks = [
        validation["uv_installation"]["uv_working"],
        validation["environment"]["environment_exists"],
        validation["environment"]["python_executable"] is not None,
        validation["dependencies"]["pyproject_toml_exists"],
        validation["scripts"]["scripts_available"]
    ]
    
    if all(all_checks):
        validation["overall_status"] = "HEALTHY"
    elif any(all_checks):
        validation["overall_status"] = "PARTIAL"
    else:
        validation["overall_status"] = "UNHEALTHY"
    
    # Generate recommendations
    recommendations = []
    
    if not validation["uv_installation"]["uv_working"]:
        recommendations.append("Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh")
    
    if not validation["environment"]["environment_exists"]:
        recommendations.append("Create UV environment: uv venv")
    
    if not validation["dependencies"]["dependencies_synced"]:
        recommendations.append("Sync dependencies: uv sync")
    
    if not validation["scripts"]["test_script_works"]:
        recommendations.append("Install test dependencies: uv sync --extra dev")
    
    validation["recommendations"] = recommendations
    
    logger.info(f"UV validation completed: {validation['overall_status']}")
    
    return validation

def generate_uv_health_report(validation: Dict[str, Any]) -> str:
    """
    Generate a human-readable UV health report.
    
    Args:
        validation: Validation results dictionary
        
    Returns:
        Formatted health report string
    """
    report = []
    report.append("=" * 60)
    report.append("UV ENVIRONMENT HEALTH REPORT")
    report.append("=" * 60)
    report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(validation['timestamp']))}")
    report.append(f"Project Root: {validation['project_root']}")
    report.append(f"Overall Status: {validation['overall_status']}")
    report.append("")
    
    # UV Installation
    report.append("UV Installation:")
    uv_install = validation["uv_installation"]
    report.append(f"  Installed: {'✅' if uv_install['uv_installed'] else '❌'}")
    if uv_install["uv_version"]:
        report.append(f"  Version: {uv_install['uv_version']}")
    if uv_install["error_message"]:
        report.append(f"  Error: {uv_install['error_message']}")
    report.append("")
    
    # Environment
    report.append("UV Environment:")
    env = validation["environment"]
    report.append(f"  Exists: {'✅' if env['environment_exists'] else '❌'}")
    report.append(f"  Python: {'✅' if env['python_executable'] else '❌'}")
    if env["python_version"]:
        report.append(f"  Version: {env['python_version']}")
    report.append(f"  Pip Available: {'✅' if env['pip_available'] else '❌'}")
    report.append(f"  Packages Installed: {env['packages_installed']}")
    
    # Core packages
    report.append("  Core Packages:")
    for pkg, installed in env["core_packages"].items():
        report.append(f"    {pkg}: {'✅' if installed else '❌'}")
    
    if env["error_message"]:
        report.append(f"  Error: {env['error_message']}")
    report.append("")
    
    # Dependencies
    report.append("Dependencies:")
    deps = validation["dependencies"]
    report.append(f"  pyproject.toml: {'✅' if deps['pyproject_toml_exists'] else '❌'}")
    report.append(f"  uv.lock: {'✅' if deps['uv_lock_exists'] else '❌'}")
    report.append(f"  Synced: {'✅' if deps['dependencies_synced'] else '❌'}")
    report.append(f"  Lock Valid: {'✅' if deps['lock_file_valid'] else '❌'}")
    
    if deps["error_message"]:
        report.append(f"  Error: {deps['error_message']}")
    report.append("")
    
    # Scripts
    report.append("UV Scripts:")
    scripts = validation["scripts"]
    report.append(f"  Available: {'✅' if scripts['scripts_available'] else '❌'}")
    report.append(f"  Test Script: {'✅' if scripts['test_script_works'] else '❌'}")
    report.append(f"  Setup Script: {'✅' if scripts['setup_script_works'] else '❌'}")
    report.append(f"  Pipeline Script: {'✅' if scripts['pipeline_script_works'] else '❌'}")
    
    if scripts["error_message"]:
        report.append(f"  Error: {scripts['error_message']}")
    report.append("")
    
    # Recommendations
    if validation["recommendations"]:
        report.append("Recommendations:")
        for i, rec in enumerate(validation["recommendations"], 1):
            report.append(f"  {i}. {rec}")
        report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)

def save_uv_validation_report(validation: Dict[str, Any], output_dir: Path) -> Path:
    """
    Save UV validation report to file.
    
    Args:
        validation: Validation results dictionary
        output_dir: Output directory for the report
        
    Returns:
        Path to saved report file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    json_path = output_dir / "uv_validation_report.json"
    with open(json_path, 'w') as f:
        json.dump(validation, f, indent=2)
    
    # Save human-readable report
    txt_path = output_dir / "uv_validation_report.txt"
    with open(txt_path, 'w') as f:
        f.write(generate_uv_health_report(validation))
    
    logger.info(f"UV validation report saved to: {json_path}")
    logger.info(f"UV health report saved to: {txt_path}")
    
    return json_path
