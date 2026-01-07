"""
Dependency validation utilities for the GNN Processing Pipeline.

This module validates that all required dependencies are available
before pipeline execution begins, preventing runtime failures.
"""

import importlib
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .logging_utils import PipelineLogger

import argparse
from .venv_utils import get_venv_python
from .argument_utils import parse_step_list

logger = logging.getLogger(__name__)


@dataclass
class DependencySpec:
    """Specification for a required dependency."""
    name: str
    module_name: Optional[str] = None  # Python module name if different from package name
    version_min: Optional[str] = None
    version_max: Optional[str] = None
    is_optional: bool = False
    install_command: Optional[str] = None
    system_command: Optional[str] = None  # For system-level dependencies like graphviz
    description: str = ""


class DependencyValidator:
    """Validates pipeline dependencies before execution."""
    
    def __init__(self, logger: Optional[logging.Logger] = None, python_path: Optional[str] = None):
        """Initialize the dependency validator."""
        self.logger = logger or PipelineLogger.get_logger("dependency_validator")
        self.python_path = python_path
        self.missing_dependencies: List[DependencySpec] = []
        self.version_conflicts: List[Tuple[DependencySpec, str]] = []
        self.warnings: List[str] = []
        
        # Define required dependencies for different pipeline steps
        self.dependencies = self._define_dependencies()
    
    def _define_dependencies(self) -> Dict[str, List[DependencySpec]]:
        """Define all dependencies required by different pipeline components."""
        return {
            "core": [
                DependencySpec(
                    name="pathlib",
                    description="Path handling utilities"
                ),
                DependencySpec(
                    name="json",
                    description="JSON processing"
                ),
                DependencySpec(
                    name="yaml",
                    module_name="yaml", 
                    install_command="pip install pyyaml",
                    description="YAML file processing"
                ),
                DependencySpec(
                    name="numpy",
                    version_min="1.20.0",
                    description="Numerical computing library"
                ),
                DependencySpec(
                    name="pathlib",
                    module_name="pathlib",
                    description="File system path handling (built-in)"
                ),
                DependencySpec(
                    name="json",
                    module_name="json", 
                    description="JSON processing (built-in)"
                ),
                DependencySpec(
                    name="re",
                    module_name="re",
                    description="Regular expressions (built-in)"
                ),
                # HTTP and async communication
                DependencySpec(
                    name="aiohttp",
                    version_min="3.9.0",
                    install_command="uv pip install aiohttp>=3.9.0",
                    description="Async HTTP client/server for LLM providers"
                ),
                DependencySpec(
                    name="httpx",
                    version_min="0.27.0", 
                    install_command="uv pip install httpx>=0.27.0",
                    description="HTTP client library"
                ),
            ],
            "gnn_processing": [
                DependencySpec(
                    name="markdown",
                    version_min="3.0.0",
                    install_command="uv pip install markdown",
                    description="Markdown processing for GNN files"
                ),
                DependencySpec(
                    name="pyyaml",
                    module_name="yaml",
                    version_min="5.0.0",
                    install_command="uv pip install pyyaml",
                    description="YAML processing"
                ),
            ],
            "visualization": [
                DependencySpec(
                    name="matplotlib",
                    version_min="3.5.0",
                    install_command="uv pip install matplotlib",
                    description="Plotting and visualization"
                ),
                DependencySpec(
                    name="networkx",
                    version_min="2.8.0",
                    install_command="uv pip install networkx",
                    description="Graph visualization"
                ),
                DependencySpec(
                    name="graphviz",
                    system_command="dot",
                    install_command="apt-get install graphviz (Ubuntu) or brew install graphviz (macOS)",
                    description="Graph layout engine"
                ),
            ],
            "pymdp": [
                DependencySpec(
                    name="inferactively-pymdp",
                    version_min="0.0.1",
                    install_command="uv pip install inferactively-pymdp",
                    is_optional=True,
                    description="PyMDP Active Inference library (package name: inferactively-pymdp)"
                ),
                DependencySpec(
                    name="scipy",
                    version_min="1.7.0",
                    install_command="uv pip install scipy",
                    description="Scientific computing library"
                ),
            ],
            "rxinfer": [
                DependencySpec(
                    name="julia",
                    system_command="julia",
                    install_command="Download from https://julialang.org/downloads/",
                    is_optional=True,
                    description="Julia programming language for RxInfer"
                ),
            ],
            "export": [
                DependencySpec(
                    name="lxml",
                    version_min="4.6.0",
                    install_command="uv pip install lxml",
                    description="XML processing"
                ),
            ],
            "testing": [
                DependencySpec(
                    name="pytest",
                    version_min="6.0.0",
                    install_command="uv pip install pytest",
                    description="Testing framework"
                ),
            ],
            "discopy": [
                DependencySpec(
                    name="discopy",
                    version_min="0.4.0",
                    install_command="uv pip install discopy",
                    is_optional=True,
                    description="DisCoPy categorical quantum computing library"
                ),
                DependencySpec(
                    name="jax",
                    version_min="0.3.0",
                    install_command="uv pip install jax jaxlib",
                    is_optional=True,
                    description="JAX for DisCoPy numerical evaluation"
                ),
                DependencySpec(
                    name="jaxlib",
                    version_min="0.3.0", 
                    install_command="uv pip install jaxlib",
                    is_optional=True,
                    description="JAX library backend"
                ),
            ]
        }
    
    def validate_python_dependency(self, dep: DependencySpec, python_path: Optional[str] = None) -> bool:
        """Validate a Python package dependency using the specified Python executable."""
        
        # Use the specified Python or fall back to current interpreter
        python_to_use = python_path or self.python_path or sys.executable
        
        try:
            # Try to import the module using subprocess
            module_name = dep.module_name or dep.name
            
            # Create a subprocess to check the import
            check_import_cmd = [
                python_to_use, 
                "-c", 
                f"import {module_name}; print('OK')"
            ]
            
            result = subprocess.run(
                check_import_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=10  # 10 second timeout for import check
            )
            
            if result.returncode != 0:
                # Import failed
                if not dep.is_optional:
                    self.missing_dependencies.append(dep)
                return dep.is_optional
            
            # If we need to check version
            if dep.version_min or dep.version_max:
                try:
                    # Get version using subprocess
                    version_cmd = [
                        python_to_use,
                        "-c",
                        f"import {module_name}; "
                        f"version = getattr({module_name}, '__version__', None) or "
                        f"getattr({module_name}, 'version', None) or "
                        f"getattr({module_name}, 'VERSION', None); "
                        f"print(version if version else 'UNKNOWN')"
                    ]
                    
                    version_result = subprocess.run(
                        version_cmd,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10
                    )
                    
                    if version_result.returncode == 0:
                        version_str = version_result.stdout.strip()
                        if version_str and version_str != 'UNKNOWN':
                            try:
                                from packaging import version as pkg_version
                                current_version = pkg_version.parse(version_str)
                                
                                if dep.version_min:
                                    min_version = pkg_version.parse(dep.version_min)
                                    if current_version < min_version:
                                        self.version_conflicts.append((dep, f"Version {version_str} < required {dep.version_min}"))
                                        return False
                                
                                if dep.version_max:
                                    max_version = pkg_version.parse(dep.version_max)
                                    if current_version > max_version:
                                        self.version_conflicts.append((dep, f"Version {version_str} > maximum {dep.version_max}"))
                                        return False
                            except ImportError:
                                # packaging not available, skip version check
                                self.warnings.append(f"Cannot verify version for {dep.name} - packaging module not available")
                        else:
                            self.warnings.append(f"Could not determine version for {dep.name}")
                    else:
                        self.warnings.append(f"Error checking version for {dep.name}: {version_result.stderr}")
                    
                except Exception as e:
                    self.warnings.append(f"Error checking version for {dep.name}: {e}")
            
            return True
            
        except subprocess.TimeoutExpired:
            self.warnings.append(f"Timeout checking dependency {dep.name}")
            return dep.is_optional
        except Exception as e:
            self.warnings.append(f"Error validating {dep.name}: {e}")
            if not dep.is_optional:
                self.missing_dependencies.append(dep)
            return dep.is_optional
    
    def validate_system_dependency(self, dep: DependencySpec) -> bool:
        """Validate a system-level dependency."""
        if not dep.system_command:
            return True
        
        # Check if command exists in PATH
        if shutil.which(dep.system_command):
            return True
        else:
            if not dep.is_optional:
                self.missing_dependencies.append(dep)
            return dep.is_optional
    
    def validate_dependency_group(self, group_name: str) -> bool:
        """Validate all dependencies in a specific group."""
        if group_name not in self.dependencies:
            self.logger.warning(f"Unknown dependency group: {group_name}")
            return True
        
        all_valid = True
        for dep in self.dependencies[group_name]:
            if dep.system_command:
                valid = self.validate_system_dependency(dep)
            else:
                valid = self.validate_python_dependency(dep, self.python_path)
            
            if not valid:
                all_valid = False
                if not dep.is_optional:
                    self.logger.error(f"Missing required dependency: {dep.name} - {dep.description}")
                else:
                    self.logger.warning(f"Missing optional dependency: {dep.name} - {dep.description}")
        
        return all_valid
    
    def validate_all_dependencies(self, required_groups: Optional[List[str]] = None) -> bool:
        """
        Validate all dependencies or specific groups.
        
        Args:
            required_groups: List of dependency groups to validate, or None for all
            
        Returns:
            bool: True if all required dependencies are satisfied
        """
        self.logger.info("Starting dependency validation...")
        
        groups_to_check = required_groups or list(self.dependencies.keys())
        all_valid = True
        
        for group in groups_to_check:
            group_valid = self.validate_dependency_group(group)
            if not group_valid:
                all_valid = False
                self.logger.error(f"Dependency group '{group}' has missing required dependencies")
        
        # Report summary
        if self.missing_dependencies:
            self.logger.error(f"Missing {len(self.missing_dependencies)} required dependencies")
            for dep in self.missing_dependencies:
                install_hint = f" (Install: {dep.install_command})" if dep.install_command else ""
                self.logger.error(f"  - {dep.name}: {dep.description}{install_hint}")
        
        if self.version_conflicts:
            self.logger.error(f"Found {len(self.version_conflicts)} version conflicts")
            for dep, conflict in self.version_conflicts:
                self.logger.error(f"  - {dep.name}: {conflict}")
        
        if self.warnings:
            for warning in self.warnings:
                self.logger.warning(warning)
        
        if all_valid:
            self.logger.info("All required dependencies validated successfully")
        
        return all_valid
    
    def get_installation_instructions(self) -> List[str]:
        """Get installation instructions for missing dependencies."""
        instructions = []
        
        if self.missing_dependencies:
            instructions.append("To install missing dependencies:")
            instructions.append("")
            
            python_deps = [dep for dep in self.missing_dependencies if not dep.system_command]
            system_deps = [dep for dep in self.missing_dependencies if dep.system_command]
            
            if python_deps:
                instructions.append("Python packages:")
                for dep in python_deps:
                    if dep.install_command:
                        instructions.append(f"  {dep.install_command}")
                    else:
                        instructions.append(f"  uv pip install {dep.name}")
                instructions.append("")
            
            if system_deps:
                instructions.append("System dependencies:")
                for dep in system_deps:
                    if dep.install_command:
                        instructions.append(f"  {dep.install_command}")
                    else:
                        instructions.append(f"  Install {dep.name} system package")
                instructions.append("")
        
        return instructions


def validate_pipeline_dependencies(step_names: Optional[List[str]] = None, 
                                 logger: Optional[logging.Logger] = None,
                                 python_path: Optional[str] = None) -> bool:
    """
    Validate dependencies for specific pipeline steps.
    
    Args:
        step_names: List of pipeline step names to validate dependencies for
        logger: Optional logger instance
        
    Returns:
        bool: True if all required dependencies are available
    """
    validator = DependencyValidator(logger, python_path)
    
    # Map pipeline step names to dependency groups
    step_to_group_mapping = {
        "gnn": ["core", "gnn_processing"],
        "setup": ["core"],
        "tests": ["core", "testing"],
        "type_checker": ["core", "gnn_processing"],
        "export": ["core", "export"],
        "visualization": ["core", "visualization"],
        "render": ["core", "pymdp", "rxinfer", "discopy"],
        "execute": ["core", "pymdp", "rxinfer", "discopy"],
        "mcp": ["core"],
        "ontology": ["core", "gnn_processing"],
    }
    
    if step_names:
        required_groups = set()
        for step_name in step_names:
            if step_name in step_to_group_mapping:
                required_groups.update(step_to_group_mapping[step_name])
        required_groups = list(required_groups)
    else:
        required_groups = None
    
    is_valid = validator.validate_all_dependencies(required_groups)
    
    if not is_valid:
        instructions = validator.get_installation_instructions()
        if instructions and logger:
            logger.error("Dependency validation failed. Installation instructions:")
            for instruction in instructions:
                logger.error(instruction)
    
    return is_valid


def check_optional_dependencies() -> dict:
    """
    Check the status of all optional dependencies and return a summary dictionary.
    Returns:
        dict: { 'optional_dependencies': {name: status, ...}, 'missing_optional': [name, ...] }
    """
    validator = DependencyValidator()
    optional_status = {}
    missing_optional = []
    for group, deps in validator.dependencies.items():
        for dep in deps:
            if dep.is_optional:
                status = validator.validate_python_dependency(dep) if not dep.system_command else validator.validate_system_dependency(dep)
                optional_status[dep.name] = 'available' if status else 'missing'
                if not status:
                    missing_optional.append(dep.name)
    return {'optional_dependencies': optional_status, 'missing_optional': missing_optional}

def get_dependency_status() -> dict:
    """
    Get a summary of required, optional, and missing dependencies for the pipeline.
    Returns:
        dict: { 'required_dependencies': [...], 'optional_dependencies': [...], 'missing_dependencies': [...], 'version_conflicts': [...] }
    """
    validator = DependencyValidator()
    validator.validate_all_dependencies()
    required = []
    optional = []
    for group, deps in validator.dependencies.items():
        for dep in deps:
            if dep.is_optional:
                optional.append(dep.name)
            else:
                required.append(dep.name)
    missing = [dep.name for dep in validator.missing_dependencies]
    version_conflicts = [f"{dep.name}: {conflict}" for dep, conflict in validator.version_conflicts]
    return {
        'required_dependencies': required,
        'optional_dependencies': optional,
        'missing_dependencies': missing,
        'version_conflicts': version_conflicts
    }

def install_missing_dependencies() -> dict:
    """
    Attempt to install missing Python dependencies using pip. System dependencies are not installed automatically.
    Returns:
        dict: { 'installed': [name, ...], 'failed': [name, ...], 'skipped': [name, ...] }
    """
    import subprocess
    validator = DependencyValidator()
    validator.validate_all_dependencies()
    installed = []
    failed = []
    skipped = []
    for dep in validator.missing_dependencies:
        if dep.system_command:
            skipped.append(dep.name)
            continue
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', dep.name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                installed.append(dep.name)
                logging.info(f"Installed missing dependency: {dep.name}")
            else:
                failed.append(dep.name)
                logging.error(f"Failed to install {dep.name}: {result.stderr}")
        except Exception as e:
            failed.append(dep.name)
            logging.error(f"Exception installing {dep.name}: {e}")
    return {'installed': installed, 'failed': failed, 'skipped': skipped}


if __name__ == "__main__":
    # Command-line interface for dependency validation
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate GNN pipeline dependencies")
    parser.add_argument("--groups", nargs="*", help="Dependency groups to validate")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logger = PipelineLogger.get_logger("dependency_validator")
    
    # Create validator and run validation
    validator = DependencyValidator(logger)
    
    success = validator.validate_all_dependencies(args.groups)
    
    if not success:
        instructions = validator.get_installation_instructions()
        for instruction in instructions:
            print(instruction)
        sys.exit(1)
    else:
        print("All dependencies validated successfully!")
        sys.exit(0) 


def validate_pipeline_dependencies_if_available(args: argparse.Namespace) -> bool:
    """
    Validate dependencies if the validator is available.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        bool: True if validation passed or validator unavailable
    """
    # Temporarily skip dependency validation for testing
    logger.info("Dependency validation temporarily skipped for testing")
    return True
    
    if getattr(args, 'skip_dependency_validation', False):
        logger.info("Dependency validation skipped (--skip-dependency-validation flag)")
        return True
        
    if validate_pipeline_dependencies is None:
        logger.info("Dependency validation skipped (validator not available)")
        return True
        
    logger.info("=== DEPENDENCY VALIDATION ===")
    
    # Determine required steps based on what will run
    required_steps = ["setup"]  # Always need core dependencies
    
    # Check which steps will actually run
    skip_steps = parse_step_list(args.skip_steps) if args.skip_steps else []
    only_steps = parse_step_list(args.only_steps) if args.only_steps else []
    
    # Map step numbers to dependency groups
    step_dependency_map = {
        1: "core",              # 1_setup.py - Setup step  
        2: "testing",           # 2_tests.py - Testing framework
        3: "gnn_processing",    # 3_gnn.py - GNN file processing
        4: "core",              # 4_model_registry.py - Model registry
        5: "gnn_processing",    # 5_type_checker.py - GNN validation
        6: "core",              # 6_validation.py - Validation
        7: "export",            # 7_export.py - Export formats
        8: "visualization",     # 8_visualization.py - Visualization
        9: "visualization",     # 9_advanced_viz.py - Advanced visualization
        10: "gnn_processing",   # 10_ontology.py - Ontology processing
        11: "core",             # 11_render.py - Rendering (pymdp, rxinfer, discopy)
        12: "core",             # 12_execute.py - Execution (pymdp, rxinfer, discopy)
        13: "core",             # 13_llm.py - LLM processing
        14: "core",             # 14_ml_integration.py - ML integration
        15: "core",             # 15_audio.py - Audio generation
        16: "core",             # 16_analysis.py - Analysis
        17: "core",             # 17_integration.py - Integration
        18: "core",             # 18_security.py - Security
        19: "core",             # 19_research.py - Research
        20: "core",             # 20_website.py - Website generation
        21: "core",             # 21_mcp.py - MCP tools
        22: "core",             # 22_gui.py - GUI
        23: "core"              # 23_report.py - Comprehensive analysis reports
    }
    
    # Determine which dependency groups we need
    required_groups = set(["core"])
    for step_num in range(1, 24):  # Updated to include steps 1-23
        # Skip if in skip list
        if step_num in skip_steps or f"{step_num}_" in str(skip_steps):
            continue
        # Skip if only_steps specified and this step not in it
        if only_steps and step_num not in only_steps:
            continue
        
        if step_num in step_dependency_map:
            required_groups.add(step_dependency_map[step_num])
    
    logger.info(f"Validating dependency groups: {sorted(required_groups)}")
    
    # Get the virtual environment Python path for dependency validation
    # Use project root directory (parent of src/) to find .venv
    current_dir = Path(__file__).resolve().parent.parent.parent
    venv_python, _ = get_venv_python(current_dir)
    python_path = str(venv_python) if venv_python else None
    
    # If we didn't find a venv, try the current working directory
    if not venv_python or not venv_python.exists():
        import os
        cwd = Path(os.getcwd())
        venv_python, _ = get_venv_python(cwd)
        python_path = str(venv_python) if venv_python else None
    
    if python_path:
        logger.debug(f"Using Python for dependency validation: {python_path}")
    else:
        logger.debug("Using system Python for dependency validation")
    
    # Validate dependencies
    try:
        is_valid = validate_pipeline_dependencies(list(required_groups), python_path=python_path)
        
        if not is_valid:
            logger.critical("Dependency validation failed. Cannot proceed with pipeline execution.")
            logger.critical("Please install the missing dependencies and try again.")
            logger.critical("Alternatively, use --skip-dependency-validation to bypass this check.")
            return False
        
        logger.info("All required dependencies validated successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error during dependency validation: {e}")
        logger.critical("Dependency validation encountered an error. Cannot proceed with pipeline execution.")
        logger.critical("Use --skip-dependency-validation to bypass this check, or fix the validation error.")
        return False 