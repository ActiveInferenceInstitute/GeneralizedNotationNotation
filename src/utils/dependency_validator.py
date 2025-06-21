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
                    install_command="pip install aiohttp>=3.9.0",
                    description="Async HTTP client/server for LLM providers"
                ),
                DependencySpec(
                    name="httpx",
                    version_min="0.27.0", 
                    install_command="pip install httpx>=0.27.0",
                    description="HTTP client library"
                ),
            ],
            "gnn_processing": [
                DependencySpec(
                    name="markdown",
                    version_min="3.0.0",
                    install_command="pip install markdown",
                    description="Markdown processing for GNN files"
                ),
                DependencySpec(
                    name="pyyaml",
                    module_name="yaml",
                    version_min="5.0.0",
                    install_command="pip install pyyaml",
                    description="YAML processing"
                ),
            ],
            "visualization": [
                DependencySpec(
                    name="matplotlib",
                    version_min="3.5.0",
                    install_command="pip install matplotlib",
                    description="Plotting and visualization"
                ),
                DependencySpec(
                    name="networkx",
                    version_min="2.8.0",
                    install_command="pip install networkx",
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
                    name="pymdp",
                    version_min="0.0.1",
                    install_command="pip install pymdp",
                    is_optional=True,
                    description="PyMDP Active Inference library"
                ),
                DependencySpec(
                    name="scipy",
                    version_min="1.7.0",
                    install_command="pip install scipy",
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
                    install_command="pip install lxml",
                    description="XML processing"
                ),
            ],
            "testing": [
                DependencySpec(
                    name="pytest",
                    version_min="6.0.0",
                    install_command="pip install pytest",
                    description="Testing framework"
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
                        instructions.append(f"  pip install {dep.name}")
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
        "gnn_type_checker": ["core", "gnn_processing"],
        "export": ["core", "export"],
        "visualization": ["core", "visualization"],
        "render": ["core", "pymdp", "rxinfer"],
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