#!/usr/bin/env python3
"""
Pipeline Dependency Manager for GNN Pipeline

This module provides comprehensive dependency management and graceful degradation
for pipeline steps when dependencies are missing or incompatible.
"""

import logging
import importlib
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class DependencyResult:
    """Result of a dependency check."""
    available: bool
    version: Optional[str] = None
    error: Optional[str] = None
    fallback_available: bool = False
    install_hint: Optional[str] = None


@dataclass
class StepDependencyInfo:
    """Information about step dependencies."""
    required: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    fallbacks: Dict[str, str] = field(default_factory=dict)
    validators: Dict[str, Callable] = field(default_factory=dict)



class PipelineDependencyManager:
    """
    Pipeline dependency manager with graceful degradation support.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dependency_cache = {}
        self.step_configs = self._initialize_step_configs()

# ... (omitting middle lines for brevity in thought process, but tool needs exact match. 
# Actually, replace_file_content needs exact match. I should use multi_replace for this file since changes are scattered.)

        
    def _initialize_step_configs(self) -> Dict[str, StepDependencyInfo]:
        """Initialize dependency configurations for each pipeline step."""
        return {
            "3_gnn": StepDependencyInfo(
                required=["pathlib", "json", "logging"],
                optional=["yaml", "toml"],
            ),
            "5_type_checker": StepDependencyInfo(
                required=["numpy"],
                optional=["sympy", "scipy"],
            ),
            "8_visualization": StepDependencyInfo(
                required=["matplotlib"],
                optional=["seaborn", "networkx", "plotly"],
                fallbacks={"seaborn": "matplotlib", "networkx": "basic_graph"},
                validators={"matplotlib": self._validate_matplotlib}
            ),
            "11_render": StepDependencyInfo(
                required=["pathlib", "json"],
                optional=["jinja2"],
            ),
            "12_execute": StepDependencyInfo(
                required=["subprocess", "pathlib"],
                optional=["pymdp", "jax", "julia", "discopy"],
                fallbacks={
                    "pymdp": "fallback_simulation", 
                    "jax": "numpy_fallback",
                    "julia": "skip_julia",
                    "discopy": "basic_category"
                }
            ),
            "15_audio": StepDependencyInfo(
                required=["pathlib"],
                optional=["librosa", "soundfile", "pedalboard", "pydub"],
                fallbacks={
                    "librosa": "basic_audio",
                    "soundfile": "wave_fallback",
                    "pedalboard": "basic_effects",
                    "pydub": "simple_audio"
                }
            ),
        }
    
    def check_dependency(self, module_name: str, use_cache: bool = True) -> DependencyResult:
        """
        Check if a dependency is available.
        
        Args:
            module_name: Name of the module to check
            use_cache: Whether to use cached results
            
        Returns:
            DependencyResult with availability information
        """
        if use_cache and module_name in self.dependency_cache:
            return self.dependency_cache[module_name]
        
        result = DependencyResult(available=False)
        
        try:
            module = importlib.import_module(module_name)
            result.available = True
            
            # Try to get version information
            version_attrs = ['__version__', 'version', 'VERSION']
            for attr in version_attrs:
                if hasattr(module, attr):
                    result.version = str(getattr(module, attr))
                    break
                    
            # Set install hint
            result.install_hint = f"uv pip install {module_name}"
            
        except ImportError as e:
            result.error = str(e)
            result.install_hint = f"uv pip install {module_name}"
        except Exception as e:
            result.error = f"Unexpected error importing {module_name}: {e}"
            
        if use_cache:
            self.dependency_cache[module_name] = result
            
        return result
    
    def check_step_dependencies(self, step_name: str) -> Dict[str, Any]:
        """
        Check all dependencies for a specific pipeline step.
        
        Args:
            step_name: Name of the pipeline step
            
        Returns:
            Dictionary with dependency check results
        """
        if step_name not in self.step_configs:
            return {
                "step": step_name,
                "status": "unknown",
                "error": f"No dependency configuration for step {step_name}"
            }
        
        config = self.step_configs[step_name]
        results = {
            "step": step_name,
            "status": "healthy",
            "required": {},
            "optional": {},
            "fallbacks_available": {},
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check required dependencies
        critical_missing = []
        for dep in config.required:
            dep_result = self.check_dependency(dep)
            results["required"][dep] = dep_result
            
            if not dep_result.available:
                critical_missing.append(dep)
                results["errors"].append(f"Required dependency missing: {dep}")
        
        # Check optional dependencies
        optional_missing = []
        for dep in config.optional:
            dep_result = self.check_dependency(dep)
            results["optional"][dep] = dep_result
            
            if not dep_result.available:
                optional_missing.append(dep)
                results["warnings"].append(f"Optional dependency missing: {dep}")
                
                # Check if fallback is available
                if dep in config.fallbacks:
                    fallback = config.fallbacks[dep]
                    results["fallbacks_available"][dep] = fallback
                    results["recommendations"].append(
                        f"Using {fallback} as fallback for {dep}"
                    )
        
        # Run custom validators
        for dep, validator in config.validators.items():
            if dep in results["required"] or dep in results["optional"]:
                try:
                    validator_result = validator(dep)
                    if not validator_result:
                        results["warnings"].append(f"Validator failed for {dep}")
                except Exception as e:
                    results["warnings"].append(f"Validator error for {dep}: {e}")
        
        # Determine overall status
        if critical_missing:
            results["status"] = "failed" 
            results["recommendations"].append(
                f"Install critical dependencies: {', '.join(critical_missing)}"
            )
        elif optional_missing:
            results["status"] = "degraded"
            results["recommendations"].append(
                f"Consider installing optional dependencies for full functionality: {', '.join(optional_missing)}"
            )
        
        return results
    
    def _validate_matplotlib(self, module_name: str) -> bool:
        """Validate matplotlib can create figures without errors."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            # Test basic figure creation
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.plot([1, 2, 3], [1, 2, 3])
            plt.close(fig)
            return True
        except Exception:
            return False
    
    @contextmanager
    def graceful_import(self, module_name: str, step_name: Optional[str] = None):
        """
        Context manager for graceful imports with fallback handling.
        
        Args:
            module_name: Module to import
            step_name: Pipeline step name for fallback lookup
            
        Yields:
            Either the imported module or None if unavailable
        """
        try:
            module = importlib.import_module(module_name)
            yield module
        except ImportError:
            self.logger.warning(f"{module_name} not available - graceful degradation enabled")
            
            # Check for fallback
            if step_name and step_name in self.step_configs:
                config = self.step_configs[step_name]
                if module_name in config.fallbacks:
                    fallback_name = config.fallbacks[module_name]
                    self.logger.info(f"Using fallback: {fallback_name}")
            
            yield None
    
    def generate_dependency_report(self, steps: List[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive dependency report for pipeline steps.
        
        Args:
            steps: List of step names to check, or None for all
            
        Returns:
            Comprehensive dependency report
        """
        if steps is None:
            steps = list(self.step_configs.keys())
        
        report = {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "python_version": sys.version,
            "steps": {},
            "summary": {
                "total_steps": len(steps),
                "healthy_steps": 0,
                "degraded_steps": 0,
                "failed_steps": 0,
                "total_dependencies": 0,
                "available_dependencies": 0,
                "missing_dependencies": 0
            }
        }
        
        for step in steps:
            step_result = self.check_step_dependencies(step)
            report["steps"][step] = step_result
            
            # Update summary
            if step_result["status"] == "healthy":
                report["summary"]["healthy_steps"] += 1
            elif step_result["status"] == "degraded":
                report["summary"]["degraded_steps"] += 1
            elif step_result["status"] == "failed":
                report["summary"]["failed_steps"] += 1
            
            # Count dependencies
            all_deps = {**step_result["required"], **step_result["optional"]}
            report["summary"]["total_dependencies"] += len(all_deps)
            report["summary"]["available_dependencies"] += sum(
                1 for dep in all_deps.values() if dep.available
            )
            report["summary"]["missing_dependencies"] += sum(
                1 for dep in all_deps.values() if not dep.available
            )
        
        return report
    
    def install_missing_dependencies(self, step_name: str, 
                                   required_only: bool = False) -> Dict[str, bool]:
        """
        Attempt to install missing dependencies for a step.
        
        Args:
            step_name: Pipeline step name
            required_only: Only install required dependencies
            
        Returns:
            Dictionary mapping dependency names to installation success
        """
        if step_name not in self.step_configs:
            return {}
        
        config = self.step_configs[step_name]
        installation_results = {}
        
        # Get dependencies to install
        deps_to_install = config.required.copy()
        if not required_only:
            deps_to_install.extend(config.optional)
        
        for dep in deps_to_install:
            dep_result = self.check_dependency(dep, use_cache=False)
            if not dep_result.available:
                try:
                    # Try to install using uv
                    cmd = ["uv", "pip", "install", dep]
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=300
                    )
                    
                    if result.returncode == 0:
                        self.logger.info(f"Successfully installed {dep}")
                        installation_results[dep] = True
                        # Clear cache to force recheck
                        self.dependency_cache.pop(dep, None)
                    else:
                        self.logger.error(f"Failed to install {dep}: {result.stderr}")
                        installation_results[dep] = False
                        
                except subprocess.TimeoutExpired:
                    self.logger.error(f"Installation of {dep} timed out")
                    installation_results[dep] = False
                except Exception as e:
                    self.logger.error(f"Error installing {dep}: {e}")
                    installation_results[dep] = False
            else:
                installation_results[dep] = True  # Already available
        
        return installation_results


def get_pipeline_dependency_manager() -> PipelineDependencyManager:
    """Get singleton instance of pipeline dependency manager."""
    if not hasattr(get_pipeline_dependency_manager, '_instance'):
        get_pipeline_dependency_manager._instance = PipelineDependencyManager()
    return get_pipeline_dependency_manager._instance
