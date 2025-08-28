#!/usr/bin/env python3
"""
Enhanced Dependency Manager for GNN Pipeline

This module provides comprehensive dependency checking and management
across all pipeline steps with intelligent fallback strategies.
"""

import sys
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import importlib.util

@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    version: Optional[str] = None
    required: bool = True
    min_version: Optional[str] = None
    install_command: Optional[str] = None
    fallback_available: bool = False
    description: Optional[str] = None

@dataclass
class DependencyGroup:
    """Group of related dependencies for a specific functionality."""
    name: str
    description: str
    dependencies: List[DependencyInfo] = field(default_factory=list)
    optional: bool = False
    
    def add_dependency(self, dep: DependencyInfo):
        """Add a dependency to this group."""
        self.dependencies.append(dep)

class DependencyManager:
    """Comprehensive dependency management for the GNN pipeline."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dependency_groups = {}
        self._setup_dependency_groups()
    
    def _setup_dependency_groups(self):
        """Setup all dependency groups for the pipeline."""
        
        # Core Dependencies (always required)
        core = DependencyGroup("core", "Core Python dependencies required for basic operation")
        core.add_dependency(DependencyInfo("numpy", min_version="1.19.0", install_command="uv add numpy", 
                                          description="Numerical computing"))
        core.add_dependency(DependencyInfo("pandas", min_version="1.1.0", install_command="uv add pandas",
                                          description="Data manipulation"))
        core.add_dependency(DependencyInfo("pyyaml", min_version="5.4.0", install_command="uv add pyyaml",
                                          description="YAML configuration files"))
        core.add_dependency(DependencyInfo("pathlib", required=False, description="Path handling (built-in)"))
        self.dependency_groups["core"] = core
        
        # Visualization Dependencies
        viz = DependencyGroup("visualization", "Visualization and plotting dependencies", optional=True)
        viz.add_dependency(DependencyInfo("matplotlib", min_version="3.3.0", install_command="uv add matplotlib",
                                         fallback_available=True, description="Basic plotting"))
        viz.add_dependency(DependencyInfo("seaborn", min_version="0.11.0", install_command="uv add seaborn",
                                         fallback_available=True, description="Statistical visualizations"))
        viz.add_dependency(DependencyInfo("plotly", min_version="5.0.0", install_command="uv add plotly",
                                         fallback_available=True, description="Interactive plots"))
        viz.add_dependency(DependencyInfo("networkx", min_version="2.5", install_command="uv add networkx",
                                         fallback_available=True, description="Graph visualization"))
        self.dependency_groups["visualization"] = viz
        
        # Simulation Dependencies
        sim = DependencyGroup("simulation", "Simulation engine dependencies", optional=True)
        sim.add_dependency(DependencyInfo("pymdp", min_version="0.0.8", install_command="uv add pymdp",
                                         fallback_available=True, description="Active Inference simulations"))
        sim.add_dependency(DependencyInfo("jax", min_version="0.3.0", install_command="uv add jax",
                                         fallback_available=True, description="JAX-based computations"))
        sim.add_dependency(DependencyInfo("flax", min_version="0.5.0", install_command="uv add flax",
                                         fallback_available=True, description="Neural network library"))
        self.dependency_groups["simulation"] = sim
        
        # Audio Dependencies
        audio = DependencyGroup("audio", "Audio processing and generation dependencies", optional=True)
        audio.add_dependency(DependencyInfo("librosa", min_version="0.8.0", install_command="uv add librosa",
                                           fallback_available=True, description="Audio analysis"))
        audio.add_dependency(DependencyInfo("soundfile", min_version="0.10.0", install_command="uv add soundfile",
                                           fallback_available=True, description="Audio I/O"))
        audio.add_dependency(DependencyInfo("pedalboard", min_version="0.5.0", install_command="uv add pedalboard",
                                           fallback_available=True, description="Audio effects"))
        self.dependency_groups["audio"] = audio
        
        # LLM Dependencies
        llm = DependencyGroup("llm", "Large Language Model integration dependencies", optional=True)
        llm.add_dependency(DependencyInfo("openai", min_version="1.0.0", install_command="uv add openai",
                                         fallback_available=True, description="OpenAI API client"))
        llm.add_dependency(DependencyInfo("anthropic", min_version="0.3.0", install_command="uv add anthropic",
                                         fallback_available=True, description="Anthropic API client"))
        llm.add_dependency(DependencyInfo("tiktoken", install_command="uv add tiktoken",
                                         fallback_available=True, description="Token counting"))
        self.dependency_groups["llm"] = llm
        
        # Machine Learning Dependencies
        ml = DependencyGroup("machine_learning", "Machine learning and statistics dependencies", optional=True)
        ml.add_dependency(DependencyInfo("scikit-learn", min_version="0.24.0", install_command="uv add scikit-learn",
                                        fallback_available=True, description="Machine learning algorithms"))
        ml.add_dependency(DependencyInfo("scipy", min_version="1.5.0", install_command="uv add scipy",
                                        fallback_available=True, description="Scientific computing"))
        ml.add_dependency(DependencyInfo("torch", min_version="1.8.0", install_command="uv add torch",
                                        fallback_available=True, description="Deep learning"))
        self.dependency_groups["machine_learning"] = ml
        
        # Development Dependencies
        dev = DependencyGroup("development", "Development and testing dependencies", optional=True)
        dev.add_dependency(DependencyInfo("pytest", min_version="6.0.0", install_command="uv add --dev pytest",
                                         fallback_available=False, description="Testing framework"))
        dev.add_dependency(DependencyInfo("black", install_command="uv add --dev black",
                                         fallback_available=False, description="Code formatting"))
        dev.add_dependency(DependencyInfo("flake8", install_command="uv add --dev flake8",
                                         fallback_available=False, description="Code linting"))
        self.dependency_groups["development"] = dev
        
        # External System Dependencies
        external = DependencyGroup("external", "External system dependencies", optional=True)
        external.add_dependency(DependencyInfo("julia", required=False, fallback_available=True,
                                              description="Julia runtime for RxInfer.jl"))
        external.add_dependency(DependencyInfo("git", required=False, fallback_available=True,
                                              description="Version control"))
        self.dependency_groups["external"] = external
    
    def check_python_dependency(self, dep: DependencyInfo) -> Tuple[bool, str, Optional[str]]:
        """Check if a Python dependency is available and get its version."""
        try:
            # Special cases for packages with different import names
            import_name = dep.name
            if dep.name == "pyyaml":
                import_name = "yaml"
            elif dep.name == "scikit-learn":
                import_name = "sklearn"
            elif dep.name == "pillow":
                import_name = "PIL"
            
            # Try to import the module
            if import_name == "pathlib":
                # Built-in module in Python 3.4+
                import pathlib
                module = pathlib
            else:
                spec = importlib.util.find_spec(import_name)
                if spec is None:
                    return False, f"Module {import_name} not found", None
                
                module = importlib.import_module(import_name)
            
            # Try to get version
            version = getattr(module, '__version__', None)
            if version is None:
                # Try alternative version attributes
                version = getattr(module, 'version', None)
                if version is None and hasattr(module, 'VERSION'):
                    version = getattr(module, 'VERSION', None)
                if version is None:
                    version = "unknown"
            
            return True, f"Available: {dep.name} {version}", version
            
        except ImportError as e:
            return False, f"Import error: {e}", None
        except Exception as e:
            return False, f"Check error: {e}", None
    
    def check_external_dependency(self, dep: DependencyInfo) -> Tuple[bool, str, Optional[str]]:
        """Check if an external system dependency is available."""
        try:
            if dep.name == "julia":
                try:
                    result = subprocess.run(["julia", "--version"], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        version_line = result.stdout.strip().split('\n')[0]
                        version = version_line.split()[-1] if version_line else "unknown"
                        return True, f"Julia {version} available", version
                    else:
                        return False, "Julia not found in PATH", None
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    return False, "Julia not installed or not in PATH", None
            
            elif dep.name == "git":
                try:
                    result = subprocess.run(["git", "--version"], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        version = result.stdout.strip().split()[-1]
                        return True, f"Git {version} available", version
                    else:
                        return False, "Git not working", None
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    return False, "Git not installed", None
            
            else:
                return False, f"Unknown external dependency: {dep.name}", None
                
        except Exception as e:
            return False, f"External check error: {e}", None
    
    def check_dependency_group(self, group_name: str) -> Dict[str, Any]:
        """Check all dependencies in a group."""
        if group_name not in self.dependency_groups:
            return {"error": f"Unknown dependency group: {group_name}"}
        
        group = self.dependency_groups[group_name]
        results = {
            "group_name": group_name,
            "description": group.description,
            "optional": group.optional,
            "dependencies": [],
            "summary": {
                "total": len(group.dependencies),
                "available": 0,
                "missing": 0,
                "errors": 0
            }
        }
        
        for dep in group.dependencies:
            if group_name == "external":
                available, message, version = self.check_external_dependency(dep)
            else:
                available, message, version = self.check_python_dependency(dep)
            
            dep_result = {
                "name": dep.name,
                "required": dep.required,
                "available": available,
                "message": message,
                "version": version,
                "min_version": dep.min_version,
                "install_command": dep.install_command,
                "fallback_available": dep.fallback_available,
                "description": dep.description
            }
            
            results["dependencies"].append(dep_result)
            
            if available:
                results["summary"]["available"] += 1
            elif not available and "error" in message.lower():
                results["summary"]["errors"] += 1
            else:
                results["summary"]["missing"] += 1
        
        return results
    
    def check_all_dependencies(self) -> Dict[str, Any]:
        """Check all dependency groups."""
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "groups": {},
            "overall_summary": {
                "total_groups": len(self.dependency_groups),
                "healthy_groups": 0,
                "degraded_groups": 0,
                "failed_groups": 0,
                "total_dependencies": 0,
                "available_dependencies": 0,
                "missing_dependencies": 0
            },
            "recommendations": []
        }
        
        for group_name in self.dependency_groups:
            group_results = self.check_dependency_group(group_name)
            all_results["groups"][group_name] = group_results
            
            # Update overall summary
            summary = group_results.get("summary", {})
            all_results["overall_summary"]["total_dependencies"] += summary.get("total", 0)
            all_results["overall_summary"]["available_dependencies"] += summary.get("available", 0)
            all_results["overall_summary"]["missing_dependencies"] += summary.get("missing", 0)
            
            # Categorize group health
            group = self.dependency_groups[group_name]
            if group.optional:
                if summary.get("available", 0) == summary.get("total", 1):
                    all_results["overall_summary"]["healthy_groups"] += 1
                elif summary.get("available", 0) > 0:
                    all_results["overall_summary"]["degraded_groups"] += 1
                else:
                    all_results["overall_summary"]["failed_groups"] += 1
            else:
                # Required group
                required_deps = [dep for dep in group.dependencies if dep.required]
                required_available = sum(1 for dep_result in group_results["dependencies"] 
                                       if dep_result["required"] and dep_result["available"])
                if required_available == len(required_deps):
                    all_results["overall_summary"]["healthy_groups"] += 1
                elif required_available > 0:
                    all_results["overall_summary"]["degraded_groups"] += 1
                else:
                    all_results["overall_summary"]["failed_groups"] += 1
        
        # Generate recommendations
        all_results["recommendations"] = self._generate_recommendations(all_results)
        
        return all_results
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate installation and configuration recommendations."""
        recommendations = []
        
        # Check for missing core dependencies
        core_results = results["groups"].get("core", {})
        missing_core = [dep for dep in core_results.get("dependencies", []) 
                       if dep["required"] and not dep["available"]]
        
        if missing_core:
            install_commands = [dep["install_command"] for dep in missing_core if dep["install_command"]]
            if install_commands:
                recommendations.append(f"Install missing core dependencies: {'; '.join(install_commands)}")
        
        # Check for visualization dependencies
        viz_results = results["groups"].get("visualization", {})
        missing_viz = [dep for dep in viz_results.get("dependencies", []) 
                      if not dep["available"]]
        if missing_viz and len(missing_viz) == len(viz_results.get("dependencies", [])):
            recommendations.append("Consider installing visualization dependencies for chart generation: uv add matplotlib seaborn plotly")
        
        # Check for simulation dependencies
        sim_results = results["groups"].get("simulation", {})
        missing_sim = [dep for dep in sim_results.get("dependencies", []) 
                      if not dep["available"]]
        if missing_sim and len(missing_sim) == len(sim_results.get("dependencies", [])):
            recommendations.append("Consider installing simulation dependencies for PyMDP/JAX execution: uv add pymdp jax flax")
        
        # Check for Julia
        external_results = results["groups"].get("external", {})
        julia_available = any(dep["available"] for dep in external_results.get("dependencies", []) 
                            if dep["name"] == "julia")
        if not julia_available:
            recommendations.append("Consider installing Julia for RxInfer.jl simulations: https://julialang.org/downloads/")
        
        # Check for LLM dependencies
        llm_results = results["groups"].get("llm", {})
        missing_llm = [dep for dep in llm_results.get("dependencies", []) 
                      if not dep["available"]]
        if missing_llm and len(missing_llm) == len(llm_results.get("dependencies", [])):
            recommendations.append("Consider installing LLM dependencies for AI analysis: uv add openai anthropic")
        
        return recommendations
    
    def get_step_dependencies(self, step_name: str) -> List[str]:
        """Get required dependency groups for a specific pipeline step."""
        step_deps = {
            "1_setup": ["core", "development"],
            "2_tests": ["core", "development"], 
            "3_gnn": ["core"],
            "4_model_registry": ["core"],
            "5_type_checker": ["core"],
            "6_validation": ["core"],
            "7_export": ["core"],
            "8_visualization": ["core", "visualization"],
            "9_advanced_viz": ["core", "visualization"],
            "10_ontology": ["core"],
            "11_render": ["core"],
            "12_execute": ["core", "simulation", "external"],
            "13_llm": ["core", "llm"],
            "14_ml_integration": ["core", "machine_learning"],
            "15_audio": ["core", "audio"],
            "16_analysis": ["core", "machine_learning"],
            "17_integration": ["core"],
            "18_security": ["core"],
            "19_research": ["core", "machine_learning"],
            "20_website": ["core", "visualization"],
            "21_mcp": ["core"],
            "22_gui": ["core", "visualization"],
            "23_report": ["core", "visualization"]
        }
        
        return step_deps.get(step_name, ["core"])
    
    def check_step_dependencies(self, step_name: str) -> Dict[str, Any]:
        """Check dependencies for a specific pipeline step."""
        required_groups = self.get_step_dependencies(step_name)
        
        step_results = {
            "step_name": step_name,
            "required_groups": required_groups,
            "group_results": {},
            "overall_status": "healthy",
            "missing_critical": [],
            "warnings": [],
            "recommendations": []
        }
        
        critical_missing = []
        warnings = []
        
        for group_name in required_groups:
            group_results = self.check_dependency_group(group_name)
            step_results["group_results"][group_name] = group_results
            
            group = self.dependency_groups[group_name]
            
            # Check for critical missing dependencies
            for dep_result in group_results.get("dependencies", []):
                if dep_result["required"] and not dep_result["available"]:
                    if not group.optional:
                        critical_missing.append(f"{dep_result['name']}: {dep_result['message']}")
                    else:
                        warnings.append(f"{dep_result['name']}: {dep_result['message']}")
                elif not dep_result["available"] and not dep_result["fallback_available"]:
                    warnings.append(f"Optional {dep_result['name']}: {dep_result['message']}")
        
        step_results["missing_critical"] = critical_missing
        step_results["warnings"] = warnings
        
        # Determine overall status
        if critical_missing:
            step_results["overall_status"] = "failed"
        elif warnings:
            step_results["overall_status"] = "degraded"
        else:
            step_results["overall_status"] = "healthy"
        
        # Generate recommendations
        if critical_missing:
            step_results["recommendations"].append(f"Install critical dependencies for {step_name}")
        if warnings:
            step_results["recommendations"].append(f"Consider installing optional dependencies for full {step_name} functionality")
        
        return step_results

# Global instance
dependency_manager = DependencyManager()

def check_dependencies_for_step(step_name: str) -> Dict[str, Any]:
    """Convenience function to check dependencies for a step."""
    return dependency_manager.check_step_dependencies(step_name)

def get_dependency_status() -> Dict[str, Any]:
    """Convenience function to get overall dependency status."""
    return dependency_manager.check_all_dependencies()

def log_dependency_status(step_name: str, logger: logging.Logger):
    """Log dependency status for a step."""
    results = check_dependencies_for_step(step_name)
    
    logger.info(f"Dependency check for {step_name}: {results['overall_status']}")
    
    if results["missing_critical"]:
        for missing in results["missing_critical"]:
            logger.error(f"❌ Critical: {missing}")
    
    if results["warnings"]:
        for warning in results["warnings"]:
            logger.warning(f"⚠️ Warning: {warning}")
    
    for rec in results["recommendations"]:
        logger.info(f"💡 Recommendation: {rec}")

if __name__ == "__main__":
    # Standalone dependency check
    results = get_dependency_status()
    print(f"Overall dependency status:")
    print(f"Groups: {results['overall_summary']['healthy_groups']} healthy, "
          f"{results['overall_summary']['degraded_groups']} degraded, "
          f"{results['overall_summary']['failed_groups']} failed")
    print(f"Dependencies: {results['overall_summary']['available_dependencies']}/"
          f"{results['overall_summary']['total_dependencies']} available")
    
    for rec in results["recommendations"]:
        print(f"💡 {rec}")
