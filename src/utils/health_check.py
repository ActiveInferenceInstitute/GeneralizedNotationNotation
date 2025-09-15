#!/usr/bin/env python3
"""
GNN Pipeline Health Check Utilities

Consolidated health checking functionality that integrates with the existing
dependency validation and pipeline monitoring systems.
"""

import sys
import json
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .dependency_validator import DependencyValidator, validate_pipeline_dependencies
from .pipeline_monitor import generate_pipeline_health_report
from .system_utils import get_system_info

logger = logging.getLogger(__name__)

class PipelineHealthChecker:
    """Comprehensive pipeline health checker."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.src_dir = self.project_root / "src"
        self.dependency_validator = DependencyValidator()
        
    def check_core_dependencies(self) -> Dict[str, Any]:
        """Check core pipeline dependencies using the existing validator."""
        try:
            # Use the existing dependency validator
            core_groups = ["core", "essential"]
            all_valid = True
            available = []
            missing = []
            
            for group in core_groups:
                if self.dependency_validator.validate_dependency_group(group):
                    # Get available dependencies from the validator
                    for dep in self.dependency_validator.dependencies.get(group, []):
                        try:
                            mod = importlib.import_module(dep.module_name or dep.name)
                            version = getattr(mod, "__version__", "unknown")
                            available.append(f"{dep.name} {version}")
                        except ImportError:
                            missing.append(f"{dep.name} {dep.version_min or ''}")
                            all_valid = False
                else:
                    all_valid = False
            
            return {
                "available": available,
                "missing": missing,
                "status": "healthy" if all_valid else "unhealthy"
            }
        except Exception as e:
            logger.error(f"Error checking core dependencies: {e}")
            return {
                "available": [],
                "missing": ["Error checking dependencies"],
                "status": "unhealthy"
            }
    
    def check_optional_dependencies(self) -> Dict[str, Any]:
        """Check optional feature dependencies."""
        optional_groups = {
            "gui": ["gradio"],
            "audio": ["librosa", "soundfile", "pedalboard"],
            "ml-ai": ["torch", "transformers"],
            "llm": ["openai", "anthropic"],
            "simulation": ["pymdp"],
            "visualization": ["plotly", "seaborn", "bokeh"],
        }
        
        results = {}
        
        for group, deps in optional_groups.items():
            group_result = {"available": [], "missing": [], "status": "available"}
            
            for dep in deps:
                try:
                    mod = importlib.import_module(dep)
                    version = getattr(mod, "__version__", "unknown")
                    group_result["available"].append(f"{dep} {version}")
                except ImportError:
                    group_result["missing"].append(dep)
            
            if group_result["missing"]:
                if len(group_result["missing"]) == len(deps):
                    group_result["status"] = "unavailable"
                else:
                    group_result["status"] = "partial"
            
            results[group] = group_result
        
        return results
    
    def check_pipeline_structure(self) -> Dict[str, Any]:
        """Check pipeline directory structure and scripts."""
        expected_steps = [
            "0_template.py", "1_setup.py", "2_tests.py", "3_gnn.py",
            "4_model_registry.py", "5_type_checker.py", "6_validation.py",
            "7_export.py", "8_visualization.py", "9_advanced_viz.py",
            "10_ontology.py", "11_render.py", "12_execute.py", "13_llm.py",
            "14_ml_integration.py", "15_audio.py", "16_analysis.py",
            "17_integration.py", "18_security.py", "19_research.py",
            "20_website.py", "21_mcp.py", "22_gui.py", "23_report.py"
        ]
        
        results = {"available": [], "missing": [], "status": "complete"}
        
        for step in expected_steps:
            step_path = self.src_dir / step
            if step_path.exists():
                results["available"].append(step)
            else:
                results["missing"].append(step)
                results["status"] = "incomplete"
        
        return results
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check system-level health metrics."""
        try:
            system_info = get_system_info()
            return {
                "status": "healthy",
                "system_info": system_info,
                "python_version": sys.version,
                "platform": sys.platform
            }
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run complete pipeline health check."""
        logger.info("🔍 Running GNN Pipeline Health Check...")
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "core_dependencies": self.check_core_dependencies(),
            "optional_dependencies": self.check_optional_dependencies(),
            "pipeline_structure": self.check_pipeline_structure(),
            "system_health": self.check_system_health(),
            "recommendations": [],
        }
        
        # Determine overall status
        if health_report["core_dependencies"]["status"] == "unhealthy":
            health_report["overall_status"] = "unhealthy"
            health_report["recommendations"].append(
                "Install missing core dependencies with: uv pip install -e ."
            )
        
        if health_report["pipeline_structure"]["status"] == "incomplete":
            health_report["overall_status"] = "degraded"
            health_report["recommendations"].append(
                "Some pipeline scripts are missing - check repository integrity"
            )
        
        if health_report["system_health"]["status"] == "unhealthy":
            health_report["overall_status"] = "degraded"
            health_report["recommendations"].append(
                "System health issues detected - check system resources"
            )
        
        # Add specific recommendations for optional features
        opt_deps = health_report["optional_dependencies"]
        if opt_deps.get("gui", {}).get("status") == "unavailable":
            health_report["recommendations"].append(
                "Install GUI support with: uv pip install -e .[gui]"
            )
        
        if opt_deps.get("simulation", {}).get("status") == "unavailable":
            health_report["recommendations"].append(
                "Install simulation support with: uv pip install pymdp"
            )
        
        return health_report
    
    def print_health_report(self, report: Dict[str, Any]):
        """Print formatted health report."""
        status_icons = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}
        
        print(f"\n{status_icons.get(report['overall_status'], '❓')} Overall Status: {report['overall_status'].upper()}")
        print(f"📅 Checked at: {report['timestamp']}")
        
        # Core dependencies
        core = report["core_dependencies"]
        print(f"\n📦 Core Dependencies: {status_icons.get(core['status'], '❓')}")
        if core["available"]:
            for dep in core["available"]:
                print(f"  ✅ {dep}")
        if core["missing"]:
            for dep in core["missing"]:
                print(f"  ❌ {dep}")
        
        # Optional dependencies
        print(f"\n🔧 Optional Features:")
        for group, data in report["optional_dependencies"].items():
            status_icon = {"available": "✅", "partial": "⚠️", "unavailable": "❌"}.get(
                data["status"], "❓"
            )
            print(f"  {status_icon} {group}: {data['status']}")
        
        # Pipeline structure
        structure = report["pipeline_structure"]
        print(f"\n📁 Pipeline Structure: {status_icons.get(structure['status'], '❓')}")
        print(f"  Available steps: {len(structure['available'])}/24")
        if structure["missing"]:
            print(f"  Missing: {', '.join(structure['missing'])}")
        
        # System health
        system = report["system_health"]
        print(f"\n💻 System Health: {status_icons.get(system['status'], '❓')}")
        if system.get("system_info"):
            print(f"  Platform: {system['system_info'].get('platform', 'unknown')}")
            print(f"  Python: {system.get('python_version', 'unknown')}")
        
        # Recommendations
        if report["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"  • {rec}")
        
        print()
    
    def save_health_report(self, report: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """Save health report to file."""
        if output_path is None:
            output_path = self.project_root / "pipeline_health_report.json"
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        return output_path

# Convenience functions for backward compatibility
def run_health_check() -> Dict[str, Any]:
    """Run complete pipeline health check (convenience function)."""
    checker = PipelineHealthChecker()
    return checker.run_comprehensive_health_check()

def print_health_report(report: Dict[str, Any]):
    """Print formatted health report (convenience function)."""
    checker = PipelineHealthChecker()
    checker.print_health_report(report)

def main():
    """Main health check entry point."""
    try:
        checker = PipelineHealthChecker()
        report = checker.run_comprehensive_health_check()
        checker.print_health_report(report)
        
        # Save detailed report
        report_file = checker.save_health_report(report)
        print(f"📊 Detailed report saved to: {report_file}")
        
        # Exit with appropriate code
        if report["overall_status"] == "healthy":
            return 0
        elif report["overall_status"] == "degraded":
            return 1
        else:
            return 2
    
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 3

if __name__ == "__main__":
    sys.exit(main())
