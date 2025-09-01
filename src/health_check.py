#!/usr/bin/env python3
"""
GNN Pipeline Health Check

Verifies all dependencies and pipeline components are properly installed and configured.
"""

import sys
import json
import importlib
from pathlib import Path
from typing import Dict, List, Any


def check_core_dependencies() -> Dict[str, Any]:
    """Check core pipeline dependencies."""
    core_deps = {
        "numpy": ">=1.21.0",
        "matplotlib": ">=3.5.0",
        "networkx": ">=2.6.0",
        "pandas": ">=1.3.0",
        "pytest": ">=6.0.0",
    }

    results = {"available": [], "missing": [], "status": "healthy"}

    for dep, version_req in core_deps.items():
        try:
            mod = importlib.import_module(dep)
            version = getattr(mod, "__version__", "unknown")
            results["available"].append(f"{dep} {version}")
        except ImportError:
            results["missing"].append(f"{dep} {version_req}")
            results["status"] = "unhealthy"

    return results


def check_optional_dependencies() -> Dict[str, Any]:
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


def check_pipeline_structure() -> Dict[str, Any]:
    """Check pipeline directory structure and scripts."""
    src_dir = Path(__file__).parent

    expected_steps = [
        "0_template.py",
        "1_setup.py",
        "2_tests.py",
        "3_gnn.py",
        "4_model_registry.py",
        "5_type_checker.py",
        "6_validation.py",
        "7_export.py",
        "8_visualization.py",
        "9_advanced_viz.py",
        "10_ontology.py",
        "11_render.py",
        "12_execute.py",
        "13_llm.py",
        "14_ml_integration.py",
        "15_audio.py",
        "16_analysis.py",
        "17_integration.py",
        "18_security.py",
        "19_research.py",
        "20_website.py",
        "21_mcp.py",
        "22_gui.py",
        "23_report.py",
    ]

    results = {"available": [], "missing": [], "status": "complete"}

    for step in expected_steps:
        step_path = src_dir / step
        if step_path.exists():
            results["available"].append(step)
        else:
            results["missing"].append(step)
            results["status"] = "incomplete"

    return results


def run_health_check() -> Dict[str, Any]:
    """Run complete pipeline health check."""
    print("🔍 Running GNN Pipeline Health Check...")

    health_report = {
        "overall_status": "healthy",
        "core_dependencies": check_core_dependencies(),
        "optional_dependencies": check_optional_dependencies(),
        "pipeline_structure": check_pipeline_structure(),
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


def print_health_report(report: Dict[str, Any]):
    """Print formatted health report."""
    status_icons = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}

    print(
        f"\n{status_icons.get(report['overall_status'], '❓')} Overall Status: {report['overall_status'].upper()}"
    )

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

    # Recommendations
    if report["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")

    print()


def main():
    """Main health check entry point."""
    try:
        report = run_health_check()
        print_health_report(report)

        # Save detailed report
        report_file = Path("pipeline_health_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
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
