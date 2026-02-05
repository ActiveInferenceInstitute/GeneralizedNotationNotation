#!/usr/bin/env python3
"""
Enhanced GNN Pipeline Health Check

Comprehensive system health validation for the GNN Processing Pipeline.
Moved from src/ to src/pipeline/ for better integration with pipeline management.

Features:
- Core dependency validation
- Optional feature dependency checking
- Pipeline structure verification
- Performance and resource monitoring
- Integration with pipeline module utilities
- Enhanced diagnostic capabilities
"""

import sys
import json
import importlib
import subprocess
import platform
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

# Optional psutil import with fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False

# Enhanced imports with fallbacks
try:
    from .config import get_pipeline_config, STEP_METADATA
    from .pipeline_validator import PipelineValidator
    from .diagnostic_enhancer import PipelineDiagnosticEnhancer
    # Use absolute import to avoid "attempted relative import beyond top-level package" warning
    import sys
    from pathlib import Path as _P
    if str(_P(__file__).parent.parent) not in sys.path:
        sys.path.insert(0, str(_P(__file__).parent.parent))
    from utils import setup_step_logging, log_step_success, log_step_warning, log_step_error
    PIPELINE_INTEGRATION = True
except ImportError as e:
    print(f"Warning: Limited pipeline integration: {e}")
    PIPELINE_INTEGRATION = False
    # Fallback logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    def setup_step_logging(name: str, verbose: bool = False) -> logging.Logger:
        return logging.getLogger(name)

    def log_step_success(logger: logging.Logger, message: str):
        logger.info(f"✅ {message}")

    def log_step_warning(logger: logging.Logger, message: str):
        logger.warning(f"⚠️ {message}")

    def log_step_error(logger: logging.Logger, message: str):
        logger.error(f"❌ {message}")


class EnhancedHealthChecker:
    """
    Enhanced health checker with comprehensive system validation.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = setup_step_logging("enhanced_health_check", verbose)
        self.results: Dict[str, Any] = {}
        self.start_time = time.time()

        # Enhanced dependency definitions
        self.core_dependencies = {
            "numpy": ">=1.21.0",
            "matplotlib": ">=3.5.0",
            "networkx": ">=2.6.0",
            "pandas": ">=1.3.0",
            "pytest": ">=6.0.0",
            "scipy": ">=1.7.0",
            "pyyaml": ">=6.0",
            "pathlib": ">=1.0.0",  # Standard library but check availability
        }

        self.optional_feature_groups = {
            "gui": {
                "dependencies": ["gradio"],
                "description": "Interactive GUI for GNN model construction",
                "critical": False
            },
            "audio": {
                "dependencies": ["librosa", "soundfile", "pedalboard"],
                "description": "Audio generation and sonification",
                "critical": False
            },
            "ml-ai": {
                "dependencies": ["torch", "transformers"],
                "description": "Machine learning and AI integration",
                "critical": False
            },
            "llm": {
                "dependencies": ["openai", "anthropic"],
                "description": "Large language model integration",
                "critical": False
            },
            "simulation": {
                "dependencies": ["pymdp"],
                "description": "Active inference simulation frameworks",
                "critical": True  # Critical for core functionality
            },
            "visualization": {
                "dependencies": ["plotly", "seaborn", "bokeh"],
                "description": "Advanced visualization capabilities",
                "critical": False
            },
            "julia": {
                "dependencies": [],  # Special case - check Julia installation
                "description": "Julia runtime for RxInfer.jl support",
                "critical": False
            }
        }

    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability."""
        self.logger.info("📊 Checking system resources...")

        # Check if psutil is available
        if not PSUTIL_AVAILABLE:
            self.logger.warning("⚠️ psutil not available - limited system resource checks")
            return {
                "status": "limited",
                "error": "psutil not installed - install with: pip install psutil",
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "python_version": sys.version
                }
            }

        try:
            # CPU information
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory information
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)

            # Disk information
            disk = psutil.disk_usage('/')
            disk_gb = disk.total / (1024**3)

            # Network check (basic connectivity)
            network_available = True
            try:
                # Simple connectivity test
                subprocess.run(['ping', '-c', '1', '8.8.8.8'],
                            capture_output=True, timeout=5)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                network_available = False

            return {
                "status": "healthy",
                "cpu": {
                    "cores": cpu_count,
                    "usage_percent": cpu_percent,
                    "status": "good" if cpu_percent < 80 else "high"
                },
                "memory": {
                    "total_gb": round(memory_gb, 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "usage_percent": memory.percent,
                    "status": "good" if memory.percent < 80 else "high"
                },
                "disk": {
                    "total_gb": round(disk_gb, 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "usage_percent": disk.percent,
                    "status": "good" if disk.percent < 85 else "high"
                },
                "network": {
                    "available": network_available,
                    "status": "connected" if network_available else "offline"
                },
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "python_version": sys.version
                }
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Could not check system resources: {e}")
            return {
                "status": "unknown",
                "error": str(e),
                "platform": {"python_version": sys.version}
            }

    def check_core_dependencies(self) -> Dict[str, Any]:
        """Enhanced core dependency checking with version validation."""
        self.logger.info("📦 Checking core dependencies...")

        results = {
            "available": [],
            "missing": [],
            "version_issues": [],
            "status": "healthy",
            "total_checked": len(self.core_dependencies)
        }

        for dep, version_req in self.core_dependencies.items():
            try:
                mod = importlib.import_module(dep)
                version = getattr(mod, "__version__", "unknown")

                # Basic version checking (could be enhanced)
                if version_req != ">=1.0.0" and version == "unknown":
                    results["version_issues"].append(f"{dep}: version unknown (required: {version_req})")
                else:
                    results["available"].append(f"{dep} {version}")

            except ImportError:
                results["missing"].append(f"{dep} {version_req}")
                results["status"] = "unhealthy"

        return results

    def check_optional_dependencies(self) -> Dict[str, Any]:
        """Enhanced optional dependency checking."""
        self.logger.info("🔧 Checking optional dependencies...")

        results = {}

        for group_name, group_info in self.optional_feature_groups.items():
            group_result = {
                "available": [],
                "missing": [],
                "status": "available",
                "description": group_info["description"],
                "critical": group_info["critical"]
            }

            for dep in group_info["dependencies"]:
                try:
                    mod = importlib.import_module(dep)
                    version = getattr(mod, "__version__", "unknown")
                    group_result["available"].append(f"{dep} {version}")
                except ImportError:
                    group_result["missing"].append(dep)

            # Special handling for Julia
            if group_name == "julia":
                try:
                    # Check if Julia is installed
                    result = subprocess.run(['julia', '--version'],
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        group_result["available"].append(f"julia {result.stdout.strip()}")
                    else:
                        group_result["missing"].append("julia")
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    group_result["missing"].append("julia")

            # Determine group status
            if group_result["missing"]:
                if len(group_result["missing"]) == len(group_info["dependencies"] or [1]):
                    group_result["status"] = "unavailable"
                else:
                    group_result["status"] = "partial"
            elif group_name == "julia" and not group_result["available"]:
                group_result["status"] = "unavailable"

            results[group_name] = group_result

        return results

    def check_pipeline_structure(self) -> Dict[str, Any]:
        """Enhanced pipeline structure validation."""
        self.logger.info("🏗️ Checking pipeline structure...")

        src_dir = Path(__file__).parent.parent  # src/
        results = {
            "available_scripts": [],
            "missing_scripts": [],
            "available_modules": [],
            "missing_modules": [],
            "status": "complete"
        }

        # Check for all numbered pipeline scripts (0-24)
        expected_scripts = []
        for step_num in range(25):  # 0-24
            # Check for both .py files and module directories
            script_path = src_dir / f"{step_num}_*.py"
            module_path = src_dir / f"{step_num}_*"

            found = False

            # Check for script file
            if list(script_path.parent.glob(f"{step_num}_*.py")):
                script_name = list(script_path.parent.glob(f"{step_num}_*.py"))[0].name
                results["available_scripts"].append(script_name)
                found = True

            # Check for module directory
            if list(module_path.parent.glob(f"{step_num}_*/")):
                module_name = list(module_path.parent.glob(f"{step_num}_*/"))[0].name
                if module_name not in results["available_modules"]:
                    results["available_modules"].append(module_name)
                found = True

            if not found:
                results["missing_scripts"].append(f"{step_num}_*.py")
                results["status"] = "incomplete"

        # Check main pipeline files
        main_files = ["main.py", "__init__.py"]
        for main_file in main_files:
            main_path = src_dir / main_file
            if main_path.exists():
                results["available_scripts"].append(main_file)
            else:
                results["missing_scripts"].append(main_file)
                results["status"] = "incomplete"

        return results

    def check_pipeline_integration(self) -> Dict[str, Any]:
        """Check integration with pipeline module utilities."""
        self.logger.info("🔗 Checking pipeline integration...")

        results = {
            "config_available": False,
            "validator_available": False,
            "diagnostic_available": False,
            "integration_status": "unknown"
        }

        if not PIPELINE_INTEGRATION:
            results["integration_status"] = "limited"
            return results

        try:
            # Test pipeline config
            config = get_pipeline_config()
            results["config_available"] = True

            # Test pipeline validator
            validator = PipelineValidator(verbose=False)
            results["validator_available"] = True

            # Test diagnostic enhancer
            enhancer = PipelineDiagnosticEnhancer()
            results["diagnostic_available"] = True

            results["integration_status"] = "full"

        except Exception as e:
            self.logger.warning(f"⚠️ Pipeline integration issue: {e}")
            results["integration_status"] = "partial"

        return results

    def run_enhanced_health_check(self) -> Dict[str, Any]:
        """Run comprehensive enhanced health check."""
        self.logger.info("🔍 Starting enhanced pipeline health check...")

        # Core system checks
        self.results["system_resources"] = self.check_system_resources()
        self.results["core_dependencies"] = self.check_core_dependencies()
        self.results["optional_dependencies"] = self.check_optional_dependencies()
        self.results["pipeline_structure"] = self.check_pipeline_structure()
        self.results["pipeline_integration"] = self.check_pipeline_integration()

        # Calculate overall health score
        self.results["health_score"] = self._calculate_overall_health()
        self.results["execution_time"] = time.time() - self.start_time

        # Generate recommendations
        self.results["recommendations"] = self._generate_recommendations()

        return self.results

    def _calculate_overall_health(self) -> Dict[str, Any]:
        """Calculate overall system health score."""
        score_components = {
            "system_resources": 20,
            "core_dependencies": 30,
            "pipeline_structure": 25,
            "pipeline_integration": 15,
            "optional_features": 10
        }

        total_score = 0
        total_possible = sum(score_components.values())

        # System resources (20 points)
        sys_resources = self.results.get("system_resources", {})
        if sys_resources.get("status") == "healthy":
            total_score += score_components["system_resources"]

        # Core dependencies (30 points)
        core_deps = self.results.get("core_dependencies", {})
        if core_deps.get("status") == "healthy":
            total_score += score_components["core_dependencies"]
        elif len(core_deps.get("missing", [])) <= 2:  # Some missing but not critical
            total_score += score_components["core_dependencies"] * 0.7

        # Pipeline structure (25 points)
        pipeline_struct = self.results.get("pipeline_structure", {})
        if pipeline_struct.get("status") == "complete":
            total_score += score_components["pipeline_structure"]
        else:
            missing_count = len(pipeline_struct.get("missing_scripts", []))
            if missing_count <= 3:  # Minor missing components
                total_score += score_components["pipeline_structure"] * 0.8

        # Pipeline integration (15 points)
        integration = self.results.get("pipeline_integration", {})
        if integration.get("integration_status") == "full":
            total_score += score_components["pipeline_integration"]
        elif integration.get("integration_status") == "partial":
            total_score += score_components["pipeline_integration"] * 0.5

        # Optional features (10 points) - bonus for having nice-to-have features
        optional_deps = self.results.get("optional_dependencies", {})
        available_features = sum(1 for group in optional_deps.values()
                               if group.get("status") == "available")
        total_score += min(available_features * 2, score_components["optional_features"])

        # Calculate percentage
        health_percentage = (total_score / total_possible) * 100

        # Determine health rating
        if health_percentage >= 90:
            rating = "excellent"
        elif health_percentage >= 75:
            rating = "good"
        elif health_percentage >= 60:
            rating = "fair"
        else:
            rating = "poor"

        return {
            "score": round(health_percentage, 1),
            "rating": rating,
            "raw_score": total_score,
            "max_score": total_possible,
            "components": score_components
        }

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []

        # Core dependency recommendations
        core_deps = self.results.get("core_dependencies", {})
        if core_deps.get("missing"):
            recommendations.append({
                "priority": "high",
                "category": "dependencies",
                "title": "Install Missing Core Dependencies",
                "description": f"Missing: {', '.join(core_deps['missing'])}",
                "action": "Run: uv pip install -e ."
            })

        # Optional feature recommendations
        optional_deps = self.results.get("optional_dependencies", {})
        for group_name, group_info in optional_deps.items():
            if group_info.get("status") == "unavailable" and group_info.get("critical"):
                recommendations.append({
                    "priority": "medium",
                    "category": "features",
                    "title": f"Install {group_name.title()} Support",
                    "description": group_info["description"],
                    "action": f"Install required packages for {group_name} functionality"
                })

        # Pipeline structure recommendations
        pipeline_struct = self.results.get("pipeline_structure", {})
        if pipeline_struct.get("missing_scripts"):
            recommendations.append({
                "priority": "high",
                "category": "pipeline",
                "title": "Complete Pipeline Structure",
                "description": f"Missing scripts: {', '.join(pipeline_struct['missing_scripts'])}",
                "action": "Ensure all numbered pipeline scripts (0-24) exist"
            })

        # System resource recommendations
        sys_resources = self.results.get("system_resources", {})
        if sys_resources.get("memory", {}).get("status") == "high":
            recommendations.append({
                "priority": "low",
                "category": "performance",
                "title": "Monitor Memory Usage",
                "description": "High memory usage detected",
                "action": "Monitor system resources during pipeline execution"
            })

        # Performance recommendations
        if self.results["execution_time"] > 5.0:
            recommendations.append({
                "priority": "low",
                "category": "performance",
                "title": "Optimize Health Check Performance",
                "description": f"Health check took {self.results['execution_time']:.2f}s",
                "action": "Consider caching results or reducing check frequency"
            })

        return recommendations

    def print_enhanced_report(self):
        """Print comprehensive health report."""
        print("\n" + "="*80)
        print("🚀 GNN PIPELINE ENHANCED HEALTH CHECK")
        print("="*80)

        health_score = self.results.get("health_score", {})
        score_icon = {
            "excellent": "🌟",
            "good": "✅",
            "fair": "⚠️",
            "poor": "❌"
        }.get(health_score.get("rating", "unknown"), "❓")

        print(f"\n{score_icon} Overall Health: {health_score.get('rating', 'unknown').upper()} ({health_score.get('score', 0)}/100)")

        # System resources
        sys_resources = self.results.get("system_resources", {})
        if sys_resources.get("status") == "healthy":
            print("\n💻 System Resources:")
            memory = sys_resources.get("memory", {})
            print(f"   CPU: {sys_resources.get('cpu', {}).get('cores', '?')} cores ({sys_resources.get('cpu', {}).get('usage_percent', '?')}% usage)")
            print(f"   Memory: {memory.get('total_gb', '?')}GB total ({memory.get('usage_percent', '?')}% usage)")
            print(f"   Disk: {sys_resources.get('disk', {}).get('free_gb', '?')}GB free")
            print(f"   Network: {sys_resources.get('network', {}).get('status', 'unknown')}")

        # Core dependencies
        core_deps = self.results.get("core_dependencies", {})
        dep_icon = "✅" if core_deps.get("status") == "healthy" else "❌"
        print(f"\n{dep_icon} Core Dependencies: {core_deps.get('status', 'unknown').upper()}")
        if core_deps.get("available"):
            print(f"   Available ({len(core_deps['available'])}/{core_deps['total_checked']}):")
            for dep in core_deps["available"][:5]:  # Show first 5
                print(f"     ✅ {dep}")
            if len(core_deps["available"]) > 5:
                print(f"     ... and {len(core_deps['available']) - 5} more")

        if core_deps.get("missing"):
            print(f"   Missing ({len(core_deps['missing'])}):")
            for dep in core_deps["missing"]:
                print(f"     ❌ {dep}")

        # Optional features
        print("\n🔧 Optional Features:")
        optional_deps = self.results.get("optional_dependencies", {})
        for group_name, group_info in optional_deps.items():
            status_icon = {
                "available": "✅",
                "partial": "⚠️",
                "unavailable": "❌"
            }.get(group_info.get("status"), "❓")
            print(f"   {status_icon} {group_name}: {group_info.get('status', 'unknown')}")

        # Pipeline structure
        pipeline_struct = self.results.get("pipeline_structure", {})
        struct_icon = "✅" if pipeline_struct.get("status") == "complete" else "❌"
        print(f"\n{struct_icon} Pipeline Structure: {pipeline_struct.get('status', 'unknown').upper()}")
        print(f"   Scripts: {len(pipeline_struct.get('available_scripts', []))}/24 available")
        print(f"   Modules: {len(pipeline_struct.get('available_modules', []))}/24 available")

        # Pipeline integration
        integration = self.results.get("pipeline_integration", {})
        int_icon = {"full": "✅", "partial": "⚠️", "limited": "❌"}.get(integration.get("integration_status"), "❓")
        print(f"\n{int_icon} Pipeline Integration: {integration.get('integration_status', 'unknown').upper()}")

        # Recommendations
        recommendations = self.results.get("recommendations", [])
        if recommendations:
            print("\n🎯 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(rec.get("priority"), "⚪")
                print(f"   {i}. {priority_icon} [{rec['category'].upper()}] {rec['title']}")
                print(f"      {rec['description']}")
                print(f"      💡 {rec['action']}")

        print(f"\n⏱️ Health check completed in {self.results['execution_time']:.2f}s")
        print("="*80)


def run_enhanced_health_check(verbose: bool = False) -> Dict[str, Any]:
    """Run the enhanced health check."""
    checker = EnhancedHealthChecker(verbose)
    results = checker.run_enhanced_health_check()

    if verbose:
        checker.print_enhanced_report()

    return results


def main():
    """Main entry point for enhanced health check."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced GNN Pipeline Health Check")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON only")
    parser.add_argument("--output-file", type=Path,
                       help="Save results to JSON file")

    args = parser.parse_args()

    # Run enhanced health check
    results = run_enhanced_health_check(args.verbose)

    # Output format
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    elif not args.verbose:
        # Brief summary for non-verbose mode
        health_score = results.get("health_score", {})
        print(f"Health Score: {health_score.get('score', 0)}/100 ({health_score.get('rating', 'unknown')})")

        core_deps = results.get("core_dependencies", {})
        print(f"Core Dependencies: {len(core_deps.get('available', []))}/{len(core_deps.get('missing', [])) + len(core_deps.get('available', []))} available")

        pipeline_struct = results.get("pipeline_structure", {})
        print(f"Pipeline Scripts: {len(pipeline_struct.get('available_scripts', []))}/24 available")

    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Results saved to: {args.output_file}")

    # Exit code based on health
    health_score = results.get("health_score", {})
    if health_score.get("rating") in ["excellent", "good"]:
        return 0
    elif health_score.get("rating") == "fair":
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
