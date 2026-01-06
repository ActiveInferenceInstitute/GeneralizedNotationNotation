#!/usr/bin/env python3
"""
GNN Pipeline Validation Script

This script validates that all pipeline steps are:
1. Using consistent logging patterns
2. Producing expected outputs  
3. Handling arguments correctly
4. Following the established patterns
5. Using centralized configuration properly

Usage:
    python pipeline_validation.py [--fix-issues]
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional
import ast
import re

# Import centralized utilities
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_warning, 
    log_step_error,
    UTILS_AVAILABLE
)

logger = setup_step_logging("pipeline_validation", verbose=True)

# Expected output patterns for each step
EXPECTED_OUTPUTS = {
    "0_template": [
        "template_processing_summary.json"
    ],
    "1_setup": [
        "setup_artifacts/installed_packages.json"
    ],
    "2_tests": [
        "test_reports/pytest_report.xml"
    ],
    "3_gnn": [
        "gnn_processing_results.json"
    ],
    "4_model_registry": [
        "model_registry_results.json"
    ],
    "5_type_checker": [
        "type_check_results.json"
    ],
    "6_validation": [
        "validation_results.json"
    ],
    "7_export": [
        "export_results.json"
    ],
    "8_visualization": [
        "visualization/"
    ],
    "9_advanced_viz": [
        "advanced_viz_summary.json"
    ],
    "10_ontology": [
        "ontology_results/"
    ],
    "11_render": [
        "gnn_rendered_simulators/"  # PyMDP, RxInfer.jl, ActiveInference.jl code
    ],
    "12_execute": [
        "execution_results/"  # PyMDP, RxInfer.jl, ActiveInference.jl results
    ],
    "13_llm": [
        "llm_results/"
    ],
    "14_ml_integration": [
        "ml_integration_summary.json"
    ],
    "15_audio": [
        "audio_processing_summary.json"
    ],
    "16_analysis": [
        "analysis_results.json"
    ],
    "17_integration": [
        "integration_summary.json"
    ],
    "18_security": [
        "security_processing_summary.json"
    ],
    "19_research": [
        "research_processing_summary.json"
    ],
    "20_website": [
        "website/"
    ],
    "21_mcp": [
        "mcp_results/"
    ],
    "22_gui": [
        "gui_processing_summary.json"
    ],
    "23_report": [
        "report_summary.json"
    ]
}

def validate_module_imports(module_path: Path) -> Dict[str, List[str]]:
    """Validate that modules properly import and use centralized utilities."""
    issues = {"errors": [], "warnings": [], "suggestions": []}
    
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for centralized utils import
        if "from utils import" not in content:
            issues["errors"].append("Missing centralized utils import")
        
        # For main.py, check for setup_main_logging instead of setup_step_logging
        if module_path.name == "main.py":
            if "setup_main_logging" not in content:
                issues["errors"].append("Missing setup_main_logging call")
        else:
            # For other modules, check for setup_step_logging
            if "setup_step_logging" not in content:
                issues["errors"].append("Missing setup_step_logging call")
        
        # Check for redundant fallback code
        if "def log_step_start(" in content and "logger.info" in content:
            issues["warnings"].append("Contains redundant fallback logging functions")
        
        # Check log function calling patterns (be more tolerant)
        log_functions = ["log_step_start", "log_step_success", "log_step_warning", "log_step_error"]
        for func in log_functions:
            # Look for calls that might not have logger as first argument
            import re
            pattern = rf'{func}\s*\(\s*"[^"]*"'  # Matches func("string"...)
            if re.search(pattern, content):
                issues["suggestions"].append(f"Check {func} calling pattern - may need logger argument")
                
    except Exception as e:
        issues["errors"].append(f"Failed to read module: {e}")
    
    return issues

def validate_output_structure(output_dir: Path) -> Dict[str, List[str]]:
    """Validate that expected outputs are being generated."""
    issues = {"missing": [], "present": [], "unexpected": []}
    
    if not output_dir.exists():
        issues["missing"].append("Output directory does not exist")
        return issues
        
    for step, expected_files in EXPECTED_OUTPUTS.items():
        for expected_file in expected_files:
            expected_path = output_dir / expected_file
            if expected_path.exists():
                issues["present"].append(f"{step}: {expected_file}")
            else:
                issues["missing"].append(f"{step}: {expected_file}")
                
    return issues

def get_pipeline_modules(src_dir: Path) -> List[Path]:
    """Get all numbered pipeline modules."""
    modules = []
    for i in range(1, 15):  # Steps 1-14
        module_path = src_dir / f"{i}_*.py"
        matching = list(src_dir.glob(f"{i}_*.py"))
        if matching:
            modules.extend(matching)
    
    # Add main.py
    main_path = src_dir / "main.py"
    if main_path.exists():
        modules.append(main_path)
        
    return sorted(modules)

def validate_centralized_imports(module_path: Path) -> Dict[str, List[str]]:
    """Enhanced validation to check for proper centralized imports."""
    issues = {"errors": [], "warnings": [], "suggestions": [], "improvements": []}
    
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for proper utils import
        if "from utils import" not in content:
            issues["errors"].append("Missing centralized utils import")
        else:
            # Check for specific required imports
            required_imports = [
                "setup_step_logging",
                "log_step_start", 
                "log_step_success",
                "log_step_warning",
                "log_step_error"
            ]
            
            missing_imports = []
            for imp in required_imports:
                if imp not in content:
                    missing_imports.append(imp)
            
            if missing_imports:
                issues["warnings"].append(f"Missing recommended imports: {', '.join(missing_imports)}")
        
        # Check for pipeline configuration usage
        if module_path.name.startswith(('1_', '2_', '3_', '4_', '5_', '6_', '7_', '8_', '9_', '10_', '11_', '12_', '13_')):
            if "from pipeline" not in content and "pipeline" not in content:
                issues["suggestions"].append("Consider using centralized pipeline configuration")
        
        # Check for hardcoded paths
        hardcoded_patterns = [
            r'Path\(["\'](?:src/|output/|\.\.)',  # Hardcoded relative paths
            r'["\'](?:/[^"\']*|[A-Za-z]:[^"\']*)["\']',  # Absolute paths
        ]
        
        import re
        for pattern in hardcoded_patterns:
            if re.search(pattern, content):
                issues["improvements"].append("Consider using centralized path configuration")
                break
        
        # Check for redundant error handling
        if "try:" in content and "except ImportError" in content and "from utils import" in content:
            issues["improvements"].append("Redundant fallback imports - utils now provides graceful fallbacks")
        
        # Check for consistent argument parsing
        if module_path.name == "main.py":
            if "ArgumentParser" not in content:
                issues["suggestions"].append("Consider using ArgumentParser for better argument handling")
        elif "__name__ == '__main__'" in content:
            if "argparse.ArgumentParser" in content and "ArgumentParser" not in content:
                issues["suggestions"].append("Consider using ArgumentParser.parse_step_arguments for consistency")
        
        # Check for performance tracking usage
        if module_path.name in ['7_export.py', '6_visualization.py', '11_render.py', '12_execute.py', '11_llm.py', '12_audio.py', '13_website.py', '14_report.py']:
            if "performance_tracker" not in content:
                issues["suggestions"].append("Consider adding performance tracking for this compute-intensive step")
                
    except Exception as e:
        issues["errors"].append(f"Failed to analyze module: {e}")
    
    return issues

def validate_configuration_consistency() -> Dict[str, List[str]]:
    """Validate that configuration is consistent across the pipeline."""
    issues = {"errors": [], "warnings": [], "suggestions": []}
    
    try:
        from pipeline.config import get_pipeline_config, STEP_METADATA
        
        # Get the centralized configuration
        config = get_pipeline_config()
        
        # Check that all configured steps have metadata
        configured_steps = set(config.steps.keys())
        metadata_steps = set(STEP_METADATA.keys()) if STEP_METADATA else set()
        
        missing_metadata = configured_steps - metadata_steps
        if missing_metadata:
            issues["warnings"].append(f"Steps missing metadata: {', '.join(missing_metadata)}")
        
        extra_metadata = metadata_steps - configured_steps
        if extra_metadata:
            issues["warnings"].append(f"Metadata for unconfigured steps: {', '.join(extra_metadata)}")
        
        # Check dependency consistency
        for step_name, step_config in config.steps.items():
            for dep in step_config.dependencies:
                if dep not in configured_steps:
                    issues["errors"].append(f"Step {step_name} depends on unconfigured step {dep}")
                dep_config = config.get_step_config(dep)
                if dep_config and not dep_config.required:
                    issues["warnings"].append(f"Step {step_name} depends on optional step {dep}")
        
        # Validate configuration consistency
        for step_name, step_config in config.steps.items():
            if not step_config.output_subdir:
                issues["suggestions"].append(f"Step {step_name} has no output subdirectory configured")
            
    except ImportError as e:
        issues["errors"].append(f"Cannot import pipeline configuration: {e}")
    except Exception as e:
        issues["errors"].append(f"Configuration validation error: {e}")
    
    return issues

def generate_improvement_recommendations(report: Dict) -> List[str]:
    """Generate specific improvement recommendations based on validation results."""
    recommendations = []
    
    # Module-specific recommendations
    module_issues = report.get("module_issues", {})
    total_modules = len(module_issues) if module_issues else 0
    
    if total_modules > 0:
        error_modules = sum(1 for issues in module_issues.values() if issues.get("errors"))
        warning_modules = sum(1 for issues in module_issues.values() if issues.get("warnings"))
        
        if error_modules > 0:
            recommendations.append(f"🔴 **Critical**: Fix import errors in {error_modules} modules")
            recommendations.append("   - Use the template in `src/utils/pipeline_template.py` as a reference")
            recommendations.append("   - Ensure all modules import from the centralized `utils` package")
        
        if warning_modules > 0:
            recommendations.append(f"🟡 **Improve**: Address warnings in {warning_modules} modules")
            recommendations.append("   - Add missing required imports (log_step_*, setup_step_logging)")
            recommendations.append("   - Consider using ArgumentParser for consistency")
    
    # Configuration recommendations
    config_issues = report.get("configuration_validation", {})
    if config_issues:
        if config_issues.get("errors"):
            recommendations.append("🔴 **Critical**: Fix configuration inconsistencies")
            for error in config_issues["errors"]:
                recommendations.append(f"   - {error}")
        
        if config_issues.get("warnings"):
            recommendations.append("🟡 **Improve**: Address configuration warnings")
            for warning in config_issues["warnings"]:
                recommendations.append(f"   - {warning}")
    
    # Performance recommendations
    performance_modules = []
    for module, issues in module_issues.items():
        if any("performance_tracker" in suggestion for suggestion in issues.get("suggestions", [])):
            performance_modules.append(module)
    
    if performance_modules:
        recommendations.append("📊 **Performance**: Add performance tracking to compute-intensive steps")
        recommendations.append(f"   - Modules to enhance: {', '.join(performance_modules)}")
    
    # Argument consistency recommendations
    arg_issues = report.get("argument_validation", {})
    if arg_issues and arg_issues.get("inconsistencies"):
        recommendations.append("🔧 **Arguments**: Fix argument inconsistencies")
        for inconsistency in arg_issues["inconsistencies"][:3]:  # Show first 3
            recommendations.append(f"   - {inconsistency}")
    
    # Dependency cycle recommendations  
    cycle_issues = report.get("dependency_validation", {})
    if cycle_issues and cycle_issues.get("cycles"):
        recommendations.append("🔄 **Dependencies**: Fix circular dependency issues")
        for cycle in cycle_issues["cycles"][:2]:  # Show first 2
            recommendations.append(f"   - Cycle: {' → '.join(cycle)}")
    
    # Output naming recommendations
    naming_issues = report.get("output_validation", {})
    if naming_issues and naming_issues.get("naming_violations"):
        recommendations.append("📁 **Naming**: Fix output directory naming violations")
        for violation in naming_issues["naming_violations"][:3]:  # Show first 3
            recommendations.append(f"   - {violation}")
    
    # General recommendations
    recommendations.extend([
        "",
        "📋 **General Improvements:**",
        "   1. Use environment variables for configuration overrides (GNN_PIPELINE_*)",
        "   2. Implement the standardized import pattern from pipeline_step_template.py",
        "   3. Add correlation IDs to logs for better debugging",
        "   4. Use centralized argument parsing where possible",
        "   5. Consider adding retry logic for network-dependent steps",
        "",
        "🔧 **Next Steps:**",
        "   1. Run `python -m utils.argument_utils --validate` to check argument consistency",
        "   2. Run `python -m utils.dependency_validator` for dependency analysis",
        "   3. Use `GNN_PIPELINE_VERBOSE=true python src/main.py` for detailed execution logs"
    ])
    
    return recommendations

def validate_argument_consistency() -> Dict[str, List[str]]:
    """Validate that argument parsing is consistent across pipeline steps."""
    issues = {"errors": [], "warnings": [], "inconsistencies": []}
    
    try:
        from utils.argument_utils import ArgumentParser, STEP_ARGUMENTS
        
        # Check that all steps define their supported arguments
        expected_steps = [f"{i}_{name}.py" for i, name in enumerate([
            "setup", "gnn", "tests", "type_checker", "export", "visualization",
            "mcp", "ontology", "render", "execute", "llm", "website", "sapf"
        ], 1)]
        
        missing_definitions = []
        for step in expected_steps:
            if step not in STEP_ARGUMENTS:
                missing_definitions.append(step)
        
        if missing_definitions:
            issues["warnings"].append(f"Steps missing argument definitions: {', '.join(missing_definitions)}")
        
        # Check for argument consistency across similar steps
        common_args = ["target_dir", "output_dir", "recursive", "verbose"]
        inconsistent_steps = []
        
        for step_name, step_args in STEP_ARGUMENTS.items():
            missing_common = [arg for arg in common_args if arg not in step_args]
            if missing_common:
                inconsistent_steps.append(f"{step_name} missing: {', '.join(missing_common)}")
        
        issues["inconsistencies"] = inconsistent_steps
        
    except ImportError as e:
        issues["errors"].append(f"Cannot import argument utilities: {e}")
    except Exception as e:
        issues["errors"].append(f"Argument validation error: {e}")
    
    return issues

def validate_dependency_cycles() -> Dict[str, List[str]]:
    """Check for circular dependencies in pipeline step configuration."""
    issues = {"errors": [], "warnings": [], "cycles": []}
    
    try:
        from pipeline.config import get_pipeline_config
        
        config = get_pipeline_config()
        
        # Build dependency graph
        dependencies = {}
        for step_name, step_config in config.steps.items():
            dependencies[step_name] = step_config.dependencies
        
        # Detect cycles using DFS
        def find_cycles(node, path, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependencies.get(node, []):
                if neighbor not in visited:
                    cycle = find_cycles(neighbor, path + [neighbor], visited, rec_stack)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
            
            rec_stack.remove(node)
            return None
        
        visited = set()
        for step in dependencies:
            if step not in visited:
                cycle = find_cycles(step, [step], visited, set())
                if cycle:
                    issues["cycles"].append(cycle)
        
        # Check for missing dependencies
        all_steps = set(dependencies.keys())
        missing_deps = []
        for step, deps in dependencies.items():
            for dep in deps:
                if dep not in all_steps:
                    missing_deps.append(f"{step} depends on non-existent {dep}")
        
        if missing_deps:
            issues["errors"].extend(missing_deps)
            
    except ImportError as e:
        issues["errors"].append(f"Cannot import pipeline config: {e}")
    except Exception as e:
        issues["errors"].append(f"Dependency validation error: {e}")
    
    return issues

def validate_output_naming_conventions() -> Dict[str, List[str]]:
    """Validate that output directory naming follows conventions."""
    issues = {"errors": [], "warnings": [], "naming_violations": []}
    
    try:
        from pipeline.config import get_pipeline_config
        
        config = get_pipeline_config()
        
        # Expected naming pattern: step name + descriptive suffix (matching config.py)
        expected_patterns = {
            "0_template.py": "0_template_output",
            "1_setup.py": "1_setup_output",
            "2_tests.py": "2_tests_output",
            "3_gnn.py": "3_gnn_output",
            "4_model_registry.py": "4_model_registry_output",
            "5_type_checker.py": "5_type_checker_output",
            "6_validation.py": "6_validation_output",
            "7_export.py": "7_export_output",
            "8_visualization.py": "8_visualization_output",
            "9_advanced_viz.py": "9_advanced_viz_output",
            "10_ontology.py": "10_ontology_output",
            "11_render.py": "11_render_output",
            "12_execute.py": "12_execute_output",
            "13_llm.py": "13_llm_output",
            "14_ml_integration.py": "14_ml_integration_output",
            "15_audio.py": "15_audio_output",
            "16_analysis.py": "16_analysis_output",
            "17_integration.py": "17_integration_output",
            "18_security.py": "18_security_output",
            "19_research.py": "19_research_output",
            "20_website.py": "20_website_output",
            "21_mcp.py": "21_mcp_output",
            "22_gui.py": "22_gui_output",
            "23_report.py": "23_report_output"
        }
        
        violations = []
        for step_name, expected_pattern in expected_patterns.items():
            step_config = config.get_step_config(step_name)
            if step_config and step_config.output_subdir != expected_pattern:
                violations.append(f"{step_name}: expected '{expected_pattern}', got '{step_config.output_subdir}'")
        
        issues["naming_violations"] = violations
        
    except ImportError as e:
        issues["errors"].append(f"Cannot import pipeline config: {e}")
    except Exception as e:
        issues["errors"].append(f"Output naming validation error: {e}")
    
    return issues

def validate_performance_tracking_coverage() -> Dict[str, List[str]]:
    """Check which steps have performance tracking enabled."""
    issues = {"warnings": [], "missing_tracking": [], "suggestions": []}
    
    try:
        from pipeline.config import get_pipeline_config
        
        config = get_pipeline_config()
        
        # Steps that should have performance tracking (compute-intensive)
        should_have_tracking = [
            "3_gnn.py", "2_tests.py", "5_type_checker.py", "7_export.py", 
            "6_visualization.py", "21_mcp.py", "10_ontology.py", "11_render.py", 
            "12_execute.py", "11_llm.py", "12_audio.py", "13_website.py", "14_report.py"
        ]
        
        missing_tracking = []
        for step_name in should_have_tracking:
            step_config = config.get_step_config(step_name)
            if step_config and not step_config.performance_tracking:
                missing_tracking.append(step_name)
        
        if missing_tracking:
            issues["missing_tracking"] = missing_tracking
            issues["suggestions"].append("Enable performance tracking for compute-intensive steps")
            
    except ImportError as e:
        issues["warnings"].append(f"Cannot import pipeline config: {e}")
    except Exception as e:
        issues["warnings"].append(f"Performance tracking validation error: {e}")
    
    return issues

def generate_validation_report(src_dir: Path, output_dir: Path) -> Dict:
    """Generate a comprehensive validation report."""
    log_step_start(logger, "Generating pipeline validation report")
    
    report = {
        "timestamp": Path(__file__).stat().st_mtime,
        "modules_checked": 0,
        "modules_with_issues": 0,
        "output_validation": {},
        "module_issues": {},
        "configuration_validation": {},
        "argument_validation": {},
        "dependency_validation": {},
        "improvement_recommendations": [],
        "summary": {}
    }
    
    # Check pipeline modules with enhanced validation
    modules = get_pipeline_modules(src_dir)
    report["modules_checked"] = len(modules)
    
    for module in modules:
        logger.debug(f"Validating module: {module.name}")
        # Use enhanced validation instead of basic imports check
        issues = validate_centralized_imports(module)
        
        if any(issues.values()):
            report["modules_with_issues"] += 1
            report["module_issues"][module.name] = issues
            
    # Check configuration consistency
    report["configuration_validation"] = validate_configuration_consistency()
    
    # Check output structure
    report["output_validation"] = validate_output_structure(output_dir)

    # Check argument consistency
    report["argument_validation"] = validate_argument_consistency()

    # Check dependency cycles
    report["dependency_validation"] = validate_dependency_cycles()

    # Check output naming conventions
    report["output_validation"]["naming_violations"] = validate_output_naming_conventions()

    # Check performance tracking coverage
    report["performance_tracking_coverage"] = validate_performance_tracking_coverage()
    
    # Generate improvement recommendations
    report["improvement_recommendations"] = generate_improvement_recommendations(report)
    
    # Generate summary
    total_issues = sum(len(issues.get("errors", [])) + len(issues.get("warnings", [])) 
                      for issues in report["module_issues"].values())
    config_errors = len(report["configuration_validation"].get("errors", []))
    missing_outputs = len(report["output_validation"].get("missing", []))
    
    # Enhanced status determination
    if total_issues == 0 and config_errors == 0 and missing_outputs == 0:
        status = "PASS"
    elif config_errors > 0 or total_issues > 0:
        status = "FAIL" 
    else:
        status = "WARN"  # Only missing outputs (expected if pipeline hasn't run)
    
    report["summary"] = {
        "total_modules": len(modules),
        "modules_with_issues": report["modules_with_issues"],
        "total_import_issues": total_issues,
        "configuration_errors": config_errors,
        "missing_outputs": missing_outputs,
        "status": status
    }
    
    log_step_success(logger, f"Validation complete. Status: {report['summary']['status']}")
    return report

def print_validation_report(report: Dict):
    """Print a human-readable validation report."""
    print("\n" + "="*80)
    print("GNN PIPELINE VALIDATION REPORT")
    print("="*80)
    
    summary = report["summary"]
    print(f"\nSUMMARY:")
    print(f"  Status: {summary['status']}")
    print(f"  Modules checked: {summary['total_modules']}")
    print(f"  Modules with issues: {summary['modules_with_issues']}")
    print(f"  Total import issues: {summary['total_import_issues']}")
    print(f"  Configuration errors: {summary['configuration_errors']}")
    print(f"  Missing outputs: {summary['missing_outputs']}")
    
    # Configuration validation
    config_val = report.get("configuration_validation", {})
    if config_val.get("errors") or config_val.get("warnings"):
        print(f"\nCONFIGURATION ISSUES:")
        for error in config_val.get("errors", []):
            print(f"  ERROR: {error}")
        for warning in config_val.get("warnings", []):
            print(f"  WARNING: {warning}")
    
    # Module issues
    if report["module_issues"]:
        print(f"\nMODULE ANALYSIS:")
        for module, issues in report["module_issues"].items():
            print(f"\n  {module}:")
            for error in issues.get("errors", []):
                print(f"    🔴 ERROR: {error}")
            for warning in issues.get("warnings", []):
                print(f"    🟡 WARNING: {warning}")
            for suggestion in issues.get("suggestions", []):
                print(f"    💡 SUGGESTION: {suggestion}")
            for improvement in issues.get("improvements", []):
                print(f"    📈 IMPROVEMENT: {improvement}")
    
    # Output validation
    output_val = report["output_validation"]
    if output_val.get("missing"):
        print(f"\nMISSING OUTPUTS:")
        for missing in output_val["missing"]:
            print(f"  - {missing}")
            
    if output_val.get("present"):
        print(f"\nPRESENT OUTPUTS:")
        for present in output_val["present"]:
            print(f"  ✓ {present}")
    
    # Argument validation
    arg_val = report.get("argument_validation", {})
    if arg_val.get("inconsistencies"):
        print(f"\nARGUMENT INCONSISTENCIES:")
        for inconsistency in arg_val["inconsistencies"]:
            print(f"  - {inconsistency}")

    # Dependency validation
    dep_val = report.get("dependency_validation", {})
    if dep_val.get("cycles"):
        print(f"\nDEPENDENCY CYCLES:")
        for cycle in dep_val["cycles"]:
            print(f"  - Cycle: {' → '.join(cycle)}")

    # Output naming violations
    naming_val = report.get("output_validation", {}).get("naming_violations")
    if naming_val:
        print(f"\nOUTPUT NAMING VIOLATIONS:")
        for violation in naming_val:
            print(f"  - {violation}")

    # Performance tracking coverage
    perf_val = report.get("performance_tracking_coverage", {})
    if perf_val.get("missing_tracking"):
        print(f"\nMISSING PERFORMANCE TRACKING:")
        for step in perf_val["missing_tracking"]:
            print(f"  - {step}")

    # Improvement recommendations
    recommendations = report.get("improvement_recommendations", [])
    if recommendations:
        print(f"\nIMPROVEMENT RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  {rec}")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Validate GNN pipeline consistency")
    parser.add_argument("--src-dir", type=Path, default=Path(__file__).parent,
                       help="Source directory containing pipeline modules")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent.parent / "output",
                       help="Output directory to validate")
    parser.add_argument("--save-report", type=Path,
                       help="Save detailed report to JSON file")
    
    args = parser.parse_args()
    
    log_step_start(logger, "Starting pipeline validation")
    
    # Generate validation report
    report = generate_validation_report(args.src_dir, args.output_dir)
    
    # Print human-readable report
    print_validation_report(report)
    
    # Save detailed report if requested
    if args.save_report:
        import json
        with open(args.save_report, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        log_step_success(logger, f"Detailed report saved to {args.save_report}")
    
    # Exit with appropriate code
    exit_code = 0 if report["summary"]["status"] == "PASS" else 1
    log_step_success(logger, f"Pipeline validation completed with exit code {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 