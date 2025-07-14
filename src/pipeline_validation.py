#!/usr/bin/env python3
"""
GNN Pipeline Validation Script

This script validates that all pipeline steps are:
1. Using consistent logging patterns
2. Producing expected outputs  
3. Handling arguments correctly
4. Following the established patterns

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
    "1_gnn": [
        "gnn_processing_step/1_gnn_discovery_report.md"
    ],
    "2_setup": [
        "directory_structure.json",
        "setup_artifacts/installed_packages.json"
    ],
    "3_tests": [
        "test_reports/pytest_report.xml"
    ],
            "4_type_checker": [
        "type_check/type_check/type_check_report.md"
    ],
    "5_export": [
        "gnn_exports/"
    ],
    "6_visualization": [
        "visualization/"
    ],
    "7_mcp": [
        "mcp_processing_step/"
    ],
    "8_ontology": [
        "ontology_processing/"
    ],
    "9_render": [
        "gnn_rendered_simulators/"
    ],
    "10_execute": [
        "execution_results/"
    ],
    "11_llm": [
        "llm_processing_step/"
    ],
    "12_site": [
        "site/"
    ],
    "13_sapf": [
        "sapf_processing_step/"
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
    for i in range(1, 14):  # Steps 1-13
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
        if module_path.name.startswith(('1_', '2_', '3_', '4_', '5_', '6_', '7_', '8_', '9_', '10_', '11_', '12_', '13_', '14_')):
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
            if "EnhancedArgumentParser" not in content:
                issues["suggestions"].append("Consider using EnhancedArgumentParser for better argument handling")
        elif "__name__ == '__main__'" in content:
            if "argparse.ArgumentParser" in content and "EnhancedArgumentParser" not in content:
                issues["suggestions"].append("Consider using EnhancedArgumentParser.parse_step_arguments for consistency")
        
        # Check for performance tracking usage
        if module_path.name in ['5_export.py', '6_visualization.py', '9_render.py', '10_execute.py', '11_llm.py']:
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
        metadata_steps = set(STEP_METADATA.keys())
        
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
            recommendations.append("   - Consider using EnhancedArgumentParser for consistency")
    
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
    
    # General recommendations
    recommendations.extend([
        "",
        "📋 **General Improvements:**",
        "   1. Use environment variables for configuration overrides (GNN_PIPELINE_*)",
        "   2. Implement the standardized import pattern from pipeline_template.py",
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