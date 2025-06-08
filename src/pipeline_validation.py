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
    "4_gnn_type_checker": [
        "gnn_type_check/gnn_type_check/type_check_report.md"
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
    "12_discopy": [
        "discopy_gnn/"
    ],
    "13_discopy_jax_eval": [
        "discopy_jax_eval/"
    ],
    "14_site": [
        "site/"
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

def generate_validation_report(src_dir: Path, output_dir: Path) -> Dict:
    """Generate a comprehensive validation report."""
    log_step_start(logger, "Generating pipeline validation report")
    
    report = {
        "timestamp": Path(__file__).stat().st_mtime,
        "modules_checked": 0,
        "modules_with_issues": 0,
        "output_validation": {},
        "module_issues": {},
        "summary": {}
    }
    
    # Check pipeline modules
    modules = get_pipeline_modules(src_dir)
    report["modules_checked"] = len(modules)
    
    for module in modules:
        logger.debug(f"Validating module: {module.name}")
        issues = validate_module_imports(module)
        
        if any(issues.values()):
            report["modules_with_issues"] += 1
            report["module_issues"][module.name] = issues
            
    # Check output structure
    report["output_validation"] = validate_output_structure(output_dir)
    
    # Generate summary
    total_issues = sum(len(issues.get("errors", [])) + len(issues.get("warnings", [])) 
                      for issues in report["module_issues"].values())
    missing_outputs = len(report["output_validation"].get("missing", []))
    
    report["summary"] = {
        "total_modules": len(modules),
        "modules_with_issues": report["modules_with_issues"],
        "total_import_issues": total_issues,
        "missing_outputs": missing_outputs,
        "status": "PASS" if total_issues == 0 and missing_outputs == 0 else "FAIL"
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
    print(f"  Missing outputs: {summary['missing_outputs']}")
    
    # Module issues
    if report["module_issues"]:
        print(f"\nMODULE ISSUES:")
        for module, issues in report["module_issues"].items():
            print(f"\n  {module}:")
            for error in issues.get("errors", []):
                print(f"    ERROR: {error}")
            for warning in issues.get("warnings", []):
                print(f"    WARNING: {warning}")
            for suggestion in issues.get("suggestions", []):
                print(f"    SUGGESTION: {suggestion}")
    
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