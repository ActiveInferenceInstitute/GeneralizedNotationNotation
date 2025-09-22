#!/usr/bin/env python3
"""
Thin Orchestrator Pattern Validator

This script validates that all numbered pipeline scripts (0-23) follow the thin orchestrator pattern.
It checks for compliance with architectural standards and provides detailed reports on violations.

Usage:
    python src/utils/thin_orchestrator_validator.py [--fix] [--verbose]

Options:
    --fix: Automatically fix minor violations where possible
    --verbose: Show detailed analysis of each script
"""

import sys
import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

@dataclass
class ScriptAnalysis:
    """Analysis results for a single pipeline script."""
    file_path: str
    line_count: int
    function_count: int
    class_count: int
    import_count: int
    violations: List[str] = field(default_factory=list)
    score: float = 0.0
    is_thin_orchestrator: bool = False
    delegation_found: bool = False
    fallback_handling: bool = False
    module_imports: List[str] = field(default_factory=list)
    long_functions: List[Tuple[str, int]] = field(default_factory=list)

@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    total_scripts: int
    compliant_scripts: int
    violating_scripts: int
    critical_violations: int
    warnings: List[str] = field(default_factory=list)
    script_analyses: Dict[str, ScriptAnalysis] = field(default_factory=dict)
    summary: str = ""

def analyze_script(file_path: Path) -> ScriptAnalysis:
    """Analyze a single script for thin orchestrator compliance."""
    analysis = ScriptAnalysis(
        file_path=str(file_path),
        line_count=0,
        function_count=0,
        class_count=0,
        import_count=0
    )

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.splitlines()
        analysis.line_count = len(lines)

        # Parse AST to analyze structure
        tree = ast.parse(content, filename=str(file_path))

        # Count functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis.function_count += 1
                # Check for long functions
                if len(lines) >= node.lineno - 1:  # Safety check
                    function_line_count = sum(1 for line in lines[node.lineno-1:node.end_lineno]
                                            if line.strip())
                    if function_line_count > 50:
                        analysis.long_functions.append((node.name, function_line_count))

            elif isinstance(node, ast.ClassDef):
                analysis.class_count += 1

        # Analyze imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    analysis.module_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    analysis.module_imports.append(node.module)

        # Check for delegation patterns
        analysis.delegation_found = check_delegation_patterns(content)

        # Check for fallback handling
        analysis.fallback_handling = check_fallback_patterns(content)

        # Analyze violations
        analysis.violations = check_violations(content, analysis)

        # Calculate compliance score
        analysis.score = calculate_compliance_score(analysis)

        # Determine if it's a thin orchestrator
        analysis.is_thin_orchestrator = determine_thin_orchestrator_status(analysis)

    except Exception as e:
        analysis.violations.append(f"Failed to analyze script: {e}")

    return analysis

def check_delegation_patterns(content: str) -> bool:
    """Check if the script properly delegates to modules."""
    # Look for module imports and function calls
    patterns = [
        r'from\s+\w+\s+import\s+\w+',  # from module import function
        r'from\s+\w+\.\w+\s+import',   # from module.submodule import
        r'import\s+\w+',               # import module
        r'\w+\.\w+\s*\(',              # module.function()
    ]

    delegation_found = False
    for pattern in patterns:
        if re.search(pattern, content):
            delegation_found = True
            break

    return delegation_found

def check_fallback_patterns(content: str) -> bool:
    """Check if the script has proper fallback handling."""
    fallback_patterns = [
        r'try:\s*.*?except\s+ImportError',
        r'except\s+ImportError',
        r'def.*fallback',
        r'if.*not.*available',
        r'logger\.warning.*not available'
    ]

    fallback_found = False
    for pattern in fallback_patterns:
        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
            fallback_found = True
            break

    return fallback_found

def check_violations(content: str, analysis: ScriptAnalysis) -> List[str]:
    """Check for thin orchestrator pattern violations."""
    violations = []

    # Check line count (should be <200 for thin orchestrator)
    if analysis.line_count > 200:
        violations.append(f"Script too long: {analysis.line_count} lines (should be <200)")
    elif analysis.line_count > 100:
        violations.append(f"Script quite long: {analysis.line_count} lines (should be <100 for ideal thin orchestrator)")

    # Check for implementation functions (indicating fat orchestrator)
    implementation_functions = []
    for line in content.splitlines():
        # Look for function definitions that suggest implementation
        if re.match(r'\s*def\s+\w+.*\(', line):
            func_name = re.search(r'def\s+(\w+)', line)
            if func_name:
                name = func_name.group(1)
                # Common implementation function patterns
                if any(pattern in name.lower() for pattern in [
                    'process_', 'execute_', 'run_', 'generate_', 'create_', 'build_',
                    'validate_', 'parse_', 'extract_', 'transform_', 'convert_'
                ]):
                    implementation_functions.append(name)

    if implementation_functions:
        violations.append(f"Contains implementation functions: {implementation_functions}")

    # Check for long functions
    if analysis.long_functions:
        violations.append(f"Contains long functions: {[(name, count) for name, count in analysis.long_functions]}")

    # Check for class definitions (should be minimal in thin orchestrators)
    if analysis.class_count > 2:
        violations.append(f"Too many classes: {analysis.class_count} (thin orchestrators should have minimal class definitions)")

    # Check for proper delegation
    if not analysis.delegation_found:
        violations.append("No clear module delegation found")

    # Check for fallback handling
    if not analysis.fallback_handling:
        violations.append("No fallback handling for missing dependencies")

    return violations

def calculate_compliance_score(analysis: ScriptAnalysis) -> float:
    """Calculate compliance score (0-100)."""
    score = 100.0

    # Deduct for violations
    for violation in analysis.violations:
        if 'too long' in violation.lower():
            if '200' in violation:
                score -= 30  # Critical violation
            else:
                score -= 15  # Warning
        elif 'implementation functions' in violation.lower():
            score -= 25  # Critical violation
        elif 'long functions' in violation.lower():
            score -= 20  # Critical violation
        elif 'no clear module delegation' in violation.lower():
            score -= 20  # Critical violation
        elif 'no fallback handling' in violation.lower():
            score -= 10  # Important but not critical
        else:
            score -= 5   # Other violations

    # Ensure score doesn't go below 0
    score = max(0, score)

    return score

def determine_thin_orchestrator_status(analysis: ScriptAnalysis) -> bool:
    """Determine if script follows thin orchestrator pattern."""
    # Must have delegation and fallback handling
    if not analysis.delegation_found or not analysis.fallback_handling:
        return False

    # Must have reasonable score (>60)
    if analysis.score < 60:
        return False

    # Must not have critical violations
    critical_violations = [
        'implementation functions',
        'no clear module delegation',
        'too long' if '200' in str(analysis.violations) else None
    ]
    critical_violations = [v for v in critical_violations if v]

    for violation in analysis.violations:
        if any(cv in violation.lower() for cv in critical_violations):
            return False

    return True

def validate_all_scripts(verbose: bool = False, fix: bool = False) -> ValidationReport:
    """Validate all numbered pipeline scripts."""
    report = ValidationReport(total_scripts=0, compliant_scripts=0, violating_scripts=0, critical_violations=0)

    # Find all numbered scripts
    src_dir = Path(__file__).parent.parent
    script_pattern = re.compile(r'^(\d+)_.*\.py$')

    script_files = []
    for file_path in src_dir.glob('*.py'):
        match = script_pattern.match(file_path.name)
        if match:
            step_number = int(match.group(1))
            if 0 <= step_number <= 23:  # Valid pipeline steps
                script_files.append(file_path)

    script_files.sort()
    report.total_scripts = len(script_files)

    for file_path in script_files:
        analysis = analyze_script(file_path)
        report.script_analyses[file_path.name] = analysis

        if analysis.is_thin_orchestrator:
            report.compliant_scripts += 1
        else:
            report.violating_scripts += 1
            if analysis.score < 50:
                report.critical_violations += 1

        if verbose:
            print(f"\n{'='*60}")
            print(f"SCRIPT: {file_path.name}")
            print(f"{'='*60}")
            print(f"Lines: {analysis.line_count}")
            print(f"Functions: {analysis.function_count}")
            print(f"Classes: {analysis.class_count}")
            print(f"Imports: {analysis.import_count}")
            print(f"Score: {analysis.score:.1f}/100")
            print(f"Status: {'✅ THIN ORCHESTRATOR' if analysis.is_thin_orchestrator else '❌ VIOLATION'}")

            if analysis.violations:
                print("\nVIOLATIONS:")
                for violation in analysis.violations:
                    print(f"  ❌ {violation}")

            if analysis.long_functions:
                print("\nLONG FUNCTIONS:")
                for name, count in analysis.long_functions:
                    print(f"  📏 {name} ({count} lines)")

    # Generate summary
    report.summary = generate_summary_report(report)
    return report

def generate_summary_report(report: ValidationReport) -> str:
    """Generate a comprehensive summary report."""
    summary = f"""
{'='*70}
THIN ORCHESTRATOR VALIDATION REPORT
{'='*70}

📊 OVERVIEW:
  Total scripts analyzed: {report.total_scripts}
  Compliant thin orchestrators: {report.compliant_scripts}
  Scripts with violations: {report.violating_scripts}
  Critical violations: {report.critical_violations}

    📈 COMPLIANCE SCORE: {report.compliant_scripts/report.total_scripts*100:.1f}%

{'='*70}
SCRIPT ANALYSIS:
{'='*70}
"""

    for script_name, analysis in report.script_analyses.items():
        status = "✅" if analysis.is_thin_orchestrator else "❌"
        summary += f"{status} {script_name:<20} | Score: {analysis.score:5.1f} | Lines: {analysis.line_count:4d}\n"

        if analysis.violations:
            for violation in analysis.violations:
                summary += f"     └─ ❌ {violation}\n"

    summary += f"""
{'='*70}
RECOMMENDATIONS:
{'='*70}
"""

    if report.critical_violations > 0:
        summary += f"""
🚨 CRITICAL ISSUES FOUND:
{report.critical_violations} scripts have critical violations that need immediate attention.

These scripts contain substantial implementation code directly in the orchestrator,
violating the thin orchestrator pattern. Consider refactoring to move implementation
code to appropriate modules.
"""
    else:
        summary += "\n✅ All scripts follow the thin orchestrator pattern correctly!"

    summary += f"""
📋 COMPLIANCE CHECKLIST:
✅ Proper module delegation: {sum(1 for a in report.script_analyses.values() if a.delegation_found)}/{report.total_scripts} scripts
✅ Fallback handling: {sum(1 for a in report.script_analyses.values() if a.fallback_handling)}/{report.total_scripts} scripts
✅ Reasonable line count: {sum(1 for a in report.script_analyses.values() if a.line_count <= 100)}/{report.total_scripts} scripts
✅ No implementation functions: {sum(1 for a in report.script_analyses.values() if not any('implementation' in v.lower() for v in a.violations))}/{report.total_scripts} scripts
"""

    return summary

def main():
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate thin orchestrator pattern compliance")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed analysis")
    parser.add_argument("--fix", "-f", action="store_true", help="Attempt to fix minor violations")
    parser.add_argument("--output", "-o", help="Output report to file")

    args = parser.parse_args()

    print("🔍 Validating thin orchestrator pattern compliance...")
    print(f"{'='*60}")

    report = validate_all_scripts(verbose=args.verbose, fix=args.fix)

    print(report.summary)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report.summary)
        print(f"\n📄 Report saved to: {args.output}")

    # Exit with appropriate code
    if report.critical_violations > 0:
        print("\n❌ VALIDATION FAILED: Critical violations found")
        sys.exit(1)
    elif report.violating_scripts > 0:
        print("\n⚠️ VALIDATION PASSED WITH WARNINGS: Some violations found")
        sys.exit(0)
    else:
        print("\n✅ VALIDATION PASSED: All scripts compliant")
        sys.exit(0)

if __name__ == "__main__":
    main()
