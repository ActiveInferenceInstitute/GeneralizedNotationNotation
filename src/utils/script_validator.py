#!/usr/bin/env python3
"""
Pipeline Script Validator

This module validates all numbered pipeline scripts to ensure they have:
- Proper safe-to-fail error handling
- All referenced methods defined in accompanying modules
- Consistent logging patterns
- Proper import statements
"""

import sys
import ast
import importlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field

@dataclass
class ValidationIssue:
    """Represents a validation issue found in a script."""
    script_name: str
    issue_type: str
    severity: str  # "error", "warning", "info"
    line_number: Optional[int]
    message: str
    suggestion: str = ""

@dataclass
class ScriptValidationResult:
    """Results of validating a single script."""
    script_name: str
    script_path: Path
    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    imported_modules: Set[str] = field(default_factory=set)
    called_functions: Set[str] = field(default_factory=set)
    has_error_handling: bool = False
    has_logging: bool = False
    has_safe_exit: bool = False

class PipelineScriptValidator:
    """Validator for pipeline scripts."""
    
    def __init__(self, src_dir: Path = None):
        self.src_dir = src_dir or Path(__file__).parent.parent
        self.logger = logging.getLogger(__name__)
        
        # Expected patterns for safe pipeline scripts
        self.required_imports = {
            "utils.pipeline_template": ["setup_step_logging", "log_step_start", "log_step_success", "log_step_error"],
            "utils.argument_utils": ["ArgumentParser"],
            "pipeline.config": ["get_output_dir_for_script", "get_pipeline_config"]
        }
        
        self.required_patterns = [
            "setup_step_logging",
            "log_step_start", 
            "try:",
            "except",
            "return 0",
            "return 1"
        ]
    
    def validate_all_scripts(self) -> Dict[str, ScriptValidationResult]:
        """Validate all numbered pipeline scripts."""
        results = {}
        
        # Find all numbered scripts (0-21)
        for i in range(22):
            # Look for scripts matching the pattern
            script_patterns = [
                f"{i}_*.py",
                f"{i:02d}_*.py"
            ]
            
            script_found = False
            for pattern in script_patterns:
                matching_scripts = list(self.src_dir.glob(pattern))
                for script_path in matching_scripts:
                    script_name = script_path.name
                    self.logger.info(f"Validating script: {script_name}")
                    results[script_name] = self.validate_script(script_path)
                    script_found = True
            
            if not script_found:
                # Create a placeholder result for missing script
                results[f"{i}_missing.py"] = ScriptValidationResult(
                    script_name=f"{i}_missing.py",
                    script_path=self.src_dir / f"{i}_missing.py",
                    is_valid=False,
                    issues=[ValidationIssue(
                        script_name=f"{i}_missing.py",
                        issue_type="missing_script",
                        severity="error",
                        line_number=None,
                        message=f"Pipeline step {i} script is missing",
                        suggestion=f"Create {i}_<module_name>.py script"
                    )]
                )
        
        return results
    
    def validate_script(self, script_path: Path) -> ScriptValidationResult:
        """Validate a single pipeline script."""
        result = ScriptValidationResult(
            script_name=script_path.name,
            script_path=script_path
        )
        
        if not script_path.exists():
            result.is_valid = False
            result.issues.append(ValidationIssue(
                script_name=script_path.name,
                issue_type="missing_file",
                severity="error",
                line_number=None,
                message="Script file does not exist"
            ))
            return result
        
        try:
            # Read and parse the script
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Analyze the script
            self._analyze_imports(tree, result)
            self._analyze_error_handling(tree, content, result)
            self._analyze_logging_patterns(tree, content, result)
            self._analyze_function_calls(tree, result)
            self._validate_module_imports(result)
            self._check_safe_fail_patterns(content, result)
            
        except SyntaxError as e:
            result.is_valid = False
            result.issues.append(ValidationIssue(
                script_name=script_path.name,
                issue_type="syntax_error",
                severity="error",
                line_number=e.lineno,
                message=f"Syntax error: {e.msg}",
                suggestion="Fix syntax errors in the script"
            ))
        except Exception as e:
            result.is_valid = False
            result.issues.append(ValidationIssue(
                script_name=script_path.name,
                issue_type="validation_error",
                severity="error",
                line_number=None,
                message=f"Validation error: {e}",
                suggestion="Review script structure and fix issues"
            ))
        
        # Determine overall validity
        error_count = len([i for i in result.issues if i.severity == "error"])
        result.is_valid = error_count == 0
        
        return result
    
    def _analyze_imports(self, tree: ast.AST, result: ScriptValidationResult):
        """Analyze import statements in the script."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result.imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    result.imported_modules.add(node.module)
                    for alias in node.names:
                        result.imported_modules.add(f"{node.module}.{alias.name}")
        
        # Check for required imports
        for module, functions in self.required_imports.items():
            if module not in result.imported_modules:
                result.issues.append(ValidationIssue(
                    script_name=result.script_name,
                    issue_type="missing_import",
                    severity="warning",
                    line_number=None,
                    message=f"Missing recommended import: {module}",
                    suggestion=f"Add: from {module} import {', '.join(functions)}"
                ))
    
    def _analyze_error_handling(self, tree: ast.AST, content: str, result: ScriptValidationResult):
        """Analyze error handling patterns."""
        has_try_except = False
        has_specific_exceptions = False
        has_general_exception = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                has_try_except = True
                for handler in node.handlers:
                    if handler.type is None:
                        has_general_exception = True
                    else:
                        has_specific_exceptions = True
        
        # For standardized pipeline scripts, check if they use create_standardized_pipeline_script
        # which includes its own error handling
        uses_standardized_pattern = "create_standardized_pipeline_script(" in content
        
        result.has_error_handling = has_try_except or uses_standardized_pattern
        
        if not has_try_except and not uses_standardized_pattern:
            result.issues.append(ValidationIssue(
                script_name=result.script_name,
                issue_type="missing_error_handling",
                severity="error",
                line_number=None,
                message="No try-except blocks found",
                suggestion="Add try-except blocks around main processing logic"
            ))
        elif not has_general_exception and not uses_standardized_pattern:
            result.issues.append(ValidationIssue(
                script_name=result.script_name,
                issue_type="incomplete_error_handling",
                severity="warning",
                line_number=None,
                message="No general exception handler found",
                suggestion="Add a general 'except Exception as e:' handler"
            ))
    
    def _analyze_logging_patterns(self, tree: ast.AST, content: str, result: ScriptValidationResult):
        """Analyze logging patterns."""
        logging_functions = ["setup_step_logging", "log_step_start", "log_step_success", "log_step_error", "log_step_warning"]
        found_logging = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in logging_functions:
                    found_logging.add(node.func.id)
        
        result.has_logging = len(found_logging) > 0
        
        # Check for required logging functions
        required_logging = ["setup_step_logging", "log_step_start"]
        for func in required_logging:
            if func not in found_logging:
                result.issues.append(ValidationIssue(
                    script_name=result.script_name,
                    issue_type="missing_logging",
                    severity="warning",
                    line_number=None,
                    message=f"Missing logging function: {func}",
                    suggestion=f"Add call to {func} in main function"
                ))
    
    def _analyze_function_calls(self, tree: ast.AST, result: ScriptValidationResult):
        """Analyze function calls to detect potential missing implementations."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    result.called_functions.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Handle method calls like module.function()
                    if isinstance(node.func.value, ast.Name):
                        result.called_functions.add(f"{node.func.value.id}.{node.func.attr}")
    
    def _validate_module_imports(self, result: ScriptValidationResult):
        """Validate that imported modules exist and have required functions."""
        # Standard library modules that don't need to be in src/
        stdlib_modules = {
            'sys', 'os', 'pathlib', 'typing', 'datetime', 'xml', 'json', 
            'logging', 'argparse', 'subprocess', 'time', 'collections',
            'enum', 'contextlib', 'dataclasses', 'functools', 'itertools',
            'pickle', 'tempfile', 'shutil', 'copy', 'math', 'random',
            'hashlib', 'uuid', 'urllib', 'http', 'socket', 'threading',
            'multiprocessing', 'concurrent', 'asyncio', 'queue', 're',
            'string', 'io', 'gzip', 'zipfile', 'tarfile', 'csv', 'sqlite3',
            'warnings', 'platform', 'stat', 'glob', 'fnmatch', 'matplotlib',
            'numpy', 'pandas', 'scipy', 'sklearn', 'torch', 'tensorflow'
        }
        
        for module_path in result.imported_modules:
            if "." in module_path and not module_path.startswith("utils.") and not module_path.startswith("pipeline."):
                # This might be a function import like "module.function"
                module_name = module_path.split(".")[0]
                
                # Skip standard library modules
                if module_name in stdlib_modules:
                    continue
                
                # Check if the module directory exists
                module_dir = self.src_dir / module_name
                if not module_dir.exists():
                    result.issues.append(ValidationIssue(
                        script_name=result.script_name,
                        issue_type="missing_module",
                        severity="error",
                        line_number=None,
                        message=f"Module directory not found: {module_name}",
                        suggestion=f"Create src/{module_name}/ directory with __init__.py"
                    ))
                    continue
                
                # Check if __init__.py exists
                init_file = module_dir / "__init__.py"
                if not init_file.exists():
                    result.issues.append(ValidationIssue(
                        script_name=result.script_name,
                        issue_type="missing_init",
                        severity="error",
                        line_number=None,
                        message=f"Missing __init__.py in module: {module_name}",
                        suggestion=f"Create src/{module_name}/__init__.py"
                    ))
                    continue
                
                # Try to validate specific function imports
                try:
                    if "." in module_path:
                        function_name = module_path.split(".")[-1]
                        # Read the init file to check if function exists
                        with open(init_file, 'r') as f:
                            init_content = f.read()
                        
                        if f"def {function_name}" not in init_content:
                            result.issues.append(ValidationIssue(
                                script_name=result.script_name,
                                issue_type="missing_function",
                                severity="warning",
                                line_number=None,
                                message=f"Function '{function_name}' not found in {module_name}",
                                suggestion=f"Implement {function_name} function in src/{module_name}/__init__.py"
                            ))
                except Exception:
                    # Can't validate function - that's ok
                    pass
    
    def _check_safe_fail_patterns(self, content: str, result: ScriptValidationResult):
        """Check for safe-to-fail patterns."""
        lines = content.split('\n')
        
        has_main_function = ("def main():" in content or 
                             "create_standardized_pipeline_script(" in content)
        has_exit_codes = ("return 0" in content and "return 1" in content) or "run_script()" in content
        has_sys_exit = ("sys.exit(main())" in content or 
                       "sys.exit(run_script())" in content)
        
        if not has_main_function:
            result.issues.append(ValidationIssue(
                script_name=result.script_name,
                issue_type="missing_main",
                severity="error",
                line_number=None,
                message="No main() function found",
                suggestion="Add a main() function as the entry point"
            ))
        
        if not has_exit_codes:
            result.issues.append(ValidationIssue(
                script_name=result.script_name,
                issue_type="missing_exit_codes",
                severity="warning",
                line_number=None,
                message="Missing proper exit codes (return 0/1)",
                suggestion="Return 0 for success, 1 for failure"
            ))
        
        if not has_sys_exit:
            result.issues.append(ValidationIssue(
                script_name=result.script_name,
                issue_type="missing_sys_exit",
                severity="warning",
                line_number=None,
                message="Missing sys.exit(main()) pattern",
                suggestion="Add 'if __name__ == \"__main__\": sys.exit(main())'"
            ))
        
        result.has_safe_exit = has_exit_codes and has_sys_exit
    
    def generate_validation_report(self, results: Dict[str, ScriptValidationResult]) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        total_scripts = len(results)
        valid_scripts = len([r for r in results.values() if r.is_valid])
        invalid_scripts = total_scripts - valid_scripts
        
        # Count issues by type and severity
        issue_counts = {"error": 0, "warning": 0, "info": 0}
        issue_types = {}
        
        for result in results.values():
            for issue in result.issues:
                issue_counts[issue.severity] = issue_counts.get(issue.severity, 0) + 1
                issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1
        
        # Generate recommendations
        recommendations = []
        if issue_counts["error"] > 0:
            recommendations.append("Fix critical errors before deploying pipeline")
        if issue_types.get("missing_module", 0) > 0:
            recommendations.append("Create missing module directories and __init__.py files")
        if issue_types.get("missing_function", 0) > 0:
            recommendations.append("Implement missing functions in module __init__.py files")
        if issue_types.get("missing_error_handling", 0) > 0:
            recommendations.append("Add comprehensive try-except blocks to all scripts")
        
        report = {
            "timestamp": Path(__file__).stat().st_mtime,
            "summary": {
                "total_scripts": total_scripts,
                "valid_scripts": valid_scripts,
                "invalid_scripts": invalid_scripts,
                "validation_success_rate": (valid_scripts / total_scripts) * 100 if total_scripts > 0 else 0
            },
            "issue_summary": {
                "total_issues": sum(issue_counts.values()),
                "by_severity": issue_counts,
                "by_type": issue_types
            },
            "script_details": {},
            "recommendations": recommendations
        }
        
        # Add individual script details
        for script_name, result in results.items():
            report["script_details"][script_name] = {
                "is_valid": result.is_valid,
                "has_error_handling": result.has_error_handling,
                "has_logging": result.has_logging,
                "has_safe_exit": result.has_safe_exit,
                "issue_count": len(result.issues),
                "issues": [
                    {
                        "type": issue.issue_type,
                        "severity": issue.severity,
                        "line": issue.line_number,
                        "message": issue.message,
                        "suggestion": issue.suggestion
                    }
                    for issue in result.issues
                ]
            }
        
        return report
    
    def fix_common_issues(self, results: Dict[str, ScriptValidationResult]) -> Dict[str, List[str]]:
        """Generate automatic fixes for common issues."""
        fixes = {}
        
        for script_name, result in results.items():
            if not result.is_valid:
                script_fixes = []
                
                for issue in result.issues:
                    if issue.issue_type == "missing_module":
                        module_name = issue.message.split(": ")[-1]
                        script_fixes.append(f"mkdir -p src/{module_name}")
                        script_fixes.append(f"touch src/{module_name}/__init__.py")
                    
                    elif issue.issue_type == "missing_function":
                        # Extract function name and module from the message
                        parts = issue.message.split("'")
                        if len(parts) >= 2:
                            function_name = parts[1]
                            module_name = issue.message.split(" in ")[-1] if " in " in issue.message else "unknown"
                            script_fixes.append(
                                f"Add function stub: def {function_name}(*args, **kwargs): pass"
                            )
                
                if script_fixes:
                    fixes[script_name] = script_fixes
        
        return fixes

def validate_pipeline_scripts(src_dir: Path = None) -> Dict[str, Any]:
    """
    Validate all pipeline scripts and return comprehensive results.
    
    Args:
        src_dir: Source directory containing pipeline scripts
        
    Returns:
        Validation report with results and recommendations
    """
    validator = PipelineScriptValidator(src_dir)
    results = validator.validate_all_scripts()
    report = validator.generate_validation_report(results)
    fixes = validator.fix_common_issues(results)
    
    report["suggested_fixes"] = fixes
    
    return report

if __name__ == "__main__":
    # Run validation on the current src directory
    src_dir = Path(__file__).parent.parent
    report = validate_pipeline_scripts(src_dir)
    
    # Print summary
    print(f"\n=== Pipeline Script Validation Report ===")
    print(f"Total Scripts: {report['summary']['total_scripts']}")
    print(f"Valid Scripts: {report['summary']['valid_scripts']}")
    print(f"Invalid Scripts: {report['summary']['invalid_scripts']}")
    print(f"Success Rate: {report['summary']['validation_success_rate']:.1f}%")
    
    print(f"\n=== Issues Summary ===")
    for severity, count in report['issue_summary']['by_severity'].items():
        if count > 0:
            print(f"{severity.title()}: {count}")
    
    print(f"\n=== Recommendations ===")
    for rec in report['recommendations']:
        print(f"- {rec}")
    
    # Show script-specific issues
    print(f"\n=== Script Details ===")
    for script_name, details in report['script_details'].items():
        if not details['is_valid'] or details['issue_count'] > 0:
            print(f"\n{script_name}:")
            print(f"  Valid: {details['is_valid']}")
            print(f"  Issues: {details['issue_count']}")
            for issue in details['issues']:
                severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[issue['severity']]
                print(f"    {severity_icon} {issue['type']}: {issue['message']}")
                if issue['suggestion']:
                    print(f"      💡 {issue['suggestion']}") 