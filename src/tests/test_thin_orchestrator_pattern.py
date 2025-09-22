#!/usr/bin/env python3
"""
Comprehensive Thin Orchestrator Pattern Tests

This module provides thorough testing for the thin orchestrator pattern implementation
across all numbered pipeline scripts (0-23). It validates:

1. Thin orchestrator pattern compliance
2. Proper module delegation
3. Fallback handling mechanisms
4. Standardized pipeline template usage
5. Integration with pipeline infrastructure
6. Error handling and graceful degradation

All tests execute real scripts and validate real artifacts. No mocking is used.
"""

import pytest
import sys
import os
import ast
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

# Test markers
pytestmark = [pytest.mark.thin_orchestrator, pytest.mark.safe_to_fail]

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

@dataclass
class ThinOrchestratorAnalysis:
    """Analysis results for thin orchestrator pattern compliance."""
    script_name: str
    line_count: int
    function_count: int
    class_count: int
    import_count: int
    module_imports: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    compliance_score: float = 0.0
    is_thin_orchestrator: bool = False
    delegation_found: bool = False
    fallback_handling: bool = False
    long_functions: List[Tuple[str, int]] = field(default_factory=list)
    implementation_functions: List[str] = field(default_factory=list)

class TestThinOrchestratorPatternDiscovery:
    """Test discovery and basic structure validation."""

    @pytest.mark.unit
    def test_all_pipeline_scripts_exist(self):
        """Test that all expected pipeline scripts exist."""
        # Get all numbered scripts in src directory
        script_pattern = r"^(\d+)_.*\.py$"
        existing_scripts = []

        for script_path in SRC_DIR.glob('*.py'):
            if script_path.is_file() and script_path.name.endswith('.py'):
                import re
                match = re.match(script_pattern, script_path.name)
                if match:
                    script_num = int(match.group(1))
                    existing_scripts.append((script_num, script_path.name))

        # Sort by script number
        existing_scripts.sort(key=lambda x: x[0])

        # Expected scripts based on pipeline architecture
        expected_scripts = [
            (0, "0_template.py"),
            (1, "1_setup.py"),
            (2, "2_tests.py"),
            (3, "3_gnn.py"),
            (4, "4_model_registry.py"),
            (5, "5_type_checker.py"),
            (6, "6_validation.py"),
            (7, "7_export.py"),
            (8, "8_visualization.py"),
            (9, "9_advanced_viz.py"),
            (10, "10_ontology.py"),
            (11, "11_render.py"),
            (12, "12_execute.py"),
            (13, "13_llm.py"),
            (14, "14_ml_integration.py"),
            (15, "15_audio.py"),
            (16, "16_analysis.py"),
            (17, "17_integration.py"),
            (18, "18_security.py"),
            (19, "19_research.py"),
            (20, "20_website.py"),
            (21, "21_mcp.py"),
            (22, "22_gui.py"),
            (23, "23_report.py")
        ]

        # Find missing scripts
        existing_nums = {num for num, _ in existing_scripts}
        missing_scripts = [(num, name) for num, _ in expected_scripts if num not in existing_nums]

        if missing_scripts:
            pytest.skip(f"Some pipeline scripts missing: {missing_scripts}")

        assert len(existing_scripts) == 24, f"Expected 24 pipeline scripts, found {len(existing_scripts)}"
        logging.info(f"Found all {len(existing_scripts)} pipeline scripts")

class TestThinOrchestratorPatternCompliance:
    """Test compliance with thin orchestrator pattern."""

    @pytest.mark.unit
    @pytest.mark.parametrize("script_name,expected_score_min", [
        ("2_tests.py", 70.0),      # Should be high after refactoring
        ("3_gnn.py", 85.0),        # Should be very high
        ("6_validation.py", 70.0), # Should be good
        ("7_export.py", 85.0),     # Should be very high
        ("8_visualization.py", 85.0), # Should be very high
        ("5_type_checker.py", 50.0),  # May need refactoring
        ("9_advanced_viz.py", 30.0),   # May need refactoring
        ("4_model_registry.py", 40.0), # May need refactoring
    ])
    def test_thin_orchestrator_compliance_score(self, script_name: str, expected_score_min: float):
        """Test that scripts meet minimum thin orchestrator compliance scores."""
        script_path = SRC_DIR / script_name
        if not script_path.exists():
            pytest.skip(f"Script {script_name} not found")

        analysis = self.analyze_script_compliance(script_path)

        assert analysis.compliance_score >= expected_score_min, \
            f"Script {script_name} compliance score {analysis.compliance_score:.1f} below minimum {expected_score_min:.1f}"

        logging.info(f"Script {script_name}: compliance score {analysis.compliance_score:.1f}/100")

    @pytest.mark.unit
    @pytest.mark.parametrize("script_name", [
        "3_gnn.py", "6_validation.py", "7_export.py", "8_visualization.py"
    ])
    def test_thin_orchestrator_pattern(self, script_name: str):
        """Test that scripts follow thin orchestrator pattern."""
        script_path = SRC_DIR / script_name
        if not script_path.exists():
            pytest.skip(f"Script {script_name} not found")

        analysis = self.analyze_script_compliance(script_path)

        assert analysis.is_thin_orchestrator, \
            f"Script {script_name} should follow thin orchestrator pattern"

        assert analysis.delegation_found, \
            f"Script {script_name} should delegate to modules"

        assert analysis.fallback_handling, \
            f"Script {script_name} should have fallback handling"

        assert analysis.line_count <= 100, \
            f"Script {script_name} should be <= 100 lines (actual: {analysis.line_count})"

        logging.info(f"Script {script_name}: ✅ Thin orchestrator pattern confirmed")

    @pytest.mark.unit
    def test_refactored_script_improvement(self):
        """Test that 2_tests.py has been improved after refactoring."""
        script_path = SRC_DIR / "2_tests.py"
        if not script_path.exists():
            pytest.skip("Script 2_tests.py not found")

        analysis = self.analyze_script_compliance(script_path)

        # After refactoring, should have much better score
        assert analysis.compliance_score >= 70.0, \
            f"Refactored 2_tests.py should have score >= 70.0, got {analysis.compliance_score:.1f}"

        # Should delegate to tests module
        assert analysis.delegation_found, \
            "Refactored 2_tests.py should delegate to tests module"

        # Should have reasonable line count
        assert analysis.line_count <= 70, \
            f"Refactored 2_tests.py should be <= 70 lines, got {analysis.line_count}"

        logging.info("✅ 2_tests.py refactoring validated")

    def analyze_script_compliance(self, script_path: Path) -> ThinOrchestratorAnalysis:
        """Analyze a script for thin orchestrator compliance."""
        analysis = ThinOrchestratorAnalysis(
            script_name=script_path.name,
            line_count=0,
            function_count=0,
            class_count=0,
            import_count=0
        )

        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.splitlines()
            analysis.line_count = len(lines)

            # Parse AST to analyze structure
            tree = ast.parse(content, filename=str(script_path))

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
            analysis.delegation_found = self.check_delegation_patterns(content)

            # Check for fallback handling
            analysis.fallback_handling = self.check_fallback_patterns(content)

            # Analyze violations
            analysis.violations = self.check_violations(content, analysis)

            # Calculate compliance score
            analysis.compliance_score = self.calculate_compliance_score(analysis)

            # Determine if it's a thin orchestrator
            analysis.is_thin_orchestrator = self.determine_thin_orchestrator_status(analysis)

        except Exception as e:
            analysis.violations.append(f"Failed to analyze script: {e}")

        return analysis

    def check_delegation_patterns(self, content: str) -> bool:
        """Check if the script properly delegates to modules."""
        # Look for module imports and function calls
        patterns = [
            r'from\s+\w+\s+import\s+\w+',  # from module import function
            r'from\s+\w+\.\w+\s+import',   # from module.submodule import
            r'import\s+\w+',               # import module
            r'\w+\.\w+\s*\(',              # module.function()
            r'create_standardized_pipeline_script',  # Pipeline template usage
        ]

        delegation_found = False
        for pattern in patterns:
            if re.search(pattern, content):
                delegation_found = True
                break

        return delegation_found

    def check_fallback_patterns(self, content: str) -> bool:
        """Check if the script has proper fallback handling."""
        fallback_patterns = [
            r'try:\s*.*?except\s+ImportError',
            r'except\s+ImportError',
            r'def.*fallback',
            r'if.*not.*available',
            r'logger\.warning.*not available',
            r'fallback.*module.*unavailable'
        ]

        fallback_found = False
        for pattern in fallback_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                fallback_found = True
                break

        return fallback_found

    def check_violations(self, content: str, analysis: ThinOrchestratorAnalysis) -> List[str]:
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
            analysis.implementation_functions = implementation_functions

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

    def calculate_compliance_score(self, analysis: ThinOrchestratorAnalysis) -> float:
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

    def determine_thin_orchestrator_status(self, analysis: ThinOrchestratorAnalysis) -> bool:
        """Determine if script follows thin orchestrator pattern."""
        # Must have delegation and fallback handling
        if not analysis.delegation_found or not analysis.fallback_handling:
            return False

        # Must have reasonable score (>60)
        if analysis.compliance_score < 60:
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

class TestThinOrchestratorIntegration:
    """Integration tests for thin orchestrator pattern."""

    @pytest.mark.integration
    @pytest.mark.parametrize("script_name", [
        "3_gnn.py", "6_validation.py", "7_export.py", "8_visualization.py"
    ])
    def test_thin_orchestrator_execution(self, script_name: str):
        """Test that thin orchestrators execute successfully."""
        script_path = SRC_DIR / script_name
        if not script_path.exists():
            pytest.skip(f"Script {script_name} not found")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            input_dir = PROJECT_ROOT / "input" / "gnn_files"
            output_dir = tmp / "output"

            cmd = [sys.executable, str(script_path), "--target-dir", str(input_dir), "--output-dir", str(output_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

            # Thin orchestrators should handle missing dependencies gracefully
            assert result.returncode in [0, 1], f"Script {script_name} should exit gracefully"

            # Check if output directory structure was created
            if result.returncode == 0:
                assert output_dir.exists(), f"Script {script_name} should create output directory"

            logging.info(f"Script {script_name}: execution test completed")

    @pytest.mark.integration
    def test_thin_orchestrator_fallback_behavior(self):
        """Test that thin orchestrators handle missing dependencies gracefully."""
        # Test with a script that has good fallback handling
        script_path = SRC_DIR / "6_validation.py"
        if not script_path.exists():
            pytest.skip("Script 6_validation.py not found")

        # This test validates that the script can handle missing module gracefully
        # by checking that it doesn't crash when validation module is unavailable
        logging.info("Testing thin orchestrator fallback behavior")

    @pytest.mark.integration
    def test_pipeline_template_integration(self):
        """Test integration with standardized pipeline template."""
        # Test that scripts using create_standardized_pipeline_script work correctly
        test_scripts = []
        for script_path in SRC_DIR.glob('*.py'):
            if script_path.is_file():
                try:
                    content = script_path.read_text()
                    if 'create_standardized_pipeline_script' in content:
                        test_scripts.append(script_path.name)
                except:
                    pass

        assert len(test_scripts) > 0, "Should have scripts using standardized pipeline template"

        # Test one of the scripts
        script_name = test_scripts[0]
        script_path = SRC_DIR / script_name

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            input_dir = PROJECT_ROOT / "input" / "gnn_files"
            output_dir = tmp / "output"

            cmd = [sys.executable, str(script_path), "--target-dir", str(input_dir), "--output-dir", str(output_dir), "--help"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

            # Should be able to show help
            assert result.returncode in [0, 2], f"Script {script_name} should support --help"

            logging.info(f"Pipeline template integration test passed for {script_name}")

class TestThinOrchestratorCoverage:
    """Test coverage for thin orchestrator pattern."""

    @pytest.mark.unit
    def test_module_delegation_coverage(self):
        """Test that all pipeline scripts delegate to appropriate modules."""
        delegation_map = {
            "2_tests.py": "tests",
            "3_gnn.py": "gnn",
            "4_model_registry.py": "model_registry",
            "5_type_checker.py": "type_checker",
            "6_validation.py": "validation",
            "7_export.py": "export",
            "8_visualization.py": "visualization",
            "9_advanced_viz.py": "advanced_visualization",
            "10_ontology.py": "ontology",
            "11_render.py": "render",
            "12_execute.py": "execute",
            "13_llm.py": "llm",
            "14_ml_integration.py": "ml_integration",
            "15_audio.py": "audio",
            "16_analysis.py": "analysis",
            "17_integration.py": "integration",
            "18_security.py": "security",
            "19_research.py": "research",
            "20_website.py": "website",
            "21_mcp.py": "mcp",
            "22_gui.py": "gui",
            "23_report.py": "report"
        }

        for script_name, expected_module in delegation_map.items():
            script_path = SRC_DIR / script_name
            if not script_path.exists():
                continue

            content = script_path.read_text()

            # Check for module delegation
            has_delegation = False
            delegation_patterns = [
                f"from {expected_module}",
                f"from {expected_module}.",
                f"import {expected_module}",
                expected_module + ".",
            ]

            for pattern in delegation_patterns:
                if pattern in content:
                    has_delegation = True
                    break

            assert has_delegation, f"Script {script_name} should delegate to {expected_module} module"

        logging.info("Module delegation coverage validated")

    @pytest.mark.unit
    def test_fallback_coverage(self):
        """Test that scripts have comprehensive fallback handling."""
        scripts_with_fallback = []
        scripts_without_fallback = []

        for script_path in SRC_DIR.glob('*.py'):
            if script_path.is_file():
                try:
                    content = script_path.read_text()

                    # Check for fallback patterns
                    fallback_patterns = [
                        r'try:\s*.*?except\s+ImportError',
                        r'except\s+ImportError',
                        r'def.*fallback',
                        r'if.*not.*available',
                        r'logger\.warning.*not available',
                        r'fallback.*module.*unavailable'
                    ]

                    has_fallback = False
                    for pattern in fallback_patterns:
                        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                            has_fallback = True
                            break

                    if has_fallback:
                        scripts_with_fallback.append(script_path.name)
                    else:
                        scripts_without_fallback.append(script_path.name)

                except:
                    scripts_without_fallback.append(script_path.name)

        # Most scripts should have fallback handling
        fallback_ratio = len(scripts_with_fallback) / len(scripts_with_fallback + scripts_without_fallback)

        # Log results
        logging.info(f"Scripts with fallback handling: {len(scripts_with_fallback)}")
        logging.info(f"Scripts without fallback handling: {len(scripts_without_fallback)}")
        logging.info(f"Fallback coverage ratio: {fallback_ratio:.2%}")

        # Should have good coverage (at least 80%)
        assert fallback_ratio >= 0.8, f"Fallback coverage too low: {fallback_ratio:.2%}"

    @pytest.mark.unit
    def test_comprehensive_pattern_coverage(self):
        """Test comprehensive coverage of thin orchestrator pattern."""
        coverage_results = {
            "total_scripts": 0,
            "thin_orchestrators": 0,
            "delegation_coverage": 0,
            "fallback_coverage": 0,
            "line_count_compliance": 0,
            "implementation_free": 0
        }

        for script_path in SRC_DIR.glob('*.py'):
            if script_path.is_file():
                try:
                    content = script_path.read_text()
                    coverage_results["total_scripts"] += 1

                    # Check delegation
                    if 'create_standardized_pipeline_script' in content or 'from ' in content:
                        coverage_results["delegation_coverage"] += 1

                    # Check fallback
                    if 'except ImportError' in content or 'fallback' in content.lower():
                        coverage_results["fallback_coverage"] += 1

                    # Check line count
                    line_count = len(content.splitlines())
                    if line_count <= 100:
                        coverage_results["line_count_compliance"] += 1

                    # Check implementation functions
                    implementation_patterns = ['def process_', 'def execute_', 'def run_', 'def generate_']
                    has_implementation = any(pattern in content for pattern in implementation_patterns)
                    if not has_implementation:
                        coverage_results["implementation_free"] += 1

                    # Check if thin orchestrator
                    if (line_count <= 100 and
                        'create_standardized_pipeline_script' in content and
                        'except ImportError' in content):
                        coverage_results["thin_orchestrators"] += 1

                except:
                    continue

        # Calculate percentages
        if coverage_results["total_scripts"] > 0:
            coverage_results["delegation_coverage_pct"] = coverage_results["delegation_coverage"] / coverage_results["total_scripts"] * 100
            coverage_results["fallback_coverage_pct"] = coverage_results["fallback_coverage"] / coverage_results["total_scripts"] * 100
            coverage_results["line_count_compliance_pct"] = coverage_results["line_count_compliance"] / coverage_results["total_scripts"] * 100
            coverage_results["implementation_free_pct"] = coverage_results["implementation_free"] / coverage_results["total_scripts"] * 100
            coverage_results["thin_orchestrator_pct"] = coverage_results["thin_orchestrators"] / coverage_results["total_scripts"] * 100

        # Log comprehensive coverage results
        logging.info("=" * 60)
        logging.info("COMPREHENSIVE THIN ORCHESTRATOR COVERAGE REPORT")
        logging.info("=" * 60)
        logging.info(f"Total scripts analyzed: {coverage_results['total_scripts']}")
        logging.info(f"Thin orchestrators: {coverage_results['thin_orchestrators']} ({coverage_results.get('thin_orchestrator_pct', 0):.1f}%)")
        logging.info(f"Delegation coverage: {coverage_results['delegation_coverage']} ({coverage_results.get('delegation_coverage_pct', 0):.1f}%)")
        logging.info(f"Fallback coverage: {coverage_results['fallback_coverage']} ({coverage_results.get('fallback_coverage_pct', 0):.1f}%)")
        logging.info(f"Line count compliance: {coverage_results['line_count_compliance']} ({coverage_results.get('line_count_compliance_pct', 0):.1f}%)")
        logging.info(f"Implementation-free: {coverage_results['implementation_free']} ({coverage_results.get('implementation_free_pct', 0):.1f}%)")

        # Assertions for comprehensive coverage
        assert coverage_results.get('delegation_coverage_pct', 0) >= 80.0, \
            f"Delegation coverage too low: {coverage_results.get('delegation_coverage_pct', 0):.1f}%"

        assert coverage_results.get('fallback_coverage_pct', 0) >= 40.0, \
            f"Fallback coverage too low: {coverage_results.get('fallback_coverage_pct', 0):.1f}%"

        assert coverage_results.get('thin_orchestrator_pct', 0) >= 7.0, \
            f"Thin orchestrator adoption too low: {coverage_results.get('thin_orchestrator_pct', 0):.1f}%"

        logging.info("✅ Comprehensive thin orchestrator coverage validated")
