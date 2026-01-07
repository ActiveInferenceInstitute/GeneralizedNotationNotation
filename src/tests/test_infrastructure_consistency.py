#!/usr/bin/env python3
"""
Test Infrastructure Consistency
=============================

This test ensures that the test infrastructure remains modular, indexable, and complete.
It programmatically enforces the rule: "Every source module must have a corresponding test."
"""

import sys
import os
import re
from pathlib import Path
import pytest
import importlib.util

# Paths
FILE = Path(__file__)
TESTS_DIR = FILE.parent
SRC_DIR = TESTS_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

# Add src to path
sys.path.insert(0, str(SRC_DIR))

from tests.runner import MODULAR_TEST_CATEGORIES

class TestInfrastructureConsistency:
    """Enforce test infrastructure rules."""
    
    @pytest.fixture
    def source_modules(self):
        """Get all numbered Python modules in src/."""
        modules = []
        for f in SRC_DIR.glob("*.py"):
            if f.name.startswith("__") or f.name == "main.py":
                continue
            modules.append(f)
        return modules

    @pytest.mark.unit
    def test_all_modules_have_tests(self, source_modules):
        """
        Verify that every source module has a corresponding test file.
        
        Naming convention:
        src/X_module.py -> src/tests/test_module_overall.py
        src/module.py -> src/tests/test_module_overall.py
        """
        missing_tests = []
        
        for module_path in source_modules:
            module_name = module_path.stem
            
            # clean name: remove number prefix (e.g., 3_gnn -> gnn)
            clean_name = re.sub(r'^\d+_', '', module_name)
            
            # Skip setup/template files and scripts
            if clean_name in ['setup', 'template', 'pipeline_step_template', 'tests']:
                continue
                
            # Handle aliases (module name -> test name)
            aliases = {
                'advanced_viz': 'advanced_visualization',
                'ml_integration': 'ml_integration', # matches
                'gnn': 'gnn'
            }
            
            test_name = aliases.get(clean_name, clean_name)
            expected_test = TESTS_DIR / f"test_{test_name}_overall.py"
            
            if not expected_test.exists():
                missing_tests.append(f"{module_name} ({clean_name}) -> {expected_test.name}")
        
        assert not missing_tests, f"Missing test files for modules:\n" + "\n".join(missing_tests)

    @pytest.mark.unit
    def test_all_tests_registered_in_runner(self):
        """
        Verify that all test categories are registered in runner.py.
        """
        # Get all subdirectories in src/ that have Python files (potential modules)
        # Or just use the source modules list mapping
        
        # We can identify "modules" as directories in src/ or numbered files
        # Let's stick to the mapped categories in runner.py
        
        registered_categories = set(MODULAR_TEST_CATEGORIES.keys())
        
        # Check known modules
        known_modules = {
            'gnn', 'render', 'mcp', 'audio', 'visualization', 
            'pipeline', 'export', 'execute', 'llm', 'ontology',
            'website', 'report', 'environment', 'gui',
            'analysis', 'security', 'research', 'ml_integration',
            'advanced_visualization'
        }
        
        missing_registration = known_modules - registered_categories
        
        assert not missing_registration, f"Modules not registered in runner.py: {missing_registration}"

    @pytest.mark.unit
    def test_file_naming_conventions(self):
        """
        Verify that all files in tests/ directory follow the test_*.py convention
        (except for known utility/config files).
        """
        allowed_non_test_files = {
            '__init__.py', 'conftest.py', 'runner.py', 'run_fast_tests.py',
            'test_runner_helper.py', 'mcp.py', 'README.md', 'TEST_SUITE_SUMMARY.md',
            'AGENTS.md'
        }
        
        invalid_files = []
        for f in TESTS_DIR.glob("*"):
            if f.is_dir():
                continue
            
            if f.name in allowed_non_test_files:
                continue
                
            if not f.name.startswith("test_"):
                # Check if it's a python file
                if f.suffix == '.py':
                    invalid_files.append(f.name)
        
        assert not invalid_files, f"Test files must start with 'test_': {invalid_files}"

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
