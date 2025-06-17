#!/usr/bin/env python3
"""
Dependency Validation System for GNN Pipeline

This module provides comprehensive validation of all dependencies
before pipeline execution to prevent runtime import failures.
"""

import importlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DependencyValidator:
    """Validates all pipeline dependencies before execution."""
    
    def __init__(self):
        self.validation_results = {}
        self.critical_failures = []
        self.warnings = []
        
    def validate_all_dependencies(self) -> Tuple[bool, Dict[str, any]]:
        """
        Comprehensive validation of all pipeline dependencies.
        
        Returns:
            Tuple of (all_valid: bool, detailed_results: dict)
        """
        logger.info("🔍 Starting comprehensive dependency validation...")
        
        # Step-specific validations
        step_validations = {
            "1_gnn": self._validate_step_1_gnn,
            "2_setup": self._validate_step_2_setup,
            "3_tests": self._validate_step_3_tests,
            "4_gnn_type_checker": self._validate_step_4_type_checker,
            "5_export": self._validate_step_5_export,
            "6_visualization": self._validate_step_6_visualization,
            "7_mcp": self._validate_step_7_mcp,
            "8_ontology": self._validate_step_8_ontology,
            "9_render": self._validate_step_9_render,
            "10_execute": self._validate_step_10_execute,
            "11_llm": self._validate_step_11_llm,
            "12_discopy": self._validate_step_12_discopy,
            "13_discopy_jax": self._validate_step_13_discopy_jax,
            "14_site": self._validate_step_14_site
        }
        
        all_valid = True
        
        for step_name, validator_func in step_validations.items():
            try:
                is_valid, details = validator_func()
                self.validation_results[step_name] = {
                    "valid": is_valid,
                    "details": details
                }
                
                if not is_valid:
                    all_valid = False
                    self.critical_failures.append(f"Step {step_name}: {details}")
                    
            except Exception as e:
                all_valid = False
                error_msg = f"Validation failed for {step_name}: {str(e)}"
                self.critical_failures.append(error_msg)
                self.validation_results[step_name] = {
                    "valid": False,
                    "details": error_msg
                }
        
        # Generate summary report
        self._generate_validation_report()
        
        return all_valid, self.validation_results
    
    def _validate_module_import(self, module_name: str, optional: bool = False) -> Tuple[bool, str]:
        """Validate that a module can be imported."""
        try:
            importlib.import_module(module_name)
            return True, f"✅ {module_name} imported successfully"
        except ImportError as e:
            if optional:
                self.warnings.append(f"Optional module {module_name} not available: {e}")
                return True, f"⚠️ Optional {module_name} not available"
            else:
                return False, f"❌ Required module {module_name} failed: {e}"
    
    def _validate_file_exists(self, file_path: str) -> Tuple[bool, str]:
        """Validate that a required file exists."""
        path = Path(file_path)
        if path.exists():
            return True, f"✅ {file_path} exists"
        else:
            return False, f"❌ Required file {file_path} not found"
    
    def _validate_step_1_gnn(self) -> Tuple[bool, str]:
        """Validate dependencies for Step 1: GNN processing."""
        dependencies = [
            ("utils", False),
            ("pathlib", False)
        ]
        
        for module, optional in dependencies:
            valid, msg = self._validate_module_import(module, optional)
            if not valid:
                return False, f"Step 1 dependency issue: {msg}"
        
        return True, "Step 1 dependencies validated"
    
    def _validate_step_2_setup(self) -> Tuple[bool, str]:
        """Validate dependencies for Step 2: Setup."""
        return True, "Step 2 dependencies validated"  # Setup handles its own deps
    
    def _validate_step_3_tests(self) -> Tuple[bool, str]:
        """Validate dependencies for Step 3: Tests."""
        valid, msg = self._validate_module_import("pytest")
        return valid, f"Step 3: {msg}"
    
    def _validate_step_4_type_checker(self) -> Tuple[bool, str]:
        """Validate dependencies for Step 4: Type Checker."""
        valid, msg = self._validate_module_import("gnn_type_checker")
        return valid, f"Step 4: {msg}"
    
    def _validate_step_5_export(self) -> Tuple[bool, str]:
        """Validate dependencies for Step 5: Export."""
        dependencies = ["export", "json", "xml.etree.ElementTree"]
        
        for module in dependencies:
            valid, msg = self._validate_module_import(module)
            if not valid:
                return False, f"Step 5 dependency issue: {msg}"
        
        return True, "Step 5 dependencies validated"
    
    def _validate_step_6_visualization(self) -> Tuple[bool, str]:
        """Validate dependencies for Step 6: Visualization."""
        dependencies = [
            ("matplotlib", False),
            ("graphviz", False),
            ("networkx", False)
        ]
        
        for module, optional in dependencies:
            valid, msg = self._validate_module_import(module, optional)
            if not valid:
                return False, f"Step 6 dependency issue: {msg}"
        
        return True, "Step 6 dependencies validated"
    
    def _validate_step_7_mcp(self) -> Tuple[bool, str]:
        """Validate dependencies for Step 7: MCP."""
        valid, msg = self._validate_module_import("mcp", optional=True)
        return valid, f"Step 7: {msg}"
    
    def _validate_step_8_ontology(self) -> Tuple[bool, str]:
        """Validate dependencies for Step 8: Ontology."""
        # Check for ontology file
        valid, msg = self._validate_file_exists("src/ontology/act_inf_ontology_terms.json")
        return valid, f"Step 8: {msg}"
    
    def _validate_step_9_render(self) -> Tuple[bool, str]:
        """Validate dependencies for Step 9: Render."""
        # Check for render functions specifically
        try:
            from render import render_gnn_to_pymdp, render_gnn_to_rxinfer_toml
            return True, "✅ Render functions available"
        except ImportError as e:
            return False, f"❌ Render functions missing: {e}"
    
    def _validate_step_10_execute(self) -> Tuple[bool, str]:
        """Validate dependencies for Step 10: Execute."""
        dependencies = [
            ("pymdp", True),  # Optional
            ("subprocess", False)
        ]
        
        missing_optional = []
        for module, optional in dependencies:
            valid, msg = self._validate_module_import(module, optional)
            if not valid and not optional:
                return False, f"Step 10 dependency issue: {msg}"
            elif not valid and optional:
                missing_optional.append(module)
        
        if missing_optional:
            return True, f"⚠️ Optional dependencies missing: {', '.join(missing_optional)}"
        
        return True, "Step 10 dependencies validated"
    
    def _validate_step_11_llm(self) -> Tuple[bool, str]:
        """Validate dependencies for Step 11: LLM."""
        dependencies = [
            ("openai", True),
            ("python-dotenv", True)
        ]
        
        # Check for LLM modules
        try:
            from llm import llm_operations
            llm_available = True
        except ImportError:
            llm_available = False
        
        if not llm_available:
            return False, "❌ LLM operations module not available"
        
        return True, "✅ LLM dependencies validated"
    
    def _validate_step_12_discopy(self) -> Tuple[bool, str]:
        """Validate dependencies for Step 12: DisCoPy."""
        valid, msg = self._validate_module_import("discopy", optional=True)
        
        # Check translator module
        try:
            from discopy_translator_module.translator import gnn_file_to_discopy_diagram
            translator_available = True
        except ImportError:
            translator_available = False
        
        if not translator_available:
            return False, "❌ DisCoPy translator module not available"
        
        return valid, f"Step 12: {msg}"
    
    def _validate_step_13_discopy_jax(self) -> Tuple[bool, str]:
        """Validate dependencies for Step 13: DisCoPy JAX."""
        dependencies = [
            ("jax", True),
            ("jaxlib", True),
            ("discopy", True)
        ]
        
        for module, optional in dependencies:
            valid, msg = self._validate_module_import(module, optional)
            if not valid:
                return False, f"Step 13 dependency issue: {msg}"
        
        return True, "Step 13 dependencies validated"
    
    def _validate_step_14_site(self) -> Tuple[bool, str]:
        """Validate dependencies for Step 14: Site generation."""
        try:
            from site.generator import generate_html_report
            return True, "✅ Site generator available"
        except ImportError:
            return True, "⚠️ Site generator not available, will use fallback"
    
    def _generate_validation_report(self):
        """Generate a comprehensive validation report."""
        logger.info("\n" + "="*60)
        logger.info("🔍 DEPENDENCY VALIDATION REPORT")
        logger.info("="*60)
        
        # Summary
        total_steps = len(self.validation_results)
        valid_steps = sum(1 for result in self.validation_results.values() if result["valid"])
        
        logger.info(f"📊 Summary: {valid_steps}/{total_steps} steps have valid dependencies")
        
        if self.critical_failures:
            logger.error(f"❌ Critical Failures ({len(self.critical_failures)}):")
            for failure in self.critical_failures:
                logger.error(f"   {failure}")
        
        if self.warnings:
            logger.warning(f"⚠️ Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"   {warning}")
        
        # Detailed results
        logger.info("\n📋 Detailed Results:")
        for step, result in self.validation_results.items():
            status = "✅" if result["valid"] else "❌"
            logger.info(f"   {status} {step}: {result['details']}")
        
        logger.info("="*60)

def validate_pipeline_dependencies() -> bool:
    """
    Main function to validate all pipeline dependencies.
    
    Returns:
        bool: True if all critical dependencies are available
    """
    validator = DependencyValidator()
    all_valid, results = validator.validate_all_dependencies()
    
    if not all_valid:
        logger.error("❌ Pipeline validation failed. Fix dependencies before running.")
        return False
    
    logger.info("✅ All pipeline dependencies validated successfully!")
    return True

if __name__ == "__main__":
    import sys
    if not validate_pipeline_dependencies():
        sys.exit(1)
    print("✅ Dependency validation passed!") 