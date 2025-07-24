#!/usr/bin/env python3
"""
Pipeline Verification Script

This script verifies that the complete GNN Processing Pipeline is working correctly.
"""

import sys
from pathlib import Path
import json
from typing import Dict, Any, List

def verify_pipeline_discovery() -> Dict[str, Any]:
    """Verify pipeline step discovery."""
    try:
        from pipeline.discovery import get_pipeline_scripts
        scripts = get_pipeline_scripts(Path(__file__).parent)
        
        expected_steps = list(range(22))  # 0-21
        found_steps = [s['num'] for s in scripts]
        
        return {
            "success": found_steps == expected_steps,
            "expected_count": len(expected_steps),
            "found_count": len(found_steps),
            "missing_steps": set(expected_steps) - set(found_steps),
            "extra_steps": set(found_steps) - set(expected_steps),
            "scripts": [s['basename'] for s in sorted(scripts, key=lambda x: x['num'])]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def verify_module_imports() -> Dict[str, Any]:
    """Verify that all modules can be imported."""
    modules_to_test = [
        "utils",
        "pipeline",
        "type_checker",
        "export",
        "tests",
        "setup",
        "gnn",
        "model_registry",
        "validation",
        "visualization",
        "advanced_visualization",
        "ontology",
        "render",
        "execute",
        "llm",
        "ml_integration",
        "audio",
        "analysis",
        "integration",
        "security",
        "research",
        "website",
        "report"
    ]
    
    results = {}
    failed_imports = []
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            results[module_name] = True
        except ImportError as e:
            results[module_name] = False
            failed_imports.append(f"{module_name}: {e}")
    
    return {
        "success": len(failed_imports) == 0,
        "total_modules": len(modules_to_test),
        "successful_imports": sum(results.values()),
        "failed_imports": len(failed_imports),
        "failed_modules": failed_imports,
        "results": results
    }

def verify_pipeline_config() -> Dict[str, Any]:
    """Verify pipeline configuration."""
    try:
        from pipeline import get_pipeline_config
        config = get_pipeline_config()
        
        return {
            "success": True,
            "total_steps": len(config.steps),
            "has_template": "0_template.py" in config.steps,
            "has_setup": "1_setup.py" in config.steps,
            "has_tests": "2_tests.py" in config.steps,
            "has_main": "main.py" in config.steps
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def verify_step_files() -> Dict[str, Any]:
    """Verify that all step files exist."""
    expected_files = [f"{i}_" + name for i, name in enumerate([
        "template.py",
        "setup.py", 
        "tests.py",
        "gnn.py",
        "model_registry.py",
        "type_checker.py",
        "validation.py",
        "export.py",
        "visualization.py",
        "advanced_viz.py",
        "ontology.py",
        "render.py",
        "execute.py",
        "llm.py",
        "ml_integration.py",
        "audio.py",
        "analysis.py",
        "integration.py",
        "security.py",
        "research.py",
        "website.py",
        "report.py"
    ])]
    
    existing_files = []
    missing_files = []
    
    for filename in expected_files:
        file_path = Path(__file__).parent / filename
        if file_path.exists():
            existing_files.append(filename)
        else:
            missing_files.append(filename)
    
    return {
        "success": len(missing_files) == 0,
        "expected_count": len(expected_files),
        "existing_count": len(existing_files),
        "missing_count": len(missing_files),
        "missing_files": missing_files,
        "existing_files": existing_files
    }

def verify_mcp_integration() -> Dict[str, Any]:
    """Verify MCP integration files."""
    modules_with_mcp = [
        "tests",
        "type_checker", 
        "export",
        "setup"
    ]
    
    results = {}
    missing_mcp = []
    
    for module_name in modules_with_mcp:
        mcp_file = Path(__file__).parent / module_name / "mcp.py"
        if mcp_file.exists():
            results[module_name] = True
        else:
            results[module_name] = False
            missing_mcp.append(module_name)
    
    return {
        "success": len(missing_mcp) == 0,
        "total_modules": len(modules_with_mcp),
        "with_mcp": len([r for r in results.values() if r]),
        "missing_mcp": len(missing_mcp),
        "missing_modules": missing_mcp,
        "results": results
    }

def verify_test_modules() -> Dict[str, Any]:
    """Verify test modules."""
    expected_test_files = [
        "__init__.py",
        "unit_tests.py",
        "integration_tests.py", 
        "performance_tests.py",
        "coverage_tests.py",
        "mcp.py"
    ]
    
    test_dir = Path(__file__).parent / "tests"
    existing_files = []
    missing_files = []
    
    for filename in expected_test_files:
        file_path = test_dir / filename
        if file_path.exists():
            existing_files.append(filename)
        else:
            missing_files.append(filename)
    
    return {
        "success": len(missing_files) == 0,
        "expected_count": len(expected_test_files),
        "existing_count": len(existing_files),
        "missing_count": len(missing_files),
        "missing_files": missing_files,
        "existing_files": existing_files
    }

def main():
    """Main verification function."""
    print("🔍 GNN Processing Pipeline Verification")
    print("=" * 50)
    
    verification_results = {}
    
    # Run all verifications
    print("\n1. Verifying pipeline discovery...")
    verification_results["pipeline_discovery"] = verify_pipeline_discovery()
    
    print("2. Verifying module imports...")
    verification_results["module_imports"] = verify_module_imports()
    
    print("3. Verifying pipeline configuration...")
    verification_results["pipeline_config"] = verify_pipeline_config()
    
    print("4. Verifying step files...")
    verification_results["step_files"] = verify_step_files()
    
    print("5. Verifying MCP integration...")
    verification_results["mcp_integration"] = verify_mcp_integration()
    
    print("6. Verifying test modules...")
    verification_results["test_modules"] = verify_test_modules()
    
    # Print results
    print("\n📊 Verification Results")
    print("=" * 50)
    
    all_successful = True
    for test_name, result in verification_results.items():
        status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
        print(f"{test_name:20} {status}")
        
        if not result.get("success", False):
            all_successful = False
            if "error" in result:
                print(f"    Error: {result['error']}")
            if "missing_files" in result and result["missing_files"]:
                print(f"    Missing: {result['missing_files']}")
            if "failed_modules" in result and result["failed_modules"]:
                print(f"    Failed: {result['failed_modules']}")
    
    # Overall status
    print("\n" + "=" * 50)
    if all_successful:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("✅ The GNN Processing Pipeline is ready for use.")
        return 0
    else:
        print("❌ SOME VERIFICATIONS FAILED!")
        print("⚠️  Please check the issues above before using the pipeline.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 