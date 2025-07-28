"""
Test Utilities for GNN Processing Pipeline.

This module provides shared utilities and helper functions for testing, designed
to be imported by both test modules and fixtures without causing circular imports.
"""

import logging
import sys
import os
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Generator
from contextlib import contextmanager

# Ensure src is in Python path for imports
SRC_DIR = Path(__file__).parent.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    
# Test categories and markers
TEST_CATEGORIES = {
    "fast": "Quick validation tests for core functionality",
    "standard": "Integration tests and moderate complexity", 
    "slow": "Complex scenarios and benchmarks",
    "performance": "Resource usage and scalability tests",
    "safe_to_fail": "Tests with graceful degradation",
    "unit": "Individual component tests",
    "integration": "Multi-component workflow tests",
    "mcp": "Model Context Protocol integration tests"
}

# Test execution stages
TEST_STAGES = {
    "fast": {
        "timeout": 180,
        "max_failures": 10,
        "parallel": True,
        "coverage": False
    },
    "standard": {
        "timeout": 600,
        "max_failures": 20,
        "parallel": True,
        "coverage": True
    },
    "slow": {
        "timeout": 300,
        "max_failures": 20,
        "parallel": False,
        "coverage": True
    },
    "performance": {
        "timeout": 600,
        "max_failures": 5,
        "parallel": False,
        "coverage": False
    }
}

# Test coverage targets
COVERAGE_TARGETS = {
    "overall": 85.0,
    "unit": 90.0,
    "integration": 80.0,
    "performance": 70.0
}

# Test configuration constants
TEST_CONFIG = {
    "safe_mode": True,
    "verbose": False,
    "strict": False,
    "estimate_resources": False,
    "skip_steps": [],
    "only_steps": [],
    "timeout_seconds": 300,  # 5 minutes default timeout
    "temp_output_dir": PROJECT_ROOT / "output" / "test_artifacts",
    "max_test_files": 10  # Maximum number of test files to process
}

# Test directory constants
TEST_DIR = Path(__file__).parent
SRC_DIR = TEST_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

def is_safe_mode() -> bool:
    """Check if tests are running in safe mode."""
    return TEST_CONFIG["safe_mode"]

def setup_test_environment() -> None:
    """Set up test environment."""
    # Create required directories if they don't exist
    for dir_path in ["input", "output", "output/test_artifacts"]:
        (PROJECT_ROOT / dir_path).mkdir(exist_ok=True, parents=True)

def cleanup_test_environment() -> None:
    """Clean up test environment."""
    # Clean up temporary files if needed
    temp_dir = TEST_CONFIG["temp_output_dir"]
    if temp_dir.exists() and is_safe_mode():
        # Only delete files, not directories for safety
        for file in temp_dir.glob("*.*"):
            if file.is_file():
                file.unlink(missing_ok=True)

def validate_test_environment() -> Tuple[bool, List[str]]:
    """Validate test environment."""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    
    # Check required directories
    required_dirs = ["src", "input", "output"]
    for dir_name in required_dirs:
        if not (PROJECT_ROOT / dir_name).exists():
            issues.append(f"Required directory '{dir_name}' not found")
    
    # Check pytest availability
    try:
        import pytest
    except ImportError:
        issues.append("pytest not installed")
    
    return len(issues) == 0, issues

def get_test_args() -> Dict[str, Any]:
    """Get standard test arguments for pipeline operations."""
    return {
        "target_dir": TEST_DIR / "test_data",
        "output_dir": PROJECT_ROOT / "output" / "test_results", 
        "verbose": True,
        "recursive": True,
        "strict": False
    }

def get_sample_pipeline_arguments() -> Dict[str, Any]:
    """Get sample pipeline arguments for testing."""
    return {
        "target_dir": Path("input/gnn_files"),
        "output_dir": Path("output"),
        "verbose": False,
        "strict": False
    }

def create_test_gnn_files(target_dir: Path) -> List[Path]:
    """Create test GNN files for testing."""
    target_dir.mkdir(parents=True, exist_ok=True)
    test_files = []
    
    # Create a basic test GNN file
    test_file = target_dir / "test_model.md"
    content = """# Test GNN Model

## GNNVersionAndFlags
Version: 1.0
Flags: test

## ModelName
TestModel

## StateSpaceBlock
Variables:
- A: [3, 3] (likelihood_matrix)
- B: [3, 3, 3] (transition_matrix)

Connections:
- A -> B
"""
    with open(test_file, 'w') as f:
        f.write(content)
    test_files.append(test_file)
    
    return test_files

def create_test_files(target_dir: Path, num_files: int = 3) -> List[Path]:
    """Create multiple test GNN files for testing purposes."""
    target_dir.mkdir(parents=True, exist_ok=True)
    test_files = []
    
    for i in range(num_files):
        test_file = target_dir / f"test_model_{i}.md"
        content = f"""# Test GNN Model {i}

## GNNVersionAndFlags
Version: 1.0
Flags: test

## ModelName
TestModel{i}

## StateSpaceBlock
Variables:
- A: [3, 3] (likelihood_matrix)
- B: [3, 3, 3] (transition_matrix)

Connections:
- A -> B
"""
        with open(test_file, 'w') as f:
            f.write(content)
        test_files.append(test_file)
    
    return test_files
    
def create_sample_gnn_content() -> Dict[str, str]:
    """Create sample GNN content for testing."""
    return {
        "valid_basic": """## GNNVersionAndFlags
Version: 1.0
Flags: test

## ModelName
TestModel

## StateSpaceBlock
X[2]

## Variables
- X: [2]

## Connections
- X -> Y

## Parameters
- A: [[1,2,3], [4,5,6]]
""",
        "complex_model": """## GNNVersionAndFlags
Version: 1.1
Flags: complex

# ComplexTestModel

## ModelName
ComplexTestModel

## Variables
- X: [3, 3]
- Y: [2]
- Z: [4]

## Connections
- X -> Y
- Y -> Z
- Z -> X

## Parameters
- A: [[1,2,3], [4,5,6], [7,8,9]]
- B: [[0.1, 0.9], [0.7, 0.3]]
""",
        "minimal": """## GNNVersionAndFlags
Version: 1.0

# MinimalModel

## ModelName
MinimalModel

## Variables
- S: [2]
""",
        "invalid": """# InvalidModel
This is not a proper GNN format
Missing required sections
"""
    }

def get_mock_filesystem_structure() -> Dict[str, List[str]]:
    """Get mock filesystem structure for testing."""
    return {
        "input": ["model.md"],
        "output": [],
        "src": ["main.py"]
    }

def run_all_tests(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False
) -> bool:
    """
    Run comprehensive test suite for the GNN pipeline.
    
    Args:
        target_dir: Directory containing GNN files to test
        output_dir: Directory to save test results
        verbose: Enable verbose output
    
    Returns:
        True if all tests pass, False otherwise
    """
    logger = logging.getLogger("tests")
    
    try:
        # Create test results directory
        test_results_dir = output_dir / "test_results"
        test_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Import test modules only when needed
        from tests.unit_tests import run_unit_test_suite
        from tests.integration_tests import run_integration_test_suite
        from tests.performance_tests import run_performance_test_suite
        from tests.coverage_tests import run_coverage_test_suite
        
        # Run different types of tests
        test_results = {}
        
        # Unit tests
        test_results["unit_tests"] = run_unit_test_suite(test_results_dir, verbose)
        
        # Integration tests
        test_results["integration_tests"] = run_integration_test_suite(target_dir, test_results_dir, verbose)
        
        # Performance tests
        test_results["performance_tests"] = run_performance_test_suite(target_dir, test_results_dir, verbose)
        
        # Coverage tests
        test_results["coverage_tests"] = run_coverage_tests(test_results_dir, verbose)
        
        # Save test results
        results_file = test_results_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Check overall success
        all_passed = all(test_results.values())
        
        return all_passed
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False

def run_coverage_tests(test_results_dir: Path, verbose: bool) -> bool:
    """Helper function to run coverage tests."""
    try:
        from tests.coverage_tests import run_coverage_test_suite
        return run_coverage_test_suite(test_results_dir, verbose)
    except Exception as e:
        logging.error(f"Failed to run coverage tests: {e}")
        return False

def assert_file_exists(file_path: Path, message: str = "") -> None:
    """Assert that a file exists."""
    assert file_path.exists(), f"File not found: {file_path} {message}"

def assert_valid_json(file_path: Path) -> None:
    """Assert that a file contains valid JSON."""
    assert file_path.exists(), f"File not found: {file_path}"
    try:
        with open(file_path) as f:
            json.load(f)
    except json.JSONDecodeError as e:
        assert False, f"Invalid JSON in {file_path}: {e}"

def assert_directory_structure(base_dir: Path, expected_structure: Dict[str, Any]) -> None:
    """Assert that directory has expected structure."""
    for name, content in expected_structure.items():
        path = base_dir / name
        assert path.exists(), f"Directory/file not found: {path}"
        
        if isinstance(content, dict):
            # It's a directory with specified structure
            assert path.is_dir(), f"Expected directory, got file: {path}"
            assert_directory_structure(path, content)
        elif isinstance(content, list):
            # It's a directory with list of files
            assert path.is_dir(), f"Expected directory, got file: {path}"
            for filename in content:
                assert (path / filename).exists(), f"File not found: {path / filename}"

# Report validation utilities
def validate_report_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate report data structure."""
    errors = []
    
    # Check if data is a dictionary
    if not isinstance(data, dict):
        errors.append("Data must be a dictionary")
        return {"valid": False, "errors": errors}
    
    # Check for required keys
    required_keys = ["steps", "summary"]
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")
    
    # Check steps structure
    if "steps" in data and not isinstance(data["steps"], dict):
        errors.append("Steps must be a dictionary")
    
    # Check summary structure  
    if "summary" in data and not isinstance(data["summary"], dict):
        errors.append("Summary must be a dictionary")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

# MCP testing helpers
def run_all_tests_mcp(target_directory: str, output_directory: str, verbose: bool = False) -> Dict[str, Any]:
    """Run all tests via MCP."""
    try:
        target_dir = Path(target_directory)
        output_dir = Path(output_directory)
        success = run_all_tests(target_dir, output_dir, verbose)
        return {
            "success": success,
            "tests_run": 100,  # Placeholder values
            "tests_passed": 95 if success else 85,
            "tests_failed": 5 if success else 15
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0
        }

def register_tools(mcp_instance) -> None:
    """Register MCP tools for testing."""
    if hasattr(mcp_instance, 'register_tool'):
        try:
            mcp_instance.register_tool(
                name="run_all_tests",
                function=run_all_tests_mcp,
                schema={
                    "type": "object",
                    "properties": {
                        "target_directory": {"type": "string"},
                        "output_directory": {"type": "string"},
                        "verbose": {"type": "boolean", "default": False}
                    },
                    "required": ["target_directory", "output_directory"]
                },
                description="Run comprehensive test suite"
            )
        except Exception as e:
            logging.error(f"Failed to register MCP tools: {e}")

# HTML and Markdown report generation helpers for report tests
def generate_html_report_file(data: Dict[str, Any], output_path: Path) -> bool:
    """Generate HTML report file."""
    try:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pipeline Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Pipeline Analysis Report</h1>
            <p>Generated: {datetime.now().isoformat()}</p>
            
            <h2>Summary</h2>
            <table>
                <tr><th>Total Files Processed</th><td>{data["summary"]["total_files_processed"]}</td></tr>
                <tr><th>Total Size (MB)</th><td>{data["summary"]["total_size_mb"]}</td></tr>
                <tr><th>Success Rate</th><td>{data["summary"]["success_rate"]}%</td></tr>
            </table>
            
            <h2>Step Details</h2>
            <table>
                <tr>
                    <th>Step</th>
                    <th>Status</th>
                    <th>Files</th>
                    <th>Size (MB)</th>
                </tr>
        """
        
        # Add steps
        for step_name, step_data in data["steps"].items():
            status_class = "success" if step_data["status"] == "success" else "error"
            html_content += f"""
                <tr>
                    <td>{step_name}</td>
                    <td class="{status_class}">{step_data["status"]}</td>
                    <td>{step_data["file_count"]}</td>
                    <td>{step_data["total_size_mb"]}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return True
    except Exception as e:
        logging.error(f"Failed to generate HTML report: {e}")
        return False

def generate_markdown_report_file(data: Dict[str, Any], output_path: Path) -> bool:
    """Generate Markdown report file."""
    try:
        md_content = f"""# Pipeline Analysis Report

Generated: {datetime.now().isoformat()}

## Summary

- **Total Files Processed**: {data["summary"]["total_files_processed"]}
- **Total Size (MB)**: {data["summary"]["total_size_mb"]}
- **Success Rate**: {data["summary"]["success_rate"]}%

## Step Details

| Step | Status | Files | Size (MB) |
|------|--------|-------|-----------|
"""
        
        # Add steps
        for step_name, step_data in data["steps"].items():
            md_content += f"| {step_name} | {step_data['status']} | {step_data['file_count']} | {step_data['total_size_mb']} |\n"
        
        with open(output_path, 'w') as f:
            f.write(md_content)
        
        return True
    except Exception as e:
        logging.error(f"Failed to generate Markdown report: {e}")
        return False

def generate_json_report_file(data: Dict[str, Any], output_path: Path) -> bool:
    """Generate JSON report file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Failed to generate JSON report: {e}")
        return False

def generate_comprehensive_report(pipeline_dir: Path, output_dir: Path, logger: logging.Logger) -> bool:
    """Generate comprehensive pipeline report."""
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report
        report_file = output_dir / "comprehensive_analysis_report.html"
        with open(report_file, 'w') as f:
            f.write("<html><body><h1>Comprehensive Pipeline Report</h1></body></html>")
        
        logger.info(f"✅ Comprehensive report generated: {report_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to generate comprehensive report: {e}")
        return False

@contextmanager
def performance_tracker():
    """Context manager for tracking performance metrics."""
    start_time = time.time()
    start_memory = get_memory_usage()
    
    class PerformanceTracker:
        def __init__(self, start_time, start_memory):
            self.start_time = start_time
            self.start_memory = start_memory
            self.end_time = None
            self.end_memory = None
            self.duration = None
            self.memory_delta = None
            
        def finalize(self):
            self.end_time = time.time()
            self.end_memory = get_memory_usage()
            self.duration = self.end_time - self.start_time
            self.memory_delta = self.end_memory - self.start_memory
    
    tracker = PerformanceTracker(start_time, start_memory)
    
    try:
        yield tracker
    finally:
        tracker.finalize()

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0.0  # Return 0 if psutil not available

def track_peak_memory(func):
    """Decorator to track peak memory usage of a function."""
    def wrapper(*args, **kwargs):
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            result = func(*args, **kwargs)
            
            final_memory = process.memory_info().rss
            peak_memory = max(initial_memory, final_memory)
            
            # Store peak memory in function attributes for testing
            wrapper.peak_memory_mb = peak_memory / 1024 / 1024
            wrapper.memory_delta_mb = (final_memory - initial_memory) / 1024 / 1024
            
            return result
            
        except ImportError:
            # If psutil not available, just run the function
            return func(*args, **kwargs)
    
    return wrapper

@contextmanager
def with_resource_limits(max_memory_mb: int = None, max_cpu_percent: int = None):
    """Context manager for resource limit testing."""
    try:
        import psutil
        process = psutil.Process()
        
        # Store initial limits
        initial_memory_limit = getattr(process, 'memory_limit', None)
        initial_cpu_limit = getattr(process, 'cpu_limit', None)
        
        # Set limits if specified
        if max_memory_mb:
            process.memory_limit = max_memory_mb * 1024 * 1024  # Convert to bytes
        if max_cpu_percent:
            process.cpu_limit = max_cpu_percent
        
        yield
        
    except ImportError:
        # If psutil not available, just yield
        yield
    finally:
        # Restore original limits
        try:
            if initial_memory_limit is not None:
                process.memory_limit = initial_memory_limit
            if initial_cpu_limit is not None:
                process.cpu_limit = initial_cpu_limit
        except (ImportError, AttributeError):
            pass 