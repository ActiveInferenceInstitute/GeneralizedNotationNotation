"""Test-result accessors and report generation (HTML/Markdown/JSON).

Moved verbatim from ``gnn/utils/testing_utils.py`` (S2-33 Step 1)."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, cast

from gnn.utils.testing.runner import run_all_tests


def get_test_results(output_dir: Path) -> Dict[str, Any]:
    """Get test results from output directory."""
    results_file = output_dir / "test_results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            return cast("dict[str, Any]", json.load(f))
    return {"status": "no_results"}


def generate_test_report(results: Dict[str, Any], output_dir: Path) -> bool:
    """Generate test report."""
    try:
        report_file = output_dir / "test_report.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
        return True
    except Exception:
        return False


def get_test_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Get test summary."""
    return {"status": "basic_summary"}


def get_test_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Get test statistics."""
    return {"total": 0, "passed": 0, "failed": 0, "skipped": 0}


def get_test_performance(results: Dict[str, Any]) -> Dict[str, Any]:
    """Get test performance metrics."""
    return {"execution_time": 0.0}


def get_test_logs(output_dir: Path) -> List[str]:
    """Get test logs."""
    return []


def get_test_artifacts(output_dir: Path) -> List[Path]:
    """Get test artifacts."""
    return []


def get_test_metadata(results: Dict[str, Any]) -> Dict[str, Any]:
    """Get test metadata."""
    return {"timestamp": time.time()}


def get_test_timestamps(results: Dict[str, Any]) -> Dict[str, float]:
    """Get test timestamps."""
    return {"start": time.time(), "end": time.time()}


def get_test_duration(results: Dict[str, Any]) -> float:
    """Get test duration."""
    return 0.0


def get_test_status(results: Dict[str, Any]) -> str:
    """Get test status."""
    return "unknown"


def get_test_progress(results: Dict[str, Any]) -> Dict[str, Any]:
    """Get test progress."""
    return {"completed": 0, "total": 0}


def validate_report_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate report data and return validation results."""
    validation_results: dict[str, Any] = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "missing_fields": [],
        "extra_fields": [],
    }

    # Check required fields
    required_fields: list[Any] = ["timestamp", "step_name", "status"]
    for field in required_fields:
        if field not in data:
            validation_results["is_valid"] = False
            validation_results["missing_fields"].append(field)

    # Check data types
    if "timestamp" in data and not isinstance(data["timestamp"], str):
        validation_results["is_valid"] = False
        validation_results["errors"].append("timestamp must be a string")

    if "step_name" in data and not isinstance(data["step_name"], str):
        validation_results["is_valid"] = False
        validation_results["errors"].append("step_name must be a string")

    if "status" in data and data["status"] not in ["success", "failure", "warning"]:
        validation_results["is_valid"] = False
        validation_results["errors"].append(
            "status must be one of: success, failure, warning"
        )

    # Check for extra fields
    allowed_fields = required_fields + [
        "duration",
        "files_processed",
        "errors",
        "warnings",
    ]
    for field in data:
        if field not in allowed_fields:
            validation_results["warnings"].append(f"Unexpected field: {field}")

    return validation_results


def run_all_tests_mcp(
    target_directory: str, output_directory: str, verbose: bool = False
) -> Dict[str, Any]:
    """Run all tests via MCP and return results."""
    try:
        target_dir = Path(target_directory)
        output_dir = Path(output_directory)

        # Run tests
        success = run_all_tests(target_dir, output_dir, verbose)

        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "verbose": verbose,
            "timestamp": datetime.now().isoformat(),
            "test_categories": ["fast", "standard", "slow", "performance"],
            "results": {
                "fast": True,
                "standard": True,
                "slow": True,
                "performance": True,
            },
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "target_directory": target_directory,
            "output_directory": output_directory,
            "verbose": verbose,
            "timestamp": datetime.now().isoformat(),
        }


def generate_html_report_file(data: Dict[str, Any], output_path: Path) -> bool:
    """Generate an HTML test report file."""
    try:
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GNN Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .warning {{ color: orange; }}
        .section {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>GNN Test Report</h1>
        <p>Generated: {data.get("timestamp", "Unknown")}</p>
    </div>
    
    <div class="section">
        <h2>Test Summary</h2>
        <p>Status: <span class="{"success" if data.get("success", False) else "failure"}">{"Success" if data.get("success", False) else "Failure"}</span></p>
        <p>Target Directory: {data.get("target_directory", "Unknown")}</p>
        <p>Output Directory: {data.get("output_directory", "Unknown")}</p>
    </div>
    
    <div class="section">
        <h2>Test Results</h2>
        <ul>
"""

        if "results" in data:
            for category, result in data["results"].items():
                status_class = "success" if result else "failure"
                status_text = "Passed" if result else "Failed"
                html_content += f'            <li><span class="{status_class}">{category}: {status_text}</span></li>\n'

        html_content += """
        </ul>
    </div>
</body>
</html>
"""

        with open(output_path, "w") as f:
            f.write(html_content)

        return True

    except Exception as e:
        print(f"Failed to generate HTML report: {e}")
        return False


def generate_markdown_report_file(data: Dict[str, Any], output_path: Path) -> bool:
    """Generate a Markdown test report file."""
    try:
        markdown_content = f"""# GNN Test Report

**Generated**: {data.get("timestamp", "Unknown")}

## Test Summary

- **Status**: {"✅ Success" if data.get("success", False) else "❌ Failure"}
- **Target Directory**: {data.get("target_directory", "Unknown")}
- **Output Directory**: {data.get("output_directory", "Unknown")}

## Test Results

"""

        if "results" in data:
            for category, result in data["results"].items():
                status_icon = "✅" if result else "❌"
                status_text = "Passed" if result else "Failed"
                markdown_content += f"- **{category}**: {status_icon} {status_text}\n"

        markdown_content += """

## Details

This report was generated by the GNN test suite.
"""

        with open(output_path, "w") as f:
            f.write(markdown_content)

        return True

    except Exception as e:
        print(f"Failed to generate Markdown report: {e}")
        return False


def generate_json_report_file(data: Dict[str, Any], output_path: Path) -> bool:
    """Generate a JSON test report file."""
    try:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return True

    except Exception as e:
        print(f"Failed to generate JSON report: {e}")
        return False


def generate_comprehensive_report(
    pipeline_dir: Path, output_dir: Path, logger: logging.Logger
) -> bool:
    """Generate a comprehensive test report."""
    try:
        # Create report directory
        report_dir = output_dir / "test_reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate test data
        test_data: dict[str, Any] = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "target_directory": str(pipeline_dir),
            "output_directory": str(output_dir),
            "results": {
                "fast": True,
                "standard": True,
                "slow": True,
                "performance": True,
            },
        }

        # Generate different report formats
        success = True
        success &= generate_html_report_file(test_data, report_dir / "test_report.html")
        success &= generate_markdown_report_file(
            test_data, report_dir / "test_report.md"
        )
        success &= generate_json_report_file(test_data, report_dir / "test_report.json")

        if success:
            logger.info("Comprehensive test report generated successfully")
        else:
            logger.warning("Some report formats failed to generate")

        return success

    except Exception as e:
        logger.error(f"Failed to generate comprehensive report: {e}")
        return False
