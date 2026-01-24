"""
Report Generation Utilities for Test Execution.

This module provides functions for generating test execution reports.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List


def generate_markdown_report(report_path: Path, summary: Dict[str, Any]):
    """Generate markdown test report."""
    try:
        with open(report_path, 'w') as f:
            f.write("# Test Execution Report\n\n")
            
            # Summary section
            f.write("## Summary\n\n")
            f.write(f"- **Total Tests**: {summary.get('total_tests_run', 0)}\n")
            f.write(f"- **Passed**: {summary.get('total_tests_passed', 0)}\n")
            f.write(f"- **Failed**: {summary.get('total_tests_failed', 0)}\n")
            f.write(f"- **Skipped**: {summary.get('total_tests_skipped', 0)}\n")
            f.write(f"- **Success Rate**: {summary.get('success_rate', 0):.1f}%\n")
            f.write(f"- **Execution Time**: {summary.get('total_execution_time', 0):.2f}s\n\n")
        
            # Stage results
            if 'stage_results' in summary:
                f.write("## Stage Results\n\n")
                for stage_name, stage_data in summary['stage_results'].items():
                    f.write(f"### {stage_name.title()}\n")
                    f.write(f"- **Status**: {'✅ Passed' if stage_data.get('success') else '❌ Failed'}\n")
                    f.write(f"- **Duration**: {stage_data.get('duration_seconds', 0):.2f}s\n")
                    f.write(f"- **Tests**: {stage_data.get('tests_passed', 0)}/{stage_data.get('tests_run', 0)} passed\n\n")
        
            # Performance metrics
            if 'performance_metrics' in summary:
                f.write("## Performance Metrics\n\n")
                perf = summary['performance_metrics']
                f.write(f"- **Peak Memory**: {perf.get('peak_memory_mb', 0):.1f}MB\n")
                f.write(f"- **Average Execution Time**: {perf.get('average_execution_time', 0):.2f}s\n")
                f.write(f"- **Success Rate**: {perf.get('success_rate', 0):.1f}%\n\n")
        
        logging.info(f"✅ Markdown report generated: {report_path}")
        
    except Exception as e:
        logging.warning(f"Failed to generate markdown report: {e}")


def generate_fallback_report(output_dir: Path, summary: Dict[str, Any]):
    """Generate fallback report when main report generation fails."""
    try:
        fallback_file = output_dir / "test_report_fallback.txt"
        with open(fallback_file, 'w') as f:
            f.write("Test Execution Summary\n")
            f.write("=====================\n\n")
            f.write(f"Total Tests: {summary.get('total_tests_run', 0)}\n")
            f.write(f"Passed: {summary.get('total_tests_passed', 0)}\n")
            f.write(f"Failed: {summary.get('total_tests_failed', 0)}\n")
            f.write(f"Skipped: {summary.get('total_tests_skipped', 0)}\n")
            f.write(f"Execution Time: {summary.get('total_execution_time', 0):.2f}s\n")
        
        logging.info(f"✅ Fallback report generated: {fallback_file}")
        
    except Exception as e:
        logging.warning(f"Failed to generate fallback report: {e}")


def generate_timeout_report(output_dir: Path, cmd: List[str], timeout: int):
    """Generate report for timeout scenarios."""
    try:
        timeout_file = output_dir / "test_timeout_report.txt"
        with open(timeout_file, 'w') as f:
            f.write("Test Execution Timeout\n")
            f.write("=====================\n\n")
            f.write(f"Timeout: {timeout} seconds\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logging.warning(f"⚠️ Timeout report generated: {timeout_file}")
        
    except Exception as e:
        logging.warning(f"Failed to generate timeout report: {e}")


def generate_error_report(output_dir: Path, cmd: List[str], error_msg: str):
    """Generate report for error scenarios."""
    try:
        error_file = output_dir / "test_error_report.txt"
        with open(error_file, 'w') as f:
            f.write("Test Execution Error\n")
            f.write("===================\n\n")
            f.write(f"Error: {error_msg}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logging.info(f"❌ Error report generated: {error_file}")
        
    except Exception as e:
        logging.warning(f"Failed to generate error report: {e}")


# Backward-compatible aliases (underscore-prefixed versions)
_generate_markdown_report = generate_markdown_report
_generate_fallback_report = generate_fallback_report
_generate_timeout_report = generate_timeout_report
_generate_error_report = generate_error_report
