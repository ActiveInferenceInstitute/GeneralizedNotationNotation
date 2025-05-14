#!/usr/bin/env python3
"""
GNN Type Checker CLI

Command-line interface for the GNN type checker.
"""

import sys
import argparse
import json
from pathlib import Path

from .checker import GNNTypeChecker
from .resource_estimator import GNNResourceEstimator


def main(args=None):
    """
    Main function to run the type checker from command line.
    
    Args:
        args: Command line arguments (if None, sys.argv[1:] is used)
        
    Returns:
        Exit code (0 for success, 1 for errors)
    """
    parser = argparse.ArgumentParser(description="GNN Type Checker")
    parser.add_argument("input_path", help="Path to GNN file or directory")
    parser.add_argument("-r", "--report-file", help="File to save the main type checking report (markdown). If only a filename is provided, it will be placed in the output directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively process directories")
    parser.add_argument("--strict", action="store_true", help="Enable strict type checking mode")
    parser.add_argument("--estimate-resources", action="store_true", help="Estimate computational resources for the GNN models")
    parser.add_argument("-o", "--output-dir", help="Base directory to save all output files (type checker reports, resource reports, visualizations). Defaults to 'output/gnn_type_checker' relative to project root.")
    parser.add_argument("--project-root", help="Absolute path to the project root, for relative path generation in reports")
    
    args = parser.parse_args(args)
    
    # Determine the base output directory
    if args.output_dir:
        # Use the user-provided output directory
        # Assuming it's relative to the original CWD or an absolute path
        # If running from src/, and path is ../output/something, it will be correct.
        actual_output_dir = Path(args.output_dir).resolve() 
    else:
        # Default output directory: <project_root>/output/gnn_type_checker
        # This needs to be careful about CWD. Let's assume CWD is project root for default.
        # If script is run from src/ via python -m, CWD might be src/.
        # A more robust way for default: Path(__file__).resolve().parent.parent.parent / "output" / "gnn_type_checker"
        # For now, let's keep it simple and rely on user providing ../output or an absolute path when needed.
        # The typical run `cd src && python3 -m gnn_type_checker ... --output-dir ../output/gnn_type_checker` handles this.
        actual_output_dir = Path("output/gnn_type_checker").resolve() # Default path, resolves relative to CWD

    actual_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine markdown report name and path
    markdown_report_name = "type_check_report.md" # Default name
    if args.report_file:
        report_file_path = Path(args.report_file)
        if report_file_path.is_absolute() or len(report_file_path.parts) > 1: # It's a path, not just a filename
            # This case is complex if we want to force it into actual_output_dir.
            # For now, let's assume if user gives a path here, they know what they're doing.
            # However, generate_report in checker now takes a base dir and a filename.
            # So, we should only pass the filename part.
            markdown_report_name = report_file_path.name
            # If report_file_path also implies a directory, we might want to use that for actual_output_dir
            # For now, let's stick to actual_output_dir for the base, and this is just the filename.
            print(f"Warning: Full path in --report-file ({args.report_file}) is not fully supported with --output-dir. Using filename: {markdown_report_name} within {actual_output_dir}")
        else: # Just a filename
            markdown_report_name = args.report_file

    # Create and run the type checker
    checker = GNNTypeChecker(strict_mode=args.strict)
    
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        is_valid, errors, warnings = checker.check_file(str(input_path))
        results = {str(input_path): {"is_valid": is_valid, "errors": errors, "warnings": warnings}}
    else:
        results = checker.check_directory(str(input_path), recursive=args.recursive)
    
    # Generate and display report - checker now handles saving all its files
    report = checker.generate_report(results, actual_output_dir, report_md_filename=markdown_report_name, project_root_path=args.project_root)
    print("\n--- Type Check Report Summary ---")
    print(report)
    print("--- End of Type Check Report Summary ---")
    
    # Resource estimation uses subdirectories within actual_output_dir
    if args.estimate_resources:
        print("\nEstimating computational resources...")
        
        # Path for resource estimator outputs
        # Estimator will create its own subdirectories like 'resources_report/' or 'html_vis/' if needed, 
        # relative to the path given to its generate_report.
        # We will give it actual_output_dir, and it can manage its own structure inside that.
        # Or, we can specify a sub-directory for it.
        # Let's make resource estimator also use a subfolder within actual_output_dir.
        resource_estimator_output_base_dir = actual_output_dir / "resource_estimates"
        resource_estimator_output_base_dir.mkdir(parents=True, exist_ok=True)

        # The estimator might try to read type_check_data.json.
        # checker.py now saves it to: actual_output_dir / "resources" / "type_check_data.json"
        type_check_data_json_path = actual_output_dir / "resources" / "type_check_data.json"

        estimator = GNNResourceEstimator(type_check_data=str(type_check_data_json_path) if type_check_data_json_path.exists() else None)
        
        if input_path.is_file():
            result = estimator.estimate_from_file(str(input_path))
            estimator.results = {str(input_path): result}
        else:
            estimator.estimate_from_directory(str(input_path), recursive=args.recursive)
        
        # Generate resource report - give it its own base output dir
        resource_report_summary = estimator.generate_report(str(resource_estimator_output_base_dir), project_root_path=args.project_root)
        
        print("\n--- Resource Estimation Report Summary ---")
        summary_lines = resource_report_summary.split('\n')[:5] # Show first 5 lines
        print('\n'.join(summary_lines))
        print(f"Resource estimation reports (markdown, JSON, HTML) saved in: {resource_estimator_output_base_dir}")
        print("--- End of Resource Estimation Report Summary ---")

    has_errors = any(not r.get("is_valid", True) for r in results.values())
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main()) 