#!/usr/bin/env python3
"""
GNN Resource Estimator

NOTE: This module is maintained for backward compatibility.
The actual implementation has been moved to the `estimation` subpackage.
"""

import sys
import argparse
import os
from pathlib import Path

# Re-export from the estimation subpackage
from .estimation.estimator import GNNResourceEstimator

__all__ = ["GNNResourceEstimator"]


def main():
    """
    Main function to run the resource estimator from command line.
    """
    parser = argparse.ArgumentParser(description="GNN Resource Estimator")
    parser.add_argument("input_path", help="Path to GNN file or directory")
    parser.add_argument("-t", "--type-check-data", help="Path to type check JSON data")
    parser.add_argument("-o", "--output-dir", help="Directory to save resource reports")
    parser.add_argument("--recursive", action="store_true", help="Recursively process directories")
    parser.add_argument("--html-only", action="store_true", help="Generate only HTML report with visualizations")

    args = parser.parse_args()

    estimator = GNNResourceEstimator(args.type_check_data)

    input_path = args.input_path
    path = Path(input_path)

    if path.is_file():
        # Estimate single file
        result = estimator.estimate_from_file(str(path))
        estimator.results = {str(path): result}
    else:
        # Estimate directory
        estimator.estimate_from_directory(str(path), recursive=args.recursive)

    # Generate and display report
    report = estimator.generate_report(args.output_dir)

    if not args.html_only:
        print(report)
    else:
        # When HTML only mode is selected, just print a simple summary and HTML location
        output_dir = args.output_dir if args.output_dir else "output/type_checker/resources"
        html_path = os.path.join(output_dir, "resource_report_detailed.html")
        print(f"Generated HTML resource report at: {html_path}")
        print(f"Analyzed {len(estimator.results)} files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
