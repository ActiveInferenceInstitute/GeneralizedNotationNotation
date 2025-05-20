#!/usr/bin/env python3
"""
Pipeline Step 99: Generate HTML Site Summary

This script generates a single HTML website that summarizes the contents of the 
 GNN pipeline's output directory. It consolidates various reports, visualizations, 
logs, and other artifacts into an accessible web page.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src directory is in Python path for `from src.site import ...`
# This is often handled by the execution environment (e.g., main.py or IDE)
# but good for robustness if script is called directly in some contexts.
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent # Assuming src/12_site.py, so two levels up
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.site.generator import generate_html_report, logger as generator_logger # Import main function and logger

# Logger for this pipeline script
logger = logging.getLogger(__name__) # __name__ will be '12_site' or '__main__'

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the site generation script."""
    parser = argparse.ArgumentParser(description="Generates an HTML summary site for GNN pipeline outputs.")
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="The main output directory of the GNN pipeline (e.g., ../output or output/). This script reads from this directory."
    )
    parser.add_argument(
        "--site-html-file",
        type=str,
        default="gnn_pipeline_summary_site.html",
        help="Filename for the generated HTML site. This file will be saved inside the --output-dir. Default: gnn_pipeline_summary_site.html"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for this script and the site generator."
    )
    return parser.parse_args()

def main():
    """Main execution function for the 12_site.py pipeline step."""
    args = parse_arguments()

    # Configure logging for this script
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        generator_logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        generator_logger.setLevel(logging.INFO)

    # If running standalone and no handlers are configured for this logger or the root logger,
    # add a default console handler to make logs visible.
    # This helps in direct testing of the script.
    if __name__ == "__main__" and not logger.hasHandlers() and not logging.getLogger().hasHandlers():
        ch_script = logging.StreamHandler(sys.stdout)
        formatter_script = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch_script.setFormatter(formatter_script)
        logger.addHandler(ch_script)
        # Also ensure generator_logger gets a handler if it doesn't have one in this standalone context
        if not generator_logger.hasHandlers():
             ch_gen = logging.StreamHandler(sys.stdout)
             formatter_gen = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             ch_gen.setFormatter(formatter_gen)
             generator_logger.addHandler(ch_gen)
             generator_logger.propagate = False # Avoid double log if root gets handler from basicConfig

    logger.info(f"Starting pipeline step: 12_site.py - Generate HTML Summary Site")
    logger.info(f"Reading from pipeline output directory: {args.output_dir}")
    logger.info(f"Generated site will be saved as: {args.output_dir / args.site_html_file}")

    # Construct the full path for the site output file
    site_output_file_path = args.output_dir.resolve() / args.site_html_file

    try:
        # Ensure the main output directory (which is input for this script) exists
        if not args.output_dir.is_dir():
            logger.error(f"Specified output directory does not exist: {args.output_dir}")
            sys.exit(1)
            
        # Ensure the parent directory for the HTML site file exists (it should be output_dir)
        site_output_file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Calling generate_html_report with output_dir='{args.output_dir.resolve()}' and site_output_file='{site_output_file_path}'")
        generate_html_report(args.output_dir.resolve(), site_output_file_path)
        
        logger.info(f"HTML summary site generated successfully: {site_output_file_path}")
        sys.exit(0) # Success
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found during site generation: {fnf_error}")
        sys.exit(1)
    except PermissionError as perm_error:
        logger.error(f"Permission error during site generation: {perm_error}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during site generation: {e}", exc_info=args.verbose)
        if args.verbose:
            logger.debug(e, exc_info=True) # Log full traceback if verbose
        sys.exit(1)

if __name__ == "__main__":
    main() 