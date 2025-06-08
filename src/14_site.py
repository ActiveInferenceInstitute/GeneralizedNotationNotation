#!/usr/bin/env python3
"""
Pipeline Step 14: Generate HTML Site Summary

This script generates a single HTML website that summarizes the contents of the 
GNN pipeline's output directory. It consolidates various reports, visualizations, 
logs, and other artifacts into an accessible web page.
"""

import argparse
import sys
from pathlib import Path
import json
import datetime

# Import centralized utilities
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    UTILS_AVAILABLE
)

# Initialize logger for this step
logger = setup_step_logging("14_site", verbose=False)

# Ensure src directory is in Python path for site generator imports
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.site.generator import generate_html_report, logger as generator_logger
except ImportError as e:
    log_step_warning(logger, f"Site generator module not found: {e}. Will use basic HTML generation.")
    generate_html_report = None
    generator_logger = None

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the site generation script."""
    parser = argparse.ArgumentParser(description="Generate HTML summary site for GNN pipeline outputs")
    
    # Define defaults for standalone execution
    script_file_path = Path(__file__).resolve()
    project_root = script_file_path.parent.parent
    default_output_dir = project_root / "output"
    
    parser.add_argument("--target-dir", type=Path,
                       help="Target directory containing GNN files (compatibility)")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir,
                       help="Main pipeline output directory")
    parser.add_argument("--site-html-filename", type=str, 
                       default="gnn_pipeline_summary_site.html",
                       help="Filename for the generated HTML site")
    parser.add_argument("--verbose", action='store_true',
                       help="Enable verbose logging")
    
    return parser.parse_args()

def generate_basic_html_report(output_dir: Path, site_output_file: Path):
    """Generate a basic HTML report when the full generator is not available."""
    log_step_start(logger, "Generating basic HTML report")
    
    # Create site directory
    site_dir = output_dir / "site"
    site_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan output directory for files
    report_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "output_directory": str(output_dir),
        "files_found": [],
        "directories": []
    }
    
    for item in output_dir.rglob("*"):
        if item.is_file():
            relative_path = item.relative_to(output_dir)
            report_data["files_found"].append({
                "path": str(relative_path),
                "size": item.stat().st_size,
                "modified": datetime.datetime.fromtimestamp(item.stat().st_mtime).isoformat()
            })
        elif item.is_dir() and item != output_dir:
            relative_path = item.relative_to(output_dir)
            report_data["directories"].append(str(relative_path))
    
    # Generate basic HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GNN Pipeline Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
            .file-list {{ max-height: 400px; overflow-y: auto; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>GNN Pipeline Summary Report</h1>
            <p>Generated: {report_data['timestamp']}</p>
            <p>Source Directory: {report_data['output_directory']}</p>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
            <p>Files Found: {len(report_data['files_found'])}</p>
            <p>Directories: {len(report_data['directories'])}</p>
        </div>
        
        <div class="section">
            <h2>Directories</h2>
            <ul>
                {''.join(f'<li>{d}</li>' for d in report_data['directories'])}
            </ul>
        </div>
        
        <div class="section file-list">
            <h2>Files</h2>
            <table>
                <tr><th>Path</th><th>Size (bytes)</th><th>Modified</th></tr>
                {''.join(f'<tr><td>{f["path"]}</td><td>{f["size"]}</td><td>{f["modified"]}</td></tr>' 
                        for f in report_data['files_found'])}
            </table>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    output_file = site_dir / "index.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Save JSON data
    json_file = site_dir / "report_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)
    
    log_step_success(logger, f"Basic HTML report generated: {output_file}")
    return output_file

def main():
    """Main execution function for the 14_site.py pipeline step."""
    args = parse_arguments()
    
    # Update logger verbosity based on args
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)
        if generator_logger:
            generator_logger.setLevel(logging.DEBUG)
    
    log_step_start(logger, "Starting HTML site generation")
    
    # Validate output directory exists
    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        log_step_error(logger, f"Output directory does not exist: {output_dir}")
        return 1
    
    # Create site output directory
    site_dir = output_dir / "site"
    if not validate_output_directory(output_dir, "site"):
        log_step_error(logger, "Failed to create site directory")
        return 1
    
    try:
        # Use full generator if available, otherwise basic generator
        if generate_html_report:
            logger.info("Using full HTML report generator")
            site_output_file = site_dir / args.site_html_filename
            generate_html_report(output_dir, site_output_file)
            log_step_success(logger, f"HTML site generated successfully: {site_output_file}")
        else:
            logger.info("Using basic HTML report generator")
            output_file = generate_basic_html_report(output_dir, site_dir / args.site_html_filename)
            log_step_success(logger, f"Basic HTML site generated: {output_file}")
        
        return 0
        
    except Exception as e:
        log_step_error(logger, f"Site generation failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 