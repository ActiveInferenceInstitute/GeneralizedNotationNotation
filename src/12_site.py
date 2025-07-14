#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 14: Site Generation

This script generates a static site from pipeline outputs:
- Consolidates reports, visualizations, and analysis results
- Creates an HTML documentation site
- Provides navigation and cross-references

Usage:
    python 14_site.py [options]
    (Typically called by main.py)
"""

import sys
from pathlib import Path
import datetime
import json
import shutil
from typing import Dict, Any, List

# Import centralized utilities
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    PipelineLogger,
    UTILS_AVAILABLE
)

# Initialize logger for this step
logger = setup_step_logging("14_site", verbose=False)

# Ensure src directory is in path to allow sibling imports
current_script_path = Path(__file__).resolve()
src_dir = current_script_path.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Try to import site generator module
try:
    # Import from our local site module (not Python's built-in site)
    import importlib.util
    
    # Direct path import to avoid conflicts with built-in site module
    site_generator_path = current_script_path.parent / "site" / "generator.py"
    spec = importlib.util.spec_from_file_location("site_generator", site_generator_path)
    site_generator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(site_generator_module)
    
    generate_html_report = site_generator_module.generate_html_report
    SITE_GENERATOR_AVAILABLE = True
except Exception as e:
    log_step_warning(logger, f"Site generator module not available: {e}")
    generate_html_report = None
    SITE_GENERATOR_AVAILABLE = False

DEFAULT_SITE_OUTPUT_DIR = "site_step"

def collect_pipeline_artifacts(output_dir: Path) -> Dict[str, Any]:
    """
    Collects artifacts from all pipeline steps for site generation.
    
    Returns:
        Dictionary containing paths to artifacts organized by step
    """
    artifacts = {
        "gnn_files": [],
        "visualizations": [],
        "exports": [],
        "type_check_reports": [],
        "llm_analysis": [],
        "discopy_diagrams": [],
        "execution_results": [],
        "test_reports": [],
        "ontology_reports": []
    }
    
    try:
        # Collect GNN files
        gnn_processing_dir = output_dir / "gnn_processing_step"
        if gnn_processing_dir.exists():
            artifacts["gnn_files"] = list(gnn_processing_dir.glob("*.md"))
        
        # Collect visualizations
        viz_dir = output_dir / "visualization"
        if viz_dir.exists():
            for subdir in viz_dir.iterdir():
                if subdir.is_dir():
                    artifacts["visualizations"].extend(list(subdir.glob("*.png")))
        
        # Collect exports
        export_dir = output_dir / "gnn_exports"
        if export_dir.exists():
            artifacts["exports"] = list(export_dir.rglob("*.json"))
        
        # Collect type check reports
        type_check_dir = output_dir / "type_check"
        if type_check_dir.exists():
            artifacts["type_check_reports"] = list(type_check_dir.glob("*.md"))
        
        # Collect LLM analysis
        llm_dir = output_dir / "llm_processing_step"
        if llm_dir.exists():
            artifacts["llm_analysis"] = list(llm_dir.rglob("*.json"))
        
        # Collect DisCoPy diagrams
        discopy_dir = output_dir / "discopy_gnn"
        if discopy_dir.exists():
            artifacts["discopy_diagrams"] = list(discopy_dir.glob("*.png"))
        
        # Collect execution results
        exec_dir = output_dir / "gnn_rendered_simulators"
        if exec_dir.exists():
            artifacts["execution_results"] = list(exec_dir.rglob("*.py"))
        
        # Collect test reports
        test_dir = output_dir / "test_reports"
        if test_dir.exists():
            artifacts["test_reports"] = list(test_dir.glob("*.xml"))
        
        # Collect ontology reports
        ontology_dir = output_dir / "ontology_processing"
        if ontology_dir.exists():
            artifacts["ontology_reports"] = list(ontology_dir.glob("*.md"))
        
        logger.info(f"Collected artifacts: {sum(len(v) for v in artifacts.values())} total files")
        
    except Exception as e:
        log_step_error(logger, f"Error collecting pipeline artifacts: {e}")
    
    return artifacts

def generate_site_content(artifacts: Dict[str, Any], site_output_dir: Path) -> bool:
    """
    Generates the static site content from collected artifacts.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if not SITE_GENERATOR_AVAILABLE:
            log_step_warning(logger, "Site generator not available, creating basic HTML index")
            return create_basic_index(artifacts, site_output_dir)
        
        # Use the full site generator
        site_output_file = site_output_dir / "index.html"
        try:
            generate_html_report(site_output_dir.parent, site_output_file)
            log_step_success(logger, f"Site generated successfully in {site_output_file}")
            return True
        except Exception as e:
            log_step_error(logger, f"Site generation failed: {e}")
            return False
        
    except Exception as e:
        log_step_error(logger, f"Error generating site content: {e}")
        return False

def create_basic_index(artifacts: Dict[str, Any], site_output_dir: Path) -> bool:
    """
    Creates a basic HTML index when the full site generator is not available.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        site_output_dir.mkdir(parents=True, exist_ok=True)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Pipeline Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .section {{ margin: 20px 0; }}
        .artifact-list {{ margin-left: 20px; }}
        .artifact-item {{ margin: 5px 0; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        .timestamp {{ color: #999; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>GNN Pipeline Results</h1>
    <p class="timestamp">Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="section">
        <h2>Pipeline Artifacts Summary</h2>
        <p>Total artifacts collected: {sum(len(v) for v in artifacts.values())}</p>
    </div>
"""
        
        for section_name, files in artifacts.items():
            if files:
                html_content += f"""
    <div class="section">
        <h2>{section_name.replace('_', ' ').title()}</h2>
        <div class="artifact-list">
"""
                for file_path in files:
                    relative_path = file_path.name
                    html_content += f'            <div class="artifact-item">{relative_path}</div>\n'
                
                html_content += """        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        index_file = site_output_dir / "index.html"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Created basic HTML index at {index_file}")
        return True
        
    except Exception as e:
        log_step_error(logger, f"Error creating basic HTML index: {e}")
        return False

def main(args) -> int:
    """
    Main function to orchestrate site generation.
    """
    # Update logger verbosity based on args
    if args.verbose:
        PipelineLogger.set_verbosity(True)
    
    log_step_start(logger, "Starting site generation step")
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        log_step_error(logger, f"Output directory does not exist: {output_dir}")
        return 1
    
    # Define site output directory
    site_output_dir = output_dir / DEFAULT_SITE_OUTPUT_DIR
    
    try:
        # Collect all pipeline artifacts
        logger.info("Collecting pipeline artifacts...")
        artifacts = collect_pipeline_artifacts(output_dir)
        
        if not any(artifacts.values()):
            log_step_warning(logger, "No pipeline artifacts found to generate site from")
            return 0
        
        # Generate the site
        logger.info(f"Generating site in {site_output_dir}...")
        success = generate_site_content(artifacts, site_output_dir)
        
        if success:
            log_step_success(logger, f"Site generation completed successfully. Output: {site_output_dir}")
            return 0
        else:
            log_step_error(logger, "Site generation failed")
            return 1
            
    except Exception as e:
        log_step_error(logger, f"Unexpected error in site generation: {e}")
        return 1

if __name__ == '__main__':
    # Enhanced argument parsing with fallback
    if UTILS_AVAILABLE:
        try:
            parsed_args = EnhancedArgumentParser.parse_step_arguments("14_site")
        except Exception as e:
            log_step_error(logger, f"Failed to parse arguments with enhanced parser: {e}")
            # Fallback to basic parser
            import argparse
            parser = argparse.ArgumentParser(description="Generate static site from GNN pipeline outputs")
            parser.add_argument("--output-dir", type=Path, required=True,
                              help="Pipeline output directory containing artifacts to include in site")
            parser.add_argument("--verbose", "-v", action="store_true", default=False,
                              help="Enable verbose logging")
            parsed_args = parser.parse_args()
    else:
        # Fallback to basic parser
        import argparse
        parser = argparse.ArgumentParser(description="Generate static site from GNN pipeline outputs")
        parser.add_argument("--output-dir", type=Path, required=True,
                          help="Pipeline output directory containing artifacts to include in site")
        parser.add_argument("--verbose", "-v", action="store_true", default=False,
                          help="Enable verbose logging")
        parsed_args = parser.parse_args()

    # Update logger for standalone execution
    if parsed_args.verbose:
        PipelineLogger.set_verbosity(True)

    exit_code = main(parsed_args)
    sys.exit(exit_code) 