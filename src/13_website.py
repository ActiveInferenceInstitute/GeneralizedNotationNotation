#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 13: Website Generation

This script generates a static website from pipeline outputs:
- Creates an HTML documentation website
- Embeds pipeline artifacts and visualizations
- Provides navigation and search functionality

Usage:
    python 13_website.py [options]
"""

import sys
import logging
from pathlib import Path
import datetime
import json
import shutil
from typing import Dict, Any, List, Optional

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

# Import centralized utilities
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    performance_tracker,
    PipelineLogger,
    UTILS_AVAILABLE
)

from utils.pipeline_template import create_standardized_pipeline_script

from pipeline import (
    get_output_dir_for_script
)

sys.path.insert(0, str(Path(__file__).parent))
from website.generator import generate_website

# Initialize logger for this step
logger = setup_step_logging("13_website", verbose=False)

def process_website_generation(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized website generation processing function.
    
    Args:
        target_dir: Directory containing pipeline artifacts
        output_dir: Output directory for generated website
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Start performance tracking
        with performance_tracker.track_operation("website_generation", {"verbose": verbose, "recursive": recursive}):
            log_step_start(logger, "Starting static website generation from pipeline outputs")
            
            # Update logger verbosity if needed
            if verbose:
                logger.setLevel(logging.DEBUG)
            
            # Set up paths - website generation uses output_dir as source of pipeline outputs
            website_output_dir = output_dir / "website"
            website_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if pipeline output directory exists
            pipeline_output_dir = output_dir.parent if output_dir.name == "website" else output_dir
            if not pipeline_output_dir.exists():
                log_step_error(logger, f"Pipeline output directory does not exist: {pipeline_output_dir}")
                return False
            
            # Generate website using the website module
            try:
                website_html_filename = kwargs.get('website_html_filename', 'index.html')
                
                # Basic website generation
                website_result = generate_website(
                    pipeline_output_dir=pipeline_output_dir,
                    website_output_dir=website_output_dir,
                    html_filename=website_html_filename,
                    logger=logger
                )
                
                if website_result.get('success', False):
                    website_file = website_output_dir / website_html_filename
                    
                    if website_file.exists():
                        file_size_mb = website_file.stat().st_size / (1024 * 1024)
                        log_step_success(logger, f"Website generated successfully: {website_file} ({file_size_mb:.2f} MB)")
                        
                        # Generate additional metadata
                        metadata = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "website_file": str(website_file),
                            "file_size_mb": round(file_size_mb, 2),
                            "source_directory": str(pipeline_output_dir),
                            "generation_details": website_result
                        }
                        
                        metadata_file = website_output_dir / "generation_metadata.json"
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        return True
                    else:
                        log_step_error(logger, f"Website generation reported success but index.html not found at {website_file}")
                        return False
                else:
                    log_step_error(logger, "Website generation failed")
                    return False
            
            except Exception as generation_error:
                log_step_warning(logger, f"Advanced website generation failed: {generation_error}")
                
                # Fallback: basic HTML generation
                return _generate_basic_website(pipeline_output_dir, website_output_dir, logger)
            
    except Exception as e:
        log_step_error(logger, f"Website failed: {e}")
        if verbose:
            import traceback
            logger.debug(f"Website generation traceback: {traceback.format_exc()}")
        return False

def _generate_basic_website(pipeline_output_dir: Path, website_output_dir: Path, logger: logging.Logger) -> bool:
    """Generate a basic HTML website as fallback."""
    try:
        logger.info("Generating basic HTML website (fallback mode)")
        
        # Create basic HTML content
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Pipeline Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .file-list {{ list-style-type: none; padding: 0; }}
        .file-item {{ padding: 5px; border-bottom: 1px solid #eee; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>GNN Pipeline Results</h1>
        <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Pipeline Outputs</h2>
        <p>Source directory: {pipeline_output_dir}</p>
        <ul class="file-list">
"""
        
        # List pipeline outputs
        for item in pipeline_output_dir.iterdir():
            if item.is_dir():
                html_content += f'<li class="file-item">📁 {item.name}/</li>\n'
            else:
                html_content += f'<li class="file-item">📄 {item.name}</li>\n'
        
        html_content += """
        </ul>
    </div>
    
    <div class="section">
        <h2>Website Generation</h2>
        <p>This website was generated using basic HTML fallback mode.</p>
        <p>For enhanced features, ensure advanced website dependencies are available.</p>
    </div>
</body>
</html>"""
        
        # Save basic website
        index_file = website_output_dir / "index.html"
        with open(index_file, 'w') as f:
            f.write(html_content)
        
        log_step_success(logger, f"Basic website generated: {index_file}")
        return True
        
    except Exception as e:
        log_step_error(logger, f"Basic website generation failed: {e}")
        return False


def _copy_website_assets(source_dir: Path, website_dir: Path, logger: logging.Logger):
    """
    Copy website assets like images and other files to make them accessible from the website.
    Excludes HTML files to prevent recursive nesting of website generations.
    
    Args:
        source_dir: Source directory containing pipeline outputs
        website_dir: Website output directory
        logger: Logger for this operation
    """
    assets_copied = 0
    
    # Asset patterns to copy - EXCLUDE HTML files to prevent nesting issues
    asset_patterns = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg"]
    
    for pattern in asset_patterns:
        for asset_file in source_dir.rglob(pattern):
            if asset_file.is_file():
                # Skip files that are already in the website directory to prevent recursion
                try:
                    asset_file.relative_to(website_dir)
                    continue  # Skip files already in website directory
                except ValueError:
                    pass  # File is not in website directory, continue processing
                
                try:
                    # Calculate relative path from source_dir
                    relative_path = asset_file.relative_to(source_dir)
                    target_file = website_dir / relative_path
                    
                    # Create parent directories if needed
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy the file if it doesn't exist or is newer
                    if not target_file.exists() or asset_file.stat().st_mtime > target_file.stat().st_mtime:
                        shutil.copy2(asset_file, target_file)
                        assets_copied += 1
                        logger.debug(f"Copied asset: {relative_path}")
                        
                except Exception as e:
                    logger.warning(f"Failed to copy asset {asset_file}: {e}")
    
    if assets_copied > 0:
        logger.info(f"Copied {assets_copied} website assets")
    else:
        logger.debug("No website assets needed copying")

run_script = create_standardized_pipeline_script(
    "13_website.py",
    process_website_generation,
    "Static HTML website generation",
    additional_arguments={
        "website_html_filename": {"type": str, "default": "gnn_pipeline_summary_website.html", "help": "Filename for generated HTML website"}
    }
)

if __name__ == '__main__':
    sys.exit(run_script()) 