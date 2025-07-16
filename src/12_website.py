#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 12: Website Generation

This script generates a static website from pipeline outputs:
- Creates an HTML documentation website
- Embeds pipeline artifacts and visualizations
- Provides navigation and search functionality

Usage:
    python 12_website.py [options]
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
    PipelineLogger,
    UTILS_AVAILABLE
)

from utils.pipeline_template import create_standardized_pipeline_script

from pipeline import (
    STEP_METADATA,
    get_output_dir_for_script
)

sys.path.insert(0, str(Path(__file__).parent))
from website.generator import generate_website

# Initialize logger for this step
logger = setup_step_logging("12_website", verbose=False)

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
        log_step_start(logger, "Starting static website generation from pipeline outputs")
        
        # Update logger verbosity if needed
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Get the website output directory for this step
        website_output_dir = get_output_dir_for_script("12_website.py", output_dir)
        website_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Website source directory: {output_dir}")
        logger.info(f"Website output directory: {website_output_dir}")
        
        # Validate that the output directory exists and has content
        if not output_dir.exists():
            log_step_error(logger, f"Pipeline output directory does not exist: {output_dir}")
            return False
        
        # Check for minimum required content
        required_items = ["pipeline_execution_summary.json"]
        missing_items = []
        for item in required_items:
            if not (output_dir / item).exists():
                missing_items.append(item)
        
        if missing_items:
            logger.warning(f"Some expected pipeline outputs are missing: {missing_items}")
            logger.info("Proceeding with website generation using available content")
        
        # Generate website using the main pipeline output directory as source
        # and the step-specific directory as destination
        success = generate_website(logger, output_dir, website_output_dir)
        
        if success:
            website_file = website_output_dir / "index.html"
            if website_file.exists():
                file_size_mb = website_file.stat().st_size / (1024 * 1024)
                log_step_success(logger, f"Website generated successfully: {website_file} ({file_size_mb:.2f} MB)")
                
                # Copy any assets from the output directory that might be referenced
                # This ensures that images, HTML files, etc. are accessible from the website
                try:
                    _copy_website_assets(output_dir, website_output_dir, logger)
                except Exception as e:
                    logger.warning(f"Failed to copy some website assets: {e}")
                
                return True
            else:
                log_step_error(logger, f"Website generation reported success but index.html not found at {website_file}")
                return False
        else:
            log_step_error(logger, "Website generation failed")
            return False
        
    except Exception as e:
        log_step_error(logger, f"Website generation failed with exception: {e}")
        if verbose:
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
    "12_website.py",
    process_website_generation,
    "Static HTML website generation",
    additional_arguments={
        "website_html_filename": {"type": str, "default": "gnn_pipeline_summary_website.html", "help": "Filename for generated HTML website"}
    }
)

if __name__ == '__main__':
    sys.exit(run_script()) 