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
        # Get the website output directory for this step
        website_output_dir = get_output_dir_for_script("12_website.py", output_dir)
        
        # Generate website using the main pipeline output directory as source
        # and the step-specific directory as destination
        success = generate_website(logger, output_dir, website_output_dir)
        
        if success:
            log_step_success(logger, f"Website generated successfully in {website_output_dir / 'index.html'}")
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Website generation failed: {e}")
        return False

run_script = create_standardized_pipeline_script(
    "12_website.py",
    process_website_generation,
    "Static HTML website generation"
)

if __name__ == '__main__':
    sys.exit(run_script()) 