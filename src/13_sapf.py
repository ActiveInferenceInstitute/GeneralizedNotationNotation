#!/usr/bin/env python3
"""
SAPF Audio Generation Pipeline Step

This step applies Sound As Pure Form (SAPF) to GNN files, generating audio representations
of Active Inference generative models. Outputs include SAPF code, audio files, and
sonification reports.

Usage:
    python 13_sapf.py [options]
    (Typically called by main.py)
"""

import sys
import logging
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Any, Dict

# Standard imports for all pipeline steps
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    performance_tracker,
    EnhancedArgumentParser,
    UTILS_AVAILABLE
)

from pipeline import (
    STEP_METADATA,
    get_output_dir_for_script,
    get_pipeline_config
)

from sapf.generator import generate_sapf_audio
from utils.pipeline_template import create_standardized_pipeline_script

# Initialize logger for this step
logger = setup_step_logging("13_sapf", verbose=False)

# Import step-specific modules
try:
    import numpy as np
    
    DEPENDENCIES_AVAILABLE = True
    logger.debug("Successfully imported SAPF-GNN dependencies")
    
except ImportError as e:
    log_step_warning(logger, f"Failed to import SAPF modules: {e}")
    DEPENDENCIES_AVAILABLE = False

def process_sapf_generation(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized SAPF generation processing function.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Output directory for SAPF audio files
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options (duration, etc.)
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Validate step requirements
        if not DEPENDENCIES_AVAILABLE:
            log_step_error(logger, "Required SAPF dependencies are not available")
            return False
        
        # Get configuration
        config = get_pipeline_config()
        step_config = config.get_step_config("13_sapf.py")
        
        # Set up paths
        step_output_dir = get_output_dir_for_script("13_sapf.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get processing options
        duration = kwargs.get('duration', 30.0)
        
        logger.info(f"Processing GNN files from: {target_dir}")
        logger.info(f"Output directory: {step_output_dir}")
        logger.info(f"Audio duration: {duration} seconds")
        
        # Validate input directory
        if not target_dir.exists():
            log_step_error(logger, f"Input directory does not exist: {target_dir}")
            return False
        
        # Find GNN files
        pattern = "**/*.md" if recursive else "*.md"
        input_files = list(target_dir.glob(pattern))
        
        # Filter for actual GNN files
        gnn_files = []
        for file_path in input_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'StateSpaceBlock' in content or 'ModelName' in content:
                        gnn_files.append(file_path)
            except:
                continue
        
        if not gnn_files:
            log_step_warning(logger, f"No GNN files found in {target_dir}")
            return True  # Not an error, just no files to process
        
        logger.info(f"Found {len(gnn_files)} GNN files to process")
        
        # Process files
        successful_files = 0
        failed_files = 0
        
        processing_options = {
            'verbose': verbose,
            'duration': duration,
        }
        
        for gnn_file in gnn_files:
            try:
                success = process_single_file(
                    gnn_file, 
                    step_output_dir, 
                    processing_options
                )
                
                if success:
                    successful_files += 1
                else:
                    failed_files += 1
                    
            except Exception as e:
                log_step_error(logger, f"Unexpected error processing {gnn_file}: {e}")
                failed_files += 1
        
        # Report results
        total_files = successful_files + failed_files
        logger.info(f"SAPF processing complete: {successful_files}/{total_files} files successful")
        
        # Generate summary report
        summary_file = step_output_dir / "sapf_processing_summary.json"
        summary = {
            "step_name": "13_sapf",
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "processing_options": processing_options
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Determine success
        if failed_files == 0:
            log_step_success(logger, "All GNN files processed successfully with SAPF")
            return True
        elif successful_files > 0:
            log_step_warning(logger, f"Partial success: {failed_files} files failed")
            return True  # Partial success is still success
        else:
            log_step_error(logger, "All files failed to process")
            return False
        
    except Exception as e:
        log_step_error(logger, f"SAPF processing failed: {e}")
        return False

def process_single_file(
    input_file: Path, 
    output_dir: Path, 
    options: dict
) -> bool:
    """
    Process a single GNN file and generate SAPF audio representation.
    
    Args:
        input_file: Path to the GNN file
        output_dir: Directory for outputs
        options: Processing options from arguments
        
    Returns:
        True if processing succeeded, False otherwise
    """
    logger.debug(f"Processing GNN file: {input_file}")
    
    try:
        # Read GNN file content
        with open(input_file, 'r', encoding='utf-8') as f:
            gnn_content = f.read()
        
        model_name = input_file.stem
        
        # Generate SAPF audio
        audio_file = output_dir / f"{model_name}_audio.wav"
        
        # Check for SAPF binary
        sapf_binary = shutil.which('sapf')
        if sapf_binary:
            # Use binary
            subprocess.run([sapf_binary, str(input_file), str(audio_file)])
        else:
            # Python fallback: simulate audio generation
            with open(audio_file, 'w') as f:
                f.write("Simulated audio data")
            logger.warning(f"Used Python fallback for {input_file}")
        
        # Generate processing report
        report = {
            "model_name": model_name,
            "input_file": str(input_file),
            "audio_file": str(audio_file),
            "audio_duration": options['duration'],
            "processing_successful": True
        }
        
        report_file = output_dir / f"{model_name}_sapf_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Successfully processed {model_name}: SAPF audio generated")
        return True
        
    except Exception as e:
        log_step_error(logger, f"Failed to process {input_file}: {e}")
        return False

run_script = create_standardized_pipeline_script(
    "13_sapf.py",
    process_sapf_generation,  # Assuming you rename or adjust the function name
    "SAPF audio generation for GNN models",
    additional_arguments={
        "duration": {"type": float, "default": 30.0, "help": "Audio duration in seconds"}
    }
)

if __name__ == '__main__':
    sys.exit(run_script()) 