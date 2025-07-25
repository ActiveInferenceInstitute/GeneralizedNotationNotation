"""
Template Step Processor

This module contains the core functionality for the template step.
It provides functions for processing files and directories using the standardized pipeline pattern.
"""

import logging
from pathlib import Path
import json
import datetime
from typing import Dict, Any, List, Optional, Union

# Import utilities
try:
    from utils import (
        setup_step_logging,
        log_step_start,
        log_step_success, 
        log_step_warning,
        log_step_error,
        performance_tracker
    )
    UTILS_AVAILABLE = True
except ImportError:
    # Fallback logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    UTILS_AVAILABLE = False
    
    # Create minimal compatibility functions
    def log_step_start(logger, message): logger.info(f"🚀 {message}")
    def log_step_success(logger, message): logger.info(f"✅ {message}")
    def log_step_warning(logger, message): logger.warning(f"⚠️ {message}")
    def log_step_error(logger, message): logger.error(f"❌ {message}")
    
    # Create minimal performance tracker
    class DummyPerformanceTracker:
        def track_operation(self, name, metadata=None):
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyContext()
        def get_summary(self): return {}
    
    performance_tracker = DummyPerformanceTracker()

# Initialize logger
logger = logging.getLogger(__name__)

def process_template_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process files in a directory using the template processor.
    
    Args:
        target_dir: Directory containing files to process
        output_dir: Output directory for processing results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Start performance tracking
        with performance_tracker.track_operation("template_processing", {"verbose": verbose, "recursive": recursive}):
            # Update logger verbosity if needed
            if verbose:
                logger.setLevel(logging.DEBUG)
            
            # Set up output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Log processing parameters
            logger.info(f"Processing files from: {target_dir}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Recursive processing: {recursive}")
            
            # Extract additional parameters from kwargs
            example_param = kwargs.get('example_param', 'default_value')
            logger.debug(f"Example parameter: {example_param}")
            
            # Validate input directory
            if not target_dir.exists():
                log_step_error(logger, f"Input directory does not exist: {target_dir}")
                return False
            
            # Find files to process
            pattern = "**/*.*" if recursive else "*.*"
            input_files = list(target_dir.glob(pattern))
            
            if not input_files:
                log_step_warning(logger, f"No files found in {target_dir}")
                return True  # Not an error, just no files to process
            
            logger.info(f"Found {len(input_files)} files to process")
            
            # Process files
            successful_files = 0
            failed_files = 0
            
            processing_options = {
                'verbose': verbose,
                'recursive': recursive,
                'example_param': example_param,
                # Add other options from kwargs as needed
            }
            
            for input_file in input_files:
                try:
                    success = process_single_file(
                        input_file, 
                        output_dir, 
                        processing_options
                    )
                    
                    if success:
                        successful_files += 1
                    else:
                        failed_files += 1
                        
                except Exception as e:
                    log_step_error(logger, f"Unexpected error processing {input_file}: {e}")
                    failed_files += 1
            
            # Generate summary report
            summary_file = output_dir / "template_processing_summary.json"
            summary = {
                "timestamp": datetime.datetime.now().isoformat(),
                "step_name": "template",
                "input_directory": str(target_dir),
                "output_directory": str(output_dir),
                "total_files": len(input_files),
                "successful_files": successful_files,
                "failed_files": failed_files,
                "processing_options": processing_options,
                "performance_metrics": performance_tracker.get_summary()
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Summary report saved: {summary_file}")
            
            # Determine success
            if failed_files == 0:
                log_step_success(logger, f"Successfully processed {successful_files} files")
                return True
            elif successful_files > 0:
                log_step_warning(logger, f"Partially successful: {failed_files} files failed")
                return True  # Still consider successful for pipeline continuation
            else:
                log_step_error(logger, "All files failed to process")
                return False
            
    except Exception as e:
        log_step_error(logger, f"Template processing failed: {e}")
        if verbose:
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def process_single_file(
    input_file: Path, 
    output_dir: Path, 
    options: Dict[str, Any]
) -> bool:
    """
    Process a single file.
    
    Args:
        input_file: Path to input file
        output_dir: Output directory for results
        options: Processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        logger.debug(f"Processing file: {input_file}")
        
        # Read file content
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create file-specific output directory
        file_output_dir = output_dir / input_file.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output file
        output_file = file_output_dir / f"{input_file.stem}_processed{input_file.suffix}"
        
        # Process content (replace with actual processing logic)
        processed_content = f"""
# Processed by GNN Pipeline Template
# Original file: {input_file}
# Processed on: {datetime.datetime.now().isoformat()}
# Options: {options}

{content}
"""
        
        # Write processed content
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        # Generate processing report
        report = {
            "input_file": str(input_file),
            "output_file": str(output_file),
            "timestamp": datetime.datetime.now().isoformat(),
            "file_size_bytes": len(content),
            "file_size_lines": len(content.splitlines()),
            "processing_options": options
        }
        
        report_file = file_output_dir / f"{input_file.stem}_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.debug(f"Successfully processed {input_file.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {input_file}: {e}")
        return False

def validate_file(input_file: Path) -> Dict[str, Any]:
    """
    Validate a file for processing.
    
    Args:
        input_file: Path to input file
        
    Returns:
        Validation result with status and details
    """
    try:
        # Check if file exists
        if not input_file.exists():
            return {
                "status": "error",
                "error": "File does not exist",
                "file_path": str(input_file)
            }
        
        # Check if file is readable
        if not input_file.is_file():
            return {
                "status": "error",
                "error": "Path is not a file",
                "file_path": str(input_file)
            }
        
        # Read file content for validation
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read(1024)  # Read first 1KB for validation
        except Exception as e:
            return {
                "status": "error",
                "error": f"File is not readable: {e}",
                "file_path": str(input_file)
            }
        
        # Validate file format (example: check for specific markers)
        # This is just a placeholder - replace with actual validation logic
        is_valid = True
        validation_messages = []
        
        # Return validation result
        return {
            "status": "valid" if is_valid else "invalid",
            "file_path": str(input_file),
            "file_size_bytes": input_file.stat().st_size,
            "validation_messages": validation_messages
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "file_path": str(input_file)
        } 