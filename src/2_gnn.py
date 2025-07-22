#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 2: GNN File Discovery and Processing

This script discovers and processes GNN files with comprehensive validation,
round-trip testing, and cross-format consistency checking.

Usage:
    python 2_gnn.py [options]
    (Typically called by main.py)
"""

import sys
import logging
import signal
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

# Import centralized utilities and configuration
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    performance_tracker,
    UTILS_AVAILABLE
)

from pipeline import (
    get_output_dir_for_script
)

from utils.pipeline_template import create_standardized_pipeline_script

# Initialize logger for this step
logger = setup_step_logging("2_gnn", verbose=False)

# Import GNN processing modules
try:
    from gnn.core_processor import GNNProcessor, ProcessingContext
    from gnn.parsers import UnifiedGNNParser
    from gnn.schema_validator import GNNValidator
    from gnn.cross_format_validator import CrossFormatValidator
    from gnn.testing import RoundTripTestStrategy
    from gnn import process_gnn_directory_lightweight
    GNN_MODULES_AVAILABLE = True
    logger.debug("Successfully imported GNN processing modules")
except ImportError as e:
    log_step_warning(logger, f"Failed to import GNN modules: {e}")
    GNN_MODULES_AVAILABLE = False

# Safety functions for handling timeouts and signals
def log_step_start_safe(logger, message):
    try:
        log_step_start(logger, message)
    except Exception:
        logger.info(f"🚀 {message}")

def log_step_success_safe(logger, message):
    try:
        log_step_success(logger, message)
    except Exception:
        logger.info(f"✅ {message}")

def log_step_warning_safe(logger, message):
    try:
        log_step_warning(logger, message)
    except Exception:
        logger.warning(f"⚠️ {message}")

def log_step_error_safe(logger, message):
    try:
        log_step_error(logger, message)
    except Exception:
        logger.error(f"❌ {message}")

def setup_signal_handler():
    """Setup signal handler for graceful timeout handling."""
    def signal_handler(signum, frame):
        log_step_warning_safe(logger, "GNN processing interrupted by timeout")
        raise TimeoutError("GNN processing timed out")
    
    # Handle both SIGALRM (timeout) and SIGINT (Ctrl+C)
    signal.signal(signal.SIGALRM, signal_handler)
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, signal_handler)
    
    return signal_handler

def clear_signal_handler():
    """Clear signal handlers and cancel any pending alarms."""
    try:
        signal.alarm(0)  # Cancel any pending alarms
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal.SIG_DFL)
    except:
        pass

def process_gnn_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    validation_level: str = "standard",
    enable_round_trip: bool = True,
    enable_cross_format: bool = True,
    test_subset: Optional[List[str]] = None,
    reference_file: Optional[str] = None,
    force_legacy: bool = False,
    **kwargs
) -> bool:
    """
    Standardized GNN processing function that follows the pipeline template pattern.
    
    Args:
        target_dir: Directory containing GNN files
        output_dir: Output directory for processing results
        logger: Logger instance for this step
        recursive: Whether to search for GNN files recursively
        verbose: Whether to enable verbose logging
        validation_level: Validation level (basic, standard, strict, research, round_trip)
        enable_round_trip: Whether to enable round-trip testing
        enable_cross_format: Whether to enable cross-format validation
        test_subset: Subset of formats to test (comma-separated string)
        reference_file: Reference file for testing
        force_legacy: Whether to force legacy processing mode
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        log_step_start_safe(logger, "Starting GNN file discovery and processing")
        
        # Update logger verbosity if needed
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Validate GNN modules availability
        if not GNN_MODULES_AVAILABLE:
            log_step_warning_safe(logger, "GNN modules not available, using lightweight processing")
            return _process_lightweight_mode(target_dir, output_dir, logger, recursive)
        
        # Parse test subset if provided as string
        parsed_test_subset = None
        if test_subset and isinstance(test_subset, str):
            parsed_test_subset = [fmt.strip() for fmt in test_subset.split(',')]
        elif test_subset and isinstance(test_subset, list):
            parsed_test_subset = test_subset
        
        # Set up processing context
        context = ProcessingContext(
            target_dir=target_dir,
            output_dir=output_dir,
            recursive=recursive,
            validation_level=validation_level,
            enable_round_trip=enable_round_trip,
            enable_cross_format=enable_cross_format,
            test_subset=parsed_test_subset,
            reference_file=reference_file,
            verbose=verbose
        )
        
        # Use modular architecture unless forced to use legacy
        if force_legacy:
            log_step_warning_safe(logger, "Using legacy processing mode (forced)")
            return _process_legacy_mode(context, logger)
        else:
            logger.info("Using enhanced modular GNN processing")
            return _process_modular_mode(context, logger)
        
    except Exception as e:
        log_step_error_safe(logger, f"GNN processing failed: {e}")
        if verbose:
            logger.exception("Detailed error:")
        return False

def _process_modular_mode(context: ProcessingContext, logger: logging.Logger) -> bool:
    """Process GNN files using the modular architecture."""
    try:
        # Initialize GNN processor
        processor = GNNProcessor(logger=logger)
        
        # Set up timeout handling
        setup_signal_handler()
        
        # Process with timeout (10 minutes for GNN processing)
        signal.alarm(600)
        success = processor.process(context)
        signal.alarm(0)  # Cancel timeout
        
        # Clear signal handlers
        clear_signal_handler()
        
        if success:
            log_step_success_safe(logger, "GNN processing completed successfully")
            return True
        else:
            log_step_warning_safe(logger, "GNN processing completed with warnings")
            return True  # Still consider successful for pipeline continuation
            
    except TimeoutError:
        log_step_error_safe(logger, "GNN processing timed out after 10 minutes")
        clear_signal_handler()
        return False
    except Exception as e:
        log_step_error_safe(logger, f"Modular GNN processing failed: {e}")
        clear_signal_handler()
        return False

def _process_legacy_mode(context: ProcessingContext, logger: logging.Logger) -> bool:
    """Process GNN files using legacy processing for compatibility."""
    try:
        # Set up output directories
        context.output_dir.mkdir(parents=True, exist_ok=True)
        round_trip_dir = context.output_dir / "round_trip_tests"
        round_trip_dir.mkdir(parents=True, exist_ok=True)
        export_dir = context.output_dir / "format_exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Process GNN files with legacy validator
        validator = GNNValidator(logger=logger)
        validator.set_validation_level(context.validation_level)
        
        # Find GNN files
        gnn_files = _find_gnn_files(context.target_dir, context.recursive)
        if not gnn_files:
            log_step_warning_safe(logger, f"No GNN files found in {context.target_dir}")
            return True
        
        logger.info(f"Found {len(gnn_files)} GNN files to process")
        
        # Process each file
        successful_files = 0
        for gnn_file in gnn_files:
            try:
                logger.debug(f"Processing {gnn_file}")
                result = validator.validate_file(gnn_file)
                if result.is_valid:
                    successful_files += 1
                else:
                    logger.warning(f"Validation issues in {gnn_file}: {result.error_summary}")
            except Exception as e:
                logger.error(f"Failed to process {gnn_file}: {e}")
        
        # Generate summary
        if successful_files == len(gnn_files):
            log_step_success_safe(logger, f"All {len(gnn_files)} GNN files processed successfully")
        elif successful_files > 0:
            log_step_warning_safe(logger, f"Processed {successful_files}/{len(gnn_files)} files successfully")
        else:
            log_step_error_safe(logger, "No files were successfully processed")
            return False
        
        return True
        
    except Exception as e:
        log_step_error_safe(logger, f"Legacy GNN processing failed: {e}")
        return False

def _process_lightweight_mode(target_dir: Path, output_dir: Path, logger: logging.Logger, recursive: bool) -> bool:
    """Lightweight processing mode when GNN modules are not available."""
    try:
        log_step_warning_safe(logger, "Using lightweight GNN processing (modules not available)")
        
        # Use lightweight processing if available
        if 'process_gnn_directory_lightweight' in globals():
            result = process_gnn_directory_lightweight(
                target_dir=target_dir,
                recursive=recursive
            )
            
            # Save lightweight result
            output_dir.mkdir(parents=True, exist_ok=True)
            summary_file = output_dir / "gnn_lightweight_summary.json"
            
            import json
            with open(summary_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            log_step_success_safe(logger, f"Lightweight processing completed: {summary_file}")
            return True
        else:
            # Basic file discovery fallback
            gnn_files = _find_gnn_files(target_dir, recursive)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            discovery_file = output_dir / "gnn_discovery_report.json"
            
            report = {
                "target_dir": str(target_dir),
                "recursive": recursive,
                "files_found": [str(f) for f in gnn_files],
                "total_files": len(gnn_files),
                "processing_mode": "discovery_only"
            }
            
            import json
            with open(discovery_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            log_step_success_safe(logger, f"Basic file discovery completed: {discovery_file}")
            return True
            
    except Exception as e:
        log_step_error_safe(logger, f"Lightweight processing failed: {e}")
        return False

def _find_gnn_files(target_dir: Path, recursive: bool = False) -> List[Path]:
    """Find GNN files in the target directory."""
    gnn_files = []
    
    # Common GNN file patterns
    patterns = ["*.md", "*.gnn", "*.json", "*.yaml", "*.yml"]
    
    if recursive:
        for pattern in patterns:
            gnn_files.extend(target_dir.rglob(pattern))
    else:
        for pattern in patterns:
            gnn_files.extend(target_dir.glob(pattern))
    
    # Filter for actual GNN files (basic check)
    filtered_files = []
    for file_path in gnn_files:
        if file_path.is_file():
            # Basic GNN content detection
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Read first 1KB
                    if any(keyword in content.lower() for keyword in 
                           ['gnn', 'active inference', 'pomdp', 'generative model']):
                        filtered_files.append(file_path)
            except (UnicodeDecodeError, IOError):
                continue
    
    return filtered_files

# Create standardized pipeline script using the template
run_script = create_standardized_pipeline_script(
    "2_gnn.py",
    process_gnn_standardized,
    "GNN file discovery, parsing, and validation with round-trip testing",
    additional_arguments={
        "validation_level": {
            "type": str,
            "choices": ["basic", "standard", "strict", "research", "round_trip"],
            "default": "standard",
            "help": "Validation level for GNN processing"
        },
        "enable_round_trip": {
            "type": bool,
            "default": True,
            "help": "Enable comprehensive round-trip testing across all formats"
        },
        "enable_cross_format": {
            "type": bool,
            "default": True,
            "help": "Enable cross-format consistency validation"
        },
        "test_subset": {
            "type": str,
            "default": None,
            "help": "Comma-separated list of formats to test (e.g., json,xml,yaml)"
        },
        "reference_file": {
            "type": str,
            "default": None,
            "help": "Specific reference file for round-trip testing"
        },
        "force_legacy": {
            "type": bool,
            "default": False,
            "help": "Force use of legacy processing mode for compatibility"
        }
    }
)

if __name__ == '__main__':
    sys.exit(run_script()) 