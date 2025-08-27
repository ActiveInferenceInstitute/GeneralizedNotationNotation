#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 4: Model Registry (Thin Orchestrator)

This step implements a centralized model registry for GNN models with versioning,
metadata management, and model lifecycle tracking.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/model_registry/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the model_registry module.

Pipeline Flow:
    main.py → 4_model_registry.py (this script) → model_registry/ (modular implementation)

Usage:
    python 4_model_registry.py [options]
    (Typically called by main.py)
"""

import sys
import logging
from pathlib import Path
import json
import datetime
from typing import Dict, Any, List, Optional, Union

# Import centralized utilities and configuration
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    ArgumentParser,
    performance_tracker
)

from pipeline import (
    get_output_dir_for_script,
    get_pipeline_config
)

from utils.pipeline_template import create_standardized_pipeline_script

# Initialize logger for this step
logger = setup_step_logging("4_model_registry", verbose=False)

# Import step-specific modules
try:
    from model_registry.registry import ModelRegistry
    
    DEPENDENCIES_AVAILABLE = True
    logger.debug("Successfully imported model registry dependencies")
    
except ImportError as e:
    log_step_warning(logger, f"Failed to import model registry dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False

def process_model_registry_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized model registry processing function.
    
    Args:
        target_dir: Directory containing GNN files to register
        output_dir: Output directory for registry data
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Start performance tracking
        with performance_tracker.track_operation("model_registry_processing", {"verbose": verbose, "recursive": recursive}):
            # Update logger verbosity if needed
            if verbose:
                logger.setLevel(logging.DEBUG)
            
            # Get configuration
            config = get_pipeline_config()
            step_config = config.get_step_config("4_model_registry") if hasattr(config, 'get_step_config') else None
            
            # Set up output directory
            step_output_dir = get_output_dir_for_script("4_model_registry.py", output_dir)
            step_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate step requirements
            if not DEPENDENCIES_AVAILABLE:
                log_step_warning(logger, "Model registry dependencies are not available, functionality will be limited")
            
            # Log processing parameters
            logger.info(f"Processing GNN files from: {target_dir}")
            logger.info(f"Output directory: {step_output_dir}")
            logger.info(f"Recursive processing: {recursive}")
            
            # Extract additional parameters from kwargs
            registry_path = kwargs.get('registry_path', step_output_dir / "model_registry.json")
            logger.debug(f"Registry path: {registry_path}")
            
            # Validate input directory
            if not target_dir.exists():
                log_step_error(logger, f"Input directory does not exist: {target_dir}")
                return False
            
            # Find GNN files
            pattern = "**/*.md" if recursive else "*.md"
            gnn_files = list(target_dir.glob(pattern))
            
            if not gnn_files:
                log_step_warning(logger, f"No GNN files found in {target_dir}")
                return True  # Not an error, just no files to process
            
            logger.info(f"Found {len(gnn_files)} GNN files to register")
            
            # Process files
            successful_files = 0
            failed_files = 0
            
            # Initialize model registry
            registry = ModelRegistry(registry_path) if DEPENDENCIES_AVAILABLE else None
            
            # Process each file
            for gnn_file in gnn_files:
                try:
                    if registry:
                        # Use actual registry implementation
                        success = registry.register_model(gnn_file)
                    else:
                        # Fallback implementation
                        success = register_model_fallback(gnn_file, step_output_dir)
                    
                    if success:
                        successful_files += 1
                    else:
                        failed_files += 1
                        
                except Exception as e:
                    log_step_error(logger, f"Unexpected error registering {gnn_file}: {e}")
                    failed_files += 1
            
            # Save registry
            if registry:
                registry.save()
            
            # Generate summary report
            summary_file = step_output_dir / "model_registry_summary.json"
            summary = {
                "timestamp": datetime.datetime.now().isoformat(),
                "step_name": "model_registry",
                "input_directory": str(target_dir),
                "output_directory": str(step_output_dir),
                "total_files": len(gnn_files),
                "successful_files": successful_files,
                "failed_files": failed_files,
                "registry_path": str(registry_path),
                "performance_metrics": performance_tracker.get_summary()
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Summary report saved: {summary_file}")
            
            # Determine success
            if failed_files == 0:
                log_step_success(logger, f"Successfully registered {successful_files} GNN models")
                return True
            elif successful_files > 0:
                log_step_warning(logger, f"Partially successful: {failed_files} files failed to register")
                return True  # Still consider successful for pipeline continuation
            else:
                log_step_error(logger, "All files failed to register")
                return False
            
    except Exception as e:
        log_step_error(logger, f"Model registry processing failed: {e}")
        if verbose:
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def register_model_fallback(
    gnn_file: Path, 
    output_dir: Path
) -> bool:
    """
    Fallback implementation for model registration when registry module is not available.
    
    Args:
        gnn_file: Path to GNN file
        output_dir: Output directory
        
    Returns:
        True if registration succeeded, False otherwise
    """
    try:
        # Read file content
        with open(gnn_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract model metadata (simple implementation)
        model_id = gnn_file.stem
        model_name = extract_model_name(content) or model_id
        
        # Create model entry
        model_entry = {
            "model_id": model_id,
            "model_name": model_name,
            "file_path": str(gnn_file),
            "file_size_bytes": gnn_file.stat().st_size,
            "registered_at": datetime.datetime.now().isoformat(),
            "version": "1.0.0"  # Default version
        }
        
        # Save model entry
        model_dir = output_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_metadata_file = model_dir / f"{model_id}_metadata.json"
        with open(model_metadata_file, 'w') as f:
            json.dump(model_entry, f, indent=2)
        
        logger.debug(f"Registered model {model_id} (fallback mode)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register model {gnn_file}: {e}")
        return False

def extract_model_name(content: str) -> Optional[str]:
    """
    Extract model name from GNN content.
    
    Args:
        content: GNN file content
        
    Returns:
        Model name if found, None otherwise
    """
    # Simple implementation - look for ModelName or title
    import re
    
    # Try to find ModelName: <name>
    model_name_match = re.search(r'ModelName:\s*([^\n]+)', content)
    if model_name_match:
        return model_name_match.group(1).strip()
    
    # Try to find # <name>
    title_match = re.search(r'^#\s+([^\n]+)', content)
    if title_match:
        return title_match.group(1).strip()
    
    return None

# Create standardized pipeline script
run_script = create_standardized_pipeline_script(
    "4_model_registry.py",
    process_model_registry_standardized,
    "Model versioning and management",
    additional_arguments={
        "registry_path": {"type": str, "help": "Path to model registry file"}
    }
)

if __name__ == '__main__':
    sys.exit(run_script()) 