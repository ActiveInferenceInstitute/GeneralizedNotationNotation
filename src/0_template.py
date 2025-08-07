#!/usr/bin/env python3
"""
Step 0: Pipeline Template with Infrastructure Demonstration (Thin Orchestrator)

This step demonstrates the thin orchestrator pattern in the GNN pipeline architecture.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/template/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the template module.

Pipeline Flow:
    main.py → 0_template.py (this script) → template/ (modular implementation)

This template serves as the foundation pattern for all other pipeline steps.

How to run:
  python src/0_template.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Template processing results in the specified output directory
  - Infrastructure demonstration and pattern validation
  - Comprehensive error handling and recovery demonstrations
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that template dependencies are installed
  - Check that src/template/ contains template modules
  - Check that the output directory is writable
  - Verify template configuration and pattern setup
"""

import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Pipeline utilities with error handling
try:
    from utils.pipeline_template import (
        setup_step_logging,
        log_step_start,
        log_step_success, 
        log_step_error,
        log_step_warning
    )
    from utils.argument_utils import EnhancedArgumentParser
    from pipeline.config import get_output_dir_for_script, get_pipeline_config
    
    # Infrastructure imports
    from utils.error_recovery import ErrorRecoverySystem, ErrorSeverity, RecoveryStrategy
    from utils.resource_manager import ResourceTracker, performance_tracker as resource_performance_tracker, get_system_info
    from utils.performance_tracker import performance_tracker
    from utils.logging_utils import PipelineLogger, set_correlation_context
    
    ENHANCED_INFRASTRUCTURE_AVAILABLE = True
    
except ImportError as e:
    # Fallback imports for minimal functionality
    import logging
    logging.basicConfig(level=logging.INFO)
    
    def setup_step_logging(name, args): return logging.getLogger(name)
    def log_step_start(logger, msg): logger.info(f"🚀 {msg}")
    def log_step_success(logger, msg): logger.info(f"✅ {msg}")
    def log_step_error(logger, msg): logger.error(f"❌ {msg}")
    def log_step_warning(logger, msg): logger.warning(f"⚠️ {msg}")
    
    ENHANCED_INFRASTRUCTURE_AVAILABLE = False
    print(f"⚠️ Infrastructure not available: {e}")

# Import core template functions from template module
try:
    from template import (
        process_template_standardized,
        generate_correlation_id,
        safe_template_execution,
        demonstrate_utility_patterns
    )
    TEMPLATE_AVAILABLE = True
except ImportError:
    TEMPLATE_AVAILABLE = False
    # Fallback function definitions if template module is not available
    def process_template_standardized(*args, **kwargs):
        return False
    
    def generate_correlation_id():
        import uuid
        return str(uuid.uuid4())[:8]
    
    def safe_template_execution(logger, correlation_id):
        from contextlib import contextmanager
        @contextmanager
        def dummy_context():
            yield {"correlation_id": correlation_id}
        return dummy_context()
    
    def demonstrate_utility_patterns(context, logger):
        return {"error": "Template module not available"}

def process_template_standardized_wrapper(
    target_dir: Path,
    output_dir: Path,
    logger,
    recursive: bool = False,
    verbose: bool = False,
    simulate_error: bool = False,
    **kwargs
) -> bool:
    """
    Standardized template processing function.
    
    Args:
        target_dir: Directory containing files to process
        output_dir: Output directory for template results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        simulate_error: Whether to simulate an error for testing
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Check if template module is available
        if not TEMPLATE_AVAILABLE:
            log_step_warning(logger, "Template module not available, using fallback functions")
        
        # Get pipeline configuration
        config = get_pipeline_config()
        step_output_dir = get_output_dir_for_script("0_template.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, "Processing template")
        
        # Generate correlation ID for this execution
        correlation_id = generate_correlation_id()
        
        # Use safe template execution context
        with safe_template_execution(logger, correlation_id) as context:
            # Demonstrate utility patterns
            demonstration_results = demonstrate_utility_patterns(context, logger)
            
            # Process template using the modular function
            success = process_template_standardized(
                target_dir=target_dir,
                output_dir=step_output_dir,
                logger=logger,
                recursive=recursive,
                verbose=verbose,
                **kwargs
            )
            
            # Save demonstration results
            demo_file = step_output_dir / "template_demonstration_results.json"
            with open(demo_file, 'w') as f:
                json.dump(demonstration_results, f, indent=2)
            
            # Save template results
            template_results = {
                "timestamp": datetime.now().isoformat(),
                "correlation_id": correlation_id,
                "source_directory": str(target_dir),
                "output_directory": str(step_output_dir),
                "success": success,
                "demonstration_results": demonstration_results,
                "infrastructure_available": ENHANCED_INFRASTRUCTURE_AVAILABLE,
                "template_module_available": TEMPLATE_AVAILABLE
            }
            
            template_results_file = step_output_dir / "template_results.json"
            with open(template_results_file, 'w') as f:
                json.dump(template_results, f, indent=2)
            
            if success:
                log_step_success(logger, "Template processing completed successfully")
            else:
                log_step_error(logger, "Template processing failed")
            
            return success
        
    except Exception as e:
        log_step_error(logger, f"Template processing failed: {e}")
        return False

def main():
    """Main template processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("0_template.py")
    
    # Setup logging
    logger = setup_step_logging("template", args)
    
    # Check if template module is available
    if not TEMPLATE_AVAILABLE:
        log_step_warning(logger, "Template module not available, using fallback functions")
    
    # Process template
    success = process_template_standardized_wrapper(
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        logger=logger,
        recursive=args.recursive,
        verbose=args.verbose,
        simulate_error=getattr(args, 'simulate_error', False)
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 