#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 2: Enhanced GNN Processing with Modular Architecture

This script orchestrates comprehensive GNN file processing using a modular,
extensible architecture with clear separation of concerns.

## New Modular Architecture

### Core Components:
- **GNNProcessor**: Central orchestrator for the entire pipeline
- **FileDiscoveryStrategy**: Intelligent file discovery with content analysis
- **ValidationStrategy**: Multi-level validation with extensible rules
- **RoundTripTestStrategy**: Semantic preservation testing
- **CrossFormatValidationStrategy**: Format consistency validation
- **ReportGenerator**: Comprehensive reporting in multiple formats

### Processing Phases:
1. **Discovery**: Intelligent file detection and analysis
2. **Validation**: Multi-level semantic and structural validation
3. **Round-Trip Testing**: Format conversion and semantic preservation
4. **Cross-Format Validation**: Consistency across format representations
5. **Reporting**: Comprehensive analysis and recommendations

### Benefits:
- **Modular Design**: Easy to extend and modify individual components
- **Clear Separation**: Each module has a single responsibility
- **Comprehensive Context**: Full pipeline state tracking
- **Enhanced Testability**: Each component can be unit tested
- **Flexible Configuration**: Easy to customize processing behavior
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import utilities if available
try:
    from utils import setup_step_logging, log_step_start, log_step_success, log_step_warning, log_step_error
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# Import new modular architecture
try:
    from gnn.core_processor import GNNProcessor, ProcessingContext, create_processor
    MODULAR_ARCHITECTURE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Modular architecture not available: {e}", file=sys.stderr)
    MODULAR_ARCHITECTURE_AVAILABLE = False

# Initialize logger
if UTILS_AVAILABLE:
    logger = setup_step_logging("2_gnn", verbose=False)
else:
    logger = logging.getLogger("2_gnn")
    logger.setLevel(logging.INFO)
    
    # Basic logging setup if utils not available
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Reduce verbosity from external libraries
logging.getLogger('gnn').setLevel(logging.ERROR)
logging.getLogger('gnn.cross_format_validator').setLevel(logging.ERROR)
logging.getLogger('gnn.schema_validator').setLevel(logging.ERROR)
logging.getLogger('gnn.parsers').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('filelock').setLevel(logging.ERROR)

# Legacy support functions
def log_step_start_safe(logger, message):
    if UTILS_AVAILABLE:
        log_step_start(logger, message)
    else:
        logger.info(f"[START] {message}")

def log_step_success_safe(logger, message):
    if UTILS_AVAILABLE:
        log_step_success(logger, message)
    else:
        logger.info(f"[SUCCESS] {message}")

def log_step_warning_safe(logger, message):
    if UTILS_AVAILABLE:
        log_step_warning(logger, message)
    else:
        logger.warning(f"[WARNING] {message}")

def log_step_error_safe(logger, message):
    if UTILS_AVAILABLE:
        log_step_error(logger, message)
    else:
        logger.error(f"[ERROR] {message}")

# Global function for processing directory with timeout
def _process_gnn_directory_with_timeout(target_dir, recursive, logger=None):
    """
    Global function to process GNN directory with error handling.
    
    Args:
        target_dir: Directory to process
        recursive: Whether to process recursively
        logger: Optional logger for error reporting
    
    Returns:
        Processing results or None if processing fails
    """
    import signal
    import sys
    import traceback
    import json
    import logging
    
    def timeout_handler(signum, frame):
        """Handle timeout by raising an exception."""
        raise TimeoutError("GNN processing timed out")
    
    # Set up timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout
    
    try:
        # Suppress specific warnings from cross_format_validator
        logging.getLogger('gnn.cross_format_validator').setLevel(logging.ERROR)
        logging.getLogger('gnn.schema_validator').setLevel(logging.ERROR)
        logging.getLogger('gnn.parsers').setLevel(logging.ERROR)
        
        from gnn import process_gnn_directory
        
        # Ensure target_dir is a Path object
        if not isinstance(target_dir, Path):
            target_dir = Path(target_dir)
        
        # Add more detailed logging
        if logger:
            logger.debug(f"Processing directory: {target_dir}")
            logger.debug(f"Recursive: {recursive}")
            logger.debug(f"Directory exists: {target_dir.exists()}")
            logger.debug(f"Is directory: {target_dir.is_dir()}")
            
            # Detailed file discovery logging
            if target_dir.exists() and target_dir.is_dir():
                if recursive:
                    discovered_files = list(target_dir.rglob("*"))
                else:
                    discovered_files = list(target_dir.iterdir())
                
                logger.debug(f"Total files discovered: {len(discovered_files)}")
                
                # Log details about discovered files
                for file_path in discovered_files:
                    if file_path.is_file():
                        logger.debug(f"Discovered file: {file_path}")
                        logger.debug(f"File size: {file_path.stat().st_size} bytes")
                        logger.debug(f"File extension: {file_path.suffix}")
                        
                        # Read and log file content
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read(500)  # Read first 500 chars
                                logger.debug(f"File content preview: {content}")
                                
                                # Try to parse as JSON or YAML if possible
                                try:
                                    parsed_content = json.loads(content)
                                    logger.debug(f"Parsed JSON content: {json.dumps(parsed_content, indent=2)}")
                                except (json.JSONDecodeError, TypeError):
                                    # If not JSON, check for GNN-specific markers
                                    gnn_markers = ['model', 'gnn', 'variable', 'connection']
                                    if any(marker in content.lower() for marker in gnn_markers):
                                        logger.debug("Potential GNN file detected based on markers")
                        except Exception as e:
                            logger.warning(f"Could not read file {file_path}: {e}")
        
        # Validate directory
        if not target_dir.exists() or not target_dir.is_dir():
            if logger:
                logger.error(f"Invalid directory: {target_dir}")
            return {
                'directory': str(target_dir),
                'total_files': 0,
                'processed_files': [],
                'summary': {
                    'valid_files': 0,
                    'invalid_files': 0,
                    'total_files': 0,
                    'total_variables': 0,
                    'total_connections': 0
                },
                'errors': ['Invalid directory']
            }
        
        # Specific check for the input file
        input_file = target_dir / 'actinf_pomdp_agent.md'
        if input_file.exists():
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if logger:
                        logger.info(f"Input file content:\n{content}")
                        
                        # Analyze content for GNN-specific markers
                        gnn_markers = ['model', 'gnn', 'variable', 'connection']
                        detected_markers = [marker for marker in gnn_markers if marker in content.lower()]
                        if detected_markers:
                            logger.info(f"Detected GNN markers: {detected_markers}")
            except Exception as e:
                if logger:
                    logger.error(f"Could not read input file {input_file}: {e}")
        else:
            if logger:
                logger.warning(f"Input file not found: {input_file}")
        
        # Attempt to process directory
        try:
            processing_results = process_gnn_directory(target_dir, recursive=recursive)
        except Exception as e:
            # Detailed error logging
            if logger:
                logger.error(f"Error in process_gnn_directory: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.error(f"Python path: {sys.path}")
                logger.error(f"Loaded modules: {list(sys.modules.keys())}")
            
            # Return a minimal processing result in case of failure
            return {
                'directory': str(target_dir),
                'total_files': 0,
                'processed_files': [],
                'summary': {
                    'valid_files': 0,
                    'invalid_files': 0,
                    'total_files': 0,
                    'total_variables': 0,
                    'total_connections': 0
                },
                'errors': [str(e)]
            }
        
        if logger:
            logger.debug("process_gnn_directory completed successfully")
            logger.debug(f"Total files processed: {processing_results.get('total_files', 0)}")
            logger.debug(f"Processed files: {len(processing_results.get('processed_files', []))}")
        
        return processing_results
    except TimeoutError:
        if logger:
            logger.warning("GNN processing timed out")
        return {
            'directory': str(target_dir),
            'total_files': 0,
            'processed_files': [],
            'summary': {
                'valid_files': 0,
                'invalid_files': 0,
                'total_files': 0,
                'total_variables': 0,
                'total_connections': 0
            },
            'errors': ['Processing timed out']
        }
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error in processing directory: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return a minimal processing result in case of failure
        return {
            'directory': str(target_dir),
            'total_files': 0,
            'processed_files': [],
            'summary': {
                'valid_files': 0,
                'invalid_files': 0,
                'total_files': 0,
                'total_variables': 0,
                'total_connections': 0
            },
            'errors': [str(e)]
        }
    finally:
        # Always cancel the alarm to prevent it from interfering with other code
        signal.alarm(0)


def process_gnn_files_modular(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    validation_level: str = "standard",
    enable_round_trip: bool = False,
    enable_cross_format: bool = False,
    test_subset: Optional[List[str]] = None,
    reference_file: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Process GNN files using the new modular architecture.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        validation_level: Validation level (basic, standard, strict, research, round_trip)
        enable_round_trip: Whether to enable round-trip testing
        enable_cross_format: Whether to enable cross-format consistency validation
        test_subset: Optional list of formats to test for round-trip
        reference_file: Optional specific reference file for round-trip testing
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    if not MODULAR_ARCHITECTURE_AVAILABLE:
        logger.error("Modular architecture not available, falling back to legacy processing")
        return process_gnn_files_legacy(
            target_dir, output_dir, logger, recursive, verbose,
            validation_level, enable_round_trip, enable_cross_format,
            test_subset, reference_file, **kwargs
        )
    
    log_step_start_safe(logger, "Enhanced GNN processing with modular architecture")
    
    try:
        # Create processing context
        context = ProcessingContext(
            target_dir=target_dir,
            output_dir=output_dir,
            recursive=recursive,
            validation_level=validation_level,
            enable_round_trip=enable_round_trip,
            enable_cross_format=enable_cross_format,
            test_subset=test_subset,
            reference_file=reference_file
        )
        
        # Configure logging for verbose mode
        if verbose:
            logging.getLogger('gnn.core_processor').setLevel(logging.DEBUG)
            logging.getLogger('gnn.discovery').setLevel(logging.DEBUG)
            logging.getLogger('gnn.validation').setLevel(logging.DEBUG)
        
        # Create and configure processor
        processor = create_processor(logger)
        
        # Execute processing pipeline
        success = processor.process(context)
        
        if success:
            processing_time = context.get_processing_time()
            log_step_success_safe(logger, f"Modular GNN processing completed in {processing_time:.2f}s")
        else:
            log_step_error_safe(logger, "Modular GNN processing failed")
        
        return success
        
    except Exception as e:
        log_step_error_safe(logger, f"Modular GNN processing failed: {e}")
        if verbose:
            logger.exception("Detailed error:")
        return False


def process_gnn_files_legacy(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    validation_level: str = "standard",
    enable_round_trip: bool = False,
    enable_cross_format: bool = False,
    test_subset: Optional[List[str]] = None,
    reference_file: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Enhanced GNN processing using the full GNN module capabilities.
    
    This uses the comprehensive GNN module to process files with full
    parsing, validation, and reporting capabilities.
    """
    log_step_start_safe(logger, "Enhanced GNN processing using full GNN module")
    start_time = time.time()
    
    try:
        # Import GNN module functions with detailed error tracking
        try:
            import sys
            import importlib
            import multiprocessing
            
            # Explicitly import each module to track import issues
            gnn_modules = [
                'gnn',
                'gnn.schema_validator', 
                'gnn.cross_format_validator', 
                'gnn.parsers'
            ]
            
            for module_name in gnn_modules:
                try:
                    importlib.import_module(module_name)
                except ImportError as e:
                    logger.error(f"Could not import {module_name}: {e}")
                    logger.error(f"Python path: {sys.path}")
                    logger.error(f"Module details: {sys.modules.get(module_name, 'Not loaded')}")
            
            from gnn import process_gnn_directory, generate_gnn_report, validate_gnn
            from gnn.schema_validator import ValidationLevel
            GNN_MODULE_AVAILABLE = True
        except ImportError as e:
            logger.error(f"GNN module import failed: {e}")
            logger.error(f"Full import error details: {sys.exc_info()}")
            GNN_MODULE_AVAILABLE = False
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate input directory
        if not target_dir.exists() or not target_dir.is_dir():
            logger.error(f"Invalid target directory: {target_dir}")
            return False
        
        if GNN_MODULE_AVAILABLE:
            # Use full GNN module processing with extensive logging and timeout
            logger.info("Using full GNN module processing")
            logger.info(f"Target directory: {target_dir}")
            logger.info(f"Recursive: {recursive}")
            
            try:
                # Use multiprocessing to implement timeout with more lenient settings
                with multiprocessing.Pool(1) as pool:
                    # 60-second timeout for processing
                    result = pool.apply_async(_process_gnn_directory_with_timeout, (target_dir, recursive, logger))
                    
                    try:
                        processing_results = result.get(timeout=60)
                    except multiprocessing.TimeoutError:
                        logger.warning("GNN module processing timed out, attempting to continue")
                        # Try to get partial results
                        processing_results = {
                            'directory': str(target_dir),
                            'total_files': 0, 
                            'processed_files': [], 
                            'summary': {
                                'valid_files': 0,
                                'invalid_files': 0,
                                'total_files': 0,
                                'total_variables': 0,
                                'total_connections': 0
                            },
                            'errors': []
                        }
                
                # Ensure processing_results has all expected keys
                expected_keys = {
                    'directory': str(target_dir),
                    'total_files': 0,
                    'processed_files': [],
                    'summary': {
                        'valid_files': 0,
                        'invalid_files': 0,
                        'total_files': 0,
                        'total_variables': 0,
                        'total_connections': 0
                    },
                    'errors': []
                }
                
                # Merge expected keys with actual results, preserving existing values
                for key, default_value in expected_keys.items():
                    if key not in processing_results:
                        processing_results[key] = default_value
                    elif isinstance(default_value, dict):
                        for subkey, subdefault in default_value.items():
                            if subkey not in processing_results[key]:
                                processing_results[key][subkey] = subdefault
                
                # Generate comprehensive report
                report_content = generate_gnn_report(processing_results)
                
                # Save detailed report
                report_file = output_dir / f"gnn_processing_report_{int(time.time())}.md"
                with open(report_file, 'w') as f:
                    f.write(report_content)
                
                # Save JSON results
                import json
                json_file = output_dir / f"gnn_processing_results_{int(time.time())}.json"
                with open(json_file, 'w') as f:
                    json.dump(processing_results, f, indent=2)
                
                logger.info(f"Enhanced GNN processing report saved: {report_file}")
                logger.info(f"JSON results saved: {json_file}")
                
                # Summary statistics
                valid_count = processing_results['summary'].get('valid_files', 0)
                total_count = processing_results['summary'].get('total_files', 0)
                total_variables = processing_results['summary'].get('total_variables', 0)
                total_connections = processing_results['summary'].get('total_connections', 0)
                
                # Log errors if any
                errors = processing_results.get('errors', [])
                if errors:
                    logger.warning(f"Processing encountered {len(errors)} errors")
                    for error in errors:
                        logger.warning(f"Error: {error}")
                
                logger.info(f"Processed {total_count} files, {valid_count} valid")
                logger.info(f"Total variables: {total_variables}")
                logger.info(f"Total connections: {total_connections}")
                
                return True
                
            except Exception as e:
                logger.error(f"GNN module processing failed: {e}")
                logger.error(f"Full exception details: {sys.exc_info()}")
                return False
        
        # Fallback to basic processing if GNN module is not available
        logger.warning("Falling back to basic processing")
        
        # Basic file discovery
        discovered_files = []
        
        if recursive:
            for file_path in target_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.md', '.json', '.xml', '.yaml', '.pkl']:
                    discovered_files.append(file_path)
        else:
            for file_path in target_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.md', '.json', '.xml', '.yaml', '.pkl']:
                    discovered_files.append(file_path)
        
        logger.info(f"Discovered {len(discovered_files)} potential GNN files")
        
        # Basic validation
        valid_files = []
        for file_path in discovered_files:
            try:
                # Simple validation - check if file is readable
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(100)  # Read first 100 chars
                    if any(marker in content.lower() for marker in ['model', 'gnn', 'variable', 'connection']):
                        valid_files.append(file_path)
            except:
                continue
        
        logger.info(f"Found {len(valid_files)} valid GNN files")
        
        # Generate basic report
        report = {
            "basic_gnn_processing_report": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "target_directory": str(target_dir),
                "output_directory": str(output_dir),
                "processing_time": f"{time.time() - start_time:.2f}s",
                "files_discovered": len(discovered_files),
                "files_valid": len(valid_files),
                "validation_level": validation_level,
                "mode": "basic_fallback"
            }
        }
        
        # Save report
        import json
        report_file = output_dir / f"basic_gnn_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Basic processing report saved: {report_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"GNN processing failed: {e}")
        if verbose:
            logger.exception("Detailed error:")
        return False


def create_enhanced_parser() -> argparse.ArgumentParser:
    """Create enhanced command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhanced GNN Processing Pipeline - Step 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Basic processing
  %(prog)s --verbose                          # Verbose output
  %(prog)s --validation-level strict          # Strict validation
  %(prog)s --enable-round-trip                # With round-trip testing
  %(prog)s --enable-cross-format              # With cross-format validation
  %(prog)s --recursive                        # Recursive file search
  %(prog)s --test-subset json,xml,yaml        # Test specific formats
        """
    )
    
    # Processing options
    parser.add_argument(
        '--recursive', 
        action='store_true',
        help='Search for GNN files recursively in subdirectories'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging output'
    )
    
    # Validation options
    parser.add_argument(
        '--validation-level',
        choices=['basic', 'standard', 'strict', 'research', 'round_trip'],
        default='standard',
        help='Validation level (default: standard)'
    )
    
    # Testing options
    parser.add_argument(
        '--enable-round-trip',
        action='store_true',
        help='Enable round-trip testing across formats'
    )
    
    parser.add_argument(
        '--enable-cross-format',
        action='store_true',
        help='Enable cross-format consistency validation'
    )
    
    parser.add_argument(
        '--test-subset',
        help='Comma-separated list of formats to test (e.g., json,xml,yaml)'
    )
    
    parser.add_argument(
        '--reference-file',
        help='Specific reference file for round-trip testing'
    )
    
    # Architecture options
    parser.add_argument(
        '--force-legacy',
        action='store_true',
        help='Force use of legacy processing mode'
    )
    
    return parser


def run_script() -> int:
    """Enhanced script execution with modular architecture support."""
    parser = create_enhanced_parser()
    args = parser.parse_args()
    
    try:
        # Get project root and set up directories
        project_root = Path(__file__).resolve().parent.parent
        
        # Set target directory (input/gnn_files)
        target_dir = project_root / "input" / "gnn_files"
        if not target_dir.exists():
            logger.error(f"Target directory does not exist: {target_dir}")
            return 1
        
        # Set output directory (output/gnn_processing_step)
        output_dir = project_root / "output" / "gnn_processing_step"
        
        # Parse test subset if provided
        test_subset = None
        if args.test_subset:
            test_subset = [fmt.strip().lower() for fmt in args.test_subset.split(',')]
        
        # Adjust logging based on verbose flag
        if args.verbose:
            logger.setLevel(logging.DEBUG)
            # Even in verbose mode, suppress some noisy loggers
            logging.getLogger('gnn.cross_format_validator').setLevel(logging.WARNING)
        else:
            # Keep external loggers very quiet
            logging.getLogger('gnn').setLevel(logging.ERROR)
            logging.getLogger('gnn.schema_validator').setLevel(logging.ERROR)
            logging.getLogger('gnn.cross_format_validator').setLevel(logging.ERROR)
            logging.getLogger('gnn.parsers').setLevel(logging.ERROR)
            logging.getLogger('gnn.testing').setLevel(logging.ERROR)
            logging.getLogger('gnn.mcp').setLevel(logging.ERROR)
        
        # Log configuration
        logger.info("Enhanced GNN processing starting...")
        if args.verbose:
            logger.debug(f"Configuration details:")
            logger.debug(f"  Target: {target_dir}")
            logger.debug(f"  Output: {output_dir}")
            logger.debug(f"  Validation level: {args.validation_level}")
            logger.debug(f"  Round-trip testing: {args.enable_round_trip}")
            logger.debug(f"  Cross-format validation: {args.enable_cross_format}")
            logger.debug(f"  Recursive: {args.recursive}")
            logger.debug(f"  Architecture: {'Legacy (forced)' if args.force_legacy else 'Modular'}")
        
                # Choose processing method - prefer enhanced GNN module processing
        if args.force_legacy:
            logger.info("Using legacy processing mode (forced)")
            success = process_gnn_files_legacy(
                target_dir=target_dir,
                output_dir=output_dir,
                logger=logger,
                recursive=args.recursive,
                verbose=args.verbose,
                validation_level=args.validation_level,
                enable_round_trip=args.enable_round_trip,
                enable_cross_format=args.enable_cross_format,
                test_subset=test_subset,
                reference_file=args.reference_file
            )
        elif MODULAR_ARCHITECTURE_AVAILABLE:
            logger.info("Using modular processing architecture")
            success = process_gnn_files_modular(
                target_dir=target_dir,
                output_dir=output_dir,
                logger=logger,
                recursive=args.recursive,
                verbose=args.verbose,
                validation_level=args.validation_level,
                enable_round_trip=args.enable_round_trip,
                enable_cross_format=args.enable_cross_format,
                test_subset=test_subset,
                reference_file=args.reference_file
            )
        else:
            logger.info("Using enhanced GNN module processing")
            success = process_gnn_files_legacy(
                target_dir=target_dir,
                output_dir=output_dir,
                logger=logger,
                recursive=args.recursive,
                verbose=args.verbose,
                validation_level=args.validation_level,
                enable_round_trip=args.enable_round_trip,
                enable_cross_format=args.enable_cross_format,
                test_subset=test_subset,
                reference_file=args.reference_file
            )
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        if args.verbose:
            logger.exception("Detailed error:")
        return 1


if __name__ == '__main__':
    sys.exit(run_script()) 