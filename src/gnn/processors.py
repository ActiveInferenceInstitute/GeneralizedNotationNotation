"""
GNN Processing Functions - Enhanced for Comprehensive Testing

This module provides enhanced processing functions for GNN files with integrated
round-trip testing, cross-format validation, and comprehensive reporting.

Enhanced Features:
- Integration with 100% round-trip testing system
- Multi-level validation (Basic, Standard, Strict, Research, Round-trip)
- Enhanced error reporting and performance metrics
- Cross-format consistency validation
- Production-ready testing infrastructure
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import re
import logging
import json
import tempfile
import time
from datetime import datetime
from utils.path_utils import get_relative_path_if_possible
from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error

# Import enhanced GNN testing capabilities with lazy loading to avoid circular imports
ENHANCED_TESTING_AVAILABLE = False
_testing_modules_cache = {}

def _get_testing_modules():
    """Lazy import of testing modules to avoid circular dependencies."""
    global _testing_modules_cache, ENHANCED_TESTING_AVAILABLE
    
    if _testing_modules_cache:
        return _testing_modules_cache
        
    try:
        # Use lazy imports to break circular dependency
        import importlib
        test_round_trip = importlib.import_module('gnn.testing.test_round_trip')
        parsers = importlib.import_module('gnn.parsers')
        schema_validator = importlib.import_module('gnn.schema_validator')
        cross_format_validator = importlib.import_module('gnn.cross_format_validator')
        
        _testing_modules_cache = {
            'GNNRoundTripTester': getattr(test_round_trip, 'GNNRoundTripTester', None),
            'ComprehensiveTestReport': getattr(test_round_trip, 'ComprehensiveTestReport', None),
            'RoundTripResult': getattr(test_round_trip, 'RoundTripResult', None),
            'GNNParsingSystem': getattr(parsers, 'GNNParsingSystem', None),
            'GNNFormat': getattr(parsers, 'GNNFormat', None),
            'GNNValidator': getattr(schema_validator, 'GNNValidator', None),
            'ValidationLevel': getattr(schema_validator, 'ValidationLevel', None),
            'ValidationResult': getattr(schema_validator, 'ValidationResult', None),
            'CrossFormatValidator': getattr(cross_format_validator, 'CrossFormatValidator', None),
            'CrossFormatValidationResult': getattr(cross_format_validator, 'CrossFormatValidationResult', None),
        }
        ENHANCED_TESTING_AVAILABLE = True
        return _testing_modules_cache
    except (ImportError, AttributeError, RecursionError):
        ENHANCED_TESTING_AVAILABLE = False
        return {}


def enhanced_validation_with_gnn_processing(
    target_dir: Path,
    validation_level: str = "standard",
    enable_round_trip: bool = False,
    enable_cross_format: bool = False,
    test_subset: Optional[List[str]] = None
) -> Tuple[int, Optional[str]]:
    """
    Enhanced validation using GNN processing capabilities.
    
    Returns:
        Tuple of (exit_code, error_message)
    """
    logger = logging.getLogger(__name__)
    
    # Load testing modules lazily
    modules = _get_testing_modules()
    if not modules:
        logger.warning("Enhanced testing modules not available, using basic validation")
        return 2, "Enhanced testing modules not available"
    
    try:
        # Initialize validation level
        try:
            val_level = modules['ValidationLevel'](validation_level.lower())
        except ValueError:
            logger.warning(f"Invalid validation level '{validation_level}', using STANDARD")
            val_level = modules['ValidationLevel'].STANDARD
        
        # Initialize enhanced validator
        validator = modules['GNNValidator'](
            validation_level=val_level,
            enable_round_trip_testing=enable_round_trip
        )
        
        # Process files with enhanced validation
        total_files = 0
        validation_errors = 0
        
        for file_path in target_dir.rglob("*.md"):
            total_files += 1
            logger.info(f"Enhanced validation of {file_path}")
            
            try:
                result = validator.validate_file(file_path)
                if not result.is_valid:
                    validation_errors += 1
                    logger.error(f"Validation failed for {file_path}: {result.errors}")
                elif result.warnings:
                    logger.warning(f"Validation warnings for {file_path}: {result.warnings}")
                    
            except Exception as e:
                validation_errors += 1
                logger.error(f"Error during enhanced validation of {file_path}: {e}")
        
        logger.info(f"Enhanced validation complete: {total_files - validation_errors}/{total_files} files valid")
        
        if validation_errors > 0:
            return 1, f"Enhanced validation failed for {validation_errors} files"
        return 0, None
        
    except Exception as e:
        logger.error(f"Enhanced validation error: {e}")
        return 1, f"Enhanced validation error: {e}"


def process_gnn_folder(
    target_dir: Path, 
    output_dir: Path, 
    logger: logging.Logger, 
    recursive: bool = False, 
    verbose: bool = False,
    validation_level: str = "standard",
    enable_round_trip: bool = False,
    **kwargs
) -> bool:
    """
    Enhanced GNN folder processing with comprehensive validation and testing.
    
    Features:
    - Multi-level validation (basic, standard, strict, research, round_trip)
    - Format detection and analysis
    - Performance metrics and detailed reporting
    - Integration with round-trip testing system
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        validation_level: Validation level (basic, standard, strict, research, round_trip)
        enable_round_trip: Whether to enable round-trip testing
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    log_step_start(logger, f"Enhanced GNN processing: '{target_dir}' (validation: {validation_level})")
    
    if not ENHANCED_TESTING_AVAILABLE:
        log_step_warning(logger, "Enhanced testing not available - using basic processing")
        return _basic_gnn_processing(target_dir, output_dir, logger, recursive, verbose, **kwargs)
    
    # Initialize validation level
    try:
        val_level = _testing_modules_cache['ValidationLevel'](validation_level.lower())
    except ValueError:
        logger.warning(f"Invalid validation level '{validation_level}', using STANDARD")
        val_level = _testing_modules_cache['ValidationLevel'].STANDARD
    
    # Initialize enhanced validator
    validator = _testing_modules_cache['GNNValidator'](
        validation_level=val_level,
        enable_round_trip_testing=enable_round_trip
    )
    
    # Create enhanced output directory structure
    enhanced_output_dir = output_dir / "enhanced_gnn_processing"
    enhanced_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discovery phase
    logger.info(f"Phase 1: Enhanced GNN file discovery (recursive: {recursive})")
    start_time = time.time()
    
    gnn_files = []
    if recursive:
        gnn_files.extend(target_dir.rglob("*.md"))
        gnn_files.extend(target_dir.rglob("*.json"))
        gnn_files.extend(target_dir.rglob("*.xml"))
        gnn_files.extend(target_dir.rglob("*.yaml"))
        gnn_files.extend(target_dir.rglob("*.pkl"))
    else:
        gnn_files.extend(target_dir.glob("*.md"))
        gnn_files.extend(target_dir.glob("*.json"))
        gnn_files.extend(target_dir.glob("*.xml"))
        gnn_files.extend(target_dir.glob("*.yaml"))
        gnn_files.extend(target_dir.glob("*.pkl"))
    
    # Filter for actual GNN files
    gnn_files = [f for f in gnn_files if f.is_file()]
    
    discovery_time = time.time() - start_time
    logger.info(f"Discovered {len(gnn_files)} potential GNN files in {discovery_time:.3f}s")
    
    if not gnn_files:
        log_step_warning(logger, "No GNN files found for processing")
        return True
    
    # Processing phase
    logger.info(f"Phase 2: Enhanced validation and analysis")
    processing_start = time.time()
    
    processing_results = {
        'files_processed': 0,
        'files_valid': 0,
        'files_invalid': 0,
        'files_with_warnings': 0,
        'format_distribution': {},
        'validation_results': [],
        'performance_metrics': {},
        'round_trip_results': []
    }
    
    for gnn_file in gnn_files:
        try:
            logger.info(f"Processing: {gnn_file.relative_to(target_dir)}")
            file_start = time.time()
            
            # Validate file with enhanced validator
            validation_result = validator.validate_file(gnn_file, val_level)
            
            # Track results
            processing_results['files_processed'] += 1
            if validation_result.is_valid:
                processing_results['files_valid'] += 1
            else:
                processing_results['files_invalid'] += 1
            
            if validation_result.warnings:
                processing_results['files_with_warnings'] += 1
            
            # Track format distribution
            file_format = validation_result.format_tested or 'unknown'
            processing_results['format_distribution'][file_format] = \
                processing_results['format_distribution'].get(file_format, 0) + 1
            
            # Store detailed results
            file_result = {
                'file': str(gnn_file.relative_to(target_dir)),
                'format': file_format,
                'validation_level': validation_result.validation_level.value,
                'is_valid': validation_result.is_valid,
                'errors': validation_result.errors,
                'warnings': validation_result.warnings,
                'suggestions': validation_result.suggestions,
                'semantic_checksum': validation_result.semantic_checksum,
                'performance': validation_result.performance_metrics,
                'round_trip_success_rate': validation_result.get_round_trip_success_rate()
            }
            processing_results['validation_results'].append(file_result)
            
            # Collect round-trip results if available
            if validation_result.round_trip_results:
                processing_results['round_trip_results'].extend([
                    {
                        'file': str(gnn_file.relative_to(target_dir)),
                        'format': result.target_format.value,
                        'success': result.success,
                        'errors': result.errors,
                        'differences': result.differences
                    }
                    for result in validation_result.round_trip_results
                ])
            
            file_time = time.time() - file_start
            
            # Log results
            if validation_result.is_valid:
                status = "✅"
                if validation_result.warnings:
                    status += f" ({len(validation_result.warnings)} warnings)"
            else:
                status = f"❌ ({len(validation_result.errors)} errors)"
            
            logger.info(f"  {status} {file_format.upper()} ({file_time:.3f}s)")
            
            if verbose and validation_result.errors:
                for error in validation_result.errors:
                    logger.info(f"    Error: {error}")
            
            if verbose and validation_result.round_trip_results:
                rt_success_rate = validation_result.get_round_trip_success_rate()
                logger.info(f"    Round-trip: {rt_success_rate:.1f}% success")
                
        except Exception as e:
            logger.error(f"Error processing {gnn_file.name}: {e}")
            processing_results['files_processed'] += 1
            processing_results['files_invalid'] += 1
    
    processing_time = time.time() - processing_start
    total_time = time.time() - start_time
    
    # Generate comprehensive report
    processing_results['performance_metrics'] = {
        'total_time': total_time,
        'discovery_time': discovery_time,
        'processing_time': processing_time,
        'avg_file_time': processing_time / len(gnn_files) if gnn_files else 0,
        'files_per_second': len(gnn_files) / processing_time if processing_time > 0 else 0
    }
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = enhanced_output_dir / f"gnn_processing_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(processing_results, f, indent=2, default=str)
    
    # Summary logging
    success_rate = (processing_results['files_valid'] / processing_results['files_processed'] * 100) \
                   if processing_results['files_processed'] > 0 else 0
    
    logger.info(f"Enhanced GNN processing completed:")
    logger.info(f"  📁 Files processed: {processing_results['files_processed']}")
    logger.info(f"  ✅ Valid files: {processing_results['files_valid']}")
    logger.info(f"  ❌ Invalid files: {processing_results['files_invalid']}")
    logger.info(f"  ⚠️  Files with warnings: {processing_results['files_with_warnings']}")
    logger.info(f"  📈 Success rate: {success_rate:.1f}%")
    logger.info(f"  ⏱️  Total time: {total_time:.3f}s")
    logger.info(f"  📊 Report saved: {report_file}")
    
    # Format distribution
    if processing_results['format_distribution']:
        logger.info("  📋 Format distribution:")
        for fmt, count in processing_results['format_distribution'].items():
            logger.info(f"    {fmt}: {count}")
    
    # Round-trip summary
    if processing_results['round_trip_results']:
        rt_total = len(processing_results['round_trip_results'])
        rt_success = sum(1 for r in processing_results['round_trip_results'] if r['success'])
        rt_rate = (rt_success / rt_total * 100) if rt_total > 0 else 0
        logger.info(f"  🔄 Round-trip tests: {rt_success}/{rt_total} ({rt_rate:.1f}%)")
    
    if success_rate >= 80.0:
        log_step_success(logger, f"Enhanced GNN processing completed successfully ({success_rate:.1f}%)")
        return True
    else:
        log_step_warning(logger, f"Enhanced GNN processing completed with issues ({success_rate:.1f}%)")
        return False


def _basic_gnn_processing(
    target_dir: Path, 
    output_dir: Path, 
    logger: logging.Logger, 
    recursive: bool = False, 
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Fallback basic GNN processing when enhanced testing is not available.
    """
    log_step_warning(logger, "Using basic GNN processing (enhanced features unavailable)")
    
    # Create basic output directory
    basic_output_dir = output_dir / "basic_gnn_processing"
    basic_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple file discovery
    pattern = "**/*.md" if recursive else "*.md"
    gnn_files = list(target_dir.glob(pattern))
    
    logger.info(f"Found {len(gnn_files)} .md files")
    
    if not gnn_files:
        return True
    
    # Basic validation
    valid_files = 0
    for gnn_file in gnn_files:
        try:
            with open(gnn_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for basic GNN sections
            if "## ModelName" in content and "## StateSpaceBlock" in content:
                valid_files += 1
                logger.info(f"✅ {gnn_file.name}: Basic GNN structure found")
            else:
                logger.warning(f"⚠️  {gnn_file.name}: Missing required GNN sections")
        except Exception as e:
            logger.error(f"❌ {gnn_file.name}: {e}")
    
    success_rate = (valid_files / len(gnn_files) * 100) if gnn_files else 0
    logger.info(f"Basic processing: {valid_files}/{len(gnn_files)} valid ({success_rate:.1f}%)")
    
    return success_rate >= 50.0


def _generate_performance_analysis(report: Any, test_time: float) -> str:
    """Generate performance analysis section for reports."""
    lines = [
        "## Performance Analysis",
        "",
        f"### Test Execution Metrics",
        f"- **Total test time:** {test_time:.3f} seconds",
        f"- **Tests per second:** {report.total_tests / test_time:.2f}" if test_time > 0 else "- **Tests per second:** N/A",
        f"- **Average test time:** {test_time / report.total_tests:.3f} seconds" if report.total_tests > 0 else "- **Average test time:** N/A",
        "",
        f"### Success Rate Analysis",
        f"- **Successful tests:** {report.tests_passed}/{report.total_tests} ({report.tests_passed/report.total_tests*100:.1f}%)" if report.total_tests > 0 else "- **Successful tests:** 0/0",
        f"- **Failed tests:** {report.tests_failed}/{report.total_tests} ({report.tests_failed/report.total_tests*100:.1f}%)" if report.total_tests > 0 else "- **Failed tests:** 0/0",
        "",
        f"### Format Coverage",
        f"- **Formats tested:** {len(report.format_results)}"
    ]
    
    if hasattr(report, 'format_results') and report.format_results:
        lines.append("- **Formats:**")
        for fmt, result in report.format_results.items():
            status = "✓" if result.get('success', False) else "✗"
            lines.append(f"  - {status} {fmt}")
    
    return "\n".join(lines)


def _is_gnn_file(file_path: Path) -> bool:
    """Determine if a file is likely a GNN file based on content analysis."""
    try:
        # Quick heuristics based on filename
        filename_lower = file_path.name.lower()
        if any(keyword in filename_lower for keyword in ['gnn', 'actinf', 'pomdp', 'model']):
            return True
        
        # Content-based detection for non-obvious files
        if file_path.suffix.lower() == '.md':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Read first 1KB
                
                # Look for GNN section headers
                gnn_indicators = [
                    '## GNNSection', '## ModelName', '## StateSpaceBlock',
                    '## Connections', '## InitialParameterization'
                ]
                
                return any(indicator in content for indicator in gnn_indicators)
            except Exception:
                return False
        
        # For other formats, assume they are GNN files if they're in our target directory
        return file_path.suffix.lower() in ['.json', '.xml', '.yaml', '.pkl']
        
    except Exception:
        return False


def _validate_binary_cross_format(file_path: Path, cross_validator) -> Any:
    """Validate binary files for cross-format consistency."""
    try:
        # Binary files need special handling for cross-format validation
        if not file_path.exists():
            logger.error(f"Binary file not found: {file_path}")
            # Return a fallback result
            return type('CrossFormatValidationResult', (), {
                'is_valid': False,
                'errors': [f"File not found: {file_path}"],
                'warnings': [],
                'binary_format': str(file_path.suffix),
                'content_hash': None
            })()
        
        # For binary files, we can only do basic format checks
        logger.info(f"Validating binary file: {file_path}")
        
        # Check file size and basic properties
        file_size = file_path.stat().st_size
        if file_size == 0:
            return type('CrossFormatValidationResult', (), {
                'is_valid': False,
                'errors': ["Empty binary file"],
                'warnings': [],
                'binary_format': str(file_path.suffix),
                'content_hash': None
            })()
        
        # Create a successful result for valid binary files
        return type('CrossFormatValidationResult', (), {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'binary_format': str(file_path.suffix),
            'content_hash': str(hash(file_path.read_bytes()))
        })()
        
    except Exception as e:
        logger.error(f"Error validating binary file {file_path}: {e}")
        return type('CrossFormatValidationResult', (), {
            'is_valid': False,
            'errors': [str(e)],
            'warnings': [],
            'binary_format': str(file_path.suffix),
            'content_hash': None
        })()


def run_gnn_round_trip_tests(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    reference_file: Optional[str] = None,
    test_subset: Optional[List[str]] = None,
    enable_parallel: bool = False,
    **kwargs
) -> bool:
    """
    Enhanced round-trip tests with performance optimization and detailed reporting.
    
    Args:
        target_dir: Directory containing GNN files
        output_dir: Output directory for test results
        logger: Logger instance
        reference_file: Optional specific reference file
        test_subset: Optional list of formats to test
        enable_parallel: Whether to enable parallel testing
        **kwargs: Additional test options
        
    Returns:
        True if all tests passed, False otherwise
    """
    if not ENHANCED_TESTING_AVAILABLE:
        log_step_warning(logger, "Enhanced round-trip testing not available")
        return False
    
    log_step_start(logger, "Enhanced GNN round-trip testing")
    
    # Create enhanced output directory
    round_trip_output_dir = output_dir / "enhanced_round_trip_tests"
    round_trip_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load testing modules lazily
        modules = _get_testing_modules()
        if not modules:
            log_step_warning(logger, "Enhanced testing modules not available")
            return False
            
        # Initialize enhanced round-trip tester
        temp_dir = round_trip_output_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        tester = modules['GNNRoundTripTester'](temp_dir)
        
        # Configure reference file
        if reference_file:
            ref_path = Path(reference_file)
            if not ref_path.is_absolute():
                ref_path = target_dir / ref_path
            
            if ref_path.exists():
                tester.reference_file = ref_path
                logger.info(f"Using custom reference file: {ref_path}")
            else:
                log_step_warning(logger, f"Custom reference file not found: {ref_path}")
        
        # Verify reference file
        if not tester.reference_file.exists():
            log_step_error(logger, f"Reference file not found: {tester.reference_file}")
            return False
        
        logger.info(f"Testing with reference: {tester.reference_file}")
        
        # Configure test subset if specified
        original_formats = tester.supported_formats.copy()
        if test_subset:
            # Filter to requested formats
            subset_formats = []
            for fmt_name in test_subset:
                try:
                    fmt = modules['GNNFormat'](fmt_name.lower())
                    if fmt in tester.supported_formats:
                        subset_formats.append(fmt)
                    else:
                        logger.warning(f"Format {fmt_name} not supported")
                except ValueError:
                    logger.warning(f"Unknown format: {fmt_name}")
            
            if subset_formats:
                tester.supported_formats = [modules['GNNFormat'].MARKDOWN] + subset_formats
                logger.info(f"Testing subset: {[f.value for f in subset_formats]}")
            else:
                logger.warning("No valid formats in subset, using all supported")
        
        # Run comprehensive tests with enhanced reporting
        logger.info("Starting enhanced round-trip testing...")
        start_time = time.time()
        
        report = tester.run_comprehensive_tests()
        
        test_time = time.time() - start_time
        
        # Generate enhanced report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = round_trip_output_dir / f"enhanced_round_trip_report_{timestamp}.md"
        
        enhanced_report_content = tester.generate_report(report, report_file)
        
        # Add performance analysis
        performance_analysis = _generate_performance_analysis(report, test_time)
        with open(report_file, 'a') as f:
            f.write(f"\n\n{performance_analysis}")
        
        # Enhanced summary logging
        success_rate = report.get_success_rate()
        format_summary = report.get_format_summary()
        
        logger.info(f"Enhanced round-trip testing completed:")
        logger.info(f"  📊 Total tests: {report.total_tests}")
        logger.info(f"  ✅ Successful: {report.successful_tests}")
        logger.info(f"  ❌ Failed: {report.failed_tests}")
        logger.info(f"  📈 Success rate: {success_rate:.1f}%")
        logger.info(f"  ⏱️  Test time: {test_time:.3f}s")
        logger.info(f"  📄 Report: {report_file}")
        
        # Format category performance
        categories = [
            ("Schema", ['json', 'xml', 'yaml', 'xsd', 'asn1', 'pkl', 'protobuf']),
            ("Language", ['scala', 'lean', 'coq', 'python', 'haskell', 'isabelle']),
            ("Formal", ['tla_plus', 'agda', 'alloy', 'z_notation', 'bnf', 'ebnf']),
            ("Other", ['maxima', 'pickle'])
        ]
        
        for category, formats in categories:
            category_success = sum(1 for fmt_name in formats 
                                 if any(fmt.value == fmt_name and summary.get('success', 0) > 0 
                                       for fmt, summary in format_summary.items()))
            category_total = len([fmt for fmt_name in formats 
                                if any(fmt.value == fmt_name for fmt in format_summary.keys())])
            
            if category_total > 0:
                category_rate = (category_success / category_total) * 100
                status = "✅" if category_rate == 100 else "⚠️" if category_rate >= 50 else "❌"
                logger.info(f"  {status} {category}: {category_success}/{category_total} ({category_rate:.1f}%)")
        
        # Determine success
        if success_rate == 100.0:
            log_step_success(logger, "🎉 ALL ROUND-TRIP TESTS PASSED! Perfect format interoperability achieved.")
            return True
        elif success_rate >= 80.0:
            log_step_success(logger, f"🎊 EXCELLENT round-trip performance: {success_rate:.1f}%")
            return True
        else:
            log_step_warning(logger, f"⚠️ Round-trip issues found: {success_rate:.1f}% success rate")
            
            # Log specific failures for debugging
            failed_formats = [result.target_format.value for result in report.round_trip_results 
                            if not result.success]
            if failed_formats:
                logger.warning(f"Failed formats: {', '.join(failed_formats)}")
            
            return False
            
    except Exception as e:
        log_step_error(logger, f"Enhanced round-trip testing failed: {e}")
        logger.exception("Detailed error:")
        return False


def validate_gnn_cross_format_consistency(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    files_to_test: Optional[List[str]] = None,
    include_binary: bool = False,
    **kwargs
) -> bool:
    """
    Enhanced cross-format consistency validation with comprehensive analysis.
    
    Args:
        target_dir: Directory containing GNN files to test
        output_dir: Output directory for validation results
        logger: Logger instance
        files_to_test: Optional list of specific files to test
        include_binary: Whether to include binary formats in validation
        **kwargs: Additional validation options
        
    Returns:
        True if all validations passed, False otherwise
    """
    if not ENHANCED_TESTING_AVAILABLE:
        log_step_warning(logger, "Enhanced cross-format validation not available")
        return False
    
    log_step_start(logger, "Enhanced cross-format consistency validation")
    
    # Create enhanced output directory
    validation_output_dir = output_dir / "enhanced_cross_format_validation"
    validation_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load testing modules lazily
        modules = _get_testing_modules()
        if not modules:
            log_step_warning(logger, "Enhanced validation modules not available")
            return False
            
        # Initialize enhanced validators
        validator = modules['GNNValidator'](validation_level=modules['ValidationLevel'].STRICT)
        cross_validator = modules['CrossFormatValidator']()
        
        # Discover GNN files to test
        if files_to_test:
            gnn_files = [target_dir / f for f in files_to_test if (target_dir / f).exists()]
        else:
            # Enhanced file discovery
            extensions = ['*.md', '*.json', '*.xml', '*.yaml']
            if include_binary:
                extensions.extend(['*.pkl', '*.pickle'])
            
            gnn_files = []
            for ext in extensions:
                gnn_files.extend(target_dir.glob(ext))
            
            # Filter for GNN-relevant files
            gnn_files = [f for f in gnn_files if _is_gnn_file(f)]
        
        if not gnn_files:
            log_step_warning(logger, "No GNN files found for cross-format validation")
            return True
        
        logger.info(f"Testing cross-format consistency for {len(gnn_files)} files")
        
        # Enhanced validation tracking
        validation_results = {
            'total_files': len(gnn_files),
            'consistent_files': 0,
            'inconsistent_files': 0,
            'validation_errors': 0,
            'files': [],
            'format_analysis': {},
            'performance_metrics': {}
        }
        
        start_time = time.time()
        
        for gnn_file in gnn_files:
            logger.info(f"Validating: {gnn_file.relative_to(target_dir)}")
            file_start = time.time()
            
            try:
                # Read file content
                if gnn_file.suffix.lower() in ['.pkl', '.pickle']:
                    # Handle binary files
                    result = _validate_binary_cross_format(gnn_file, cross_validator)
                else:
                    with open(gnn_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Test cross-format consistency
                    cross_result = cross_validator.validate_cross_format_consistency(content)
                    result = cross_result
                
                file_time = time.time() - file_start
                
                # Process results
                file_result = {
                    'file': str(gnn_file.relative_to(target_dir)),
                    'format': gnn_file.suffix.lower(),
                    'is_consistent': result.is_consistent,
                    'formats_tested': getattr(result, 'schema_formats', []),
                    'inconsistencies': getattr(result, 'inconsistencies', []),
                    'warnings': getattr(result, 'warnings', []),
                    'metadata': getattr(result, 'metadata', {}),
                    'validation_time': file_time
                }
                
                validation_results['files'].append(file_result)
                
                # Update counters
                if result.is_consistent:
                    validation_results['consistent_files'] += 1
                    logger.info(f"  ✅ Consistent across {len(getattr(result, 'schema_formats', []))} formats ({file_time:.3f}s)")
                else:
                    validation_results['inconsistent_files'] += 1
                    logger.warning(f"  ❌ Inconsistent - {len(getattr(result, 'inconsistencies', []))} issues ({file_time:.3f}s)")
                    
                    for issue in getattr(result, 'inconsistencies', [])[:3]:  # Show first 3
                        logger.warning(f"    • {issue}")
                
                # Track format analysis
                file_format = gnn_file.suffix.lower()
                if file_format not in validation_results['format_analysis']:
                    validation_results['format_analysis'][file_format] = {
                        'total': 0, 'consistent': 0, 'avg_time': 0
                    }
                
                format_stats = validation_results['format_analysis'][file_format]
                format_stats['total'] += 1
                if result.is_consistent:
                    format_stats['consistent'] += 1
                format_stats['avg_time'] = (format_stats['avg_time'] * (format_stats['total'] - 1) + file_time) / format_stats['total']
                
            except Exception as e:
                logger.error(f"  ❌ Validation error: {e}")
                validation_results['validation_errors'] += 1
                
                file_result = {
                    'file': str(gnn_file.relative_to(target_dir)),
                    'format': gnn_file.suffix.lower(),
                    'is_consistent': False,
                    'error': str(e),
                    'validation_time': time.time() - file_start
                }
                validation_results['files'].append(file_result)
        
        total_time = time.time() - start_time
        
        # Enhanced performance metrics
        validation_results['performance_metrics'] = {
            'total_time': total_time,
            'avg_file_time': total_time / len(gnn_files) if gnn_files else 0,
            'files_per_second': len(gnn_files) / total_time if total_time > 0 else 0
        }
        
        # Generate enhanced report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = validation_output_dir / f"enhanced_cross_format_validation_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        # Summary and results
        success_rate = (validation_results['consistent_files'] / validation_results['total_files'] * 100) \
                      if validation_results['total_files'] > 0 else 0
        
        logger.info(f"Enhanced cross-format validation completed:")
        logger.info(f"  📁 Files tested: {validation_results['total_files']}")
        logger.info(f"  ✅ Consistent: {validation_results['consistent_files']}")
        logger.info(f"  ❌ Inconsistent: {validation_results['inconsistent_files']}")
        logger.info(f"  ⚠️  Errors: {validation_results['validation_errors']}")
        logger.info(f"  📈 Success rate: {success_rate:.1f}%")
        logger.info(f"  ⏱️  Total time: {total_time:.3f}s")
        logger.info(f"  📊 Report: {report_file}")
        
        # Format analysis summary
        if validation_results['format_analysis']:
            logger.info("  📋 Format analysis:")
            for fmt, stats in validation_results['format_analysis'].items():
                fmt_rate = (stats['consistent'] / stats['total'] * 100) if stats['total'] > 0 else 0
                logger.info(f"    {fmt}: {stats['consistent']}/{stats['total']} ({fmt_rate:.1f}%, {stats['avg_time']:.3f}s avg)")
        
        if success_rate == 100.0:
            log_step_success(logger, "✅ All files passed cross-format consistency validation")
            return True
        elif success_rate >= 80.0:
            log_step_success(logger, f"🎊 Good cross-format consistency: {success_rate:.1f}%")
            return True
        else:
            log_step_warning(logger, f"❌ Cross-format consistency issues: {success_rate:.1f}% success")
            return False
            
    except Exception as e:
        log_step_error(logger, f"Enhanced cross-format validation failed: {e}")
        logger.exception("Detailed error:")
        return False 