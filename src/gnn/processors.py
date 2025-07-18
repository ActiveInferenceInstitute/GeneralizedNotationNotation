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

# Import GNN testing capabilities
try:
    from gnn.testing.test_round_trip import GNNRoundTripTester, ComprehensiveTestReport
    from gnn.parsers import GNNParsingSystem, GNNFormat
    from gnn.schema_validator import GNNValidator
    from gnn.cross_format_validator import CrossFormatValidator
    ROUND_TRIP_AVAILABLE = True
except ImportError:
    ROUND_TRIP_AVAILABLE = False

def process_gnn_folder(
    target_dir: Path, 
    output_dir: Path, 
    logger: logging.Logger, 
    recursive: bool = False, 
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process the GNN folder:
    - Discover .md files.
    - Perform basic parsing for key GNN sections.
    - Log findings and simple statistics to a report file.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    log_step_start(logger, f"Processing GNN files in directory: '{target_dir}'")
    
    if recursive:
        logger.info("Recursive mode enabled: searching in subdirectories.")
    else:
        logger.info("Recursive mode disabled: searching in top-level directory only.")

    gnn_target_path_abs = target_dir.resolve()

    if not target_dir.is_dir():
        log_step_warning(logger, f"GNN target directory '{gnn_target_path_abs}' not found or not a directory. Skipping GNN processing for this target.")
        return False

    # Use centralized output directory configuration
    step_output_dir = get_output_dir_for_script("1_gnn.py", output_dir)
    
    # Create the step output directory
    try:
        step_output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created output directory: {step_output_dir}")
    except Exception as e:
        log_step_error(logger, f"Failed to create GNN processing output directory '{step_output_dir}': {e}")
        return False

    report_file_path = step_output_dir / "1_gnn_discovery_report.md"
    report_file_path_abs = report_file_path.resolve()

    processed_files_summary = []
    file_pattern = "**/*.md" if recursive else "*.md"
    
    # Counters for summary
    found_model_name_count = 0
    found_statespace_count = 0
    found_connections_count = 0
    files_with_errors_count = 0

    logger.debug(f"Searching for GNN files matching pattern '{file_pattern}' in '{gnn_target_path_abs}'")
    gnn_files = list(target_dir.glob(file_pattern))

    if not gnn_files:
        logger.info(f"No .md files found in '{gnn_target_path_abs}' with pattern '{file_pattern}'.")
        try:
            with open(report_file_path, "w", encoding="utf-8") as f_report:
                f_report.write("# GNN File Discovery Report\n\n")
                f_report.write(f"No .md files found in `{gnn_target_path_abs}` using pattern `{file_pattern}`.\n")
            logger.info(f"Empty report saved to: {report_file_path_abs}")
        except IOError as e:
            log_step_error(logger, f"Failed to write empty report to {report_file_path_abs}: {e}")
        return True  # Return True for empty directory (not an error)

    logger.info(f"Found {len(gnn_files)} .md file(s) to process in '{gnn_target_path_abs}'.")

    for gnn_file_path_obj in gnn_files:
        resolved_gnn_file_path = gnn_file_path_obj.resolve() 
        path_for_report_str = str(resolved_gnn_file_path.relative_to(gnn_target_path_abs)) if resolved_gnn_file_path.is_relative_to(gnn_target_path_abs) else str(resolved_gnn_file_path)
        
        logger.debug(f"Processing file: {path_for_report_str}")
        
        file_summary = {
            "file_name": resolved_gnn_file_path.name,
            "path": path_for_report_str,
            "model_name": "Not found",
            "sections_found": [],
            "model_parameters": {},
            "errors": []
        }
        
        try:
            with open(resolved_gnn_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"Successfully read content from {path_for_report_str}.")
            
            # ModelName parsing
            model_name_section_header_text = "ModelName"
            parsed_model_name = "Not found" 

            _model_name_regex_string = rf"^##\s*{re.escape(model_name_section_header_text)}\s*$\r?"
            model_name_header_pattern = re.compile(_model_name_regex_string, re.IGNORECASE | re.MULTILINE)
            model_name_header_match = model_name_header_pattern.search(content)

            if model_name_header_match:
                logger.debug(f"  Found '## {model_name_section_header_text}' header in {path_for_report_str}")
                found_model_name_count += 1
                
                content_after_header = content[model_name_header_match.end():]
                next_section_header_match = re.search(r"^##\s+\w+", content_after_header, re.MULTILINE)
                
                if next_section_header_match:
                    name_region_content = content_after_header[:next_section_header_match.start()]
                else:
                    name_region_content = content_after_header
                
                extracted_name_candidate = ""
                for line in name_region_content.splitlines():
                    stripped_line = line.strip()
                    if stripped_line and not stripped_line.startswith("#"):
                        extracted_name_candidate = stripped_line
                        break
                
                if extracted_name_candidate:
                    parsed_model_name = extracted_name_candidate
                    logger.debug(f"    Extracted {model_name_section_header_text}: '{parsed_model_name}' from {path_for_report_str}")
                else:
                    parsed_model_name = "(Header found, but name line empty or only comments)"
                    logger.debug(f"    '## {model_name_section_header_text}' header found, but no suitable name line in {path_for_report_str}")

            file_summary["model_name"] = parsed_model_name
            file_summary["sections_found"].append(f"ModelName: {'Found: ' + parsed_model_name if parsed_model_name != 'Not found' else 'Not found'}")

            # StateSpaceBlock parsing
            statespace_section_header_text = "StateSpaceBlock"
            statespace_search_pattern = rf"^##\s*{re.escape(statespace_section_header_text)}\s*(?:#.*)?$"
            statespace_match = re.search(statespace_search_pattern, content, re.MULTILINE | re.IGNORECASE)
            if statespace_match:
                file_summary["sections_found"].append("StateSpaceBlock: Found")
                logger.debug(f"  Found {statespace_section_header_text} section in {path_for_report_str}")
                found_statespace_count += 1
            else:
                file_summary["sections_found"].append("StateSpaceBlock: Not found")
                logger.debug(f"  {statespace_section_header_text} section not found in {path_for_report_str}")

            # Connections parsing
            connections_section_header_text = "Connections"
            connections_search_pattern = rf"^##\s*{re.escape(connections_section_header_text)}\s*(?:#.*)?$"
            connections_match = re.search(connections_search_pattern, content, re.MULTILINE | re.IGNORECASE)
            if connections_match:
                file_summary["sections_found"].append("Connections: Found")
                logger.debug(f"  Found {connections_section_header_text} section in {path_for_report_str}")
                found_connections_count += 1
            else:
                file_summary["sections_found"].append("Connections: Not found")
                logger.debug(f"  {connections_section_header_text} section not found in {path_for_report_str}")

            # ModelParameters parsing
            parameters_section_header_text = "ModelParameters"
            parameters_search_pattern = rf"^##\s*{re.escape(parameters_section_header_text)}\s*(?:#.*)?$"
            parameters_match = re.search(parameters_search_pattern, content, re.MULTILINE | re.IGNORECASE)
            if parameters_match:
                logger.debug(f"  Found {parameters_section_header_text} section in {path_for_report_str}")
                
                content_after_header = content[parameters_match.end():]
                next_section_header_match = re.search(r"^##\s+\w+", content_after_header, re.MULTILINE)
                
                if next_section_header_match:
                    parameters_region_content = content_after_header[:next_section_header_match.start()]
                else:
                    parameters_region_content = content_after_header
                
                # Simple parameter extraction - look for key = value patterns
                parameter_pattern = r"^\s*(\w+)\s*=\s*(.+?)(?:\s*###.*)?$"
                for line in parameters_region_content.splitlines():
                    stripped_line = line.strip()
                    if stripped_line and not stripped_line.startswith("#"):
                        param_match = re.match(parameter_pattern, stripped_line)
                        if param_match:
                            param_name = param_match.group(1)
                            param_value_str = param_match.group(2).strip()
                            
                            # Try to parse the value
                            try:
                                import ast
                                param_value = ast.literal_eval(param_value_str)
                            except (ValueError, SyntaxError):
                                param_value = param_value_str
                            
                            file_summary["model_parameters"][param_name] = param_value
                            logger.debug(f"    Parsed ModelParameter: {param_name} = {param_value}")

        except Exception as e:
            error_msg = f"Error processing {path_for_report_str}: {e}"
            file_summary["errors"].append(error_msg)
            log_step_warning(logger, error_msg)
            files_with_errors_count += 1

        processed_files_summary.append(file_summary)

    # Generate report
    try:
        with open(report_file_path, "w", encoding="utf-8") as f_report:
            f_report.write("# GNN File Discovery Report\n\n")
            f_report.write(f"**Target Directory:** `{gnn_target_path_abs}`\n")
            f_report.write(f"**Search Pattern:** `{file_pattern}`\n")
            f_report.write(f"**Files Found:** {len(gnn_files)}\n\n")
            
            f_report.write("## Summary Statistics\n\n")
            f_report.write(f"- **Files with ModelName:** {found_model_name_count}\n")
            f_report.write(f"- **Files with StateSpaceBlock:** {found_statespace_count}\n")
            f_report.write(f"- **Files with Connections:** {found_connections_count}\n")
            f_report.write(f"- **Files with Errors:** {files_with_errors_count}\n\n")
            
            f_report.write("## Detailed File Analysis\n\n")
            for file_summary in processed_files_summary:
                f_report.write(f"### {file_summary['file_name']}\n\n")
                f_report.write(f"**Path:** `{file_summary['path']}`\n")
                f_report.write(f"**Model Name:** {file_summary['model_name']}\n\n")
                
                if file_summary['sections_found']:
                    f_report.write("**Sections Found:**\n")
                    for section in file_summary['sections_found']:
                        f_report.write(f"- {section}\n")
                    f_report.write("\n")
                
                if file_summary['model_parameters']:
                    f_report.write("**Model Parameters:**\n")
                    for param_name, param_value in file_summary['model_parameters'].items():
                        f_report.write(f"- `{param_name}` = `{param_value}`\n")
                    f_report.write("\n")
                
                if file_summary['errors']:
                    f_report.write("**Errors:**\n")
                    for error in file_summary['errors']:
                        f_report.write(f"- {error}\n")
                    f_report.write("\n")
                
                f_report.write("---\n\n")
        
        logger.info(f"Report saved to: {report_file_path_abs}")
        
        # Log summary
        if files_with_errors_count == 0:
            log_step_success(logger, f"Successfully processed {len(gnn_files)} GNN files without errors")
            return True
        elif files_with_errors_count < len(gnn_files):
            log_step_warning(logger, f"Processed {len(gnn_files)} GNN files with {files_with_errors_count} files having errors")
            return True
        else:
            log_step_error(logger, f"All {len(gnn_files)} GNN files had processing errors")
            return False
            
    except IOError as e:
        log_step_error(logger, f"Failed to write report to {report_file_path_abs}: {e}")
        return False


def run_gnn_round_trip_tests(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    reference_file: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Run comprehensive round-trip tests for GNN format conversion.
    
    This function tests the ability to read the reference POMDP model,
    convert it to all supported formats, read them back, and verify
    complete semantic equivalence.
    
    Args:
        target_dir: Directory containing GNN files (will look for reference)
        output_dir: Output directory for test results
        logger: Logger instance
        reference_file: Optional specific reference file (defaults to actinf_pomdp_agent.md)
        **kwargs: Additional test options
        
    Returns:
        True if all tests passed, False otherwise
    """
    if not ROUND_TRIP_AVAILABLE:
        log_step_warning(logger, "Round-trip testing not available - required modules not found")
        return False
    
    log_step_start(logger, "Running comprehensive GNN round-trip tests")
    
    # Create output directory for round-trip tests
    round_trip_output_dir = output_dir / "round_trip_tests"
    round_trip_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize the round-trip tester
        temp_dir = round_trip_output_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        tester = GNNRoundTripTester(temp_dir)
        
        # Set reference file if specified
        if reference_file:
            ref_path = Path(reference_file)
            if not ref_path.is_absolute():
                ref_path = target_dir / ref_path
            
            if ref_path.exists():
                tester.reference_file = ref_path
                logger.info(f"Using custom reference file: {ref_path}")
            else:
                log_step_warning(logger, f"Custom reference file not found: {ref_path}, using default")
        
        # Verify reference file exists
        if not tester.reference_file.exists():
            log_step_error(logger, f"Reference file not found: {tester.reference_file}")
            return False
        
        logger.info(f"Testing with reference file: {tester.reference_file}")
        
        # Run comprehensive tests
        logger.info("Starting comprehensive round-trip format conversion tests...")
        report = tester.run_comprehensive_tests()
        
        # Generate and save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = round_trip_output_dir / f"round_trip_report_{timestamp}.md"
        
        report_content = tester.generate_report(report, report_file)
        
        # Log summary
        success_rate = report.get_success_rate()
        logger.info(f"Round-trip tests completed:")
        logger.info(f"  Total tests: {report.total_tests}")
        logger.info(f"  Successful: {report.successful_tests}")
        logger.info(f"  Failed: {report.failed_tests}")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        logger.info(f"  Report saved: {report_file}")
        
        if success_rate == 100.0:
            log_step_success(logger, "🎉 ALL ROUND-TRIP TESTS PASSED! 100% confidence in format conversion.")
            return True
        else:
            log_step_warning(logger, f"⚠️ {report.failed_tests} round-trip tests failed. See report for details.")
            
            # Log specific failures
            for result in report.round_trip_results:
                if not result.success:
                    logger.warning(f"  {result.target_format.value}: {len(result.errors)} errors, {len(result.differences)} differences")
            
            return False
            
    except Exception as e:
        log_step_error(logger, f"Round-trip testing failed with exception: {e}")
        logger.exception("Round-trip test error details:")
        return False


def validate_gnn_cross_format_consistency(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    files_to_test: Optional[List[str]] = None,
    **kwargs
) -> bool:
    """
    Validate cross-format consistency for GNN files.
    
    This function tests that all schema formats (JSON, YAML, XSD, etc.)
    validate the same GNN models consistently.
    
    Args:
        target_dir: Directory containing GNN files to test
        output_dir: Output directory for validation results
        logger: Logger instance
        files_to_test: Optional list of specific files to test
        **kwargs: Additional validation options
        
    Returns:
        True if all validations passed, False otherwise
    """
    if not ROUND_TRIP_AVAILABLE:
        log_step_warning(logger, "Cross-format validation not available - required modules not found")
        return False
    
    log_step_start(logger, "Validating GNN cross-format consistency")
    
    # Create output directory
    validation_output_dir = output_dir / "cross_format_validation"
    validation_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize validators
        validator = GNNValidator()
        cross_validator = CrossFormatValidator()
        
        # Find GNN files to test
        if files_to_test:
            gnn_files = [target_dir / f for f in files_to_test if (target_dir / f).exists()]
        else:
            gnn_files = list(target_dir.glob("**/*.md"))
            # Filter for actual GNN files
            gnn_files = [f for f in gnn_files if "actinf" in f.name.lower() or "pomdp" in f.name.lower()]
        
        if not gnn_files:
            log_step_warning(logger, "No GNN files found for cross-format validation")
            return True
        
        logger.info(f"Testing cross-format consistency for {len(gnn_files)} files")
        
        all_consistent = True
        validation_results = []
        
        for gnn_file in gnn_files:
            logger.info(f"Validating cross-format consistency: {gnn_file.name}")
            
            try:
                # Read file content
                with open(gnn_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Test cross-format consistency
                cross_result = cross_validator.validate_cross_format_consistency(content)
                
                file_result = {
                    'file': str(gnn_file.relative_to(target_dir)),
                    'is_consistent': cross_result.is_consistent,
                    'formats_tested': cross_result.schema_formats,
                    'inconsistencies': cross_result.inconsistencies,
                    'warnings': cross_result.warnings,
                    'metadata': cross_result.metadata
                }
                
                validation_results.append(file_result)
                
                if cross_result.is_consistent:
                    logger.info(f"  ✅ {gnn_file.name}: Consistent across {len(cross_result.schema_formats)} formats")
                else:
                    logger.warning(f"  ❌ {gnn_file.name}: Inconsistent - {len(cross_result.inconsistencies)} issues")
                    for issue in cross_result.inconsistencies:
                        logger.warning(f"    - {issue}")
                    all_consistent = False
                
            except Exception as e:
                logger.error(f"  Error validating {gnn_file.name}: {e}")
                all_consistent = False
        
        # Generate validation report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = validation_output_dir / f"cross_format_validation_{timestamp}.json"
        
        report_data = {
            'timestamp': timestamp,
            'total_files': len(gnn_files),
            'consistent_files': sum(1 for r in validation_results if r['is_consistent']),
            'inconsistent_files': sum(1 for r in validation_results if not r['is_consistent']),
            'overall_consistent': all_consistent,
            'files': validation_results
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Cross-format validation report saved: {report_file}")
        
        if all_consistent:
            log_step_success(logger, "✅ All files passed cross-format consistency validation")
            return True
        else:
            inconsistent_count = sum(1 for r in validation_results if not r['is_consistent'])
            log_step_warning(logger, f"❌ {inconsistent_count} files failed cross-format consistency validation")
            return False
            
    except Exception as e:
        log_step_error(logger, f"Cross-format validation failed with exception: {e}")
        logger.exception("Cross-format validation error details:")
        return False


def run_comprehensive_gnn_testing(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    **kwargs
) -> bool:
    """
    Run comprehensive GNN testing including discovery, round-trip, and cross-format validation.
    
    This is the main function that orchestrates all GNN testing capabilities:
    1. File discovery and basic parsing
    2. Round-trip format conversion testing
    3. Cross-format schema consistency validation
    
    Args:
        target_dir: Directory containing GNN files
        output_dir: Output directory for all test results
        logger: Logger instance
        **kwargs: Additional options for all tests
        
    Returns:
        True if all tests passed, False otherwise
    """
    log_step_start(logger, "Running comprehensive GNN testing suite")
    
    # Create main output directory
    comprehensive_output_dir = output_dir / "comprehensive_gnn_tests"
    comprehensive_output_dir.mkdir(parents=True, exist_ok=True)
    
    all_tests_passed = True
    test_results = {}
    
    # Step 1: File discovery and basic parsing
    logger.info("Step 1: GNN file discovery and parsing")
    discovery_success = process_gnn_folder(
        target_dir, comprehensive_output_dir, logger, **kwargs
    )
    test_results['discovery'] = discovery_success
    all_tests_passed = all_tests_passed and discovery_success
    
    # Step 2: Round-trip format conversion testing
    logger.info("Step 2: Round-trip format conversion testing")
    round_trip_success = run_gnn_round_trip_tests(
        target_dir, comprehensive_output_dir, logger, **kwargs
    )
    test_results['round_trip'] = round_trip_success
    all_tests_passed = all_tests_passed and round_trip_success
    
    # Step 3: Cross-format consistency validation
    logger.info("Step 3: Cross-format consistency validation")
    cross_format_success = validate_gnn_cross_format_consistency(
        target_dir, comprehensive_output_dir, logger, **kwargs
    )
    test_results['cross_format'] = cross_format_success
    all_tests_passed = all_tests_passed and cross_format_success
    
    # Generate comprehensive summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = comprehensive_output_dir / f"comprehensive_test_summary_{timestamp}.md"
    
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive GNN Testing Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Target Directory:** `{target_dir}`\n")
            f.write(f"**Output Directory:** `{comprehensive_output_dir}`\n\n")
            
            f.write("## Test Results\n\n")
            
            status_icon = lambda success: "✅" if success else "❌"
            
            f.write(f"- **File Discovery & Parsing:** {status_icon(test_results['discovery'])} {'PASS' if test_results['discovery'] else 'FAIL'}\n")
            f.write(f"- **Round-Trip Conversion:** {status_icon(test_results['round_trip'])} {'PASS' if test_results['round_trip'] else 'FAIL'}\n")
            f.write(f"- **Cross-Format Consistency:** {status_icon(test_results['cross_format'])} {'PASS' if test_results['cross_format'] else 'FAIL'}\n\n")
            
            f.write(f"**Overall Result:** {status_icon(all_tests_passed)} {'ALL TESTS PASSED' if all_tests_passed else 'SOME TESTS FAILED'}\n\n")
            
            if all_tests_passed:
                f.write("🎉 **Congratulations!** All GNN tests passed. The system has 100% confidence in:\n")
                f.write("- Reading and parsing GNN model files\n")
                f.write("- Converting between all supported formats\n")
                f.write("- Maintaining semantic equivalence in round-trip conversions\n")
                f.write("- Cross-format schema consistency\n\n")
            else:
                f.write("⚠️ **Action Required:** Some tests failed. Please review the detailed reports:\n\n")
                if not test_results['discovery']:
                    f.write("- Check the file discovery report for parsing issues\n")
                if not test_results['round_trip']:
                    f.write("- Review the round-trip test report for format conversion problems\n")
                if not test_results['cross_format']:
                    f.write("- Examine the cross-format validation report for schema inconsistencies\n")
            
            f.write("## File Locations\n\n")
            f.write("- **Discovery Report:** `1_gnn_discovery_report.md`\n")
            f.write("- **Round-Trip Reports:** `round_trip_tests/`\n")
            f.write("- **Cross-Format Reports:** `cross_format_validation/`\n")
            f.write("- **This Summary:** `comprehensive_test_summary_*.md`\n")
        
        logger.info(f"Comprehensive test summary saved: {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate summary report: {e}")
    
    # Final logging
    if all_tests_passed:
        log_step_success(logger, "🎉 ALL COMPREHENSIVE TESTS PASSED! 100% confidence in GNN system.")
    else:
        failed_tests = [name for name, success in test_results.items() if not success]
        log_step_warning(logger, f"❌ Comprehensive testing failed. Failed tests: {', '.join(failed_tests)}")
    
    return all_tests_passed 