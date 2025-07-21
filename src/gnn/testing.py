#!/usr/bin/env python3
"""
GNN Testing Strategy Module

This module provides comprehensive testing strategies for GNN models,
including round-trip testing, format conversion validation, and
semantic preservation verification.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a testing operation."""
    test_name: str
    success: bool
    execution_time: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RoundTripTestStrategy:
    """
    Round-trip testing strategy for GNN models.
    
    Tests semantic preservation across format conversions
    and validates model consistency.
    """
    
    def __init__(self):
        self.test_subset = None
        self.reference_file = None
        self.output_dir = None
        self.round_trip_tester = None
        self._initialize_tester()
    
    def configure(self, test_subset: Optional[List[str]] = None,
                 reference_file: Optional[str] = None,
                 output_dir: Optional[Path] = None):
        """Configure testing parameters."""
        self.test_subset = test_subset
        self.reference_file = reference_file
        self.output_dir = output_dir
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_tester(self):
        """Initialize round-trip testing components."""
        try:
            from .testing.test_round_trip import GNNRoundTripTester
            self.round_trip_tester = GNNRoundTripTester()
            logger.debug("Round-trip tester initialized")
        except ImportError as e:
            logger.warning(f"Round-trip tester not available: {e}")
            self.round_trip_tester = None
    
    def test(self, files: List[Path]) -> Dict[str, Any]:
        """
        Run round-trip tests on provided files.
        
        Args:
            files: List of file paths to test
            
        Returns:
            Dictionary with test results and metadata
        """
        if not self.round_trip_tester:
            return {
                'success': False,
                'error': 'Round-trip tester not available',
                'tests_run': 0,
                'files_tested': 0
            }
        
        logger.info(f"Running round-trip tests on {len(files)} files")
        
        results = {
            'success': True,
            'tests_run': 0,
            'files_tested': len(files),
            'file_results': {},
            'summary': {}
        }
        
        for file_path in files:
            try:
                file_result = self._test_file_round_trip(file_path)
                results['file_results'][str(file_path)] = file_result
                results['tests_run'] += 1
                
                if not file_result['success']:
                    results['success'] = False
                    
            except Exception as e:
                logger.error(f"Round-trip test failed for {file_path}: {e}")
                results['file_results'][str(file_path)] = {
                    'success': False,
                    'error': str(e)
                }
                results['success'] = False
        
        # Generate summary
        results['summary'] = self._generate_test_summary(results['file_results'])
        
        # Save results if output directory is configured
        if self.output_dir:
            self._save_test_results(results)
        
        return results
    
    def _test_file_round_trip(self, file_path: Path) -> Dict[str, Any]:
        """Test round-trip conversion for a single file."""
        logger.debug(f"Testing round-trip for {file_path}")
        
        try:
            # Run comprehensive round-trip tests
            report = self.round_trip_tester.run_comprehensive_tests()
            
            # Extract results for this file
            file_results = []
            for result in report.round_trip_results:
                if hasattr(result, 'source_file') and result.source_file == str(file_path):
                    file_results.append({
                        'target_format': result.target_format.value if hasattr(result.target_format, 'value') else str(result.target_format),
                        'success': result.success,
                        'errors': result.errors,
                        'warnings': result.warnings,
                        'test_time': result.test_time
                    })
            
            success_rate = sum(1 for r in file_results if r['success']) / max(len(file_results), 1) * 100
            
            return {
                'success': success_rate >= 80.0,  # 80% success threshold
                'success_rate': success_rate,
                'format_results': file_results,
                'total_formats_tested': len(file_results)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'success_rate': 0.0,
                'format_results': [],
                'total_formats_tested': 0
            }
    
    def _generate_test_summary(self, file_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of test results."""
        summary = {
            'total_files': len(file_results),
            'successful_files': 0,
            'failed_files': 0,
            'average_success_rate': 0.0,
            'format_performance': {},
            'common_errors': []
        }
        
        total_success_rate = 0.0
        all_errors = []
        
        for file_path, result in file_results.items():
            if result.get('success', False):
                summary['successful_files'] += 1
            else:
                summary['failed_files'] += 1
            
            success_rate = result.get('success_rate', 0.0)
            total_success_rate += success_rate
            
            # Collect format performance data
            for format_result in result.get('format_results', []):
                fmt = format_result['target_format']
                if fmt not in summary['format_performance']:
                    summary['format_performance'][fmt] = {'success': 0, 'total': 0}
                
                summary['format_performance'][fmt]['total'] += 1
                if format_result['success']:
                    summary['format_performance'][fmt]['success'] += 1
                
                all_errors.extend(format_result.get('errors', []))
        
        # Calculate average success rate
        if summary['total_files'] > 0:
            summary['average_success_rate'] = total_success_rate / summary['total_files']
        
        # Find common errors
        error_counts = {}
        for error in all_errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        
        summary['common_errors'] = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return summary
    
    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results to output directory."""
        try:
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed results
            results_file = self.output_dir / f"round_trip_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save summary report
            summary_file = self.output_dir / f"round_trip_summary_{timestamp}.md"
            with open(summary_file, 'w') as f:
                self._write_summary_report(f, results)
            
            logger.info(f"Test results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
    
    def _write_summary_report(self, file, results: Dict[str, Any]):
        """Write human-readable summary report."""
        summary = results['summary']
        
        file.write("# GNN Round-Trip Test Summary\n\n")
        file.write(f"**Test Date:** {results.get('timestamp', 'Unknown')}\n")
        file.write(f"**Files Tested:** {summary['total_files']}\n")
        file.write(f"**Success Rate:** {summary['average_success_rate']:.1f}%\n\n")
        
        file.write("## Results Overview\n\n")
        file.write(f"- ✅ Successful: {summary['successful_files']}\n")
        file.write(f"- ❌ Failed: {summary['failed_files']}\n\n")
        
        if summary['format_performance']:
            file.write("## Format Performance\n\n")
            for fmt, perf in summary['format_performance'].items():
                success_rate = (perf['success'] / perf['total']) * 100 if perf['total'] > 0 else 0
                file.write(f"- **{fmt}**: {success_rate:.1f}% ({perf['success']}/{perf['total']})\n")
            file.write("\n")
        
        if summary['common_errors']:
            file.write("## Common Errors\n\n")
            for error, count in summary['common_errors']:
                file.write(f"- `{error}` (occurred {count} times)\n")


class CrossFormatValidationStrategy:
    """
    Cross-format validation strategy for GNN models.
    
    Validates consistency across different format representations.
    """
    
    def __init__(self):
        self.output_dir = None
        self.cross_validator = None
        self._initialize_validator()
    
    def configure(self, output_dir: Optional[Path] = None):
        """Configure validation parameters."""
        self.output_dir = output_dir
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_validator(self):
        """Initialize cross-format validation components."""
        try:
            from .cross_format_validator import CrossFormatValidator
            self.cross_validator = CrossFormatValidator()
            logger.debug("Cross-format validator initialized")
        except ImportError as e:
            logger.warning(f"Cross-format validator not available: {e}")
            self.cross_validator = None
    
    def validate(self, files: List[Path]) -> Dict[str, Any]:
        """
        Run cross-format validation on provided files.
        
        Args:
            files: List of file paths to validate
            
        Returns:
            Dictionary with validation results and metadata
        """
        if not self.cross_validator:
            return {
                'success': False,
                'error': 'Cross-format validator not available',
                'files_validated': 0
            }
        
        logger.info(f"Running cross-format validation on {len(files)} files")
        
        results = {
            'success': True,
            'files_validated': len(files),
            'file_results': {},
            'summary': {}
        }
        
        for file_path in files:
            try:
                file_result = self._validate_file_cross_format(file_path)
                results['file_results'][str(file_path)] = file_result
                
                if not file_result['success']:
                    results['success'] = False
                    
            except Exception as e:
                logger.error(f"Cross-format validation failed for {file_path}: {e}")
                results['file_results'][str(file_path)] = {
                    'success': False,
                    'error': str(e)
                }
                results['success'] = False
        
        # Generate summary
        results['summary'] = self._generate_validation_summary(results['file_results'])
        
        # Save results if output directory is configured
        if self.output_dir:
            self._save_validation_results(results)
        
        return results
    
    def _validate_file_cross_format(self, file_path: Path) -> Dict[str, Any]:
        """Validate cross-format consistency for a single file."""
        logger.debug(f"Validating cross-format consistency for {file_path}")
        
        try:
            content = file_path.read_text(encoding='utf-8')
            validation_result = self.cross_validator.validate_cross_format_consistency(content)
            
            return {
                'success': validation_result.is_consistent,
                'consistency_rate': validation_result.get_consistency_rate(),
                'inconsistencies': validation_result.inconsistencies,
                'warnings': validation_result.warnings,
                'format_results': validation_result.format_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'consistency_rate': 0.0,
                'inconsistencies': [],
                'warnings': [],
                'format_results': {}
            }
    
    def _generate_validation_summary(self, file_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of validation results."""
        summary = {
            'total_files': len(file_results),
            'consistent_files': 0,
            'inconsistent_files': 0,
            'average_consistency_rate': 0.0,
            'common_inconsistencies': []
        }
        
        total_consistency_rate = 0.0
        all_inconsistencies = []
        
        for file_path, result in file_results.items():
            if result.get('success', False):
                summary['consistent_files'] += 1
            else:
                summary['inconsistent_files'] += 1
            
            consistency_rate = result.get('consistency_rate', 0.0)
            total_consistency_rate += consistency_rate
            
            all_inconsistencies.extend(result.get('inconsistencies', []))
        
        # Calculate average consistency rate
        if summary['total_files'] > 0:
            summary['average_consistency_rate'] = total_consistency_rate / summary['total_files']
        
        # Find common inconsistencies
        inconsistency_counts = {}
        for inconsistency in all_inconsistencies:
            inconsistency_counts[inconsistency] = inconsistency_counts.get(inconsistency, 0) + 1
        
        summary['common_inconsistencies'] = sorted(inconsistency_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return summary
    
    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results to output directory."""
        try:
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed results
            results_file = self.output_dir / f"cross_format_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Cross-format validation results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}") 