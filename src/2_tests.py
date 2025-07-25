#!/usr/bin/env python3
"""
Step 2: Test Suite Execution

This step runs comprehensive tests for the GNN pipeline with staged execution,
progressive timeouts, and detailed reporting. Tests are executed in stages:

1. Fast tests (< 30 seconds) - Basic functionality validation
2. Standard tests (< 5 minutes) - Comprehensive module testing
3. Slow tests (< 15 minutes) - Integration and performance testing

Each stage has appropriate timeouts and detailed progress reporting.
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config
from tests.runner import run_tests, check_test_dependencies

class TestStage(Enum):
    """Test execution stages with different characteristics."""
    FAST = "fast"
    STANDARD = "standard" 
    SLOW = "slow"
    PERFORMANCE = "performance"

@dataclass
class TestStageConfig:
    """Configuration for a test execution stage."""
    name: str
    markers: List[str]
    timeout_seconds: int
    description: str
    max_failures: int = 10
    parallel: bool = True
    coverage: bool = True

# Define test execution stages
TEST_STAGES = {
    TestStage.FAST: TestStageConfig(
        name="Fast Tests",
        markers=["fast"],
        timeout_seconds=180,  # 3 minutes
        description="Quick validation tests for core functionality",
        max_failures=5,
        parallel=True,
        coverage=False  # Skip coverage for speed
    ),
    TestStage.STANDARD: TestStageConfig(
        name="Standard Tests", 
        markers=["not slow", "not performance"],
        timeout_seconds=600,  # 10 minutes
        description="Comprehensive module and integration tests",
        max_failures=15,
        parallel=True,
        coverage=True
    ),
    TestStage.SLOW: TestStageConfig(
        name="Slow Tests",
        markers=["slow"],
        timeout_seconds=900,  # 15 minutes
        description="Integration tests and complex scenarios",
        max_failures=20,
        parallel=False,  # May have resource conflicts
        coverage=True
    ),
    TestStage.PERFORMANCE: TestStageConfig(
        name="Performance Tests",
        markers=["performance"],
        timeout_seconds=1200,  # 20 minutes
        description="Performance benchmarks and resource usage tests",
        max_failures=5,
        parallel=False,
        coverage=False
    )
}

class StagedTestRunner:
    """Enhanced test runner with staged execution and comprehensive reporting."""
    
    def __init__(self, args, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.output_dir = get_output_dir_for_script("2_tests.py", Path(args.output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize execution tracking
        self.execution_start_time = time.time()
        self.stage_results: Dict[TestStage, Dict[str, Any]] = {}
        self.overall_stats = {
            "total_stages": 0,
            "successful_stages": 0,
            "failed_stages": 0,
            "total_tests_run": 0,
            "total_tests_passed": 0,
            "total_tests_failed": 0,
            "total_execution_time": 0.0
        }
        
    def should_run_stage(self, stage: TestStage) -> bool:
        """Determine if a stage should be executed based on arguments."""
        if getattr(self.args, 'fast_only', False):
            return stage == TestStage.FAST
        if getattr(self.args, 'include_slow', False):
            return True  # Run all stages
        if getattr(self.args, 'include_performance', False):
            return stage != TestStage.PERFORMANCE
        
        # Default behavior: run fast and standard tests
        return stage in [TestStage.FAST, TestStage.STANDARD]
    
    def run_test_stage(self, stage: TestStage, config: TestStageConfig) -> Dict[str, Any]:
        """Execute a single test stage with appropriate configuration."""
        stage_start_time = time.time()
        
        self.logger.info(f"🚀 Starting {config.name}")
        self.logger.info(f"   📋 {config.description}")
        self.logger.info(f"   ⏱️  Timeout: {config.timeout_seconds} seconds")
        self.logger.info(f"   🎯 Markers: {', '.join(config.markers)}")
        
        # Prepare stage-specific arguments
        stage_args = self._prepare_stage_args(config)
        
        try:
            # Create stage output directory
            stage_output_dir = self.output_dir / f"stage_{stage.value}"
            stage_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Execute tests using the runner infrastructure
            success = run_tests(
                logger=self.logger,
                output_dir=stage_output_dir,
                verbose=self.args.verbose,
                include_slow=(stage == TestStage.SLOW),
                fast_only=(stage == TestStage.FAST),
                generate_coverage=config.coverage
            )
            
            stage_duration = time.time() - stage_start_time
            
            # Parse results
            stage_results = self._parse_stage_results(stage_output_dir, success, stage_duration)
            
            if success:
                self.logger.info(f"✅ {config.name} completed successfully in {stage_duration:.1f}s")
                self.overall_stats["successful_stages"] += 1
            else:
                self.logger.warning(f"⚠️ {config.name} completed with issues in {stage_duration:.1f}s")
                self.overall_stats["failed_stages"] += 1
            
            return stage_results
            
        except Exception as e:
            stage_duration = time.time() - stage_start_time
            error_result = {
                "stage": stage.value,
                "success": False,
                "error": str(e),
                "duration_seconds": stage_duration,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0
            }
            
            self.logger.error(f"❌ {config.name} failed with error: {e}")
            self.overall_stats["failed_stages"] += 1
            
            return error_result
    
    def _prepare_stage_args(self, config: TestStageConfig) -> Dict[str, Any]:
        """Prepare arguments specific to the test stage."""
        return {
            "markers": config.markers,
            "timeout": config.timeout_seconds,
            "max_failures": config.max_failures,
            "parallel": config.parallel,
            "coverage": config.coverage,
            "verbose": self.args.verbose
        }
    
    def _parse_stage_results(self, stage_output_dir: Path, success: bool, duration: float) -> Dict[str, Any]:
        """Parse results from a completed test stage."""
        result = {
            "success": success,
            "duration_seconds": duration,
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "output_files": []
        }
        
        # Try to parse detailed results from test output files
        try:
            # Look for test summary files
            summary_file = stage_output_dir / "test_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                    test_stats = summary_data.get("test_statistics", {})
                    result.update({
                        "tests_run": test_stats.get("total_tests", 0),
                        "tests_passed": test_stats.get("passed", 0),
                        "tests_failed": test_stats.get("failed", 0) + test_stats.get("errors", 0)
                    })
            
            # Collect output files
            result["output_files"] = [str(f) for f in stage_output_dir.glob("*") if f.is_file()]
            
        except Exception as e:
            self.logger.warning(f"Could not parse detailed stage results: {e}")
        
        # Update overall statistics
        self.overall_stats["total_tests_run"] += result["tests_run"]
        self.overall_stats["total_tests_passed"] += result["tests_passed"] 
        self.overall_stats["total_tests_failed"] += result["tests_failed"]
        
        return result
    
    def run_all_stages(self) -> bool:
        """Execute all configured test stages in order."""
        log_step_start(self.logger, "Running staged test suite execution")
        
        # Check test dependencies first
        self.logger.info("🔍 Checking test dependencies...")
        dependencies = check_test_dependencies(self.logger)
        self._save_dependency_report(dependencies)
        
        if not dependencies.get("pytest", {}).get("available", False):
            log_step_error(self.logger, "pytest is not available - cannot run tests")
            return False
        
        # Determine which stages to run
        stages_to_run = [stage for stage in TestStage if self.should_run_stage(stage)]
        self.overall_stats["total_stages"] = len(stages_to_run)
        
        self.logger.info(f"📋 Running {len(stages_to_run)} test stages: {[s.value for s in stages_to_run]}")
        
        # Execute each stage
        overall_success = True
        for i, stage in enumerate(stages_to_run, 1):
            config = TEST_STAGES[stage]
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📍 Stage {i}/{len(stages_to_run)}: {config.name}")
            self.logger.info(f"{'='*60}")
            
            stage_result = self.run_test_stage(stage, config)
            self.stage_results[stage] = stage_result
            
            if not stage_result["success"]:
                overall_success = False
                
                # Check if we should continue with remaining stages
                if stage == TestStage.FAST:
                    self.logger.warning("⚠️ Fast tests failed - continuing with remaining stages")
                elif stage == TestStage.STANDARD:
                    self.logger.warning("⚠️ Standard tests failed - continuing with remaining stages")
            
            # Save intermediate results
            self._save_intermediate_results()
        
        # Generate final report
        self.overall_stats["total_execution_time"] = time.time() - self.execution_start_time
        self._generate_final_report()
        
        # Log final summary
        self._log_final_summary(overall_success)
        
        return overall_success
    
    def _save_dependency_report(self, dependencies: Dict[str, Any]):
        """Save dependency check results."""
        dependency_file = self.output_dir / "test_dependencies.json"
        with open(dependency_file, 'w') as f:
            json.dump(dependencies, f, indent=2)
    
    def _save_intermediate_results(self):
        """Save intermediate results after each stage."""
        results_file = self.output_dir / "staged_test_results.json"
        
        results_data = {
            "execution_start_time": self.execution_start_time,
            "current_time": time.time(),
            "overall_stats": self.overall_stats,
            "stage_results": {stage.value: result for stage, result in self.stage_results.items()},
            "args": {
                "verbose": self.args.verbose,
                "fast_only": getattr(self.args, 'fast_only', False),
                "include_slow": getattr(self.args, 'include_slow', False),
                "target_dir": str(self.args.target_dir),
                "output_dir": str(self.args.output_dir)
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def _generate_final_report(self):
        """Generate comprehensive final test report."""
        # Generate detailed markdown report
        report_file = self.output_dir / "staged_test_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Staged Test Execution Report\n\n")
            f.write(f"**Execution Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Duration**: {self.overall_stats['total_execution_time']:.1f} seconds\n")
            f.write(f"**Stages Executed**: {self.overall_stats['total_stages']}\n")
            f.write(f"**Overall Success**: {'✅ PASSED' if self.overall_stats['successful_stages'] == self.overall_stats['total_stages'] else '❌ FAILED'}\n\n")
            
            # Overall statistics
            f.write("## Overall Statistics\n\n")
            f.write(f"- **Total Tests Run**: {self.overall_stats['total_tests_run']}\n")
            f.write(f"- **Tests Passed**: {self.overall_stats['total_tests_passed']}\n")
            f.write(f"- **Tests Failed**: {self.overall_stats['total_tests_failed']}\n")
            f.write(f"- **Success Rate**: {(self.overall_stats['total_tests_passed'] / max(1, self.overall_stats['total_tests_run']) * 100):.1f}%\n")
            f.write(f"- **Successful Stages**: {self.overall_stats['successful_stages']}/{self.overall_stats['total_stages']}\n\n")
            
            # Stage details
            f.write("## Stage Details\n\n")
            for stage, result in self.stage_results.items():
                config = TEST_STAGES[stage]
                status = "✅ PASSED" if result["success"] else "❌ FAILED"
                
                f.write(f"### {config.name} - {status}\n\n")
                f.write(f"- **Description**: {config.description}\n")
                f.write(f"- **Duration**: {result['duration_seconds']:.1f} seconds\n")
                f.write(f"- **Tests Run**: {result['tests_run']}\n")
                f.write(f"- **Tests Passed**: {result['tests_passed']}\n")
                f.write(f"- **Tests Failed**: {result['tests_failed']}\n")
                
                if result['tests_run'] > 0:
                    success_rate = (result['tests_passed'] / result['tests_run']) * 100
                    f.write(f"- **Success Rate**: {success_rate:.1f}%\n")
                
                f.write(f"- **Output Files**: {len(result.get('output_files', []))}\n\n")
            
            # Configuration
            f.write("## Test Configuration\n\n")
            f.write(f"- **Target Directory**: {self.args.target_dir}\n")
            f.write(f"- **Output Directory**: {self.args.output_dir}\n")
            f.write(f"- **Verbose Mode**: {self.args.verbose}\n")
            f.write(f"- **Fast Only**: {getattr(self.args, 'fast_only', False)}\n")
            f.write(f"- **Include Slow**: {getattr(self.args, 'include_slow', False)}\n")
    
    def _log_final_summary(self, success: bool):
        """Log final execution summary."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("📊 STAGED TEST EXECUTION SUMMARY")
        self.logger.info(f"{'='*80}")
        
        if success:
            log_step_success(self.logger, "All test stages completed successfully")
        else:
            log_step_warning(self.logger, "Some test stages failed or had issues")
        
        # Summary statistics
        stats = self.overall_stats
        self.logger.info(f"📈 Execution Statistics:")
        self.logger.info(f"   • Total Duration: {stats['total_execution_time']:.1f} seconds")
        self.logger.info(f"   • Stages: {stats['successful_stages']}/{stats['total_stages']} successful")
        self.logger.info(f"   • Tests: {stats['total_tests_passed']}/{stats['total_tests_run']} passed")
        
        if stats['total_tests_run'] > 0:
            success_rate = (stats['total_tests_passed'] / stats['total_tests_run']) * 100
            self.logger.info(f"   • Success Rate: {success_rate:.1f}%")
        
        # Stage breakdown
        self.logger.info(f"🎯 Stage Breakdown:")
        for stage, result in self.stage_results.items():
            config = TEST_STAGES[stage]
            status = "✅" if result["success"] else "❌"
            duration = result["duration_seconds"]
            test_count = result["tests_run"]
            
            self.logger.info(f"   {status} {config.name}: {test_count} tests in {duration:.1f}s")
        
        # Output location
        self.logger.info(f"📁 Detailed reports saved to: {self.output_dir}")
        self.logger.info(f"{'='*80}")

def parse_enhanced_arguments():
    """Parse command line arguments with enhanced test options."""
    parser = EnhancedArgumentParser.get_parser()
    
    # Add test-specific arguments
    parser.add_argument("--fast-only", action="store_true", 
                       help="Run only fast tests (< 3 minutes)")
    parser.add_argument("--include-slow", action="store_true",
                       help="Include slow integration tests")
    parser.add_argument("--include-performance", action="store_true", 
                       help="Include performance benchmark tests")
    parser.add_argument("--no-coverage", action="store_true",
                       help="Disable coverage reporting for faster execution")
    parser.add_argument("--max-failures", type=int, default=20,
                       help="Maximum number of test failures before stopping")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Enable parallel test execution where supported")
    
    # Use standard argparse for robustness
    args, _ = parser.parse_known_args()
    
    # Set defaults for missing attributes if not coming from main orchestrator
    if not hasattr(args, 'target_dir'):
        args.target_dir = Path("input/gnn_files")
    if not hasattr(args, 'output_dir'):
        args.output_dir = Path("output")
    if not hasattr(args, 'verbose'):
        args.verbose = False
    
    return args

def main():
    """Main test execution function with staged test running."""
    # Parse arguments
    args = parse_enhanced_arguments()
    
    # Setup logging
    logger = setup_step_logging("tests", args)
    
    try:
        # Create and run staged test execution
        runner = StagedTestRunner(args, logger)
        success = runner.run_all_stages()
        
        # Return appropriate exit code
        return 0 if success else 1
        
    except KeyboardInterrupt:
        log_step_error(logger, "Test execution interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.exception(f"An unexpected error occurred in 2_tests.py: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 