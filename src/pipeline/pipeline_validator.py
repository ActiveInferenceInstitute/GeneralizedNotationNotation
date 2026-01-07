#!/usr/bin/env python3
"""
Pipeline Validator

This module provides comprehensive validation and testing for the GNN pipeline
to ensure all improvements are working correctly and SUCCESS status is achieved.
"""

import logging
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.pipeline_dependencies import get_pipeline_dependency_manager
    from utils.logging_utils import setup_step_logging, log_step_success, log_step_warning, log_step_error
    from utils.pipeline import get_output_dir_for_script
except ImportError as e:
    print(f"Warning: Could not import pipeline utilities: {e}")


class PipelineValidator:
    """Comprehensive pipeline validator and improvement tester."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.dependency_manager = get_pipeline_dependency_manager()
        self.validation_results = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the validator."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger
    
    def validate_code_generation_fixes(self) -> Dict[str, Any]:
        """Validate that code generation issues have been fixed."""
        self.logger.info("🔧 Validating code generation fixes...")
        
        fixes_validation = {
            "pymdp_import_fix": False,
            "jax_flax_fix": False,
            "julia_matrix_fix": False,
            "dependency_handling": False
        }
        
        try:
            # Test PyMDP import fix
            pymdp_renderer_path = Path("src/render/pymdp/pymdp_renderer.py")
            if pymdp_renderer_path.exists():
                content = pymdp_renderer_path.read_text()
                # Should NOT contain configure_from_gnn_spec import
                if "configure_from_gnn_spec" not in content:
                    fixes_validation["pymdp_import_fix"] = True
                    self.logger.info("✅ PyMDP import fix validated")
                else:
                    self.logger.warning("⚠️ PyMDP import fix not applied")
            
            # Test JAX Flax fix  
            jax_renderer_path = Path("src/render/jax/jax_renderer.py")
            if jax_renderer_path.exists():
                content = jax_renderer_path.read_text()
                # Should use variables parameter in get_model_summary
                if "get_model_summary(num_states" in content:
                    fixes_validation["jax_flax_fix"] = True
                    self.logger.info("✅ JAX Flax attribute access fix validated")
                else:
                    self.logger.warning("⚠️ JAX Flax fix not applied")
            
            # Test Julia matrix formatting fix
            julia_renderer_path = Path("src/render/activeinference_jl/activeinference_renderer.py")
            if julia_renderer_path.exists():
                content = julia_renderer_path.read_text()
                # Should have improved matrix conversion function
                if "convert_element" in content and "tuple, list" in content:
                    fixes_validation["julia_matrix_fix"] = True
                    self.logger.info("✅ Julia matrix formatting fix validated")
                else:
                    self.logger.warning("⚠️ Julia matrix fix not applied")
            
            # Test pipeline dependency manager
            dep_manager_path = Path("src/utils/pipeline_dependencies.py")
            if dep_manager_path.exists():
                fixes_validation["dependency_handling"] = True
                self.logger.info("✅ Pipeline dependency manager created")
            else:
                self.logger.warning("⚠️ Pipeline dependency manager missing")
            
        except Exception as e:
            self.logger.error(f"❌ Error validating code generation fixes: {e}")
        
        return fixes_validation
    
    def test_dependency_validation(self) -> Dict[str, Any]:
        """Test the enhanced dependency validation system."""
        self.logger.info("📋 Testing enhanced dependency validation...")
        
        try:
            # Test key pipeline steps
            test_steps = ["8_visualization", "11_render", "12_execute", "15_audio"]
            step_results = {}
            
            for step in test_steps:
                result = self.dependency_manager.check_step_dependencies(step)
                step_results[step] = {
                    "status": result["status"],
                    "required_satisfied": all(
                        dep.available for dep in result["required"].values()
                    ),
                    "optional_count": len(result["optional"]),
                    "fallbacks_available": len(result["fallbacks_available"]),
                    "warnings": len(result["warnings"]),
                    "errors": len(result["errors"])
                }
                
                status_emoji = {
                    "healthy": "✅",
                    "degraded": "⚠️", 
                    "failed": "❌",
                    "unknown": "❓"
                }.get(result["status"], "❓")
                
                self.logger.info(
                    f"{status_emoji} {step}: {result['status'].upper()} "
                    f"({len(result['errors'])} errors, {len(result['warnings'])} warnings)"
                )
            
            return {
                "validation_successful": True,
                "step_results": step_results,
                "overall_health": self._calculate_overall_health(step_results)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Dependency validation test failed: {e}")
            return {"validation_successful": False, "error": str(e)}
    
    def _calculate_overall_health(self, step_results: Dict[str, Any]) -> str:
        """Calculate overall pipeline health from step results."""
        if not step_results:
            return "unknown"
        
        statuses = [result["status"] for result in step_results.values()]
        
        if all(status == "healthy" for status in statuses):
            return "healthy"
        elif any(status == "failed" for status in statuses):
            return "failed"
        else:
            return "degraded"
    
    def test_pipeline_execution(self, steps: List[str] = None) -> Dict[str, Any]:
        """Test actual pipeline execution with improvements."""
        self.logger.info("🚀 Testing pipeline execution...")
        
        if steps is None:
            steps = ["3", "5", "7", "8", "11", "15"]  # Key steps for testing
        
        test_results = {
            "execution_successful": False,
            "step_results": {},
            "execution_time": 0,
            "memory_usage": 0,
            "warnings_count": 0,
            "errors_count": 0
        }
        
        try:
            start_time = datetime.now()
            
            # Use main.py to execute pipeline steps
            cmd = [
                sys.executable, "src/main.py",
                "--target-dir", "input/gnn_files",
                "--output-dir", "output",
                "--only-steps", ",".join(steps),
                "--verbose"
            ]
            
            self.logger.info(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=Path.cwd()
            )
            
            end_time = datetime.now()
            test_results["execution_time"] = (end_time - start_time).total_seconds()
            
            # Parse execution results
            if result.returncode == 0:
                test_results["execution_successful"] = True
                self.logger.info("✅ Pipeline execution completed successfully")
            else:
                self.logger.error(f"❌ Pipeline execution failed with code {result.returncode}")
            
            # Count warnings and errors in output
            combined_output = result.stdout + result.stderr
            test_results["warnings_count"] = combined_output.lower().count("warning")
            test_results["errors_count"] = combined_output.lower().count("error")
            
            # Parse individual step results if summary file exists
            summary_file = Path("output/pipeline_execution_summary.json")
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        summary_data = json.load(f)
                    
                    for step_data in summary_data.get("steps", []):
                        step_name = step_data.get("script_name", "unknown")
                        test_results["step_results"][step_name] = {
                            "status": step_data.get("status", "unknown"),
                            "duration": step_data.get("duration_seconds", 0),
                            "exit_code": step_data.get("exit_code", -1),
                            "memory_usage": step_data.get("memory_usage_mb", 0)
                        }
                except Exception as e:
                    self.logger.warning(f"Could not parse summary file: {e}")
            
            test_results["stdout"] = result.stdout
            test_results["stderr"] = result.stderr
            
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Pipeline execution timed out")
            test_results["error"] = "Execution timed out"
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution error: {e}")
            test_results["error"] = str(e)
        
        return test_results
    
    def validate_success_status(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that pipeline steps achieve SUCCESS status instead of SUCCESS_WITH_WARNINGS."""
        self.logger.info("📊 Validating SUCCESS status achievement...")
        
        validation = {
            "success_rate_improved": False,
            "warnings_reduced": False,
            "no_critical_failures": False,
            "step_analysis": {}
        }
        
        try:
            if execution_results.get("execution_successful") and execution_results.get("step_results"):
                step_results = execution_results["step_results"]
                
                # Analyze step statuses
                success_count = 0
                success_with_warnings_count = 0
                failed_count = 0
                
                for step_name, result in step_results.items():
                    status = result.get("status", "unknown")
                    validation["step_analysis"][step_name] = status
                    
                    if status == "SUCCESS":
                        success_count += 1
                    elif status == "SUCCESS_WITH_WARNINGS":
                        success_with_warnings_count += 1
                    elif status == "FAILED":
                        failed_count += 1
                
                total_steps = len(step_results)
                success_rate = success_count / total_steps if total_steps > 0 else 0
                
                # Consider improvement if success rate is high
                validation["success_rate_improved"] = success_rate >= 0.7
                validation["warnings_reduced"] = success_with_warnings_count <= total_steps * 0.3
                validation["no_critical_failures"] = failed_count == 0
                
                self.logger.info(f"📈 Success rate: {success_rate:.1%} ({success_count}/{total_steps})")
                self.logger.info(f"⚠️ Warnings: {success_with_warnings_count}")
                self.logger.info(f"❌ Failures: {failed_count}")
                
            else:
                self.logger.warning("⚠️ No execution results to validate")
        
        except Exception as e:
            self.logger.error(f"❌ Error validating success status: {e}")
        
        return validation
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation and improvement report."""
        self.logger.info("📋 Generating comprehensive validation report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "validator_version": "1.0.0",
            "improvements_validated": False,
            "pipeline_health": "unknown",
            "recommendations": []
        }
        
        try:
            # 1. Validate code generation fixes
            code_fixes = self.validate_code_generation_fixes()
            report["code_generation_fixes"] = code_fixes
            
            # 2. Test dependency validation
            dependency_test = self.test_dependency_validation()
            report["dependency_validation"] = dependency_test
            
            # 3. Test pipeline execution
            execution_test = self.test_pipeline_execution()
            report["pipeline_execution"] = execution_test
            
            # 4. Validate success status
            success_validation = self.validate_success_status(execution_test)
            report["success_status_validation"] = success_validation
            
            # 5. Determine overall improvement status
            code_fixes_success = all(code_fixes.values())
            dependency_success = dependency_test.get("validation_successful", False)
            execution_success = execution_test.get("execution_successful", False)
            success_status_improved = any(success_validation.values())
            
            report["improvements_validated"] = (
                code_fixes_success and 
                dependency_success and 
                execution_success and
                success_status_improved
            )
            
            # 6. Calculate pipeline health
            if execution_test.get("step_results"):
                health_scores = []
                for result in execution_test["step_results"].values():
                    if result["status"] == "SUCCESS":
                        health_scores.append(1.0)
                    elif result["status"] == "SUCCESS_WITH_WARNINGS":
                        health_scores.append(0.7)
                    else:
                        health_scores.append(0.0)
                
                avg_health = sum(health_scores) / len(health_scores) if health_scores else 0
                if avg_health >= 0.8:
                    report["pipeline_health"] = "excellent"
                elif avg_health >= 0.6:
                    report["pipeline_health"] = "good"
                elif avg_health >= 0.4:
                    report["pipeline_health"] = "fair" 
                else:
                    report["pipeline_health"] = "poor"
            
            # 7. Generate recommendations
            recommendations = []
            
            if not code_fixes_success:
                recommendations.append("Complete remaining code generation fixes")
            
            if execution_test.get("warnings_count", 0) > 0:
                recommendations.append(f"Address {execution_test['warnings_count']} remaining warnings")
            
            if execution_test.get("errors_count", 0) > 0:
                recommendations.append(f"Fix {execution_test['errors_count']} remaining errors")
            
            if not success_status_improved:
                recommendations.append("Continue improving SUCCESS status achievement")
            
            if not recommendations:
                recommendations.append("Pipeline improvements successfully validated!")
            
            report["recommendations"] = recommendations
            
            # Log final results
            status_emoji = {
                "excellent": "🌟",
                "good": "✅", 
                "fair": "⚠️",
                "poor": "❌",
                "unknown": "❓"
            }.get(report["pipeline_health"], "❓")
            
            self.logger.info(f"{status_emoji} Pipeline Health: {report['pipeline_health'].upper()}")
            self.logger.info(f"🔧 Improvements Validated: {'YES' if report['improvements_validated'] else 'NO'}")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating report: {e}")
            report["error"] = str(e)
        
        return report
    
    def save_validation_report(self, report: Dict[str, Any], output_path: Path = None) -> Path:
        """Save validation report to file."""
        if output_path is None:
            output_path = Path("output/pipeline_validation_report.json")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📄 Validation report saved to: {output_path}")
        return output_path


def main():
    """Main function to run comprehensive pipeline validation."""
    validator = PipelineValidator(verbose=True)
    
    print("🔍 Starting comprehensive pipeline validation...")
    print("=" * 60)
    
    # Generate comprehensive report
    report = validator.generate_comprehensive_report()
    
    # Save report
    report_path = validator.save_validation_report(report)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Pipeline Health: {report['pipeline_health'].upper()}")
    print(f"Improvements Validated: {'✅ YES' if report['improvements_validated'] else '❌ NO'}")
    print(f"Report saved to: {report_path}")
    
    if report.get("recommendations"):
        print("\n🎯 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    return 0 if report["improvements_validated"] else 1


if __name__ == "__main__":
    sys.exit(main())
