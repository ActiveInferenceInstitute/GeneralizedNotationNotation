#!/usr/bin/env python3
"""
Pipeline Error Recovery System

This module provides comprehensive error recovery strategies and automatic
suggestions for common pipeline failures, enabling self-healing and guided
troubleshooting across all pipeline steps.
"""

import re
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    CRITICAL = "critical"      # Pipeline cannot continue
    HIGH = "high"             # Step fails but pipeline might continue
    MEDIUM = "medium"         # Warning that should be addressed
    LOW = "low"              # Informational, no action needed

class RecoveryStrategy(Enum):
    """Types of recovery strategies."""
    RETRY = "retry"                    # Retry the operation
    SKIP = "skip"                     # Skip this step and continue
    FALLBACK = "fallback"             # Use alternative implementation
    MANUAL = "manual"                 # Requires manual intervention
    AUTO_FIX = "auto_fix"            # Automatically fix the issue

@dataclass
class ErrorPattern:
    """Definition of an error pattern and its recovery strategy."""
    pattern: str                      # Regex pattern to match error
    error_type: str                  # Type classification
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    description: str                 # Human-readable description
    suggestion: str                  # Specific suggestion for this error
    auto_fix_function: Optional[Callable] = None  # Function to auto-fix if possible
    related_components: List[str] = field(default_factory=list)

@dataclass
class RecoveryAction:
    """A specific recovery action that can be taken."""
    action_type: RecoveryStrategy
    description: str
    command: Optional[str] = None      # Command to run
    script_path: Optional[str] = None  # Script to execute
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_time: str = "unknown"   # Estimated time to complete
    success_probability: float = 0.5  # Estimated success probability

class ErrorRecoverySystem:
    """Comprehensive error recovery system for the GNN pipeline."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_patterns = self._initialize_error_patterns()
        self.recovery_history: List[Dict[str, Any]] = []
        
    def _initialize_error_patterns(self) -> List[ErrorPattern]:
        """Initialize common error patterns and their recovery strategies."""
        return [
            # Dependency errors
            ErrorPattern(
                pattern=r"ModuleNotFoundError.*'([^']+)'",
                error_type="missing_dependency",
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.AUTO_FIX,
                description="Missing Python package dependency",
                suggestion="Install missing package with pip",
                auto_fix_function=self._auto_fix_missing_package
            ),
            
            ErrorPattern(
                pattern=r"ImportError.*No module named '([^']+)'",
                error_type="missing_dependency",
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.AUTO_FIX,
                description="Missing Python module",
                suggestion="Install missing module with pip",
                auto_fix_function=self._auto_fix_missing_package
            ),
            
            # File system errors
            ErrorPattern(
                pattern=r"FileNotFoundError.*'([^']+)'",
                error_type="missing_file",
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.MANUAL,
                description="Required file not found",
                suggestion="Verify file path and ensure file exists"
            ),
            
            ErrorPattern(
                pattern=r"PermissionError.*",
                error_type="permission_denied",
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.MANUAL,
                description="Insufficient permissions",
                suggestion="Check file/directory permissions and user access rights"
            ),
            
            # Memory errors
            ErrorPattern(
                pattern=r"MemoryError.*",
                error_type="memory_exhausted",
                severity=ErrorSeverity.CRITICAL,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                description="Insufficient memory",
                suggestion="Reduce batch size, close other applications, or increase system memory"
            ),
            
            ErrorPattern(
                pattern=r"out of memory.*",
                error_type="memory_exhausted",
                severity=ErrorSeverity.CRITICAL,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                description="Out of memory",
                suggestion="Reduce data size or increase available memory"
            ),
            
            # Timeout errors
            ErrorPattern(
                pattern=r"TimeoutExpired.*",
                error_type="timeout",
                severity=ErrorSeverity.MEDIUM,
                recovery_strategy=RecoveryStrategy.RETRY,
                description="Operation timed out",
                suggestion="Increase timeout or check system performance"
            ),
            
            # Network errors
            ErrorPattern(
                pattern=r"ConnectionError.*",
                error_type="network_error",
                severity=ErrorSeverity.MEDIUM,
                recovery_strategy=RecoveryStrategy.RETRY,
                description="Network connection failed",
                suggestion="Check network connectivity and retry"
            ),
            
            # Syntax errors
            ErrorPattern(
                pattern=r"SyntaxError.*",
                error_type="syntax_error",
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.MANUAL,
                description="Python syntax error",
                suggestion="Check code syntax and fix syntax errors"
            ),
            
            # JSON errors
            ErrorPattern(
                pattern=r"JSONDecodeError.*",
                error_type="json_parse_error",
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                description="Invalid JSON format",
                suggestion="Verify JSON file format and fix syntax errors"
            ),
            
            # GNN-specific errors
            ErrorPattern(
                pattern=r".*GNN.*parsing.*failed.*",
                error_type="gnn_parse_error",
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                description="GNN file parsing failed",
                suggestion="Check GNN file format and syntax"
            ),
            
            # PyMDP-specific errors
            ErrorPattern(
                pattern=r".*pymdp.*",
                error_type="pymdp_error",
                severity=ErrorSeverity.MEDIUM,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                description="PyMDP simulation error",
                suggestion="Check PyMDP installation and model parameters",
                related_components=["pymdp", "simulation"]
            ),
            
            # Visualization errors
            ErrorPattern(
                pattern=r".*matplotlib.*backend.*",
                error_type="matplotlib_backend_error",
                severity=ErrorSeverity.MEDIUM,
                recovery_strategy=RecoveryStrategy.AUTO_FIX,
                description="Matplotlib backend issue",
                suggestion="Set matplotlib backend to non-interactive mode",
                auto_fix_function=self._auto_fix_matplotlib_backend
            ),
            
            # Generic errors
            ErrorPattern(
                pattern=r".*",
                error_type="unknown_error",
                severity=ErrorSeverity.MEDIUM,
                recovery_strategy=RecoveryStrategy.MANUAL,
                description="Unknown error",
                suggestion="Review error details and check logs for more information"
            )
        ]
    
    def analyze_error(self, error_message: str, step_name: str = "", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze an error and provide recovery suggestions.
        
        Args:
            error_message: The error message to analyze
            step_name: Name of the pipeline step where error occurred
            context: Additional context about the error
            
        Returns:
            Dictionary with error analysis and recovery suggestions
        """
        if context is None:
            context = {}
            
        # Find matching error pattern
        matched_pattern = None
        extracted_info = {}
        
        for pattern in self.error_patterns:
            match = re.search(pattern.pattern, error_message, re.IGNORECASE)
            if match:
                matched_pattern = pattern
                if match.groups():
                    extracted_info = {"matched_text": match.group(0), "groups": match.groups()}
                break
        
        if not matched_pattern:
            # Use the generic error pattern as fallback
            matched_pattern = self.error_patterns[-1]
        
        # Generate recovery actions
        recovery_actions = self._generate_recovery_actions(matched_pattern, extracted_info, step_name, context)
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "step_name": step_name,
            "error_type": matched_pattern.error_type,
            "severity": matched_pattern.severity.value,
            "description": matched_pattern.description,
            "suggestion": matched_pattern.suggestion,
            "recovery_strategy": matched_pattern.recovery_strategy.value,
            "recovery_actions": recovery_actions,
            "extracted_info": extracted_info,
            "context": context,
            "auto_fixable": matched_pattern.auto_fix_function is not None
        }
        
        self.logger.info(f"Error analyzed: {matched_pattern.error_type} in {step_name}")
        return analysis
    
    def _generate_recovery_actions(self, pattern: ErrorPattern, extracted_info: Dict[str, Any], 
                                 step_name: str, context: Dict[str, Any]) -> List[RecoveryAction]:
        """Generate specific recovery actions based on the error pattern."""
        actions = []
        
        if pattern.recovery_strategy == RecoveryStrategy.AUTO_FIX and pattern.auto_fix_function:
            actions.append(RecoveryAction(
                action_type=RecoveryStrategy.AUTO_FIX,
                description=f"Automatically fix {pattern.error_type}",
                estimated_time="30 seconds",
                success_probability=0.8
            ))
        
        if pattern.recovery_strategy == RecoveryStrategy.RETRY:
            actions.append(RecoveryAction(
                action_type=RecoveryStrategy.RETRY,
                description=f"Retry {step_name} with exponential backoff",
                estimated_time="1-5 minutes",
                success_probability=0.6
            ))
        
        if pattern.recovery_strategy == RecoveryStrategy.FALLBACK:
            actions.append(RecoveryAction(
                action_type=RecoveryStrategy.FALLBACK,
                description=f"Use fallback implementation for {step_name}",
                estimated_time="30 seconds",
                success_probability=0.7
            ))
        
        # Add step-specific actions
        if step_name in ["1_setup", "setup"]:
            actions.extend(self._get_setup_recovery_actions(pattern, extracted_info))
        elif step_name in ["3_gnn", "gnn"]:
            actions.extend(self._get_gnn_recovery_actions(pattern, extracted_info))
        elif step_name in ["8_visualization", "visualization"]:
            actions.extend(self._get_visualization_recovery_actions(pattern, extracted_info))
        elif step_name in ["12_execute", "execute"]:
            actions.extend(self._get_execution_recovery_actions(pattern, extracted_info))
        
        return actions
    
    def _get_setup_recovery_actions(self, pattern: ErrorPattern, extracted_info: Dict[str, Any]) -> List[RecoveryAction]:
        """Get recovery actions specific to setup step."""
        actions = []
        
        if pattern.error_type == "missing_dependency":
            if extracted_info.get("groups"):
                package_name = extracted_info["groups"][0]
                actions.append(RecoveryAction(
                    action_type=RecoveryStrategy.AUTO_FIX,
                    description=f"Install missing package: {package_name}",
                    command=f"pip install {package_name}",
                    estimated_time="1-3 minutes",
                    success_probability=0.9
                ))
        
        return actions
    
    def _get_gnn_recovery_actions(self, pattern: ErrorPattern, extracted_info: Dict[str, Any]) -> List[RecoveryAction]:
        """Get recovery actions specific to GNN processing step."""
        actions = []
        
        if pattern.error_type == "gnn_parse_error":
            actions.append(RecoveryAction(
                action_type=RecoveryStrategy.FALLBACK,
                description="Use basic GNN parser",
                estimated_time="30 seconds",
                success_probability=0.6
            ))
        
        return actions
    
    def _get_visualization_recovery_actions(self, pattern: ErrorPattern, extracted_info: Dict[str, Any]) -> List[RecoveryAction]:
        """Get recovery actions specific to visualization step."""
        actions = []
        
        if pattern.error_type == "matplotlib_backend_error":
            actions.append(RecoveryAction(
                action_type=RecoveryStrategy.AUTO_FIX,
                description="Set matplotlib to non-interactive backend",
                estimated_time="10 seconds",
                success_probability=0.95
            ))
        
        return actions
    
    def _get_execution_recovery_actions(self, pattern: ErrorPattern, extracted_info: Dict[str, Any]) -> List[RecoveryAction]:
        """Get recovery actions specific to execution step."""
        actions = []
        
        if pattern.error_type == "timeout":
            actions.append(RecoveryAction(
                action_type=RecoveryStrategy.RETRY,
                description="Retry with increased timeout",
                parameters={"timeout_multiplier": 2},
                estimated_time="Variable",
                success_probability=0.7
            ))
        
        if pattern.error_type == "memory_exhausted":
            actions.append(RecoveryAction(
                action_type=RecoveryStrategy.FALLBACK,
                description="Use reduced batch size",
                parameters={"batch_size_reduction": 0.5},
                estimated_time="Same as original",
                success_probability=0.8
            ))
        
        return actions
    
    def _auto_fix_missing_package(self, package_name: str) -> bool:
        """Automatically install missing package."""
        try:
            import subprocess
            result = subprocess.run(
                ["pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Failed to auto-install {package_name}: {e}")
            return False
    
    def _auto_fix_matplotlib_backend(self) -> bool:
        """Fix matplotlib backend issues."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            return True
        except Exception as e:
            self.logger.error(f"Failed to set matplotlib backend: {e}")
            return False
    
    def apply_auto_fix(self, error_analysis: Dict[str, Any]) -> bool:
        """Apply automatic fix if available."""
        if not error_analysis.get("auto_fixable"):
            return False
        
        error_type = error_analysis["error_type"]
        extracted_info = error_analysis.get("extracted_info", {})
        
        try:
            if error_type == "missing_dependency" and extracted_info.get("groups"):
                package_name = extracted_info["groups"][0]
                return self._auto_fix_missing_package(package_name)
            elif error_type == "matplotlib_backend_error":
                return self._auto_fix_matplotlib_backend()
            
        except Exception as e:
            self.logger.error(f"Auto-fix failed for {error_type}: {e}")
        
        return False
    
    def get_recovery_suggestions(self, step_name: str, error_message: str, 
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get comprehensive recovery suggestions for a pipeline step error.
        
        Args:
            step_name: Name of the pipeline step
            error_message: The error message
            context: Additional context
            
        Returns:
            Dictionary with recovery suggestions
        """
        error_analysis = self.analyze_error(error_message, step_name, context)
        
        # Record this error for learning
        self.recovery_history.append({
            "timestamp": error_analysis["timestamp"],
            "step_name": step_name,
            "error_type": error_analysis["error_type"],
            "severity": error_analysis["severity"],
            "auto_fixable": error_analysis["auto_fixable"]
        })
        
        # Generate comprehensive suggestions
        suggestions = {
            "immediate_actions": [],
            "preventive_measures": [],
            "escalation_steps": []
        }
        
        # Immediate actions
        for action in error_analysis["recovery_actions"]:
            if action.action_type in [RecoveryStrategy.AUTO_FIX, RecoveryStrategy.RETRY]:
                suggestions["immediate_actions"].append({
                    "description": action.description,
                    "command": action.command,
                    "estimated_time": action.estimated_time,
                    "success_probability": action.success_probability
                })
        
        # Preventive measures
        if error_analysis["error_type"] == "missing_dependency":
            suggestions["preventive_measures"].append(
                "Add dependency checks to setup validation"
            )
        elif error_analysis["error_type"] == "memory_exhausted":
            suggestions["preventive_measures"].append(
                "Implement resource monitoring and early warnings"
            )
        elif error_analysis["error_type"] == "timeout":
            suggestions["preventive_measures"].append(
                "Add performance benchmarking and timeout estimation"
            )
        
        # Escalation steps
        if error_analysis["severity"] in ["critical", "high"]:
            suggestions["escalation_steps"] = [
                "Review system requirements and configuration",
                "Check for known issues in documentation",
                "Contact support with error details and context"
            ]
        
        return {
            "error_analysis": error_analysis,
            "suggestions": suggestions,
            "can_auto_fix": error_analysis["auto_fixable"],
            "recommended_action": self._get_recommended_action(error_analysis)
        }
    
    def _get_recommended_action(self, error_analysis: Dict[str, Any]) -> str:
        """Get the recommended action based on error analysis."""
        if error_analysis["auto_fixable"]:
            return "auto_fix"
        elif error_analysis["recovery_strategy"] == "retry":
            return "retry"
        elif error_analysis["recovery_strategy"] == "fallback":
            return "fallback"
        else:
            return "manual_intervention"
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about encountered errors."""
        if not self.recovery_history:
            return {"message": "No errors recorded yet"}
        
        total_errors = len(self.recovery_history)
        error_types = {}
        severity_counts = {}
        auto_fixable_count = 0
        
        for error in self.recovery_history:
            error_types[error["error_type"]] = error_types.get(error["error_type"], 0) + 1
            severity_counts[error["severity"]] = severity_counts.get(error["severity"], 0) + 1
            if error["auto_fixable"]:
                auto_fixable_count += 1
        
        return {
            "total_errors": total_errors,
            "error_types": error_types,
            "severity_distribution": severity_counts,
            "auto_fixable_percentage": (auto_fixable_count / total_errors) * 100,
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }


class ErrorReporter:
    """Simple error reporter used by tests to collect errors."""
    def __init__(self):
        self._errors: list[dict[str, Any]] = []

    def collect_error(self, error_type: str, message: str) -> None:
        self._errors.append({"type": error_type, "message": message})

    def get_errors(self) -> list[dict[str, Any]]:
        return list(self._errors)

# Global instance for easy access
error_recovery = ErrorRecoverySystem()

def analyze_pipeline_error(step_name: str, error_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze pipeline errors.
    
    Args:
        step_name: Name of the pipeline step
        error_message: The error message
        context: Additional context
        
    Returns:
        Error analysis with recovery suggestions
    """
    return error_recovery.get_recovery_suggestions(step_name, error_message, context)

def log_error_with_recovery(logger: logging.Logger, step_name: str, error: Exception, 
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Log an error with recovery suggestions.
    
    Args:
        logger: Logger instance
        step_name: Name of the pipeline step
        error: The exception that occurred
        context: Additional context
        
    Returns:
        Error analysis with recovery suggestions
    """
    error_message = str(error)
    full_traceback = traceback.format_exc()
    
    # Analyze the error
    recovery_info = analyze_pipeline_error(step_name, error_message, context)
    
    # Log the error with recovery information
    logger.error(f"❌ Error in {step_name}: {error_message}")
    logger.error(f"Error type: {recovery_info['error_analysis']['error_type']}")
    logger.error(f"Severity: {recovery_info['error_analysis']['severity']}")
    
    if recovery_info["can_auto_fix"]:
        logger.info("🔧 Auto-fix available - attempting automatic recovery")
    
    # Log recovery suggestions
    suggestions = recovery_info["suggestions"]
    if suggestions["immediate_actions"]:
        logger.info("💡 Immediate recovery actions:")
        for action in suggestions["immediate_actions"]:
            logger.info(f"  - {action['description']} (success rate: {action['success_probability']:.0%})")
    
    if suggestions["preventive_measures"]:
        logger.info("🛡️ Preventive measures:")
        for measure in suggestions["preventive_measures"]:
            logger.info(f"  - {measure}")
    
    # Log full traceback at debug level
    logger.debug(f"Full traceback:\n{full_traceback}")
    
    return recovery_info


# Pipeline Step-Specific Recovery Functions

def attempt_step_recovery(script_name: str, step_result: Dict[str, Any], 
                          args, logger) -> Optional[Dict[str, Any]]:
    """Attempt to recover from step failures using fallback strategies."""
    recovery_strategies = {
        "8_visualization.py": recover_visualization_step,
        "12_execute.py": recover_execution_step,
        "15_audio.py": recover_audio_step,
        "13_llm.py": recover_llm_step
    }
    
    if script_name in recovery_strategies:
        try:
            return recovery_strategies[script_name](step_result, args, logger)
        except Exception as e:
            logger.warning(f"Recovery attempt failed for {script_name}: {e}")
            return None
    
    return None

def recover_visualization_step(step_result: Dict[str, Any], args, logger) -> Dict[str, Any]:
    """Recovery strategy for visualization step failures."""
    recovery_result = {"success": False, "warnings": []}
    
    # Check if matplotlib backend issues
    stderr = step_result.get("stderr", "")
    if "matplotlib" in stderr.lower() or "renderer" in stderr.lower():
        recovery_result["warnings"].append("Matplotlib backend issues detected - visualization may use fallback rendering")
        # Check if any visualization files were actually created despite the error
        viz_output_dir = args.output_dir / "8_visualization_output"
        if viz_output_dir.exists():
            viz_files = list(viz_output_dir.rglob("*.png")) + list(viz_output_dir.rglob("*.svg"))
            if viz_files:
                recovery_result["success"] = True
                recovery_result["warnings"].append(f"Found {len(viz_files)} visualization files despite errors")
    
    return recovery_result

def recover_execution_step(step_result: Dict[str, Any], args, logger) -> Dict[str, Any]:
    """Recovery strategy for execution step failures."""
    recovery_result = {"success": False, "warnings": []}
    
    stderr = step_result.get("stderr", "")
    if "PyMDP not available" in stderr or "render output directory not found" in stderr:
        # Create informational execution report even without actual execution
        exec_output_dir = args.output_dir / "12_execute_output"
        exec_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a fallback execution report
        fallback_report = {
            "status": "degraded",
            "message": "Execution completed with graceful degradation",
            "missing_dependencies": [],
            "available_simulators": [],
            "recommendations": ["Install PyMDP for full simulation capabilities", "Run render step (11) before execute step (12)"]
        }
        
        # Check which simulators are available
        if "PyMDP not available" in stderr:
            fallback_report["missing_dependencies"].append("PyMDP")
        
        # Write fallback report
        import json
        report_file = exec_output_dir / "execution_fallback_report.json"
        with open(report_file, 'w') as f:
            json.dump(fallback_report, f, indent=2)
            
        recovery_result["success"] = True
        recovery_result["warnings"].append("Created fallback execution report with installation recommendations")
    
    return recovery_result

def recover_audio_step(step_result: Dict[str, Any], args, logger) -> Dict[str, Any]:
    """Recovery strategy for audio step failures."""
    recovery_result = {"success": True, "warnings": []}
    
    # Audio step is typically optional and can gracefully degrade
    recovery_result["warnings"].append("Audio generation skipped - non-critical for pipeline completion")
    
    return recovery_result

def recover_llm_step(step_result: Dict[str, Any], args, logger) -> Dict[str, Any]:
    """Recovery strategy for LLM step failures."""
    recovery_result = {"success": False, "warnings": []}
    
    stderr = step_result.get("stderr", "")
    if "API" in stderr or "key" in stderr.lower() or "authentication" in stderr.lower():
        # Create fallback analysis without LLM
        llm_output_dir = args.output_dir / "13_llm_output"
        llm_output_dir.mkdir(parents=True, exist_ok=True)
        
        fallback_analysis = """
# GNN Analysis Report (Fallback Mode)

This analysis was generated without LLM integration due to missing API credentials.

## Recommendations
1. Configure API keys for enhanced AI analysis
2. Review GNN specifications manually
3. Consider local LLM alternatives
        """
        
        analysis_file = llm_output_dir / "fallback_analysis.md"
        with open(analysis_file, 'w') as f:
            f.write(fallback_analysis)
            
        recovery_result["success"] = True
        recovery_result["warnings"].append("Created fallback analysis without LLM integration")
    
    return recovery_result

def is_failure_recoverable(script_name: str, step_result: Dict[str, Any]) -> bool:
    """Determine if a step failure is recoverable for downstream processing."""
    # Steps that are critical for downstream processing
    critical_steps = {
        "3_gnn.py": "GNN parsing is critical for all downstream steps",
        "1_setup.py": "Environment setup is critical for pipeline execution"
    }
    
    if script_name in critical_steps:
        return False
        
    # Check exit code - some exit codes indicate recoverable errors
    exit_code = step_result.get("exit_code", -1)
    if exit_code == 2:  # Often used for warnings/partial success
        return True
        
    # Check stderr for recoverable error patterns
    stderr = step_result.get("stderr", "")
    recoverable_patterns = [
        "not available",
        "optional dependency",
        "warning",
        "fallback",
        "degraded"
    ]
    
    return any(pattern in stderr.lower() for pattern in recoverable_patterns) 