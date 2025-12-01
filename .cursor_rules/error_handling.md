# Error Handling and Recovery Patterns

> **Environment Note**: Error handling patterns are validated through real testing via `uv run pytest`. All error scenarios must be tested with actual failure conditions.

## Overview

This document defines comprehensive error handling strategies for the GNN pipeline. The pipeline implements a "fail gracefully, continue when possible" philosophy with robust recovery mechanisms.

---

## Exit Code Conventions

### Standard Exit Codes

| Code | Meaning | Pipeline Behavior | Use Case |
|------|---------|-------------------|----------|
| 0 | Success | Continue to next step | Step completed successfully |
| 1 | Critical Error | Stop pipeline | Unrecoverable failure |
| 2 | Success with Warnings | Continue to next step | Completed but issues detected |

### Implementation Pattern

```python
def main() -> int:
    """Main function with proper exit codes."""
    try:
        success = process_step()
        
        if success:
            logger.info("Step completed successfully")
            return 0  # SUCCESS
        else:
            logger.warning("Step completed with warnings")
            return 2  # SUCCESS_WITH_WARNINGS
            
    except CriticalError as e:
        logger.error(f"Critical failure: {e}")
        return 1  # CRITICAL_ERROR - stops pipeline
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # For most steps, return 0 to allow pipeline to continue
        return 0  # Safe-to-fail: pipeline continues

if __name__ == "__main__":
    sys.exit(main())
```

---

## Safe-to-Fail Pattern

### Overview

Steps 8 (Visualization), 9 (Advanced Visualization), and 12 (Execute) implement comprehensive safe-to-fail patterns. These steps **never stop the pipeline** regardless of internal failures.

### Implementation Structure

```python
def safe_to_fail_step(target_dir: Path, output_dir: Path, logger) -> bool:
    """
    Safe-to-fail step implementation.
    
    This step will NEVER return a non-zero exit code.
    All failures are handled internally with fallbacks.
    """
    results = {
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "fallback_used": False,
    }
    
    try:
        # Level 1: Full functionality
        result = attempt_full_processing(target_dir, output_dir)
        results["attempted"] += 1
        if result:
            results["succeeded"] += 1
            return True
    except Exception as e:
        logger.warning(f"Full processing failed: {e}")
        results["failed"] += 1
    
    try:
        # Level 2: Reduced functionality
        result = attempt_reduced_processing(target_dir, output_dir)
        results["fallback_used"] = True
        if result:
            results["succeeded"] += 1
            return True
    except Exception as e:
        logger.warning(f"Reduced processing failed: {e}")
        results["failed"] += 1
    
    try:
        # Level 3: Minimal fallback (always succeeds)
        generate_fallback_report(output_dir, results)
        logger.info("Generated fallback report")
        return True  # ALWAYS return True
    except Exception as e:
        logger.error(f"Even fallback failed: {e}")
        # Still return True - never stop pipeline
        return True
```

### Fallback HTML Report

When processing fails, generate an informative fallback:

```python
def generate_fallback_html(
    output_dir: Path,
    step_name: str,
    error: str,
    suggestions: List[str]
) -> Path:
    """Generate fallback HTML when processing fails."""
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{step_name} - Fallback Report</title>
    <style>
        body {{ font-family: system-ui, sans-serif; padding: 2rem; }}
        .warning {{ background: #fff3cd; padding: 1rem; border-radius: 8px; }}
        .suggestion {{ margin: 0.5rem 0; padding: 0.5rem; background: #f8f9fa; }}
    </style>
</head>
<body>
    <h1>{step_name}</h1>
    <div class="warning">
        <h2>Processing Completed with Fallback</h2>
        <p>Full processing was not available. Error: {error}</p>
    </div>
    <h3>Suggestions</h3>
    {"".join(f'<div class="suggestion">{s}</div>' for s in suggestions)}
    <h3>Next Steps</h3>
    <ul>
        <li>The pipeline has continued successfully</li>
        <li>This step's advanced features were skipped</li>
        <li>Check dependencies listed above for full functionality</li>
    </ul>
</body>
</html>"""
    
    output_file = output_dir / f"{step_name}_fallback.html"
    output_file.write_text(html_content)
    return output_file
```

---

## Graceful Degradation Pattern

### Dependency Availability Checking

```python
def check_dependency_availability() -> Dict[str, bool]:
    """Check which optional dependencies are available."""
    deps = {}
    
    # Check PyMDP
    try:
        import pymdp
        deps["pymdp"] = True
    except ImportError:
        deps["pymdp"] = False
        logger.info("PyMDP not available - will use fallback")
    
    # Check JAX
    try:
        import jax
        deps["jax"] = True
    except ImportError:
        deps["jax"] = False
    
    # Check Julia
    try:
        import subprocess
        result = subprocess.run(
            ["julia", "--version"],
            capture_output=True,
            timeout=5
        )
        deps["julia"] = result.returncode == 0
    except Exception:
        deps["julia"] = False
    
    return deps
```

### Tiered Functionality Pattern

```python
class TieredProcessor:
    """Processor with tiered functionality based on available dependencies."""
    
    def __init__(self):
        self.deps = check_dependency_availability()
        self.tier = self._determine_tier()
    
    def _determine_tier(self) -> str:
        """Determine functionality tier."""
        if all(self.deps.values()):
            return "full"
        elif self.deps.get("jax") or self.deps.get("pymdp"):
            return "standard"
        else:
            return "basic"
    
    def process(self, model: Dict[str, Any]) -> ProcessingResult:
        """Process with appropriate tier."""
        if self.tier == "full":
            return self._full_process(model)
        elif self.tier == "standard":
            return self._standard_process(model)
        else:
            return self._basic_process(model)
    
    def _full_process(self, model):
        """Full processing with all features."""
        ...
    
    def _standard_process(self, model):
        """Standard processing with core features."""
        ...
    
    def _basic_process(self, model):
        """Basic processing - always available."""
        ...
```

---

## Error Classification

### Error Types and Handling

```python
from enum import Enum, auto
from typing import Optional

class ErrorType(Enum):
    """Classification of error types."""
    
    # Recoverable errors - continue with degraded functionality
    DEPENDENCY_MISSING = auto()     # Optional dep not installed
    RESOURCE_UNAVAILABLE = auto()   # Network, disk space, etc.
    TIMEOUT = auto()                # Operation took too long
    PARTIAL_FAILURE = auto()        # Some items failed, others ok
    
    # Critical errors - may stop pipeline
    SYNTAX_ERROR = auto()           # Invalid GNN syntax
    CONFIGURATION_ERROR = auto()    # Invalid configuration
    PERMISSION_ERROR = auto()       # File/directory permissions
    CRITICAL_DEPENDENCY = auto()    # Required dep missing

class GNNError(Exception):
    """Base exception for GNN pipeline errors."""
    
    def __init__(
        self,
        message: str,
        error_type: ErrorType,
        recoverable: bool = True,
        suggestion: Optional[str] = None
    ):
        self.message = message
        self.error_type = error_type
        self.recoverable = recoverable
        self.suggestion = suggestion
        super().__init__(message)
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)

# Specific error classes
class DependencyError(GNNError):
    """Error due to missing dependency."""
    
    def __init__(self, package: str, install_cmd: str):
        super().__init__(
            f"{package} not available",
            ErrorType.DEPENDENCY_MISSING,
            recoverable=True,
            suggestion=f"Install with: {install_cmd}"
        )

class SyntaxError(GNNError):
    """Error in GNN file syntax."""
    
    def __init__(self, message: str, line: Optional[int] = None):
        line_info = f" at line {line}" if line else ""
        super().__init__(
            f"Syntax error{line_info}: {message}",
            ErrorType.SYNTAX_ERROR,
            recoverable=False,
            suggestion="Check GNN syntax documentation"
        )
```

### Error Handler Implementation

```python
class ErrorHandler:
    """Centralized error handling for pipeline steps."""
    
    def __init__(self, logger: logging.Logger, step_name: str):
        self.logger = logger
        self.step_name = step_name
        self.errors: List[GNNError] = []
        self.warnings: List[str] = []
    
    def handle(self, error: Exception) -> bool:
        """
        Handle an error and determine if processing can continue.
        
        Returns:
            True if processing can continue, False if must stop.
        """
        if isinstance(error, GNNError):
            self.errors.append(error)
            
            if error.recoverable:
                self.logger.warning(
                    f"[{self.step_name}] {error.error_type.name}: {error}"
                )
                return True  # Can continue
            else:
                self.logger.error(
                    f"[{self.step_name}] CRITICAL {error.error_type.name}: {error}"
                )
                return False  # Must stop
        else:
            # Unknown error - log and attempt to continue
            self.logger.error(
                f"[{self.step_name}] Unexpected error: {type(error).__name__}: {error}"
            )
            self.warnings.append(str(error))
            return True  # Try to continue
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error handling summary."""
        return {
            "step": self.step_name,
            "total_errors": len(self.errors),
            "recoverable_errors": sum(1 for e in self.errors if e.recoverable),
            "critical_errors": sum(1 for e in self.errors if not e.recoverable),
            "warnings": len(self.warnings),
            "error_types": [e.error_type.name for e in self.errors],
        }
```

---

## Logging Patterns for Errors

### Structured Error Logging

```python
def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Dict[str, Any]
) -> None:
    """Log error with full context for debugging."""
    
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        **context
    }
    
    # Log structured data
    logger.error(
        f"Error in {context.get('operation', 'unknown')}: {error}",
        extra=error_data
    )
    
    # Also log to error file
    error_file = context.get("output_dir", Path(".")) / "errors.json"
    try:
        existing = json.loads(error_file.read_text()) if error_file.exists() else []
        existing.append({
            "timestamp": datetime.now().isoformat(),
            **error_data
        })
        error_file.write_text(json.dumps(existing, indent=2))
    except Exception:
        pass  # Don't fail on error logging
```

### Actionable Error Messages

```python
def format_actionable_error(
    error: Exception,
    file_path: Optional[Path] = None,
    operation: Optional[str] = None
) -> str:
    """Format error message with actionable guidance."""
    
    lines = []
    
    # Error summary
    lines.append(f"Error: {type(error).__name__}: {error}")
    
    # Context
    if file_path:
        lines.append(f"File: {file_path}")
    if operation:
        lines.append(f"Operation: {operation}")
    
    # Specific guidance based on error type
    if isinstance(error, FileNotFoundError):
        lines.append("Suggestion: Check that the file path exists")
        lines.append(f"  - Verify: ls -la {file_path.parent if file_path else '.'}")
    
    elif isinstance(error, ImportError):
        module = str(error).split("'")[1] if "'" in str(error) else "unknown"
        lines.append(f"Suggestion: Install missing module")
        lines.append(f"  - Run: uv pip install {module}")
    
    elif isinstance(error, json.JSONDecodeError):
        lines.append("Suggestion: Check JSON syntax")
        lines.append(f"  - Line: {error.lineno}, Column: {error.colno}")
    
    elif isinstance(error, TimeoutError):
        lines.append("Suggestion: Increase timeout or check resource usage")
        lines.append("  - Set: export STEP_TIMEOUT=600")
    
    return "\n".join(lines)
```

---

## Recovery Patterns

### Retry with Exponential Backoff

```python
import time
from functools import wraps

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds.
        backoff_factor: Multiplier for delay after each retry.
        exceptions: Tuple of exceptions to catch.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        )
            
            raise last_exception
        return wrapper
    return decorator

# Usage
@retry_with_backoff(max_retries=3, exceptions=(ConnectionError, TimeoutError))
def fetch_remote_resource(url: str) -> bytes:
    """Fetch resource with automatic retry."""
    ...
```

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    """
    Circuit breaker for protecting against repeated failures.
    
    States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Failures exceeded threshold, requests fail immediately
        - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpen("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self.failures = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
```

---

## Pipeline Recovery

### Checkpoint and Resume

```python
@dataclass
class PipelineCheckpoint:
    """Checkpoint for pipeline recovery."""
    
    step_number: int
    step_name: str
    timestamp: datetime
    outputs: Dict[str, Path]
    state: Dict[str, Any]
    
    def save(self, checkpoint_dir: Path) -> Path:
        """Save checkpoint to disk."""
        checkpoint_file = checkpoint_dir / f"checkpoint_step_{self.step_number}.json"
        data = {
            "step_number": self.step_number,
            "step_name": self.step_name,
            "timestamp": self.timestamp.isoformat(),
            "outputs": {k: str(v) for k, v in self.outputs.items()},
            "state": self.state,
        }
        checkpoint_file.write_text(json.dumps(data, indent=2))
        return checkpoint_file
    
    @classmethod
    def load(cls, checkpoint_file: Path) -> "PipelineCheckpoint":
        """Load checkpoint from disk."""
        data = json.loads(checkpoint_file.read_text())
        return cls(
            step_number=data["step_number"],
            step_name=data["step_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            outputs={k: Path(v) for k, v in data["outputs"].items()},
            state=data["state"],
        )

def resume_from_checkpoint(checkpoint_dir: Path) -> Optional[int]:
    """Find latest checkpoint and return step to resume from."""
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.json"))
    if not checkpoints:
        return None
    
    latest = PipelineCheckpoint.load(checkpoints[-1])
    logger.info(f"Resuming from step {latest.step_number + 1} ({latest.step_name})")
    return latest.step_number + 1
```

---

## Testing Error Handling

### Testing Error Scenarios

```python
class TestErrorHandling:
    """Test error handling patterns."""
    
    def test_graceful_degradation(self):
        """Test graceful degradation when dependency missing."""
        # Simulate missing dependency
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "optional_package", None)
            
            result = process_with_optional_dep()
            
            # Should succeed with fallback
            assert result.success is True
            assert result.used_fallback is True
    
    def test_safe_to_fail_always_succeeds(self):
        """Test safe-to-fail step never returns failure."""
        # Even with broken input, should return success
        result = safe_to_fail_step(
            target_dir=Path("/nonexistent"),
            output_dir=Path("/tmp/test"),
            logger=logging.getLogger()
        )
        
        # Should always return True
        assert result is True
    
    def test_error_classification(self):
        """Test error classification and handling."""
        handler = ErrorHandler(logging.getLogger(), "test_step")
        
        # Recoverable error
        dep_error = DependencyError("pymdp", "pip install pymdp")
        can_continue = handler.handle(dep_error)
        assert can_continue is True
        
        # Critical error
        syntax_error = SyntaxError("Invalid matrix dimensions", line=42)
        can_continue = handler.handle(syntax_error)
        assert can_continue is False
```

---

**Last Updated**: December 2025  
**Status**: Production Standard

