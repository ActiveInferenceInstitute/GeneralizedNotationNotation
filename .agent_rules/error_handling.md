# Error Handling and Safe-to-Fail

> Steps 8 (Visualization), 9 (Advanced Viz), 12 (Execute) **must never stop the pipeline**.

## Exit Code Conventions

| Code | Meaning | When to Use |
|------|---------|-------------|
| `0` | Success | Normal completion, partial results, all optional steps |
| `1` | Critical Error | Only Steps 1 and 3 (setup/GNN parse failures halt pipeline) |
| `2` | Success with Warnings | Missing files, partial processing, degraded output |

---

## Safe-to-Fail Implementation (Steps 8, 9, 12)

```python
def safe_to_fail_step(target_dir: Path, output_dir: Path, logger) -> bool:
    """Safe-to-fail step — NEVER returns False."""

    # Level 1: Full processing
    try:
        result = full_processing(target_dir, output_dir)
        if result:
            return True
    except Exception as e:
        logger.warning(f"Full processing failed: {e}")

    # Level 2: Reduced processing
    try:
        return reduced_processing(target_dir, output_dir)
    except Exception as e:
        logger.warning(f"Reduced processing failed: {e}")

    # Level 3: Explicit degraded status report
    try:
        write_degraded_status_report(output_dir)
    except Exception as report_error:
        logger.error(f"Could not write degraded status report: {report_error}")

    return True  # ALWAYS True — never stop the pipeline

def main() -> int:
    """Entry point — always returns 0 for safe-to-fail steps."""
    try:
        safe_to_fail_step(target_dir, output_dir, logger)
    except Exception as e:
        logger.error(f"Step failed completely: {e}")
    return 0  # Never return 1
```

---

## Graceful Degradation Pattern

```python
# Tiered functionality — try requested options, report explicit outcomes
def process_with_explicit_status(model: Dict[str, Any]) -> ExecutionResult:
    """Run requested frameworks and report skipped/failed status with reasons."""
    frameworks = [
        ("jax", JAX_AVAILABLE, _execute_jax),
        ("pymdp", PYMDP_AVAILABLE, _execute_pymdp),
        ("julia", JULIA_AVAILABLE, _execute_julia),
    ]
    for name, available, executor in frameworks:
        if available:
            try:
                return executor(model)
            except Exception as e:
                logger.warning(f"{name} failed: {e}")

    logger.error("No requested framework executed")
    return ExecutionResult(success=False, executed=False,
                          message="No requested framework executed; see dependency diagnostics")
```

---

## Error Classification

| Type | Handling | Example |
|------|---------|---------|
| `ImportError` | Skip gracefully | Optional dep not installed |
| `FileNotFoundError` | Log warning, continue | Missing input file |
| `ValueError` | Log error, return False | Invalid parameter |
| `subprocess.TimeoutExpired` | Log, return failed/degraded status | Simulation timeout |
| `MemoryError` | Log critical, return False | Model too large |
| `Exception` | Log full traceback, continue | Unexpected error |

---

## Retry with Exponential Backoff

```python
import time
from functools import wraps

def with_retry(max_retries: int = 3, base_delay: float = 1.0):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {delay}s")
                    time.sleep(delay)
        return wrapper
    return decorator
```

---

## Structured Error Logging

```python
# Always include: what failed, why, what to do
logger.error(
    f"GNN parsing failed for {file_path}: {e}",
    extra={
        "error_type": type(e).__name__,
        "file": str(file_path),
        "suggestion": "Verify GNN file format (see gnn_standards.md)"
    }
)
```

---

## Timeout Management

```python
STEP_TIMEOUTS = {
    "fast": 30,          # parse, validate, export
    "standard": 120,     # render, ontology, registry
    "slow": 600,         # tests, execute
    "llm": 360,          # LLM processing
    "visualization": 180,# viz generation
}

# Environment override: STEP_TIMEOUT_RENDER=60 python src/11_render.py
```

---

## Degraded Status Report Generation

When a step fails completely, generate a meaningful HTML report:

```python
def write_degraded_status_report(output_dir: Path, error: str = "") -> None:
    """Create an explicit status artifact in the output directory."""
    report = output_dir / "degraded_status_report.html"
    report.write_text(f"""
    <html><body>
    <h1>Step Status: Degraded</h1>
    <p>Processing failed: {error}</p>
    <p>Check logs for details. Pipeline continues.</p>
    </body></html>
    """)
```

---

**Last Updated**: 2026-05-20 | **Status**: Maintained Standard
