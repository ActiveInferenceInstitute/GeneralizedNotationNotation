# Performance Optimization Guide

> **Environment Note**: Performance monitoring uses `uv` for consistent execution. Run benchmarks with `uv run python` to ensure accurate measurements.

## Overview

This document defines performance standards, optimization patterns, and monitoring practices for the GNN pipeline. The goal is to maintain fast, efficient execution while ensuring scientific accuracy.

---

## Performance Baselines

### Pipeline Execution Targets

| Metric | Target | Maximum | Notes |
|--------|--------|---------|-------|
| Full pipeline execution | <3 minutes | 30 minutes | Standard workload |
| Single step execution | <30 seconds | 5 minutes | Except tests/LLM |
| Test suite (fast) | <2 minutes | 10 minutes | `-m "not slow"` |
| Test suite (full) | <10 minutes | 20 minutes | All tests |
| Memory usage (peak) | <500MB | 2GB | Per step |
| Memory usage (total) | <1GB | 2GB | Full pipeline |

### Step-Specific Benchmarks

Based on actual pipeline execution (November 2025):

| Step | Description | Target Time | Actual Time |
|------|-------------|-------------|-------------|
| 0 | Template | <1s | 0.13s |
| 1 | Setup | <5s | 2.46s |
| 2 | Tests | <120s | 93.5s |
| 3 | GNN Processing | <1s | 0.09s |
| 4 | Model Registry | <1s | 0.06s |
| 5 | Type Checker | <1s | 0.06s |
| 6 | Validation | <1s | 0.07s |
| 7 | Export | <1s | 0.07s |
| 8 | Visualization | <1s | 0.32s |
| 9 | Advanced Viz | <15s | 8.25s |
| 10 | Ontology | <1s | 0.17s |
| 11 | Render | <1s | 0.11s |
| 12 | Execute | <60s | 29.8s |
| 13 | LLM | <60s | 0.41s |
| 14 | ML Integration | <5s | 1.74s |
| 15 | Audio | <1s | 0.10s |
| 16 | Analysis | <1s | 0.28s |
| 17-19 | Integration/Security/Research | <1s each | 0.06s each |
| 20 | Website | <1s | 0.06s |
| 21 | MCP | <15s | 10.5s |
| 22 | GUI | <2s | 1.19s |
| 23 | Report | <1s | 0.07s |

---

## Memory Optimization

### Memory Usage Patterns

```python
import psutil
import gc
from contextlib import contextmanager

@contextmanager
def memory_tracked_operation(operation_name: str, logger):
    """
    Context manager for tracking memory usage.
    
    Usage:
        with memory_tracked_operation("parse_models", logger):
            models = parse_all_models(files)
    """
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        yield
    finally:
        final_memory = process.memory_info().rss / 1024 / 1024
        delta = final_memory - initial_memory
        
        logger.debug(
            f"Memory: {operation_name} | "
            f"Initial: {initial_memory:.1f}MB | "
            f"Final: {final_memory:.1f}MB | "
            f"Delta: {delta:+.1f}MB"
        )
        
        # Warn if significant memory increase
        if delta > 100:
            logger.warning(f"High memory usage in {operation_name}: +{delta:.1f}MB")
```

### Large File Processing

```python
def process_large_file_streaming(file_path: Path, chunk_size: int = 8192):
    """
    Process large files using streaming to minimize memory.
    
    Instead of loading entire file into memory, process in chunks.
    """
    with open(file_path, 'r') as f:
        buffer = ""
        for chunk in iter(lambda: f.read(chunk_size), ""):
            buffer += chunk
            # Process complete lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                yield process_line(line)
        
        # Process remaining
        if buffer:
            yield process_line(buffer)
```

### Memory Cleanup

```python
def cleanup_after_step(large_objects: List[Any]) -> None:
    """
    Explicit cleanup after memory-intensive operations.
    
    Call at the end of steps that process large datasets.
    """
    for obj in large_objects:
        del obj
    
    # Force garbage collection
    gc.collect()
    
    # Log current memory state
    process = psutil.Process()
    logger.debug(f"Post-cleanup memory: {process.memory_info().rss / 1024 / 1024:.1f}MB")
```

---

## Execution Time Optimization

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Callable, TypeVar

T = TypeVar('T')
R = TypeVar('R')

def parallel_process(
    items: List[T],
    processor: Callable[[T], R],
    max_workers: int = 4,
    use_threads: bool = True
) -> List[R]:
    """
    Process items in parallel for improved throughput.
    
    Args:
        items: Items to process.
        processor: Function to apply to each item.
        max_workers: Maximum parallel workers.
        use_threads: Use threads (I/O bound) or processes (CPU bound).
    
    Returns:
        List of processed results.
    """
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    with Executor(max_workers=max_workers) as executor:
        results = list(executor.map(processor, items))
    
    return results

# Usage for I/O bound operations (file reading)
results = parallel_process(files, parse_file, max_workers=8, use_threads=True)

# Usage for CPU bound operations (computation)
results = parallel_process(models, compute_matrices, max_workers=4, use_threads=False)
```

### Caching Patterns

```python
from functools import lru_cache
from pathlib import Path
import hashlib
import json

# In-memory LRU cache for repeated calls
@lru_cache(maxsize=128)
def parse_gnn_cached(file_content_hash: str, file_path: str) -> Dict[str, Any]:
    """
    Parse GNN file with caching based on content hash.
    
    The file_content_hash ensures cache invalidation when file changes.
    """
    return _do_parse(Path(file_path))

def get_cached_parse(file_path: Path) -> Dict[str, Any]:
    """Get parsed result with automatic cache management."""
    content = file_path.read_text()
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return parse_gnn_cached(content_hash, str(file_path))


# Disk-based cache for expensive operations
class DiskCache:
    """Persistent disk cache for expensive computations."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Set cached value with TTL."""
        cache_file = self.cache_dir / f"{key}.json"
        data = {
            "value": value,
            "expires": time.time() + ttl_seconds
        }
        cache_file.write_text(json.dumps(data))
    
    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl_seconds: int = 3600
    ) -> Any:
        """Get cached value or compute and cache."""
        cached = self.get(key)
        if cached and cached.get("expires", 0) > time.time():
            return cached["value"]
        
        value = compute_fn()
        self.set(key, value, ttl_seconds)
        return value
```

### Lazy Loading

```python
class LazyLoader:
    """
    Lazy loading wrapper for expensive imports/initializations.
    
    The actual loading only happens when the object is first accessed.
    """
    
    def __init__(self, loader: Callable[[], Any]):
        self._loader = loader
        self._instance = None
        self._loaded = False
    
    def __getattr__(self, name: str) -> Any:
        if not self._loaded:
            self._instance = self._loader()
            self._loaded = True
        return getattr(self._instance, name)

# Usage
def _load_heavy_model():
    """Load heavy ML model - only when needed."""
    import transformers
    return transformers.AutoModel.from_pretrained("model-name")

heavy_model = LazyLoader(_load_heavy_model)

# Model is NOT loaded until first access:
# result = heavy_model.predict(data)  # Now it loads
```

---

## I/O Optimization

### Efficient File Operations

```python
from pathlib import Path
from typing import Iterator, List

def batch_read_files(
    file_paths: List[Path],
    batch_size: int = 10
) -> Iterator[List[tuple[Path, str]]]:
    """
    Read files in batches to optimize I/O.
    
    Yields batches of (path, content) tuples.
    """
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        yield [(p, p.read_text()) for p in batch]

def write_with_buffer(
    output_path: Path,
    lines: Iterator[str],
    buffer_size: int = 65536
) -> None:
    """
    Write lines with buffering for efficiency.
    
    More efficient than writing line-by-line.
    """
    with open(output_path, 'w', buffering=buffer_size) as f:
        for line in lines:
            f.write(line)
            f.write('\n')
```

### JSON Optimization

```python
import json
from typing import Any

# Use orjson for faster JSON operations (if available)
try:
    import orjson
    
    def fast_json_dumps(obj: Any) -> str:
        """Fast JSON serialization."""
        return orjson.dumps(obj).decode()
    
    def fast_json_loads(s: str) -> Any:
        """Fast JSON parsing."""
        return orjson.loads(s)

except ImportError:
    # Fallback to standard json
    def fast_json_dumps(obj: Any) -> str:
        return json.dumps(obj)
    
    def fast_json_loads(s: str) -> Any:
        return json.loads(s)
```

---

## Timeout Management

### Step Timeouts

```python
import signal
from functools import wraps

class TimeoutError(Exception):
    """Operation timed out."""
    pass

def timeout(seconds: int):
    """
    Decorator to enforce timeout on function execution.
    
    Args:
        seconds: Maximum execution time in seconds.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(
                    f"{func.__name__} timed out after {seconds} seconds"
                )
            
            # Set alarm
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator

# Usage
@timeout(60)
def execute_simulation(script_path: Path) -> bool:
    """Execute simulation with 60-second timeout."""
    ...
```

### Configurable Timeouts

```python
# Default timeouts per step type
STEP_TIMEOUTS = {
    "fast": 30,           # Quick steps (parse, validate)
    "standard": 120,      # Normal steps (render, export)
    "slow": 600,          # Slow steps (tests, execute)
    "llm": 360,           # LLM processing
    "visualization": 180, # Visualization generation
}

def get_step_timeout(step_name: str) -> int:
    """
    Get timeout for a step, respecting environment override.
    
    Environment variables:
        STEP_TIMEOUT: Override all timeouts
        STEP_TIMEOUT_{STEP_NAME}: Override specific step
    """
    # Check environment override
    env_timeout = os.environ.get(f"STEP_TIMEOUT_{step_name.upper()}")
    if env_timeout:
        return int(env_timeout)
    
    global_timeout = os.environ.get("STEP_TIMEOUT")
    if global_timeout:
        return int(global_timeout)
    
    # Determine step category
    if step_name in ["template", "registry", "type_checker"]:
        return STEP_TIMEOUTS["fast"]
    elif step_name in ["tests"]:
        return STEP_TIMEOUTS["slow"]
    elif step_name in ["llm"]:
        return STEP_TIMEOUTS["llm"]
    else:
        return STEP_TIMEOUTS["standard"]
```

---

## Performance Monitoring

### Built-in Performance Tracking

```python
import time
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class PerformanceMetrics:
    """Collected performance metrics."""
    
    operation: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0
    sub_operations: List["PerformanceMetrics"] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def memory_delta_mb(self) -> float:
        return self.memory_end_mb - self.memory_start_mb

class PerformanceTracker:
    """Track performance across operations."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self._stack: List[PerformanceMetrics] = []
    
    @contextmanager
    def track(self, operation: str):
        """Context manager for tracking operation."""
        process = psutil.Process()
        metric = PerformanceMetrics(
            operation=operation,
            memory_start_mb=process.memory_info().rss / 1024 / 1024
        )
        
        self._stack.append(metric)
        
        try:
            yield metric
        finally:
            metric.end_time = time.time()
            metric.memory_end_mb = process.memory_info().rss / 1024 / 1024
            
            self._stack.pop()
            
            if self._stack:
                self._stack[-1].sub_operations.append(metric)
            else:
                self.metrics.append(metric)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "total_operations": len(self.metrics),
            "total_time_seconds": sum(m.duration_seconds for m in self.metrics),
            "peak_memory_mb": max(
                (m.memory_end_mb for m in self.metrics),
                default=0
            ),
            "operations": [
                {
                    "name": m.operation,
                    "duration_seconds": m.duration_seconds,
                    "memory_delta_mb": m.memory_delta_mb,
                }
                for m in self.metrics
            ]
        }

# Global tracker
performance_tracker = PerformanceTracker()
```

### Performance Assertions in Tests

```python
import pytest

def test_gnn_parsing_performance():
    """Ensure GNN parsing meets performance requirements."""
    from gnn import parse_gnn_file
    
    test_file = Path("tests/fixtures/large_model.md")
    
    start = time.time()
    result = parse_gnn_file(test_file)
    duration = time.time() - start
    
    # Parsing should complete in under 1 second
    assert duration < 1.0, f"Parsing took {duration:.2f}s, expected <1s"
    assert result is not None

def test_memory_usage():
    """Ensure step doesn't exceed memory limits."""
    import tracemalloc
    
    tracemalloc.start()
    
    # Run operation
    result = process_large_batch(files)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_mb = peak / 1024 / 1024
    assert peak_mb < 500, f"Peak memory {peak_mb:.1f}MB exceeds 500MB limit"
```

---

## Profiling Guidelines

### CPU Profiling

```bash
# Profile a specific step
python -m cProfile -o profile.prof src/11_render.py --target-dir input/gnn_files

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.prof')
p.sort_stats('cumulative').print_stats(20)
"

# Visual profiling with snakeviz
uv pip install snakeviz
snakeviz profile.prof
```

### Memory Profiling

```bash
# Memory profiling with memory_profiler
uv pip install memory_profiler

# Profile memory line-by-line
python -m memory_profiler src/11_render.py --target-dir input/gnn_files

# Or use decorator in code:
from memory_profiler import profile

@profile
def memory_intensive_function():
    ...
```

### Line Profiling

```bash
# Line-by-line timing
uv pip install line_profiler

# Add @profile decorator to functions, then:
kernprof -l -v src/script.py
```

---

## Optimization Checklist

Before submitting performance-sensitive code:

- [ ] Measured baseline performance
- [ ] Identified bottlenecks with profiling
- [ ] Implemented caching where appropriate
- [ ] Used streaming for large files
- [ ] Applied parallel processing where safe
- [ ] Added timeout protection
- [ ] Verified memory cleanup
- [ ] Added performance tests
- [ ] Documented performance characteristics

---

**Last Updated**: December 2025  
**Status**: Production Standard


