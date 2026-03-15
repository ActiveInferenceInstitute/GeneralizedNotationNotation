# Performance Optimization

> Full pipeline target: <3 minutes | Peak memory: <500MB

## Step Benchmarks (Actual — March 2026)

| Step | Description | Target | Actual |
|------|-------------|--------|--------|
| 0 | Template | <1s | 0.13s |
| 1 | Setup | <5s | 2.46s |
| 2 | Tests | <120s | 93.5s |
| 3 | GNN Processing | <1s | 0.09s |
| 4–7 | Registry/Type/Val/Export | <1s each | ~0.06s |
| 8 | Visualization | <5s | 0.32s |
| 9 | Advanced Viz | <15s | 8.25s |
| 12 | Execute | <60s | 29.8s |
| 13 | LLM | <60s | 0.41s |
| 14 | ML Integration | <5s | 1.74s |
| 21 | MCP | <15s | 10.5s |
| 22 | GUI | <2s | 1.19s |
| **Total** | **Full pipeline** | **<3 min** | **~2m53s** |

---

## Memory Tracking

```python
import psutil
from contextlib import contextmanager

@contextmanager
def memory_tracked(operation: str, logger):
    """Track memory usage of an operation."""
    proc = psutil.Process()
    start_mb = proc.memory_info().rss / 1024 / 1024
    try:
        yield
    finally:
        end_mb = proc.memory_info().rss / 1024 / 1024
        delta = end_mb - start_mb
        logger.debug(f"Memory [{operation}]: {start_mb:.1f}→{end_mb:.1f}MB (Δ{delta:+.1f}MB)")
        if delta > 100:
            logger.warning(f"High memory usage in {operation}: +{delta:.1f}MB")

# Usage
with memory_tracked("parse_models", logger):
    models = [parse_gnn_file(f) for f in files]
```

---

## Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=128)
def parse_gnn_cached(content_hash: str, file_path: str) -> Dict[str, Any]:
    """Parse GNN with LRU cache keyed on content hash."""
    return _do_parse(Path(file_path))

def get_cached_parse(path: Path) -> Dict[str, Any]:
    content = path.read_text()
    h = hashlib.md5(content.encode()).hexdigest()
    return parse_gnn_cached(h, str(path))
```

---

## Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def parallel_process(items, processor, max_workers=4, use_threads=True):
    """Process items in parallel."""
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with Executor(max_workers=max_workers) as ex:
        return list(ex.map(processor, items))

# I/O bound (file reading) → threads
results = parallel_process(files, parse_file, max_workers=8, use_threads=True)

# CPU bound (matrix computation) → processes
results = parallel_process(models, compute_matrices, max_workers=4, use_threads=False)
```

---

## Timeout Protection

```python
import signal
from functools import wraps

def timeout(seconds: int):
    """Decorator enforcing execution timeout (Unix only)."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")
            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
        return wrapper
    return decorator

@timeout(60)
def execute_simulation(script_path: Path) -> bool:
    ...
```

---

## Performance Optimization Checklist

- [ ] Measure baseline before optimizing
- [ ] Profile bottlenecks (`python -m cProfile -o profile.prof src/11_render.py`)
- [ ] Cache repeated expensive calls (`@lru_cache`)
- [ ] Stream large files instead of loading fully into memory
- [ ] Use threads for I/O, processes for CPU-bound work
- [ ] Add timeout protection for external calls
- [ ] Clean up large objects explicitly (`del obj; gc.collect()`)
- [ ] Add timing assertions in performance tests

---

## Memory Cleanup

```python
import gc

def cleanup_large_objects(*objects) -> None:
    """Explicit cleanup after memory-intensive operations."""
    for obj in objects:
        del obj
    gc.collect()
    proc = psutil.Process()
    logger.debug(f"Post-cleanup: {proc.memory_info().rss / 1024 / 1024:.1f}MB")
```

---

## Profiling Commands

```bash
# CPU profiling
python -m cProfile -o profile.prof src/11_render.py --target-dir input/gnn_files
python -c "import pstats; p = pstats.Stats('profile.prof'); p.sort_stats('cumulative').print_stats(20)"

# Memory profiling
uv pip install memory_profiler
python -m memory_profiler src/11_render.py --target-dir input/gnn_files

# Snakeviz visualization
uv pip install snakeviz && snakeviz profile.prof
```

---

**Last Updated**: March 2026 | **Status**: Production Standard
