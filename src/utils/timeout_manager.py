#!/usr/bin/env python3
"""
Timeout Manager for Long-Running Operations

This module provides comprehensive timeout handling for various operations
including LLM calls, external processes, and network requests with intelligent
retry strategies and graceful degradation.
"""

import asyncio
import time
import signal
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union, Coroutine
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
from enum import Enum
import functools
import threading
import subprocess

class TimeoutStrategy(Enum):
    """Timeout handling strategies."""
    FAIL_FAST = "fail_fast"           # Fail immediately on timeout
    RETRY_EXPONENTIAL = "retry_exponential"  # Exponential backoff retries
    RETRY_LINEAR = "retry_linear"     # Linear backoff retries
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Return partial results

@dataclass
class TimeoutConfig:
    """Configuration for timeout handling."""
    base_timeout: float = 30.0        # Base timeout in seconds
    max_timeout: float = 300.0        # Maximum timeout in seconds
    max_retries: int = 3              # Maximum number of retries
    strategy: TimeoutStrategy = TimeoutStrategy.RETRY_EXPONENTIAL
    retry_delay: float = 1.0          # Base retry delay in seconds
    exponential_base: float = 2.0     # Exponential backoff base
    graceful_fallback: Optional[Callable] = None  # Fallback function for graceful degradation
    log_retries: bool = True          # Whether to log retry attempts

@dataclass
class TimeoutResult:
    """Result of a timeout-managed operation."""
    success: bool = False
    result: Any = None
    error: Optional[str] = None
    attempts: int = 0
    total_time: float = 0.0
    timed_out: bool = False
    used_fallback: bool = False

class TimeoutManager:
    """Comprehensive timeout management for various operation types."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._active_operations: Dict[str, float] = {}  # Track active operations
    
    @contextmanager
    def sync_timeout(self, 
                    operation_name: str,
                    config: TimeoutConfig,
                    operation: Callable[..., Any],
                    *args, **kwargs):
        """Synchronous timeout context manager."""
        start_time = time.time()
        operation_id = f"{operation_name}_{id(threading.current_thread())}"
        self._active_operations[operation_id] = start_time
        
        try:
            result = self._execute_with_timeout_sync(
                operation_name, config, operation, *args, **kwargs
            )
            yield result
        finally:
            self._active_operations.pop(operation_id, None)
    
    @asynccontextmanager
    async def async_timeout(self,
                           operation_name: str,
                           config: TimeoutConfig,
                           operation: Callable[..., Coroutine],
                           *args, **kwargs):
        """Asynchronous timeout context manager."""
        start_time = time.time()
        operation_id = f"{operation_name}_{id(asyncio.current_task())}"
        self._active_operations[operation_id] = start_time
        
        try:
            result = await self._execute_with_timeout_async(
                operation_name, config, operation, *args, **kwargs
            )
            yield result
        finally:
            self._active_operations.pop(operation_id, None)
    
    def _execute_with_timeout_sync(self,
                                  operation_name: str,
                                  config: TimeoutConfig,
                                  operation: Callable,
                                  *args, **kwargs) -> TimeoutResult:
        """Execute synchronous operation with timeout and retry logic."""
        result = TimeoutResult()
        start_time = time.time()
        
        for attempt in range(config.max_retries + 1):
            result.attempts = attempt + 1
            attempt_start = time.time()
            
            # Calculate timeout for this attempt
            if config.strategy == TimeoutStrategy.RETRY_EXPONENTIAL:
                current_timeout = min(
                    config.base_timeout * (config.exponential_base ** attempt),
                    config.max_timeout
                )
            elif config.strategy == TimeoutStrategy.RETRY_LINEAR:
                current_timeout = min(
                    config.base_timeout + (config.retry_delay * attempt),
                    config.max_timeout
                )
            else:
                current_timeout = config.base_timeout
            
            try:
                if config.log_retries and attempt > 0:
                    self.logger.info(f"Retry {attempt} for {operation_name} with timeout {current_timeout:.1f}s")
                
                # Execute with timeout
                if sys.platform != "win32":
                    # Unix-like systems - use signal for timeout
                    result_value = self._execute_with_signal_timeout(
                        operation, current_timeout, *args, **kwargs
                    )
                else:
                    # Windows - use threading timeout
                    result_value = self._execute_with_thread_timeout(
                        operation, current_timeout, *args, **kwargs
                    )
                
                result.success = True
                result.result = result_value
                result.total_time = time.time() - start_time
                return result
                
            except TimeoutError as e:
                result.timed_out = True
                result.error = f"Operation timed out after {current_timeout:.1f}s"
                
                if config.log_retries:
                    self.logger.warning(f"{operation_name} timed out on attempt {attempt + 1}")
                
                if attempt < config.max_retries:
                    # Wait before retry
                    retry_delay = config.retry_delay * (config.exponential_base ** attempt) \
                                 if config.strategy == TimeoutStrategy.RETRY_EXPONENTIAL \
                                 else config.retry_delay
                    time.sleep(retry_delay)
                    continue
                else:
                    break
                    
            except Exception as e:
                result.error = str(e)
                if config.log_retries:
                    self.logger.error(f"{operation_name} failed on attempt {attempt + 1}: {e}")
                
                if attempt < config.max_retries and not isinstance(e, (KeyboardInterrupt, SystemExit)):
                    retry_delay = config.retry_delay * (config.exponential_base ** attempt) \
                                 if config.strategy == TimeoutStrategy.RETRY_EXPONENTIAL \
                                 else config.retry_delay
                    time.sleep(retry_delay)
                    continue
                else:
                    break
        
        # All attempts failed - check for graceful degradation
        if config.strategy == TimeoutStrategy.GRACEFUL_DEGRADATION and config.graceful_fallback:
            try:
                self.logger.info(f"Using graceful fallback for {operation_name}")
                fallback_result = config.graceful_fallback(*args, **kwargs)
                result.success = True
                result.result = fallback_result
                result.used_fallback = True
                result.error = None
            except Exception as e:
                result.error = f"Fallback also failed: {e}"
        
        result.total_time = time.time() - start_time
        return result
    
    async def _execute_with_timeout_async(self,
                                         operation_name: str,
                                         config: TimeoutConfig,
                                         operation: Callable,
                                         *args, **kwargs) -> TimeoutResult:
        """Execute asynchronous operation with timeout and retry logic."""
        result = TimeoutResult()
        start_time = time.time()
        
        for attempt in range(config.max_retries + 1):
            result.attempts = attempt + 1
            
            # Calculate timeout for this attempt
            if config.strategy == TimeoutStrategy.RETRY_EXPONENTIAL:
                current_timeout = min(
                    config.base_timeout * (config.exponential_base ** attempt),
                    config.max_timeout
                )
            elif config.strategy == TimeoutStrategy.RETRY_LINEAR:
                current_timeout = min(
                    config.base_timeout + (config.retry_delay * attempt),
                    config.max_timeout
                )
            else:
                current_timeout = config.base_timeout
            
            try:
                if config.log_retries and attempt > 0:
                    self.logger.info(f"Retry {attempt} for {operation_name} with timeout {current_timeout:.1f}s")
                
                # Execute with asyncio timeout
                result_value = await asyncio.wait_for(
                    operation(*args, **kwargs), timeout=current_timeout
                )
                
                result.success = True
                result.result = result_value
                result.total_time = time.time() - start_time
                return result
                
            except asyncio.TimeoutError:
                result.timed_out = True
                result.error = f"Operation timed out after {current_timeout:.1f}s"
                
                if config.log_retries:
                    self.logger.warning(f"{operation_name} timed out on attempt {attempt + 1}")
                
                if attempt < config.max_retries:
                    # Wait before retry
                    retry_delay = config.retry_delay * (config.exponential_base ** attempt) \
                                 if config.strategy == TimeoutStrategy.RETRY_EXPONENTIAL \
                                 else config.retry_delay
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    break
                    
            except Exception as e:
                result.error = str(e)
                if config.log_retries:
                    self.logger.error(f"{operation_name} failed on attempt {attempt + 1}: {e}")
                
                if attempt < config.max_retries and not isinstance(e, (KeyboardInterrupt, SystemExit)):
                    retry_delay = config.retry_delay * (config.exponential_base ** attempt) \
                                 if config.strategy == TimeoutStrategy.RETRY_EXPONENTIAL \
                                 else config.retry_delay
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    break
        
        # All attempts failed - check for graceful degradation
        if config.strategy == TimeoutStrategy.GRACEFUL_DEGRADATION and config.graceful_fallback:
            try:
                self.logger.info(f"Using graceful fallback for {operation_name}")
                if asyncio.iscoroutinefunction(config.graceful_fallback):
                    fallback_result = await config.graceful_fallback(*args, **kwargs)
                else:
                    fallback_result = config.graceful_fallback(*args, **kwargs)
                result.success = True
                result.result = fallback_result
                result.used_fallback = True
                result.error = None
            except Exception as e:
                result.error = f"Fallback also failed: {e}"
        
        result.total_time = time.time() - start_time
        return result
    
    def _execute_with_signal_timeout(self, operation: Callable, timeout: float, *args, **kwargs):
        """Execute operation with signal-based timeout (Unix only)."""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
        
        # Set up signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = operation(*args, **kwargs)
            signal.alarm(0)  # Cancel the alarm
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
    
    def _execute_with_thread_timeout(self, operation: Callable, timeout: float, *args, **kwargs):
        """Execute operation with thread-based timeout (cross-platform)."""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(operation, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Operation timed out after {timeout} seconds")
    
    def get_active_operations(self) -> Dict[str, float]:
        """Get currently active operations and their start times."""
        current_time = time.time()
        return {op_id: current_time - start_time 
                for op_id, start_time in self._active_operations.items()}

# Specialized timeout managers for different operation types

class LLMTimeoutManager(TimeoutManager):
    """Specialized timeout manager for LLM operations."""
    
    def __init__(self):
        super().__init__()
        self.default_config = TimeoutConfig(
            base_timeout=60.0,
            max_timeout=300.0,
            max_retries=2,
            strategy=TimeoutStrategy.RETRY_EXPONENTIAL,
            retry_delay=2.0
        )
    
    async def llm_call_with_timeout(self,
                                   llm_function: Callable,
                                   prompt: str,
                                   model: str = "default",
                                   config: Optional[TimeoutConfig] = None,
                                   **kwargs) -> TimeoutResult:
        """Execute LLM call with specialized timeout handling."""
        config = config or self.default_config
        
        # Add graceful fallback for LLM calls
        def llm_fallback(*args, **kwargs):
            return f"LLM request timed out. Model: {model}, Prompt length: {len(prompt)} chars"
        
        config.graceful_fallback = llm_fallback
        config.strategy = TimeoutStrategy.GRACEFUL_DEGRADATION
        
        async with self.async_timeout("llm_call", config, llm_function, prompt, model, **kwargs) as result:
            return result

class ProcessTimeoutManager(TimeoutManager):
    """Specialized timeout manager for subprocess operations."""
    
    def __init__(self):
        super().__init__()
        self.default_config = TimeoutConfig(
            base_timeout=120.0,
            max_timeout=600.0,
            max_retries=1,
            strategy=TimeoutStrategy.FAIL_FAST,
            retry_delay=5.0
        )
    
    def run_with_timeout(self,
                        command: List[str],
                        config: Optional[TimeoutConfig] = None,
                        **subprocess_kwargs) -> TimeoutResult:
        """Run subprocess with timeout handling."""
        config = config or self.default_config
        
        def subprocess_operation(*args, **kwargs):
            return subprocess.run(
                command,
                timeout=config.base_timeout,
                check=False,
                **subprocess_kwargs
            )
        
        with self.sync_timeout("subprocess", config, subprocess_operation) as result:
            return result

# Global timeout manager instances
_timeout_manager = TimeoutManager()
_llm_timeout_manager = LLMTimeoutManager()
_process_timeout_manager = ProcessTimeoutManager()

def get_timeout_manager() -> TimeoutManager:
    """Get the global timeout manager instance."""
    return _timeout_manager

def get_llm_timeout_manager() -> LLMTimeoutManager:
    """Get the specialized LLM timeout manager."""
    return _llm_timeout_manager

def get_process_timeout_manager() -> ProcessTimeoutManager:
    """Get the specialized process timeout manager."""
    return _process_timeout_manager

# Convenience decorators
def with_timeout(config: TimeoutConfig):
    """Decorator for adding timeout handling to functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_timeout_manager()
            with manager.sync_timeout(func.__name__, config, func, *args, **kwargs) as result:
                if result.success:
                    return result.result
                else:
                    raise RuntimeError(result.error or "Operation failed")
        return wrapper
    return decorator

def with_async_timeout(config: TimeoutConfig):
    """Decorator for adding timeout handling to async functions."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_timeout_manager()
            async with manager.async_timeout(func.__name__, config, func, *args, **kwargs) as result:
                if result.success:
                    return result.result
                else:
                    raise RuntimeError(result.error or "Operation failed")
        return wrapper
    return decorator

if __name__ == "__main__":
    # Test the timeout manager
    import asyncio
    
    async def test_async_timeout():
        """Test async timeout functionality."""
        manager = get_llm_timeout_manager()
        
        async def slow_operation(delay: float):
            await asyncio.sleep(delay)
            return f"Completed after {delay}s"
        
        config = TimeoutConfig(base_timeout=2.0, max_retries=1)
        result = await manager._execute_with_timeout_async("test", config, slow_operation, 1.0)
        print(f"Fast operation result: {result.result if result.success else result.error}")
        
        result = await manager._execute_with_timeout_async("test", config, slow_operation, 5.0)
        print(f"Slow operation result: {result.result if result.success else result.error}")
    
    asyncio.run(test_async_timeout())
