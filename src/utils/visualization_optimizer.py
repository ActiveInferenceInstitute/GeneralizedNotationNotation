#!/usr/bin/env python3
"""
Visualization Performance Optimizer

This module provides performance optimizations for visualization processing,
including caching, parallel processing, and intelligent data sampling.
"""

import os
import sys
import hashlib
import json
import time
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class VisualizationCache:
    """Cache management for visualization artifacts."""
    cache_dir: Path
    max_age_hours: int = 24
    max_cache_size_mb: int = 500
    
    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception:
            pass  # Continue if metadata save fails
    
    def get_cache_key(self, content: str, params: Dict[str, Any]) -> str:
        """Generate cache key from content and parameters."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        params_str = json.dumps(sorted(params.items()), sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        return f"{content_hash}_{params_hash}"
    
    def is_cached(self, cache_key: str) -> bool:
        """Check if visualization is cached and still valid."""
        if cache_key not in self.metadata:
            return False
        
        entry = self.metadata[cache_key]
        cache_time = datetime.fromisoformat(entry['timestamp'])
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600
        
        if age_hours > self.max_age_hours:
            self._remove_cache_entry(cache_key)
            return False
        
        # Check if files still exist
        for file_path in entry.get('files', []):
            if not Path(file_path).exists():
                self._remove_cache_entry(cache_key)
                return False
        
        return True
    
    def get_cached_files(self, cache_key: str) -> List[str]:
        """Get list of cached visualization files."""
        if cache_key in self.metadata:
            return self.metadata[cache_key].get('files', [])
        return []
    
    def cache_visualization(self, cache_key: str, files: List[str]):
        """Cache visualization results."""
        self.metadata[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'files': files,
            'size_mb': sum(Path(f).stat().st_size for f in files if Path(f).exists()) / (1024 * 1024)
        }
        self._save_metadata()
        self._cleanup_old_cache()
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove a cache entry and its files."""
        if cache_key in self.metadata:
            entry = self.metadata[cache_key]
            for file_path in entry.get('files', []):
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception:
                    pass
            del self.metadata[cache_key]
            self._save_metadata()
    
    def _cleanup_old_cache(self):
        """Clean up old cache entries to stay within size limits."""
        # Calculate total cache size
        total_size_mb = sum(entry.get('size_mb', 0) for entry in self.metadata.values())
        
        if total_size_mb <= self.max_cache_size_mb:
            return
        
        # Sort entries by timestamp (oldest first)
        entries = [(key, datetime.fromisoformat(entry['timestamp'])) 
                  for key, entry in self.metadata.items()]
        entries.sort(key=lambda x: x[1])
        
        # Remove oldest entries until we're under the size limit
        for cache_key, _ in entries:
            if total_size_mb <= self.max_cache_size_mb * 0.8:  # Leave some margin
                break
            
            entry = self.metadata[cache_key]
            total_size_mb -= entry.get('size_mb', 0)
            self._remove_cache_entry(cache_key)

@dataclass
class DataSampler:
    """Smart data sampling for large datasets."""
    max_nodes: int = 1000
    max_edges: int = 5000
    max_matrix_size: int = 500
    sampling_strategy: str = "random"  # "random", "importance", "cluster"
    
    def should_sample(self, data: Dict[str, Any]) -> bool:
        """Determine if data should be sampled."""
        node_count = len(data.get('nodes', []))
        edge_count = len(data.get('edges', []))
        matrix_size = max(data.get('matrix_dimensions', [0, 0]))
        
        return (node_count > self.max_nodes or 
                edge_count > self.max_edges or 
                matrix_size > self.max_matrix_size)
    
    def sample_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sample large datasets for visualization."""
        if not self.should_sample(data):
            return data
        
        sampled_data = data.copy()
        
        # Sample nodes
        nodes = data.get('nodes', [])
        if len(nodes) > self.max_nodes:
            if self.sampling_strategy == "random":
                import random
                sampled_nodes = random.sample(nodes, self.max_nodes)
            else:
                # Keep first N nodes (may be more important)
                sampled_nodes = nodes[:self.max_nodes]
            sampled_data['nodes'] = sampled_nodes
            sampled_data['_sampling_applied'] = True
            sampled_data['_original_node_count'] = len(nodes)
        
        # Sample edges (only those connecting sampled nodes)
        if 'edges' in data and '_sampling_applied' in sampled_data:
            sampled_node_ids = set(node.get('id') for node in sampled_data['nodes'])
            sampled_edges = [edge for edge in data['edges'] 
                           if edge.get('source') in sampled_node_ids and 
                              edge.get('target') in sampled_node_ids]
            sampled_data['edges'] = sampled_edges[:self.max_edges]
            sampled_data['_original_edge_count'] = len(data.get('edges', []))
        
        # Sample matrices
        if 'matrices' in data:
            sampled_matrices = []
            for matrix in data['matrices']:
                if isinstance(matrix, list) and len(matrix) > self.max_matrix_size:
                    # Sample rows and columns
                    step = len(matrix) // self.max_matrix_size
                    sampled_matrix = [row[::step] for row in matrix[::step]]
                    sampled_matrices.append(sampled_matrix)
                else:
                    sampled_matrices.append(matrix)
            sampled_data['matrices'] = sampled_matrices
        
        return sampled_data

class ParallelVisualizationProcessor:
    """Parallel processing for visualization tasks."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
        self.logger = logging.getLogger(__name__)
    
    def process_files_parallel(self, 
                             files: List[Path], 
                             processing_func: Callable,
                             output_dir: Path,
                             **kwargs) -> Dict[str, Any]:
        """Process multiple files in parallel."""
        results = {
            "processed_files": [],
            "failed_files": [],
            "total_time": 0,
            "parallel_efficiency": 0
        }
        
        start_time = time.time()
        
        # Prepare tasks
        tasks = []
        for file_path in files:
            task = {
                'file_path': file_path,
                'output_dir': output_dir,
                'model_name': file_path.stem,
                **kwargs
            }
            tasks.append(task)
        
        # Execute in parallel
        if len(tasks) == 1:
            # Single file - no need for parallelization overhead
            try:
                result = processing_func(**tasks[0])
                results["processed_files"].append(result)
            except Exception as e:
                results["failed_files"].append({"file": str(tasks[0]['file_path']), "error": str(e)})
        else:
            # Multiple files - use parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {executor.submit(self._safe_process_task, processing_func, task): task 
                                 for task in tasks}
                
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        if result.get('success'):
                            results["processed_files"].append(result)
                        else:
                            results["failed_files"].append({
                                "file": str(task['file_path']), 
                                "error": result.get('error', 'Unknown error')
                            })
                    except Exception as e:
                        results["failed_files"].append({"file": str(task['file_path']), "error": str(e)})
        
        total_time = time.time() - start_time
        results["total_time"] = total_time
        
        # Calculate parallel efficiency (rough estimate)
        if len(files) > 1:
            sequential_estimate = total_time * len(files)
            results["parallel_efficiency"] = min(1.0, sequential_estimate / (total_time * self.max_workers))
        else:
            results["parallel_efficiency"] = 1.0
        
        return results
    
    def _safe_process_task(self, processing_func: Callable, task: Dict[str, Any]) -> Dict[str, Any]:
        """Safely execute a processing task."""
        try:
            result = processing_func(**task)
            return {"success": True, "result": result, "task": task}
        except Exception as e:
            return {"success": False, "error": str(e), "task": task}

class VisualizationOptimizer:
    """Main visualization optimization coordinator."""
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 max_workers: Optional[int] = None,
                 enable_sampling: bool = True,
                 enable_caching: bool = True):
        
        self.enable_caching = enable_caching
        self.enable_sampling = enable_sampling
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache
        if self.enable_caching:
            cache_dir = cache_dir or Path.cwd() / ".visualization_cache"
            self.cache = VisualizationCache(cache_dir)
        
        # Initialize sampler
        if self.enable_sampling:
            self.sampler = DataSampler()
        
        # Initialize parallel processor
        self.parallel_processor = ParallelVisualizationProcessor(max_workers)
    
    def optimize_single_file_processing(self, 
                                      content: str,
                                      model_name: str,
                                      output_dir: Path,
                                      processing_func: Callable,
                                      processing_params: Dict[str, Any]) -> Tuple[List[str], bool]:
        """Optimize processing of a single file with caching and sampling."""
        
        # Check cache first
        if self.enable_caching:
            cache_key = self.cache.get_cache_key(content, processing_params)
            if self.cache.is_cached(cache_key):
                self.logger.debug(f"Using cached visualization for {model_name}")
                cached_files = self.cache.get_cached_files(cache_key)
                return cached_files, True  # True indicates cache hit
        
        # Parse and optimize data
        optimized_content = content
        if self.enable_sampling:
            try:
                # This would need to be adapted based on the actual data format
                # For now, we'll pass the content through
                pass
            except Exception as e:
                self.logger.warning(f"Data sampling failed for {model_name}: {e}")
        
        # Process visualization
        try:
            start_time = time.time()
            result_files = processing_func(
                content=optimized_content,
                model_name=model_name, 
                output_dir=output_dir,
                **processing_params
            )
            processing_time = time.time() - start_time
            
            # Cache results
            if self.enable_caching and result_files:
                self.cache.cache_visualization(cache_key, result_files)
                
            self.logger.debug(f"Generated {len(result_files)} visualizations for {model_name} in {processing_time:.2f}s")
            return result_files, False  # False indicates no cache hit
            
        except Exception as e:
            self.logger.error(f"Visualization processing failed for {model_name}: {e}")
            return [], False
    
    def optimize_batch_processing(self,
                                files: List[Path],
                                output_dir: Path,
                                processing_func: Callable,
                                **kwargs) -> Dict[str, Any]:
        """Optimize batch processing with parallel execution."""
        
        self.logger.info(f"Processing {len(files)} files with optimization (parallel={self.parallel_processor.max_workers} workers)")
        
        # Use parallel processing for batch operations
        results = self.parallel_processor.process_files_parallel(
            files, processing_func, output_dir, **kwargs
        )
        
        # Add optimization statistics
        results["optimization_stats"] = {
            "caching_enabled": self.enable_caching,
            "sampling_enabled": self.enable_sampling,
            "parallel_workers": self.parallel_processor.max_workers,
            "cache_hits": 0,  # This would need to be tracked during processing
            "sampling_applied": 0  # This would need to be tracked during processing
        }
        
        return results
    
    def clear_cache(self):
        """Clear the visualization cache."""
        if self.enable_caching:
            for cache_key in list(self.cache.metadata.keys()):
                self.cache._remove_cache_entry(cache_key)
            self.logger.info("Visualization cache cleared")

# Global optimizer instance
_global_optimizer = None

def get_visualization_optimizer(**kwargs) -> VisualizationOptimizer:
    """Get the global visualization optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = VisualizationOptimizer(**kwargs)
    return _global_optimizer

def optimize_visualization_processing(files: List[Path], 
                                    output_dir: Path,
                                    processing_func: Callable,
                                    **kwargs) -> Dict[str, Any]:
    """Convenience function for optimized visualization processing."""
    optimizer = get_visualization_optimizer()
    return optimizer.optimize_batch_processing(files, output_dir, processing_func, **kwargs)

# Performance monitoring utilities
def monitor_visualization_performance(func: Callable) -> Callable:
    """Decorator to monitor visualization performance."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = get_memory_usage()
            
            # Log performance metrics
            logger = logging.getLogger(func.__module__)
            logger.debug(f"Performance: {func.__name__} - "
                        f"Time: {end_time - start_time:.2f}s, "
                        f"Memory: {end_memory - start_memory:.1f}MB, "
                        f"Success: {success}")
        
        return result
    return wrapper

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

if __name__ == "__main__":
    # Test the optimization system
    optimizer = VisualizationOptimizer()
    print(f"Visualization optimizer initialized with {optimizer.parallel_processor.max_workers} workers")
    
    if optimizer.enable_caching:
        print(f"Caching enabled at {optimizer.cache.cache_dir}")
    
    if optimizer.enable_sampling:
        print(f"Sampling enabled (max nodes: {optimizer.sampler.max_nodes})")
