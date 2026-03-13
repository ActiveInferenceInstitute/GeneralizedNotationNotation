"""
Distributed Execution Module for GNN

Provides Ray-based parallel dispatching for script execution and parameter sweeps.
"""
import logging
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class RayDispatcher:
    """
    Dispatcher for Ray-based distributed parameter sweeps and script execution.
    """
    def __init__(self, address: Optional[str] = None, num_cpus: Optional[int] = None):
        """Initialize connection to Ray cluster."""
        self.address = address
        self.num_cpus = num_cpus
        self._initialized = False
        
    def initialize(self) -> bool:
        """Connect to Ray cluster."""
        try:
            import ray
            if not ray.is_initialized():
                ray.init(address=self.address, num_cpus=self.num_cpus, ignore_reinit_error=True)
            self._initialized = True
            logger.info(f"Successfully connected to Ray cluster (Active Nodes: {len(ray.nodes())})")
            return True
        except ImportError:
            logger.warning("Ray is not installed. Run: uv sync --extra scaling")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            return False
            
    def shutdown(self):
        """Shutdown Ray connection."""
        if self._initialized:
            try:
                import ray
                ray.shutdown()
                self._initialized = False
            except ImportError:
                pass
                
    def run_scripts_parallel(self, script_infos: List[Dict[str, Any]], execute_fn: Callable, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute multiple scripts in parallel across Ray workers.
        """
        if not self._initialized and not self.initialize():
            logger.warning("Falling back to sequential execution due to Ray initialization failure.")
            return [execute_fn(info, **kwargs) for info in script_infos]
            
        import ray
        
        # Ray Remote wrapper for the execution function
        @ray.remote
        def _remote_execute(script_info, kwargs_dict):
            # Pass dictionary of primitive types to remote functions
            return execute_fn(script_info, **kwargs_dict)
            
        logger.info(f"Dispatching {len(script_infos)} scripts to Ray cluster...")
        
        # Dispatch all tasks
        futures = [_remote_execute.remote(info, kwargs) for info in script_infos]
        
        # Wait for all to complete and collect results
        results = ray.get(futures)
        return results
        
    def parameter_sweep(self, model_fn: Callable, param_grid: List[Dict[str, Any]]) -> List[Any]:
        """
        Execute a parameter sweep by running model_fn across all params in param_grid.
        """
        if not self._initialized and not self.initialize():
            logger.warning("Falling back to sequential parameter sweep.")
            return [model_fn(**params) for params in param_grid]
            
        import ray
        
        @ray.remote
        def _remote_eval(params):
            return model_fn(**params)
            
        logger.info(f"Dispatching {len(param_grid)} parameter combinations for sweep...")
        futures = [_remote_eval.remote(p) for p in param_grid]
        return ray.get(futures)
