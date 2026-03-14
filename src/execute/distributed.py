"""
Distributed Execution Module for GNN

Provides Ray and Dask-based parallel dispatching for script execution and parameter sweeps.
Includes robust retry semantics for node failure in external cloud instances.
"""
import logging
from typing import List, Dict, Any, Callable, Optional, Literal

logger = logging.getLogger(__name__)

class Dispatcher:
    """
    Dispatcher for distributed parameter sweeps and script execution.
    Supports both Ray and Dask backends.
    """
    def __init__(self, backend: Literal["ray", "dask"] = "ray", address: Optional[str] = None, num_cpus: Optional[int] = None, max_retries: int = 3):
        """Initialize connection to distributed cluster."""
        self.backend = backend
        self.address = address
        self.num_cpus = num_cpus
        self.max_retries = max_retries
        self._initialized = False
        self.client = None
        
    def initialize(self) -> bool:
        """Connect to distributed cluster."""
        if self.backend == "ray":
            try:
                import ray
                if not ray.is_initialized():
                    ray.init(address=self.address, num_cpus=self.num_cpus, ignore_reinit_error=True)
                self._initialized = True
                logger.info(f"Successfully connected to Ray cluster (Active Nodes: {len(ray.nodes())})")
                return True
            except ImportError:
                logger.warning("Ray is not installed. Run: pip install ray")
                return False
            except Exception as e:
                logger.error(f"Failed to initialize Ray: {e}")
                return False
        elif self.backend == "dask":
            try:
                from dask.distributed import Client, LocalCluster
                if self.address:
                    self.client = Client(self.address)
                else:
                    cluster = LocalCluster(n_workers=self.num_cpus if self.num_cpus else 4)
                    self.client = Client(cluster)
                self._initialized = True
                logger.info(f"Successfully connected to Dask cluster: {self.client}")
                return True
            except ImportError:
                logger.warning("Dask is not installed. Run: pip install dask distributed")
                return False
            except Exception as e:
                logger.error(f"Failed to initialize Dask: {e}")
                return False
        return False
            
    def shutdown(self):
        """Shutdown connection."""
        if self._initialized:
            try:
                if self.backend == "ray":
                    import ray
                    ray.shutdown()
                elif self.backend == "dask" and self.client:
                    self.client.close()
                self._initialized = False
            except ImportError:
                pass
                
    def run_scripts_parallel(self, script_infos: List[Dict[str, Any]], execute_fn: Callable, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute multiple scripts in parallel across workers with robust retries.
        """
        if not self._initialized and not self.initialize():
            logger.warning("Falling back to sequential execution due to initialization failure.")
            return [execute_fn(info, **kwargs) for info in script_infos]
            
        logger.info(f"Dispatching {len(script_infos)} scripts to {self.backend.capitalize()} cluster...")
        
        if self.backend == "ray":
            import ray
            
            # Context switch to a remote function with robust retries
            @ray.remote(max_retries=self.max_retries, retry_exceptions=True)
            def _remote_execute(script_info, kwargs_dict):
                return execute_fn(script_info, **kwargs_dict)
                
            futures = [_remote_execute.remote(info, kwargs) for info in script_infos]
            return ray.get(futures)
            
        elif self.backend == "dask":
            # Use retries parameter if manually providing the tuple logic
            futures = [self.client.submit(execute_fn, info, **kwargs) for info in script_infos]
            return self.client.gather(futures)
            
        return []
        
    def parameter_sweep(self, model_fn: Callable, param_grid: List[Dict[str, Any]]) -> List[Any]:
        """
        Execute a parameter sweep with built-in retry semantics.
        """
        if not self._initialized and not self.initialize():
            logger.warning("Falling back to sequential parameter sweep.")
            return [model_fn(**params) for params in param_grid]
            
        logger.info(f"Dispatching {len(param_grid)} parameter combinations for sweep using {self.backend.capitalize()}...")
        
        if self.backend == "ray":
            import ray
            
            @ray.remote(max_retries=self.max_retries, retry_exceptions=True)
            def _remote_eval(params):
                return model_fn(**params)
                
            futures = [_remote_eval.remote(p) for p in param_grid]
            return ray.get(futures)
            
        elif self.backend == "dask":
            futures = [self.client.submit(model_fn, **p) for p in param_grid]
            return self.client.gather(futures)

# Backward compatibility alias
RayDispatcher = Dispatcher
