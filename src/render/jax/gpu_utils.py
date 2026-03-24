"""
JAX GPU/TPU Hardware Inspector

Provides utilities for querying and verifying JAX hardware accelerators (GPU/TPU)
and applying performance configurations.
"""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def get_hardware_topology() -> Dict[str, Any]:
    """
    Inspect local JAX devices and return hardware topology.
    """
    try:
        import jax
        devices = jax.local_devices()
        
        # Determine GPU/TPU availability without failing if backends don't exist
        gpu_devices = []
        tpu_devices = []
        
        try:
            gpu_devices = jax.local_devices(backend="gpu")
        except RuntimeError:
            logger.debug("GPU backend not available")

        try:
            tpu_devices = jax.local_devices(backend="tpu")
        except RuntimeError:
            logger.debug("TPU backend not available")
        
        topology = {
            "total_devices": len(devices),
            "device_types": [str(d.device_kind) for d in devices],
            "gpus_available": len(gpu_devices) > 0,
            "tpus_available": len(tpu_devices) > 0,
            "has_accelerator": len(gpu_devices) > 0 or len(tpu_devices) > 0
        }
        
        logger.info(f"Hardware Inspector: Found {len(devices)} devices. GPU: {topology['gpus_available']}, TPU: {topology['tpus_available']}")
        return topology
        
    except ImportError:
        logger.warning("JAX not installed, cannot inspect hardware topology.")
        return {"total_devices": 0, "has_accelerator": False}
    except Exception as e:
        logger.warning(f"Failed to query JAX hardware topology: {e}")
        return {"total_devices": 0, "has_accelerator": False}

def configure_xla_runtime() -> bool:
    """
    Configure XLA runtime for optimal performance based on available hardware.
    
    Returns True if an accelerator is found and configured, False otherwise.
    """
    import os
    topology = get_hardware_topology()
    if topology["has_accelerator"]:
        # Set memory growth to avoid completely locking GPU memory
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        logger.info("Configured XLA runtime for accelerator usage.")
        return True
    return False
