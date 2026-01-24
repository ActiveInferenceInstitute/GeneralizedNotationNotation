"""
ML Integration module for GNN Processing Pipeline.

This module provides machine learning model integration capabilities.
"""

__version__ = "1.1.3"
FEATURES = {
    "model_training": True,
    "model_inference": True,
    "pipeline_integration": True,
    "mcp_integration": True
}

# Import processor functions - single source of truth
from .processor import process_ml_integration


def check_ml_frameworks():
    """Check availability of ML frameworks."""
    import logging
    frameworks = {}

    # Check PyTorch
    try:
        import torch
        if not hasattr(torch, '__version__'):
            logging.getLogger(__name__).warning(
                f"Imported 'torch' module has no '__version__'. Path: {getattr(torch, '__file__', 'unknown')}"
            )
            frameworks['pytorch'] = {'available': False, 'version': None}
        else:
            frameworks['pytorch'] = {
                'available': True,
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available() if hasattr(torch, 'cuda') else False
            }
    except ImportError:
        frameworks['pytorch'] = {'available': False, 'version': None}
    except Exception as e:
        logging.getLogger(__name__).warning(f"Error checking PyTorch: {e}")
        frameworks['pytorch'] = {'available': False, 'version': None}

    # Check TensorFlow
    try:
        import tensorflow as tf
        frameworks['tensorflow'] = {
            'available': True,
            'version': tf.__version__
        }
    except ImportError:
        frameworks['tensorflow'] = {'available': False, 'version': None}

    # Check JAX
    try:
        import jax
        frameworks['jax'] = {
            'available': True,
            'version': jax.__version__
        }
    except ImportError:
        frameworks['jax'] = {'available': False, 'version': None}

    # Check scikit-learn
    try:
        import sklearn
        frameworks['sklearn'] = {
            'available': True,
            'version': sklearn.__version__
        }
    except ImportError:
        frameworks['sklearn'] = {'available': False, 'version': None}

    return frameworks


__all__ = [
    'process_ml_integration',
    'check_ml_frameworks',
    'FEATURES',
    '__version__'
]
