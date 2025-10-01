# ML Integration module

import logging
from pathlib import Path
from typing import Optional

# Import processor functions
from .processor import (
    process_ml_integration
)


def process_ml_integration(target_dir, output_dir, verbose=False, logger=None, **kwargs):
    """
    Main processing function for ml_integration.
    
    Args:
        target_dir: Directory containing files to process
        output_dir: Output directory for results
        verbose: Whether to enable verbose logging
        logger: Logger instance
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    import logging
    import json
    from pathlib import Path
    from datetime import datetime
    
    if logger is None:
        logger = logging.getLogger(__name__)
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    try:
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing ml_integration for files in {target_dir}")
        
        # Gather ML framework information
        ml_frameworks = check_ml_frameworks()
        
        # Create processing summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "target_dir": str(target_dir),
            "output_dir": str(output_dir),
            "ml_frameworks_available": ml_frameworks,
            "processing_status": "completed",
            "frameworks_detected": [fw for fw, available in ml_frameworks.items() if available],
            "message": "ML integration module ready for model training and evaluation"
        }
        
        # Save summary
        summary_file = output_dir / "ml_integration_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"📊 ML integration summary saved to: {summary_file}")
        
        # Save framework details
        frameworks_file = output_dir / "ml_frameworks_status.json"
        with open(frameworks_file, 'w') as f:
            json.dump(ml_frameworks, f, indent=2)
        logger.info(f"🔧 ML frameworks status saved to: {frameworks_file}")
        
        logger.info(f"✅ ML integration processing completed")
        return True
    except Exception as e:
        logger.error(f"❌ ML integration processing failed: {e}")
        return False

def check_ml_frameworks():
    """Check availability of ML frameworks."""
    frameworks = {}
    
    # Check PyTorch
    try:
        import torch
        frameworks['pytorch'] = {
            'available': True,
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available() if hasattr(torch, 'cuda') else False
        }
    except ImportError:
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
    'process_ml_integration'
]
