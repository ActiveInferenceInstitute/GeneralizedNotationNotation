import os
import sys
from pathlib import Path
import logging
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from typing import Dict, Any

logger = logging.getLogger(__name__)

def get_system_info() -> Dict[str, Any]:
    """Gather comprehensive system information for pipeline tracking."""
    try:
        base_info = {
            "python_version": sys.version,
            "platform": os.name,
            "cpu_count": os.cpu_count(),
            "working_directory": str(Path.cwd()),
            "user": os.getenv('USER', 'unknown')
        }
        
        # Add psutil-dependent info if available
        if PSUTIL_AVAILABLE:
            base_info.update({
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "disk_free_gb": round(psutil.disk_usage('.').free / (1024**3), 2)
            })
        else:
            base_info.update({
                "memory_total_gb": "unavailable (psutil not installed)",
                "disk_free_gb": "unavailable (psutil not installed)"
            })
        
        return base_info
    except Exception as e:
        logger.warning(f"Failed to gather complete system info: {e}")
        return {"error": str(e)} 