"""Earlier name; implementation moved to
``gnn/utils/config_io/io_utils.py`` (S2-33 Step 7, family 2/3)."""

import warnings

warnings.warn(
    "gnn.utils.io_utils is the earlier name; import gnn.utils.config_io.io_utils instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.config_io.io_utils import (  # noqa: E402,F401
    batch_read_files,
    batch_write_files,
    cleanup_temp_files,
    create_temp_file_with_content,
    get_file_performance_metrics,
    verify_directory_writable,
)

__all__ = [
    "batch_read_files",
    "batch_write_files",
    "cleanup_temp_files",
    "create_temp_file_with_content",
    "get_file_performance_metrics",
    "verify_directory_writable",
]
