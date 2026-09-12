"""Config/IO concern package (S2-33 Step 7, family 2/3): canonical home of
file/config I/O — YAML pipeline configuration loading and validation, batch
file read/write helpers with the shared writable-directory probe, generated-file
code metrics, and relative-path formatting — code that used to live in the
``gnn/utils`` top-level grab-bag.

Eager re-export note (§4.3.1): this family's ``__init__`` re-exports every
public name as a real object. Eager is safe here because all four leaves are
stdlib-only at module scope — the one third-party import (PyYAML in
``config_loader``) is guarded by ``try/except ImportError`` and degrades to a
``YAML_AVAILABLE = False`` sentinel, so importing this package never pulls a
heavy dependency (contrast ``runtime_safety``, whose leaves carry psutil and
therefore resolve lazily).

Cross-family imports go through leaf modules, never through any facade (I5).
The old top-level paths (``gnn/utils/config_loader.py`` etc.) are deprecation
facades over this package.
"""

from gnn.utils.config_io.code_metrics import count_code_metrics
from gnn.utils.config_io.config_loader import (
    YAML_AVAILABLE,
    GNNPipelineConfig,
    LLMConfig,
    ModelConfig,
    OntologyConfig,
    PipelineConfig,
    SAPFConfig,
    SetupConfig,
    TypeCheckerConfig,
    WebsiteConfig,
    get_config_value,
    load_config,
    save_config,
    set_config_value,
    validate_config,
)
from gnn.utils.config_io.io_utils import (
    batch_read_files,
    batch_write_files,
    cleanup_temp_files,
    create_temp_file_with_content,
    get_file_performance_metrics,
    verify_directory_writable,
)
from gnn.utils.config_io.path_utils import get_relative_path_if_possible

__all__ = [
    "GNNPipelineConfig",
    "LLMConfig",
    "ModelConfig",
    "OntologyConfig",
    "PipelineConfig",
    "SAPFConfig",
    "SetupConfig",
    "TypeCheckerConfig",
    "WebsiteConfig",
    "YAML_AVAILABLE",
    "batch_read_files",
    "batch_write_files",
    "cleanup_temp_files",
    "count_code_metrics",
    "create_temp_file_with_content",
    "get_config_value",
    "get_file_performance_metrics",
    "get_relative_path_if_possible",
    "load_config",
    "save_config",
    "set_config_value",
    "validate_config",
    "verify_directory_writable",
]
