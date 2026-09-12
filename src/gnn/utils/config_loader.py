"""Earlier name; implementation moved to
``gnn/utils/config_io/config_loader.py`` (S2-33 Step 7, family 2/3)."""

import warnings

warnings.warn(
    "gnn.utils.config_loader is the earlier name; import gnn.utils.config_io.config_loader instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.config_io.config_loader import (  # noqa: E402,F401
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
    "get_config_value",
    "load_config",
    "save_config",
    "set_config_value",
    "validate_config",
]
