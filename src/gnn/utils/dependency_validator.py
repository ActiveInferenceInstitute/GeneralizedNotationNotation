"""Earlier name; implementation moved to
``gnn/utils/runtime_safety/dependency_validator.py`` (S2-33 Step 4)."""

import warnings

warnings.warn(
    "gnn.utils.dependency_validator is the earlier name; import gnn.utils.runtime_safety.dependency_validator instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.runtime_safety.dependency_validator import (  # noqa: E402,F401
    DependencySpec,
    DependencyValidator,
    check_optional_dependencies,
    get_dependency_status,
    install_missing_dependencies,
    validate_pipeline_dependencies,
    validate_pipeline_dependencies_if_available,
)

__all__ = [
    "DependencySpec",
    "DependencyValidator",
    "check_optional_dependencies",
    "get_dependency_status",
    "install_missing_dependencies",
    "validate_pipeline_dependencies",
    "validate_pipeline_dependencies_if_available",
]
