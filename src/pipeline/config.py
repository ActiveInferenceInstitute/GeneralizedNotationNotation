#!/usr/bin/env python3
"""
Pipeline configuration module.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, cast

logger = logging.getLogger(__name__)

# Make PyYAML optional to avoid hard failures during import time
try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    yaml = cast(Any, None)
    _YAML_AVAILABLE = False

# Pipeline configuration derived from canonical step registry
from pipeline.step_registry import (
    STEP_METADATA_DICT as STEP_METADATA,  # noqa: E402,F401
)


class StepConfig:
    """Configuration for a pipeline step."""

    def __init__(self, step_name: str, **kwargs: Any) -> None:
        """Initialize the instance."""
        self.step_name = step_name
        self.enabled = kwargs.get("enabled", True)
        self.timeout = kwargs.get("timeout", 3600)
        self.retries = kwargs.get("retries", 3)
        self.dependencies = kwargs.get("dependencies", [])
        self.parameters = kwargs.get("parameters", {})
        self.required = kwargs.get("required", True)
        self.output_subdir = kwargs.get(
            "output_subdir", f"{step_name.removesuffix('.py')}_output"
        )
        self.performance_tracking = kwargs.get("performance_tracking", True)


class PipelineConfig:
    """Pipeline configuration manager."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize the instance."""
        self.config_path = config_path or Path("pipeline_config.yaml")
        self.config = self._load_settings_from_path()

    def _load_settings_from_path(self) -> Dict[str, Any]:
        """Load pipeline settings dict from ``config_path`` (YAML or JSON)."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    if self.config_path.suffix in (".yaml", ".yml"):
                        if _YAML_AVAILABLE:
                            return yaml.safe_load(f) or {}
                        else:
                            # Gracefully degrade: cannot parse YAML; return empty config
                            # Downstream code should use sensible defaults
                            return {}
                    else:
                        return cast("dict[str, Any]", json.load(f))
            except (json.JSONDecodeError, OSError, ValueError) as e:
                logger.debug("Could not parse config file %s: %s", self.config_path, e)
                return {}
        return {}

    @property
    def steps(self) -> Dict[str, StepConfig]:
        """Return configured step objects keyed by canonical step script name."""
        config_steps = self.config.get("steps", {})
        if isinstance(config_steps, dict):
            return {
                step_name: StepConfig(step_name, **step_data)
                for step_name, step_data in config_steps.items()
                if isinstance(step_data, dict)
            }
        return {
            f"{step_name}.py": self.get_step_config(f"{step_name}.py")
            for step_name in STEP_METADATA
        }

    def get_step_config(self, step_name: str) -> StepConfig:
        """Get configuration for a specific step."""
        step_data = self.config.get(step_name, {})
        return StepConfig(step_name, **step_data)

    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_path, "w") as f:
                if self.config_path.suffix in (".yaml", ".yml") and _YAML_AVAILABLE:
                    yaml.dump(self.config, f)
                else:
                    json.dump(self.config, f, indent=2)
        except (OSError, ValueError) as e:
            logger.debug("Config save failed (%s), attempting JSON recovery", e)
            try:
                json_path = self.config_path.with_suffix(".json")
                with open(json_path, "w") as jf:
                    json.dump(self.config, jf, indent=2)
            except (OSError, TypeError, ValueError) as e:
                logger.debug("JSON recovery config save also failed: %s", e)


def get_pipeline_config() -> dict:
    """Get pipeline configuration as a plain dict for compatibility with tests."""
    cfg = PipelineConfig()
    data = cfg.config if isinstance(cfg.config, dict) else {}
    # Ensure required keys exist for tests with sensible defaults
    if "steps" not in data:
        data["steps"] = list(STEP_METADATA.keys())
    if "timeout" not in data:
        data["timeout"] = 3600
    if "parallel" not in data:
        data["parallel"] = True
    return data


def get_pipeline_config_dict() -> Dict[str, Any]:
    """Get the pipeline configuration as a plain dict (compatibility helper)."""
    cfg = PipelineConfig()
    return cfg.config if isinstance(cfg.config, dict) else {}


def set_pipeline_config(config: PipelineConfig) -> None:
    """Set the pipeline configuration."""
    config.save_config()


def get_output_dir_for_script(script_name: str, base_output_dir: Path) -> Path:
    """Get output directory for a specific script.

    Use a consistent numbered '<N_name>_output' directory for every step to
    keep the pipeline coherent and simple.

    Args:
        script_name: Name of the pipeline script (e.g., "3_gnn.py" or "3_gnn")
        base_output_dir: Base output directory (usually "output/")

    Returns:
        Path to step-specific output directory

    Note:
        This function prevents nested directories by detecting if base_output_dir
        already ends with the expected output directory name.
    """
    from pipeline.step_registry import output_dir_for_stem

    script_stem = Path(script_name).stem
    normalized = script_stem  # e.g., '7_export'

    # Try exact match from step registry
    mapped = output_dir_for_stem(normalized)
    if mapped is not None:
        result = base_output_dir / mapped
        # Prevent nesting
        if base_output_dir.name == mapped:
            return base_output_dir
        return result

    # Accept '.py' suffix keys as well
    if script_name.endswith(".py"):
        normalized = script_name[:-3]
        mapped = output_dir_for_stem(normalized)
        if mapped is not None:
            result = base_output_dir / mapped
            if base_output_dir.name == mapped:
                return base_output_dir
            return result

    # Get expected output directory name for this script
    expected_dir_name = f"{normalized}_output"

    # Check if base_output_dir already ends with the expected directory name
    # This prevents nested directories like "10_ontology_output/10_ontology_output"
    if base_output_dir.name == expected_dir_name:
        # Already at the correct output directory - return as is
        return base_output_dir

    # Check if base_output_dir is already inside a step output directory
    # (e.g., "output/10_ontology_output/subdir" should not create another layer)
    if "_output" in base_output_dir.name and base_output_dir.parent.name != "output":
        # We're inside a step output directory - use the parent's parent as base
        # This handles cases like passing "output/10_ontology_output/results" as base
        actual_base = base_output_dir
        while (
            actual_base.name.endswith("_output") or "_output" in actual_base.parts[-2:]
        ):
            if actual_base.parent.name == "output":
                break
            actual_base = actual_base.parent
        base_output_dir = (
            actual_base.parent if actual_base.name.endswith("_output") else actual_base
        )

    # Default recovery
    return base_output_dir / expected_dir_name
