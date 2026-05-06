"""Step timeout configuration for the GNN pipeline."""

import logging
import os

logger = logging.getLogger(__name__)

# Timeout configuration in seconds
STEP_TIMEOUTS = {
    "2_tests.py": {"default": 900, "comprehensive": 1200},
    "9_advanced_viz.py": 300,
    "13_llm.py": 900,       # 72 LLM calls (9 prompts × 8 files), ~12s each
    "16_analysis.py": 300,  # 9+ models × multi-framework visualization generation
    "17_integration.py": 300,  # Dependency graph + system checks
    "22_gui.py": 600,
    "11_render.py": 300,
    "12_execute.py": 3600,
}

DEFAULT_TIMEOUT = 180


def get_step_timeout(script_name: str, comprehensive: bool = False) -> int:
    """Get timeout for a pipeline step.

    Supports environment variable override: GNN_STEP_TIMEOUT_{STEP_NUMBER}
    e.g., GNN_STEP_TIMEOUT_2=1800 overrides 2_tests.py timeout.
    """
    # Check env var override first
    step_num = script_name.split("_")[0] if "_" in script_name else ""
    env_key = f"GNN_STEP_TIMEOUT_{step_num}"
    env_val = os.environ.get(env_key)
    if env_val:
        try:
            return int(env_val)
        except ValueError as e:
            logger.debug("Invalid timeout value in %s: %s", env_key, e)

    timeout_config = STEP_TIMEOUTS.get(script_name)
    if timeout_config is None:
        return DEFAULT_TIMEOUT
    if isinstance(timeout_config, dict):
        return timeout_config.get("comprehensive" if comprehensive else "default", DEFAULT_TIMEOUT)
    return timeout_config
