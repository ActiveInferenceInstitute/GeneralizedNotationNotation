"""Step timeout configuration for the GNN pipeline."""

import os
from typing import Optional

# Timeout configuration in seconds
STEP_TIMEOUTS = {
    "2_tests.py": {"default": 900, "comprehensive": 1200},
    "13_llm.py": 600,
    "22_gui.py": 600,
    "11_render.py": 300,
    "12_execute.py": 300,
}

DEFAULT_TIMEOUT = 120


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
        except ValueError:
            pass

    timeout_config = STEP_TIMEOUTS.get(script_name)
    if timeout_config is None:
        return DEFAULT_TIMEOUT
    if isinstance(timeout_config, dict):
        return timeout_config.get("comprehensive" if comprehensive else "default", DEFAULT_TIMEOUT)
    return timeout_config
