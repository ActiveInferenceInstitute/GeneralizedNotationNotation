# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

```python
import json
from typing import List

def gnn(input: List[List[float]], output: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """GNN implementation for the T-Maze Epistemic Foraging Agent.

    Args:
        input (list): A list of locations to explore and reward/cue observations.
        output (list): A list of probabilities that indicate whether agent visited a location or not.

    Returns:
        List[Dict]: A dictionary containing the GNN inference results for each location
    """
    # Initialize state matrices
    s_loc = [[0] * 4 for _ in range(num_locations)]
    o_loc = [[0] * num_locations + 1 for _ in range(num_contexts)]

    # Initialize transition matrices
    B_loc = [[[]] * num_location_obs + 2 for _ in range(num_location_obs)]
    C_loc = [[[[]] * num_location_obs] + [0.5] * num_locations + 1 for _ in range(num_contexts)]

    # Initialize action matrices
    D_loc = [[[]] * num_action_states + 2 for _ in range(num_actions)]
    G_epi = [[[[]] * num_location_obs] + [0.5] * num_locations + 1 for _ in range(num_contexts)]

    # Initialize reward matrices
    F_loc = [[[]] * num_reward_states + 2 for _ in range(num_actions)]
    G_ins = [[[[]] * num_location_obs] + [0.5] * num_locations + 1 for _ in range(num_contexts)]

    # Initialize action matrices
    D_curr = [[[]] * num_action_states + 2 for _ in range(num_actions)]
    G_ins = [[[[]] * num_location_obs] + [0.5] * num_locations + 1 for _ in range(num_contexts)]

    # Initialize reward matrices
    F_curr = [[[]] * num_reward_states + 2 for _ in range(num_actions)]
    G_ins = [[[[]] * num_location_obs] + [0.5] * num_locations + 1 for _ in