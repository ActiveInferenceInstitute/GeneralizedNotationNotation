# Nontechnical Description

Prompt:

> Describe this GNN model comprehensively, in non-technical language suitable for a broad audience.

Response:

```python
import numpy as np
from scipy import stats

def t_maze(num_locations, num_contexts):
    """
    T-Maze Epistemic Foraging Agent

    Args:
        num_locations (int): Number of locations to explore.
        num_contexts (int): Number of contexts to explore.
        num_location_obs (int): Number of locations in each context.
        num_reward_obs (int): Number of rewards per location within a context.
        num_actions (int): Number of actions taken by the agent during exploration.
        num_timesteps (int): Number of steps to explore at any given time step.
    """

    # Initialize state matrices for each type of action
    A = np.zeros((num_locations, 4))
    B = np.zeros(num_locations)
    C = np.zeros(num_contexts)
    D = np.zeros(num_location_obs)

    # Initialize the reward matrix
    G = stats.gauss(0., 1.)

    # Initialize the prior distribution for each action
    P = np.zeros((num_actions, num_locations))

    # Initialize the reward vector for each action
    R = np.zeros((num_actions, num_locations))

    # Initialize the reward matrix for each context
    B[np.arange(num_contexts), A] = stats.gauss(0., 1.)
    B[np.arange(num_contexts), B] = stats.gauss(0., 1.)

    # Initialize the prior distribution for each action
    P[np.arange(num_actions), A] = stats.gaussian(0, 1)
    P[np.arange(num_actions), B] = stats.gaussian(0, 1)

    # Initialize the reward vector for each context
    R[np.arange(num_contexts), A] = stats.gauss(0., 1.)
    R[np.arange(num_contexts), B] = stats.gauss(0., 1.)

    # Initialize the reward matrix for each action
    P[np.arange(num_actions), A] = stats.gaussian(0, 1)
    P[np.arange(num_actions), B] = stats.gaussian(0, 1)
```