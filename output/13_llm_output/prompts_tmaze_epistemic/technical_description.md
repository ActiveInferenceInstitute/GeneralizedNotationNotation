# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

```python
import json
from typing import List, Dict, Tuple

class TmazeEpistemicForagingAgent:
    """Class for T-Maze Epistemic Foragers.

    Attributes:
        state_space(list[Dict[str, float]]): A list of 4 locations representing the agent's current location and reward/reward pair.
            Each location is represented by a dictionary with keys "center", "left" and "right". The values are floats.
            Each value represents the probability that the agent will visit the given arm if it has not already visited it before,
            or to explore the next arm based on its current state space.
    """

    def __init__(self):
        self._state_space = list(range(10))  # Initialize with a random location and reward pair

        self._actions = []  # List of actions for each arm

        self._reward_map = {}  # Map reward to the corresponding action in the agent's state space

    def _get_location(self, x):
        """Return the next location from the agent's current position."""
        if x == "center":
            return (x, 1.0)
        elif x == "left":
            return (x - 1, 1.0)
        else:
            raise ValueError("Invalid action")

    def _get_reward(self):
        """Return the reward from the agent's current actions."""
        if self._actions[0] is not None and self._actions[0].get_value() == "left":
            return (self._reward_map.get(self._actions[1], 0))

    def _update_state_space(self):
        """Update the state space of the agent."""

        if len(self._actions) > 3:
            raise ValueError("Agent cannot visit more than three actions")

        self._action = self._get_location()
        self.move_to_next_arm()

    def _move_to_next_arm(self):
        """Move the agent to a new arm."""

        if len(self._actions) > 3:
            raise ValueError("Agent cannot visit more than three actions")

        for action in range(len(self._actions)):
            self.move_to_new_arm()

    def _get_reward_next_arm(self):
        """Return the reward from the agent's next arm."""

        if len(self