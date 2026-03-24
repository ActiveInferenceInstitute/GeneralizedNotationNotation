# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

```python
import numpy as np
from typing import List, Tuple

class MultiAgentCooperativeActiveInference(object):
    """Multi-agent cooperative active inference implemented using multi-agent GNN."""

    def __init__(self) -> None:
        self.agents = []

        for i in range(len(self.agents)):
            self.agents[i] = [
                Agent(
                    name=f'Agent{i+1}',
                    actions=[
                        (
                            AgentActionType('A', 'EFE'),
                            (
                                AgentActionType('B', 'EFA')
                            ),
                            (
                                AgentActionType('C', 'ECF')
                            )
                        ),
                    ],
                ]
            for action in self.agents[i].actions

            self.agent_probabilities = np.zeros(len(self.agents), dtype=np.float)

        self.policy_probabilities = np.zeros(len(self.agents))

    def set_action(self, agent: Agent):
        """Set the action for agent 1."""
        self.agents[agent].set_action('A')

    def set_observation(self, observation: Tuple[List[Tuple[int, int]]]):
        """Set the observed state of agent 1 to a new state."""

        # Set all actions in the policy vector as future actions
        for action in self.policy_probabilities:
            if isinstance(action, AgentActionType):
                action = (
                    AgentActionType('EFE', 'ECF')
                )

            self.agents[agent].set_observation((tuple([int]) * len(self.actions)) + observation)

        # Set all actions in the policy vector as future actions
        for action in self.policy_probabilities:
            if isinstance(action, AgentActionType):
                action = (
                    AgentActionType('EFA', 'ECF')
                )

            self.agents[agent].set_observation((tuple([int]) * len(self.actions)) + observation)

        # Set all actions in the policy vector as future actions
        for action in self.policy_probabilities:
            if isinstance(action, AgentActionType):
                action = (
                    AgentActionType('EFE', 'ECF')
                )

            self.agents[agent].set_observation((tuple([int]) * len(self.actions)) + observation)

    def set_next_state(self, state: Tuple[