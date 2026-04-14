# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

```python
import json
from typing import List

class MultiAgentCooperativeActiveInference:
    def __init__(self):
        self._model = {}

    def add_agent(self, agent1, agent2):
        """Add a new agent to the graph."""

        if isinstance(agent1, (int, float)):
            self.add_agent(
                Agent(
                    "A",
                    {"x": 0},
                    {"y": 0}
                )
        elif isinstance(agent2, (int, float)):
            self.add_agent(
                Agent(
                    "B",
                    {"x": 1},
                    {"y": 1}
                )
        else:
            raise ValueError("Invalid agent type")

    def add_agent(self, agent):
        """Add a new agent to the graph."""

        if isinstance(agent, (int, float)):
            self.add_agent(
                Agent(
                    "A",
                    {"x": 0},
                    {"y": 1}
                )
        elif isinstance(agent, (list[float], list[int])):
            for i in range(len(agent) + 1):
                if isinstance(agent[i], (int, float)):
                    self.add_agent(
                        Agent(
                            "B",
                            {"x": 0},
                            {"y": 1}
                        )
        else:
            raise ValueError("Invalid agent type")

    def add_agent_with_policy(self):
        """Add a new agent with policy."""

        if isinstance(self.add_agent, (list[float], list[int])):
            for i in range(len(self)):
                self.add_agent([
                    Agent(
                        "A",
                        {"x": 0},
                        {"y": 1}
                    )
                ]).append(Agent("B"))

    def add_agent_with_policy(self, agent):
        """Add a new agent with policy."""

        if isinstance(self.add_agent, (list[float], list[int])):
            for i in range(len(self)):
                self.add_agent([
                    Agent(
                        "A",
                        {"x": 0},
                        {"y": 1}
                    )
                ]).append(Agent("B"))

    def add_agent_with_policy(self, agent):
        """Add a new agent with policy."""

        if isinstance(self.add_agent