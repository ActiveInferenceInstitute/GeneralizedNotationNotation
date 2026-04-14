# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

```python
import json
from typing import List


class GNNModel:
    """GNN Representation for dynamic perception model."""

    def __init__(self):
        self.hidden_states = [
            {
                "type": "float",
                "value": 0,
                "name": "time"
            },
            {
                "type": "int",
                "value": 1,
                "name": "num_hidden_states"
            },
            {
                "type": "list[str]",
                "values": [
                    {"key": "state", "description": "The initial state of the observer.", "data": []},
                    {"key": "observation", "data": []}
                ]
            }
        ]

        self.belief_inference = {
            "time": 0,
            "observations": [],
            "actions": [
                {
                    "type": "float",
                    "value": 1,
                    "name": "state"
                },
                {
                    "type": "int",
                    "value": 2,
                    "name": "num_obs"
                }
            ]
        }

        self.belief = {
            "time": 0,
            "observations": [],
            "actions": [
                {
                    "type": "float",
                    "value": 1,
                    "name": "state"
                },
                {
                    "type": "int",
                    "value": 2,
                    "name": "num_obs"
                }
            ]
        }

        self.belief_inference = {
            "time": 0,
            "observations": [],
            "actions": [
                {
                    "type": "float",
                    "value": 1,
                    "name": "state"
                },
                {
                    "type": "int",
                    "value": 2,
                    "name": "num_obs"
                }
            ]
        }

        self.belief = {
            "time": 0,
            "observations": [],
            "actions": [
                {
                    "type": "float",
                    "value": 1,
                    "name": "state"
                },
                {
                    "type": "int",
                    "value": 2,
                    "name": "num_obs"
                }
            ]
        }

        self.belief = {
            "time": 0,
            "observations": [],
            "actions": [
                {
                    "type": "float",