"""
GNN Model: Stigmergic Swarm Active Inference
Three Active Inference agents coordinating via stigmergy (environmental traces):

- No direct communication between agents — coordination emerges from environment
- Agents deposit and sense environmental signals (pheromone analogy)
- Shared 3x3 grid environment with signal intensity at each cell
- Each agent navigates independently while responding to accumulated signals
- Signal deposition: actions leave traces that other agents can observe
- Signal decay: environmental signals decay over time (volatility)
- Demonstrates emergent collective behavior from individual free energy minimization
- Models ant colony foraging, distributed robotics, and decentralized coordination
Generated: 2026-09-05T20:30:46.265837
"""

import numpy as np
from typing import Dict, List, Any

class StigmergicSwarmActiveInferenceModel:
    """GNN Model: Stigmergic Swarm Active Inference"""

    def __init__(self):
        self.model_name = "Stigmergic Swarm Active Inference"
        self.version = "1.0"
        self.annotation = "Three Active Inference agents coordinating via stigmergy (environmental traces):\n\n- No direct communication between agents \u2014 coordination emerges from environment\n- Agents deposit and sense environmental signals (pheromone analogy)\n- Shared 3x3 grid environment with signal intensity at each cell\n- Each agent navigates independently while responding to accumulated signals\n- Signal deposition: actions leave traces that other agents can observe\n- Signal decay: environmental signals decay over time (volatility)\n- Demonstrates emergent collective behavior from individual free energy minimization\n- Models ant colony foraging, distributed robotics, and decentralized coordination"

        # Variables
        self.variables = {
            "A_agent1": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 9],
                "description": "Agent 1 likelihood: P(obs | position on 3x3 grid)",
            },
            "A_agent2": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 9],
                "description": "Agent 2 likelihood",
            },
            "A_agent3": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 9],
                "description": "Agent 3 likelihood",
            },
            "B_agent1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9, 9, 4],
                "description": "Agent 1 transitions: (9 positions × 4 actions: N/S/E/W)",
            },
            "B_agent2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9, 9, 4],
                "description": "Agent 2 transitions",
            },
            "B_agent3": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9, 9, 4],
                "description": "Agent 3 transitions",
            },
            "C_agent1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 1 preferences over observations",
            },
            "C_agent2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 2 preferences",
            },
            "C_agent3": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 3 preferences",
            },
            "D_agent1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9],
                "description": "Agent 1 position prior",
            },
            "D_agent2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9],
                "description": "Agent 2 position prior",
            },
            "D_agent3": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9],
                "description": "Agent 3 position prior",
            },
            "G1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Agent 1 EFE",
            },
            "G2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Agent 2 EFE",
            },
            "G3": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Agent 3 EFE",
            },
            "env_obs_likelihood": {
                "type": "observation",
                "data_type": "float",
                "dimensions": [4, 3],
                "description": "P(obs category | local signal level: none/low/high)",
            },
            "env_signal": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9, 1],
                "description": "Signal intensity at each grid cell (0.0 to 1.0)",
            },
            "env_signal_prior": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [3],
                "description": "prior over local signal level (none/low/high)",
            },
            "o_agent1": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [4, 1],
                "description": "Agent 1 observation: (empty, signal_low, signal_high, goal)",
            },
            "o_agent2": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [4, 1],
                "description": "Agent 2 observation",
            },
            "o_agent3": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [4, 1],
                "description": "Agent 3 observation",
            },
            "pi1": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 1 policy",
            },
            "pi2": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 2 policy",
            },
            "pi3": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 3 policy",
            },
            "s_agent1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9, 1],
                "description": "Agent 1 position belief (9 grid cells)",
            },
            "s_agent2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9, 1],
                "description": "Agent 2 position belief",
            },
            "s_agent3": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9, 1],
                "description": "Agent 3 position belief",
            },
            "signal_decay": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Signal decay rate per timestep",
            },
            "signal_seek": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "signal-seeking gain applied to action selection",
            },
            "t": {
                "type": "hidden_state",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Discrete time step",
            },
            "u_agent1": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Agent 1 action",
            },
            "u_agent2": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Agent 2 action",
            },
            "u_agent3": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Agent 3 action",
            },
        }

        # Parameters
        self.parameters = {
            "A_agent1": [[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7]],
            "A_agent2": [[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7]],
            "A_agent3": [[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7]],
            "B_agent1": [[[1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]], [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0]], [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0]], [[1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]],
            "B_agent2": [[[1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]], [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0]], [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0]], [[1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]],
            "B_agent3": [[[1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]], [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0]], [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0]], [[1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]],
            "C_agent1": [[-0.5, 0.5, 1.5, 3.0]],
            "C_agent2": [[-0.5, 0.5, 1.5, 3.0]],
            "C_agent3": [[-0.5, 0.5, 1.5, 3.0]],
            "D_agent1": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "D_agent2": [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "D_agent3": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            "env_obs_likelihood": [[0.7, 0.1, 0.05], [0.15, 0.7, 0.15], [0.1, 0.15, 0.75], [0.05, 0.05, 0.05]],
            "env_signal": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "env_signal_prior": [[0.7, 0.2, 0.1]],
            "grid_size": 9,
            "num_actions": 4,
            "num_actions": 4,
            "num_agents": 3,
            "num_hidden_states": 729,
            "num_obs": 64,
            "num_obs": 4,
            "num_timesteps": 30,
            "signal_decay": [[0.9]],
            "signal_decay_rate": 0.9,
            "signal_deposit_rate": 0.3,
            "signal_seek": [[2.0]],
        }

# MODEL_DATA: {"model_name":"Stigmergic Swarm Active Inference","annotation":"Three Active Inference agents coordinating via stigmergy (environmental traces):\n\n- No direct communication between agents \u2014 coordination emerges from environment\n- Agents deposit and sense environmental signals (pheromone analogy)\n- Shared 3x3 grid environment with signal intensity at each cell\n- Each agent navigates independently while responding to accumulated signals\n- Signal deposition: actions leave traces that other agents can observe\n- Signal decay: environmental signals decay over time (volatility)\n- Demonstrates emergent collective behavior from individual free energy minimization\n- Models ant colony foraging, distributed robotics, and decentralized coordination","variables":[{"name":"A_agent1","var_type":"action","data_type":"float","dimensions":[4,9]},{"name":"B_agent1","var_type":"hidden_state","data_type":"float","dimensions":[9,9,4]},{"name":"C_agent1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_agent1","var_type":"hidden_state","data_type":"float","dimensions":[9]},{"name":"s_agent1","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"o_agent1","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"pi1","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"u_agent1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A_agent2","var_type":"action","data_type":"float","dimensions":[4,9]},{"name":"B_agent2","var_type":"hidden_state","data_type":"float","dimensions":[9,9,4]},{"name":"C_agent2","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_agent2","var_type":"hidden_state","data_type":"float","dimensions":[9]},{"name":"s_agent2","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"o_agent2","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"pi2","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"u_agent2","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A_agent3","var_type":"action","data_type":"float","dimensions":[4,9]},{"name":"B_agent3","var_type":"hidden_state","data_type":"float","dimensions":[9,9,4]},{"name":"C_agent3","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_agent3","var_type":"hidden_state","data_type":"float","dimensions":[9]},{"name":"s_agent3","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"o_agent3","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"pi3","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"u_agent3","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G3","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"env_signal","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"signal_decay","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"env_obs_likelihood","var_type":"observation","data_type":"float","dimensions":[4,3]},{"name":"env_signal_prior","var_type":"prior_vector","data_type":"float","dimensions":[3]},{"name":"signal_seek","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["D_agent1"],"target_variables":["s_agent1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_agent1"],"target_variables":["A_agent1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_agent1"],"target_variables":["o_agent1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_agent1"],"target_variables":["G1"],"connection_type":"directed"},{"annotation":null,"source_variables":["G1"],"target_variables":["pi1"],"connection_type":"directed"},{"annotation":null,"source_variables":["pi1"],"target_variables":["u_agent1"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_agent1"],"target_variables":["u_agent1"],"connection_type":"directed"},{"annotation":null,"source_variables":["D_agent2"],"target_variables":["s_agent2"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_agent2"],"target_variables":["A_agent2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_agent2"],"target_variables":["o_agent2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_agent2"],"target_variables":["G2"],"connection_type":"directed"},{"annotation":null,"source_variables":["G2"],"target_variables":["pi2"],"connection_type":"directed"},{"annotation":null,"source_variables":["pi2"],"target_variables":["u_agent2"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_agent2"],"target_variables":["u_agent2"],"connection_type":"directed"},{"annotation":null,"source_variables":["D_agent3"],"target_variables":["s_agent3"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_agent3"],"target_variables":["A_agent3"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_agent3"],"target_variables":["o_agent3"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_agent3"],"target_variables":["G3"],"connection_type":"directed"},{"annotation":null,"source_variables":["G3"],"target_variables":["pi3"],"connection_type":"directed"},{"annotation":null,"source_variables":["pi3"],"target_variables":["u_agent3"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_agent3"],"target_variables":["u_agent3"],"connection_type":"directed"},{"annotation":null,"source_variables":["u_agent1"],"target_variables":["env_signal"],"connection_type":"directed"},{"annotation":null,"source_variables":["u_agent2"],"target_variables":["env_signal"],"connection_type":"directed"},{"annotation":null,"source_variables":["u_agent3"],"target_variables":["env_signal"],"connection_type":"directed"},{"annotation":null,"source_variables":["env_signal"],"target_variables":["A_agent1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["env_signal"],"target_variables":["A_agent2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["env_signal"],"target_variables":["A_agent3"],"connection_type":"undirected"},{"annotation":null,"source_variables":["signal_decay"],"target_variables":["env_signal"],"connection_type":"directed"}],"parameters":[{"name":"A_agent1","value":[[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.7]],"param_type":"constant"},{"name":"A_agent2","value":[[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.7]],"param_type":"constant"},{"name":"A_agent3","value":[[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.7]],"param_type":"constant"},{"name":"C_agent1","value":[[-0.5,0.5,1.5,3.0]],"param_type":"constant"},{"name":"C_agent2","value":[[-0.5,0.5,1.5,3.0]],"param_type":"constant"},{"name":"C_agent3","value":[[-0.5,0.5,1.5,3.0]],"param_type":"constant"},{"name":"D_agent1","value":[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],"param_type":"constant"},{"name":"D_agent2","value":[[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]],"param_type":"constant"},{"name":"D_agent3","value":[[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]],"param_type":"constant"},{"name":"B_agent1","value":[[[1.0,0.0,0.0,0.9,0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.9,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0,0.9,0.0,0.0,0.0],[0.0,0.0,0.0,0.1,0.0,0.0,0.9,0.0,0.0],[0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.9,0.0],[0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.9],[0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1]],[[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0],[0.9,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0],[0.0,0.9,0.0,0.0,0.1,0.0,0.0,0.0,0.0],[0.0,0.0,0.9,0.0,0.0,0.1,0.0,0.0,0.0],[0.0,0.0,0.0,0.9,0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.9,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.9,0.0,0.0,1.0]],[[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.9,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.9,1.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.9,0.1,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.9,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.9,0.1,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.9,1.0]],[[1.0,0.9,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.1,0.9,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.9,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.1,0.9,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.9,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.9],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1]]],"param_type":"constant"},{"name":"B_agent2","value":[[[1.0,0.0,0.0,0.9,0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.9,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0,0.9,0.0,0.0,0.0],[0.0,0.0,0.0,0.1,0.0,0.0,0.9,0.0,0.0],[0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.9,0.0],[0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.9],[0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1]],[[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0],[0.9,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0],[0.0,0.9,0.0,0.0,0.1,0.0,0.0,0.0,0.0],[0.0,0.0,0.9,0.0,0.0,0.1,0.0,0.0,0.0],[0.0,0.0,0.0,0.9,0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.9,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.9,0.0,0.0,1.0]],[[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.9,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.9,1.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.9,0.1,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.9,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.9,0.1,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.9,1.0]],[[1.0,0.9,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.1,0.9,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.9,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.1,0.9,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.9,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.9],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1]]],"param_type":"constant"},{"name":"B_agent3","value":[[[1.0,0.0,0.0,0.9,0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.9,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0,0.9,0.0,0.0,0.0],[0.0,0.0,0.0,0.1,0.0,0.0,0.9,0.0,0.0],[0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.9,0.0],[0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.9],[0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1]],[[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0],[0.9,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0],[0.0,0.9,0.0,0.0,0.1,0.0,0.0,0.0,0.0],[0.0,0.0,0.9,0.0,0.0,0.1,0.0,0.0,0.0],[0.0,0.0,0.0,0.9,0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.9,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.9,0.0,0.0,1.0]],[[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.9,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.9,1.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.9,0.1,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.9,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.9,0.1,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.9,1.0]],[[1.0,0.9,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.1,0.9,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.9,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.1,0.9,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.9,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.9],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1]]],"param_type":"constant"},{"name":"env_signal","value":[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],"param_type":"constant"},{"name":"signal_decay","value":[[0.9]],"param_type":"constant"},{"name":"env_obs_likelihood","value":[[0.7,0.1,0.05],[0.15,0.7,0.15],[0.1,0.15,0.75],[0.05,0.05,0.05]],"param_type":"constant"},{"name":"env_signal_prior","value":[[0.7,0.2,0.1]],"param_type":"constant"},{"name":"signal_seek","value":[[2.0]],"param_type":"constant"},{"name":"num_hidden_states","value":729,"param_type":"constant"},{"name":"num_obs","value":64,"param_type":"constant"},{"name":"num_actions","value":4,"param_type":"constant"},{"name":"num_agents","value":3,"param_type":"constant"},{"name":"grid_size","value":9,"param_type":"constant"},{"name":"num_obs","value":4,"param_type":"constant"},{"name":"num_actions","value":4,"param_type":"constant"},{"name":"signal_decay_rate","value":0.9,"param_type":"constant"},{"name":"signal_deposit_rate","value":0.3,"param_type":"constant"},{"name":"num_timesteps","value":30,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":30,"step_size":null},"ontology_mappings":[{"variable_name":"A_agent1","ontology_term":"Agent1LikelihoodMatrix","description":null},{"variable_name":"C_agent1","ontology_term":"Agent1PreferenceVector","description":null},{"variable_name":"D_agent1","ontology_term":"Agent1PositionPrior","description":null},{"variable_name":"s_agent1","ontology_term":"Agent1PositionState","description":null},{"variable_name":"o_agent1","ontology_term":"Agent1Observation","description":null},{"variable_name":"pi1","ontology_term":"Agent1PolicyVector","description":null},{"variable_name":"u_agent1","ontology_term":"Agent1Action","description":null},{"variable_name":"G1","ontology_term":"Agent1ExpectedFreeEnergy","description":null},{"variable_name":"A_agent2","ontology_term":"Agent2LikelihoodMatrix","description":null},{"variable_name":"C_agent2","ontology_term":"Agent2PreferenceVector","description":null},{"variable_name":"D_agent2","ontology_term":"Agent2PositionPrior","description":null},{"variable_name":"s_agent2","ontology_term":"Agent2PositionState","description":null},{"variable_name":"o_agent2","ontology_term":"Agent2Observation","description":null},{"variable_name":"pi2","ontology_term":"Agent2PolicyVector","description":null},{"variable_name":"u_agent2","ontology_term":"Agent2Action","description":null},{"variable_name":"G2","ontology_term":"Agent2ExpectedFreeEnergy","description":null},{"variable_name":"A_agent3","ontology_term":"Agent3LikelihoodMatrix","description":null},{"variable_name":"C_agent3","ontology_term":"Agent3PreferenceVector","description":null},{"variable_name":"D_agent3","ontology_term":"Agent3PositionPrior","description":null},{"variable_name":"s_agent3","ontology_term":"Agent3PositionState","description":null},{"variable_name":"o_agent3","ontology_term":"Agent3Observation","description":null},{"variable_name":"pi3","ontology_term":"Agent3PolicyVector","description":null},{"variable_name":"u_agent3","ontology_term":"Agent3Action","description":null},{"variable_name":"G3","ontology_term":"Agent3ExpectedFreeEnergy","description":null},{"variable_name":"env_signal","ontology_term":"EnvironmentalSignal","description":null},{"variable_name":"signal_decay","ontology_term":"SignalDecayRate","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
