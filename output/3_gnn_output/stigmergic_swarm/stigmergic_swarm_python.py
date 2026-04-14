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
Generated: 2026-04-14T11:53:25.425888
"""

import numpy as np
from typing import Dict, List, Any

class StigmergicSwarmActiveInferenceModel:
    """GNN Model: Stigmergic Swarm Active Inference"""

    def __init__(self):
        self.model_name = "Stigmergic Swarm Active Inference"
        self.version = "1.0"
        self.annotation = "Three Active Inference agents coordinating via stigmergy (environmental traces):

- No direct communication between agents — coordination emerges from environment
- Agents deposit and sense environmental signals (pheromone analogy)
- Shared 3x3 grid environment with signal intensity at each cell
- Each agent navigates independently while responding to accumulated signals
- Signal deposition: actions leave traces that other agents can observe
- Signal decay: environmental signals decay over time (volatility)
- Demonstrates emergent collective behavior from individual free energy minimization
- Models ant colony foraging, distributed robotics, and decentralized coordination"

        # Variables
        self.variables = {
            "A1": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 9],
                "description": "Agent 1 likelihood: P(obs | position on 3x3 grid)",
            },
            "A2": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 9],
                "description": "Agent 2 likelihood",
            },
            "A3": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 9],
                "description": "Agent 3 likelihood",
            },
            "B1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9, 9, 4],
                "description": "Agent 1 transitions: (9 positions × 4 actions: N/S/E/W)",
            },
            "B2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9, 9, 4],
                "description": "Agent 2 transitions",
            },
            "B3": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9, 9, 4],
                "description": "Agent 3 transitions",
            },
            "C1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 1 preferences over observations",
            },
            "C2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 2 preferences",
            },
            "C3": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 3 preferences",
            },
            "D1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9],
                "description": "Agent 1 position prior",
            },
            "D2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9],
                "description": "Agent 2 position prior",
            },
            "D3": {
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
            "env_signal": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9, 1],
                "description": "Signal intensity at each grid cell (0.0 to 1.0)",
            },
            "o1": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [4, 1],
                "description": "Agent 1 observation: (empty, signal_low, signal_high, goal)",
            },
            "o2": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [4, 1],
                "description": "Agent 2 observation",
            },
            "o3": {
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
            "s1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9, 1],
                "description": "Agent 1 position belief (9 grid cells)",
            },
            "s2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9, 1],
                "description": "Agent 2 position belief",
            },
            "s3": {
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
            "t": {
                "type": "hidden_state",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Discrete time step",
            },
            "u1": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Agent 1 action",
            },
            "u2": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Agent 2 action",
            },
            "u3": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Agent 3 action",
            },
        }

        # Parameters
        self.parameters = {
            "A1": [[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7]],
            "A2": [[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7]],
            "A3": [[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7]],
            "C1": [[-0.5, 0.5, 1.5, 3.0]],
            "C2": [[-0.5, 0.5, 1.5, 3.0]],
            "C3": [[-0.5, 0.5, 1.5, 3.0]],
            "D1": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "D2": [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "D3": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            "env_signal": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "signal_decay": [[0.9]],
        }

# MODEL_DATA: {"model_name":"Stigmergic Swarm Active Inference","annotation":"Three Active Inference agents coordinating via stigmergy (environmental traces):\n\n- No direct communication between agents \u2014 coordination emerges from environment\n- Agents deposit and sense environmental signals (pheromone analogy)\n- Shared 3x3 grid environment with signal intensity at each cell\n- Each agent navigates independently while responding to accumulated signals\n- Signal deposition: actions leave traces that other agents can observe\n- Signal decay: environmental signals decay over time (volatility)\n- Demonstrates emergent collective behavior from individual free energy minimization\n- Models ant colony foraging, distributed robotics, and decentralized coordination","variables":[{"name":"A1","var_type":"action","data_type":"float","dimensions":[4,9]},{"name":"B1","var_type":"hidden_state","data_type":"float","dimensions":[9,9,4]},{"name":"C1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D1","var_type":"hidden_state","data_type":"float","dimensions":[9]},{"name":"s1","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"o1","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"pi1","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"u1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A2","var_type":"action","data_type":"float","dimensions":[4,9]},{"name":"B2","var_type":"hidden_state","data_type":"float","dimensions":[9,9,4]},{"name":"C2","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D2","var_type":"hidden_state","data_type":"float","dimensions":[9]},{"name":"s2","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"o2","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"pi2","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"u2","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A3","var_type":"action","data_type":"float","dimensions":[4,9]},{"name":"B3","var_type":"hidden_state","data_type":"float","dimensions":[9,9,4]},{"name":"C3","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D3","var_type":"hidden_state","data_type":"float","dimensions":[9]},{"name":"s3","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"o3","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"pi3","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"u3","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G3","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"env_signal","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"signal_decay","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D1"],"target_variables":["s1"],"connection_type":"directed"},{"source_variables":["s1"],"target_variables":["A1"],"connection_type":"undirected"},{"source_variables":["A1"],"target_variables":["o1"],"connection_type":"undirected"},{"source_variables":["C1"],"target_variables":["G1"],"connection_type":"directed"},{"source_variables":["G1"],"target_variables":["pi1"],"connection_type":"directed"},{"source_variables":["pi1"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["B1"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["D2"],"target_variables":["s2"],"connection_type":"directed"},{"source_variables":["s2"],"target_variables":["A2"],"connection_type":"undirected"},{"source_variables":["A2"],"target_variables":["o2"],"connection_type":"undirected"},{"source_variables":["C2"],"target_variables":["G2"],"connection_type":"directed"},{"source_variables":["G2"],"target_variables":["pi2"],"connection_type":"directed"},{"source_variables":["pi2"],"target_variables":["u2"],"connection_type":"directed"},{"source_variables":["B2"],"target_variables":["u2"],"connection_type":"directed"},{"source_variables":["D3"],"target_variables":["s3"],"connection_type":"directed"},{"source_variables":["s3"],"target_variables":["A3"],"connection_type":"undirected"},{"source_variables":["A3"],"target_variables":["o3"],"connection_type":"undirected"},{"source_variables":["C3"],"target_variables":["G3"],"connection_type":"directed"},{"source_variables":["G3"],"target_variables":["pi3"],"connection_type":"directed"},{"source_variables":["pi3"],"target_variables":["u3"],"connection_type":"directed"},{"source_variables":["B3"],"target_variables":["u3"],"connection_type":"directed"},{"source_variables":["u1"],"target_variables":["env_signal"],"connection_type":"directed"},{"source_variables":["u2"],"target_variables":["env_signal"],"connection_type":"directed"},{"source_variables":["u3"],"target_variables":["env_signal"],"connection_type":"directed"},{"source_variables":["env_signal"],"target_variables":["A1"],"connection_type":"undirected"},{"source_variables":["env_signal"],"target_variables":["A2"],"connection_type":"undirected"},{"source_variables":["env_signal"],"target_variables":["A3"],"connection_type":"undirected"},{"source_variables":["signal_decay"],"target_variables":["env_signal"],"connection_type":"directed"}],"parameters":[{"name":"A1","value":[[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.7]],"param_type":"constant"},{"name":"A2","value":[[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.7]],"param_type":"constant"},{"name":"A3","value":[[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.7]],"param_type":"constant"},{"name":"C1","value":[[-0.5,0.5,1.5,3.0]],"param_type":"constant"},{"name":"C2","value":[[-0.5,0.5,1.5,3.0]],"param_type":"constant"},{"name":"C3","value":[[-0.5,0.5,1.5,3.0]],"param_type":"constant"},{"name":"D1","value":[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],"param_type":"constant"},{"name":"D2","value":[[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]],"param_type":"constant"},{"name":"D3","value":[[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]],"param_type":"constant"},{"name":"env_signal","value":[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],"param_type":"constant"},{"name":"signal_decay","value":[[0.9]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":30,"step_size":null},"ontology_mappings":[{"variable_name":"A1","ontology_term":"Agent1LikelihoodMatrix","description":null},{"variable_name":"B1","ontology_term":"Agent1TransitionMatrix","description":null},{"variable_name":"C1","ontology_term":"Agent1PreferenceVector","description":null},{"variable_name":"D1","ontology_term":"Agent1PositionPrior","description":null},{"variable_name":"s1","ontology_term":"Agent1PositionState","description":null},{"variable_name":"o1","ontology_term":"Agent1Observation","description":null},{"variable_name":"pi1","ontology_term":"Agent1PolicyVector","description":null},{"variable_name":"u1","ontology_term":"Agent1Action","description":null},{"variable_name":"G1","ontology_term":"Agent1ExpectedFreeEnergy","description":null},{"variable_name":"A2","ontology_term":"Agent2LikelihoodMatrix","description":null},{"variable_name":"B2","ontology_term":"Agent2TransitionMatrix","description":null},{"variable_name":"C2","ontology_term":"Agent2PreferenceVector","description":null},{"variable_name":"D2","ontology_term":"Agent2PositionPrior","description":null},{"variable_name":"s2","ontology_term":"Agent2PositionState","description":null},{"variable_name":"o2","ontology_term":"Agent2Observation","description":null},{"variable_name":"pi2","ontology_term":"Agent2PolicyVector","description":null},{"variable_name":"u2","ontology_term":"Agent2Action","description":null},{"variable_name":"G2","ontology_term":"Agent2ExpectedFreeEnergy","description":null},{"variable_name":"A3","ontology_term":"Agent3LikelihoodMatrix","description":null},{"variable_name":"B3","ontology_term":"Agent3TransitionMatrix","description":null},{"variable_name":"C3","ontology_term":"Agent3PreferenceVector","description":null},{"variable_name":"D3","ontology_term":"Agent3PositionPrior","description":null},{"variable_name":"s3","ontology_term":"Agent3PositionState","description":null},{"variable_name":"o3","ontology_term":"Agent3Observation","description":null},{"variable_name":"pi3","ontology_term":"Agent3PolicyVector","description":null},{"variable_name":"u3","ontology_term":"Agent3Action","description":null},{"variable_name":"G3","ontology_term":"Agent3ExpectedFreeEnergy","description":null},{"variable_name":"env_signal","ontology_term":"EnvironmentalSignal","description":null},{"variable_name":"signal_decay","ontology_term":"SignalDecayRate","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
