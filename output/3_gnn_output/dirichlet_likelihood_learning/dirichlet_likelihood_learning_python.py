"""
GNN Model: Dirichlet Likelihood Learning Agent
This model describes a discrete POMDP agent that learns its observation model:

- 3 hidden states, 3 observation outcomes, 2 actions (cycle, stay).
- The likelihood matrix A is NOT fixed: it is a latent DirichletCollection
  variable with prior pseudo-counts declared in dirichlet_A.
- The A values under InitialParameterization are the GROUND-TRUTH likelihood
  used by the environment to simulate observations; the agent never sees them
  directly and must recover them in q(A).
- The Dirichlet prior is identity-biased (diagonal 3.0, off-diagonal 1.0):
  the agent starts believing observations weakly track states. A fully
  uniform prior leaves the column-permutation symmetry unbroken and
  variational inference converges to a label-switched optimum.
- Transitions B are near-deterministic and known, so states are
  well-determined by actions and likelihood learning is well-conditioned.
- Inference: structured VMP with mean-field cut q(s, A) = q(s)q(A),
  q(A) initialized at the prior counts, q(s) initialized uniform.
Generated: 2026-09-08T06:53:45.803863
"""

import numpy as np
from typing import Dict, List, Any

class DirichletLikelihoodLearningAgentModel:
    """GNN Model: Dirichlet Likelihood Learning Agent"""

    def __init__(self):
        self.model_name = "Dirichlet Likelihood Learning Agent"
        self.version = "1.0"
        self.annotation = "This model describes a discrete POMDP agent that learns its observation model:\n\n- 3 hidden states, 3 observation outcomes, 2 actions (cycle, stay).\n- The likelihood matrix A is NOT fixed: it is a latent DirichletCollection\n  variable with prior pseudo-counts declared in dirichlet_A.\n- The A values under InitialParameterization are the GROUND-TRUTH likelihood\n  used by the environment to simulate observations; the agent never sees them\n  directly and must recover them in q(A).\n- The Dirichlet prior is identity-biased (diagonal 3.0, off-diagonal 1.0):\n  the agent starts believing observations weakly track states. A fully\n  uniform prior leaves the column-permutation symmetry unbroken and\n  variational inference converges to a label-switched optimum.\n- Transitions B are near-deterministic and known, so states are\n  well-determined by actions and likelihood learning is well-conditioned.\n- Inference: structured VMP with mean-field cut q(s, A) = q(s)q(A),\n  q(A) initialized at the prior counts, q(s) initialized uniform."

        # Variables
        self.variables = {
            "A": {
                "type": "likelihood_matrix",
                "data_type": "float",
                "dimensions": [3, 3],
                "description": "Ground-truth observation model P(o|s); latent in the agent",
            },
            "B": {
                "type": "transition_matrix",
                "data_type": "float",
                "dimensions": [3, 3, 2],
                "description": "State transitions given previous state and action",
            },
            "C": {
                "type": "preference_vector",
                "data_type": "float",
                "dimensions": [3],
                "description": "Log-preferences over observations",
            },
            "D": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [3],
                "description": "Prior over initial hidden states (known start)",
            },
            "G": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [1],
                "description": "Expected Free Energy (per policy)",
            },
            "dirichlet_A": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 3],
                "description": "Prior counts for q(A) ~ DirichletCollection",
            },
            "o": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [3, 1],
                "description": "Current observation index",
            },
            "s": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Current hidden state distribution",
            },
            "s_prime": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Next hidden state distribution",
            },
            "t": {
                "type": "hidden_state",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Discrete time step",
            },
            "u": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Action taken",
            },
            "π": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [2],
                "description": "Policy (distribution over actions)",
            },
        }

        # Parameters
        self.parameters = {
            "A": [[0.85, 0.05, 0.1], [0.1, 0.9, 0.05], [0.05, 0.05, 0.85]],
            "B": [[[0.1, 0.9], [0.0, 0.05], [0.9, 0.05]], [[0.9, 0.05], [0.1, 0.9], [0.0, 0.05]], [[0.0, 0.05], [0.9, 0.05], [0.1, 0.9]]],
            "C": [[0.0, 0.0, 1.0]],
            "D": [[1.0, 0.0, 0.0]],
            "dirichlet_A": [[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]],
            "inference_iterations": 40,
            "num_actions": 2,
            "num_hidden_states": 3,
            "num_obs": 3,
            "num_timesteps": 15,
        }

# MODEL_DATA: {"model_name":"Dirichlet Likelihood Learning Agent","annotation":"This model describes a discrete POMDP agent that learns its observation model:\n\n- 3 hidden states, 3 observation outcomes, 2 actions (cycle, stay).\n- The likelihood matrix A is NOT fixed: it is a latent DirichletCollection\n  variable with prior pseudo-counts declared in dirichlet_A.\n- The A values under InitialParameterization are the GROUND-TRUTH likelihood\n  used by the environment to simulate observations; the agent never sees them\n  directly and must recover them in q(A).\n- The Dirichlet prior is identity-biased (diagonal 3.0, off-diagonal 1.0):\n  the agent starts believing observations weakly track states. A fully\n  uniform prior leaves the column-permutation symmetry unbroken and\n  variational inference converges to a label-switched optimum.\n- Transitions B are near-deterministic and known, so states are\n  well-determined by actions and likelihood learning is well-conditioned.\n- Inference: structured VMP with mean-field cut q(s, A) = q(s)q(A),\n  q(A) initialized at the prior counts, q(s) initialized uniform.","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[3,3]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[3,3,2]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[3]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[3]},{"name":"dirichlet_A","var_type":"hidden_state","data_type":"float","dimensions":[3,3]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[3,1]},{"name":"\u03c0","var_type":"policy","data_type":"float","dimensions":[2]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"annotation":null,"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s"],"target_variables":["s_prime"],"connection_type":"directed"},{"annotation":null,"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s"],"target_variables":["B"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C"],"target_variables":["G"],"connection_type":"directed"},{"annotation":null,"source_variables":["G"],"target_variables":["\u03c0"],"connection_type":"directed"},{"annotation":null,"source_variables":["\u03c0"],"target_variables":["u"],"connection_type":"directed"},{"annotation":null,"source_variables":["B"],"target_variables":["u"],"connection_type":"directed"},{"annotation":null,"source_variables":["u"],"target_variables":["s_prime"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[[0.85,0.05,0.1],[0.1,0.9,0.05],[0.05,0.05,0.85]],"param_type":"constant"},{"name":"B","value":[[[0.1,0.9],[0.0,0.05],[0.9,0.05]],[[0.9,0.05],[0.1,0.9],[0.0,0.05]],[[0.0,0.05],[0.9,0.05],[0.1,0.9]]],"param_type":"constant"},{"name":"C","value":[[0.0,0.0,1.0]],"param_type":"constant"},{"name":"D","value":[[1.0,0.0,0.0]],"param_type":"constant"},{"name":"dirichlet_A","value":[[3.0,1.0,1.0],[1.0,3.0,1.0],[1.0,1.0,3.0]],"param_type":"constant"},{"name":"num_hidden_states","value":3,"param_type":"constant"},{"name":"num_obs","value":3,"param_type":"constant"},{"name":"num_actions","value":2,"param_type":"constant"},{"name":"num_timesteps","value":15,"param_type":"constant"},{"name":"inference_iterations","value":40,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"dirichlet_A","ontology_term":"LikelihoodMatrixConcentrationParameters","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"\u03c0","ontology_term":"PolicyVector","description":null},{"variable_name":"u","ontology_term":"Action","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
