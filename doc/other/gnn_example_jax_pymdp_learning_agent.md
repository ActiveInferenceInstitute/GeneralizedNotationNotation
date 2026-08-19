# GNN Example: JAX PyMDP Learning Agent in a Grid World
# Format: Markdown representation of a JAX PyMDP learning agent model.
# Version: 1.0
# This file is machine-readable and based on a JAX learning example.

## GNNSection
JaxPyMDPLearningAgentGridWorld

## GNNVersionAndFlags
GNN v1

## ModelName
JAX PyMDP Grid World Learning Agent v1

## ModelAnnotation
This model represents a PyMDP agent implemented in JAX, learning its transition model (B) and initial state distribution (D) within a 7x7 grid world.
- Observation modality: Fully observable location (49 outcomes).
- Hidden state factor: Location on the grid (49 states).
- Control: 5 actions (N, E, S, W, Stay) affecting location.
- Learning: `learn_B=True`, `learn_D=True`, `learn_A=False`.
- Priors: `pB` and `pD` are Dirichlet distributions.
- Agent behavior: Driven by minimizing Expected Free Energy (EFE), incorporating state and parameter information gain.

## StateSpaceBlock
# Grid dimensions (metadata, used to define num_states etc.)
# num_grid_rows: 7
# num_grid_cols: 7
# num_actions: 5 (N,E,S,W,Stay)

# Hidden States (s_factorIndex[num_states_factor, 1, type=dataType])
s_f0[49,1,type=int]   # Hidden State Factor 0: Location (0-48)

# Observations (o_modalityIndex[num_outcomes_modality, 1, type=dataType])
o_m0[49,1,type=int]   # Observation Modality 0: Observed Location (0-48)

# Control Factors / Policies (pi_controlFactorIndex[num_actions_factor, type=dataType])
pi_c0[5,type=float]  # Policy for Control Factor 0: Movement Actions

# Actions (u_controlFactorIndex[1, type=dataType]) - chosen actions
u_c0[1,type=int]     # Chosen action for Movement

# Likelihood Mapping (A_modalityIndex[outcomes, factor0_states, ..., type=dataType])
A_m0[49,49,type=float] # Likelihood: o_m0 outcomes given s_f0 states

# Transition Dynamics (B_factorIndex[next_states, prev_states, control0_actions, ..., type=dataType])
B_f0[49,49,5,type=float] # Location (s_f0) transitions, depends on s_f0 and u_c0

# Preferences (C_modalityIndex[outcomes, type=dataType]) - Log preferences
C_m0[49,type=float]   # Preferences for o_m0 outcomes

# Priors over Initial Hidden States (D_factorIndex[num_states_factor, type=dataType])
D_f0[49,type=float]   # Prior belief for s_f0

# Dirichlet Prior Concentration Parameters (for learnable matrices)
pA_m0_conc[49,49,type=float] # Dirichlet concentrations for A_m0 (if learn_A=True)
pB_f0_conc[49,49,5,type=float]# Dirichlet concentrations for B_f0
pD_f0_conc[49,type=float]    # Dirichlet concentrations for D_f0

# Expected Free Energy
G[1,type=float]      # Overall Expected Free Energy of chosen policy/actions

# Time
t[1,type=int]

## Connections
(D_f0) -> (s_f0)
(s_f0) -> (A_m0)
(A_m0) -> (o_m0)
(s_f0, u_c0) -> (B_f0)
(B_f0) -> s_f0_next # Implied next state for s_f0
(C_m0, s_f0, A_m0) > G # Simplified EFE dependency, involving current beliefs and parameters
G > (pi_c0)
(pi_c0) -> u_c0

# Connections for learning (priors influence matrix updates)
(pB_f0_conc) -> B_f0 # Denotes B_f0 is learnable using this prior
(pD_f0_conc) -> D_f0 # Denotes D_f0 is learnable using this prior
# (pA_m0_conc) -> A_m0 # If learn_A were True

## InitialParameterization
# Note: `n_batches` (e.g., 20) from example implies multiple agents; GNN typically defines one.
#       Parameters below are for a single agent instance.

# A_m0: Identity matrix (fully observable environment)
# A_m0_values: "np.eye(49)"
A_m0={
  # Represents np.eye(49). Too large to list explicitly.
  # Example for 3x3: ((1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0))
  "description": "Identity matrix of shape (49, 49), representing fully observable states."
}

# B_f0: Initial values (mean of the Dirichlet prior pB_f0_conc). Uniform.
# B_f0_initial_values: "np.ones((49,49,5)) / 49.0"
B_f0={
  "description": "Uniform transition probabilities. Each B_f0[s_next, s_prev, action] = 1/49. Shape (49,49,5)."
}

# C_m0: Preferences (vector of 49 values). Zero preferences in the learning example.
# C_m0_values: "np.zeros(49)"
C_m0={
  "description": "Zero vector of shape (49,), indicating no specific outcome preferences."
}

# D_f0: Initial belief over states (vector of 49 values). Uniform initial belief.
# D_f0_initial_values: "np.ones(49) / 49.0"
D_f0={
  "description": "Uniform prior belief over initial states. Each D_f0[state] = 1/49. Shape (49,)."
}

# pA_m0_conc: Dirichlet concentration parameters for A_m0.
# Not strictly used as learn_A=False, but for completeness if it were True.
# pA_m0_conc_values: "np.ones((49,49)) * (1.0/49.0)" (example of a weak uniform prior)
pA_m0_conc={
  "description": "Dirichlet concentration parameters for A_m0. All values e.g., 1/49. Shape (49,49). Relevant if learn_A=True."
}

# pB_f0_conc: Dirichlet concentration parameters for B_f0.
# From code: pB = [jnp.ones_like(B_true[0]) / num_states[0]]
# pB_f0_conc_values: "np.ones((49,49,5)) * (1.0/49.0)"
pB_f0_conc={
  "description": "Dirichlet concentration parameters for B_f0. All values are 1/49. Shape (49,49,5)."
}

# pD_f0_conc: Dirichlet concentration parameters for D_f0.
# Assuming default of 1.0 for each concentration parameter for a vague prior.
# pD_f0_conc_values: "np.ones(49) * 1.0"
pD_f0_conc={
  "description": "Dirichlet concentration parameters for D_f0. All values are 1.0. Shape (49,)."
}

## AgentHyperparameters
# These parameters are mostly for the JAX `AIFAgent`
learning_A_enabled: False
learning_B_enabled: True
learning_D_enabled: True
learning_E_enabled: False  # E is not used in this example

policy_length: 3
use_utility_in_efe: False
use_states_info_gain_in_efe: True
use_param_info_gain_in_efe: True

# `gamma` in JAX AIFAgent: precision of EFE distribution for policy selection
efe_policy_precision_gamma: 1.0

# `alpha` in JAX AIFAgent: precision parameter for updating Dirichlet posteriors
learning_dirichlet_update_precision_alpha: "Varies (e.g., 0.0, 0.2, ..., 0.8 in example script)"

# `kappa` in JAX AIFAgent: decay parameter for Dirichlet updates (not explicitly set, uses default)
learning_dirichlet_decay_kappa: None

action_selection_method: "stochastic" # Corresponds to `action_selection`
state_inference_algorithm: "ovf"     # Corresponds to `inference_algo`
state_inference_num_iter: 1          # Corresponds to `num_iter`
expect_onehot_observations: False    # Corresponds to `onehot_obs`

## SimulationParameters
# Parameters for the simulation environment and learning loop
num_grid_rows: 7
num_grid_columns: 7
num_actions_available: 5 # (N, E, S, W, Stay)

num_agents_batched: 20 # Number of agents run in parallel in the JAX example
timesteps_per_rollout: 50
num_learning_epochs: 40 # Number of blocks/updates
initial_agent_grid_coords_true_state: "(3,3)" # For environment initialization, maps to state_idx 24

## Equations
# Standard Active Inference equations for:
# 1. State inference: q(s_t) using variational message passing (e.g., OVF algorithm).
# 2. Policy evaluation: Expected Free Energy G(π) considering epistemic (information gain about states and parameters) and instrumental value (preferences C).
# 3. Action selection: Softmax over -G(π) scaled by `efe_policy_precision_gamma`.
# 4. Parameter learning: Bayesian updates of Dirichlet posteriors (pA, pB, pD) based on experiences, scaled by `learning_dirichlet_update_precision_alpha`.

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon: "Unbounded for agent definition; rollouts are for `timesteps_per_rollout`."

## ActInfOntologyAnnotation
s_f0=HiddenStateLocationGrid
o_m0=ObservationLocationGrid
pi_c0=PolicyMovementGridActions
u_c0=ActionMovementGrid
A_m0=LikelihoodMatrixLocationObsGivenState
B_f0=TransitionMatrixLocationStateGivenAction
C_m0=LogPreferenceLocationOutcomes
D_f0=PriorBeliefOverLocationInitialState
pA_m0_conc=DirichletPriorConcentrationLikelihoodA
pB_f0_conc=DirichletPriorConcentrationTransitionB
pD_f0_conc=DirichletPriorConcentrationInitialStateD
G=ExpectedFreeEnergyScalar
t=TimeStepInteger

# Agent Hyperparameters
learning_A_enabled=FlagLearnLikelihoodMatrix
learning_B_enabled=FlagLearnTransitionMatrix
learning_D_enabled=FlagLearnInitialStateDistribution
policy_length=PolicySearchHorizonDepth
use_utility_in_efe=FlagUseUtilityEFE
use_states_info_gain_in_efe=FlagUseStateInfoGainEFE
use_param_info_gain_in_efe=FlagUseParamInfoGainEFE
efe_policy_precision_gamma=GammaParameterEFEPolicySelection
learning_dirichlet_update_precision_alpha=AlphaParameterDirichletUpdatePrecision

## ModelParameters
num_hidden_states_factors: [49]      # Location
num_obs_modalities: [49]             # Observed Location
num_control_factors_actions: [5]     # N, E, S, W, Stay actions for Location factor
num_batched_agents_in_example: 20    # From JAX example setup

## Footer
JAX PyMDP Grid World Learning Agent v1 - GNN Representation.
This GNN describes an agent learning its model of the world (B and D matrices).
Initial matrix parameterizations are large and specified descriptively.
AgentHyperparameters are reflective of the JAX AIFAgent.

## Signature
Creator: AI Assistant for GNN
Date: 2024-07-26
Status: Example based on JAX PyMDP learning script.
Reference: User-provided Python script for JAX PyMDP learning.
Context: For comprehensive GNN specification of a learning PyMDP agent. 