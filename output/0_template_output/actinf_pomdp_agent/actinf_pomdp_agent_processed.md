
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/actinf_pomdp_agent.md
# Processed on: 2026-01-07T06:26:04.436590
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Active Inference POMDP Agent
# GNN Version: 1.0
# This file is machine-readable and specifies a classic Active Inference agent for a discrete POMDP with one observation modality and one hidden state factor. The model is suitable for rendering into various simulation or inference backends.

## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Classic Active Inference POMDP Agent v1

## ModelAnnotation
This model describes a classic Active Inference agent for a discrete POMDP:
- One observation modality ("state_observation") with 3 possible outcomes.
- One hidden state factor ("location") with 3 possible states.
- The hidden state is fully controllable via 3 discrete actions.
- The agent's preferences are encoded as log-probabilities over observations.
- The agent has an initial policy prior (habit) encoded as log-probabilities over actions.

## StateSpaceBlock
# Likelihood matrix: A[observation_outcomes, hidden_states]
A[3,3,type=float]   # Likelihood mapping hidden states to observations

# Transition matrix: B[states_next, states_previous, actions]
B[3,3,3,type=float]   # State transitions given previous state and action

# Preference vector: C[observation_outcomes]
C[3,type=float]       # Log-preferences over observations

# Prior vector: D[states]
D[3,type=float]       # Prior over initial hidden states

# Habit vector: E[actions]
E[3,type=float]       # Initial policy prior (habit) over actions

# Hidden State
s[3,1,type=float]     # Current hidden state distribution
s_prime[3,1,type=float] # Next hidden state distribution
F[π,type=float]       # Variational Free Energy for belief updating from observations

# Observation
o[3,1,type=int]     # Current observation (integer index)

# Policy and Control
π[3,type=float]       # Policy (distribution over actions), no planning
u[1,type=int]         # Action taken
G[π,type=float]       # Expected Free Energy (per policy)

# Time
t[1,type=int]         # Discrete time step

## Connections
D>s
s-A
s>s_prime
A-o
s-B
C>G
E>π
G>π
π>u
B>u
u>s_prime

## InitialParameterization
# A: 3 observations x 3 hidden states. Identity mapping (each state deterministically produces a unique observation). Rows are observations, columns are hidden states.
A={
  (0.9, 0.05, 0.05),
  (0.05, 0.9, 0.05),
  (0.05, 0.05, 0.9)
}

# B: 3 states x 3 previous states x 3 actions. Each action deterministically moves to a state. For each slice, rows are previous states, columns are next states. Each slice is a transition matrix corresponding to a different action selection.
B={
  ( (1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0) ),
  ( (0.0,1.0,0.0), (1.0,0.0,0.0), (0.0,0.0,1.0) ),
  ( (0.0,0.0,1.0), (0.0,1.0,0.0), (1.0,0.0,0.0) )
}

# C: 3 observations. Preference in terms of log-probabilities over observations.
C={(0.1, 0.1, 1.0)}

# D: 3 states. Uniform prior over hidden states. Rows are hidden states, columns are prior probabilities.
D={(0.33333, 0.33333, 0.33333)}

# E: 3 actions. Uniform habit used as initial policy prior.
E={(0.33333, 0.33333, 0.33333)}

## Equations
# Standard Active Inference update equations for POMDPs:
# - State inference using Variational Free Energy with infer_states()
# - Policy inference using Expected Free Energy = with infer_policies()
# - Action selection from policy posterior: action = sample_action()
# - Belief updating using Variational Free Energy with update_beliefs()

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=Unbounded # The agent is defined for an unbounded time horizon; simulation runs may specify a finite horizon.

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
E=Habit
F=VariationalFreeEnergy
G=ExpectedFreeEnergy
s=HiddenState
s_prime=NextHiddenState
o=Observation
π=PolicyVector # Distribution over actions
u=Action       # Chosen action
t=Time

## ModelParameters
num_hidden_states: 3  # s[3]
num_obs: 3           # o[3]
num_actions: 3       # B actions_dim=3 (controlled by π)

## Footer
Active Inference POMDP Agent v1 - GNN Representation. 
Currently there is a planning horizon of 1 step (no deep planning), no precision modulation, no hierarchical nesting. 

## Signature
Cryptographic signature goes here 
