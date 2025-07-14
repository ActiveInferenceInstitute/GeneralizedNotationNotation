# GNN Example: Classic Active Inference POMDP Agent
# Format: Markdown representation of a single-modality, single-factor POMDP agent in Active Inference format
# Version: 1.0
# This file is machine-readable and specifies a classic Active Inference agent for a discrete POMDP with one observation modality and one hidden state factor. The model is suitable for rendering into various simulation or inference backends.

## GNNSection
ClassicPOMDPAgent

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
- All parameterizations are explicit and suitable for translation to code or simulation in any Active Inference framework.

## StateSpaceBlock
# Likelihood matrix: A[observation_outcomes, hidden_states]
A[3,3,type=float]   # Likelihood mapping hidden states to observations

# Transition matrix: B[states_next, states_previous, actions]
B[3,3,3,type=float]   # State transitions given previous state and action

# Preference vector: C[observation_outcomes]
C[3,type=float]       # Log-preferences over observations

# Prior vector: D[states]
D[3,type=float]       # Prior over initial hidden states

# Hidden State
s[3,1,type=float]     # Current hidden state distribution
s_prime[3,1,type=float] # Next hidden state distribution

# Observation
o[3,1,type=float]     # Current observation

# Policy and Control
π[3,type=float]       # Policy (distribution over actions)
u[1,type=int]         # Action taken
G[1,type=float]       # Expected Free Energy (scalar or per policy)
t[1,type=int]         # Time step

## Connections
D-s
s-A
A-o
(s,u)-B
B-s_prime
C>G
G>π
π-u
G=ExpectedFreeEnergy
t=Time

## InitialParameterization
# A: 3 observations x 3 hidden states. Identity mapping (each state deterministically produces a unique observation).
A={
  (1.0, 0.0, 0.0),  # obs=0; state=0,1,2
  (0.0, 1.0, 0.0),  # obs=1
  (0.0, 0.0, 1.0)   # obs=2
}

# B: 3 states x 3 previous states x 3 actions. Each action deterministically moves to a state.
B={
  ( (1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0) ), # s_next=0; actions 0,1,2
  ( (0.0,1.0,0.0), (1.0,0.0,0.0), (0.0,0.0,1.0) ), # s_next=1
  ( (0.0,0.0,1.0), (0.0,1.0,0.0), (1.0,0.0,0.0) )  # s_next=2
}

# C: 3 observations. Preference for observing state 2.
C={(0.0, 0.0, 1.0)}

# D: 3 states. Uniform prior.
D={(0.33333, 0.33333, 0.33333)}

## Equations
# Standard Active Inference update equations for POMDPs:
# - State inference: qs = infer_states(o)
# - Policy evaluation: q_pi, efe = infer_policies()
# - Action selection: action = sample_action()

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=Unbounded # The agent is defined for an unbounded time horizon; simulation runs may specify a finite horizon.

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
s=HiddenState
s_prime=NextHiddenState
o=Observation
π=PolicyVector # Distribution over actions
u=Action       # Chosen action
G=ExpectedFreeEnergy

## ModelParameters
num_hidden_states: 3  # s[3]
num_obs: 3           # o[3]
num_actions: 3       # B actions_dim=3 (controlled by π)

## Footer
Classic Active Inference POMDP Agent v1 - GNN Representation

## Signature
NA 