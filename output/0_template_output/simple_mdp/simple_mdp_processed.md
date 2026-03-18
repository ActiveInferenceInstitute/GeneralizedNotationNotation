
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/discrete/simple_mdp.md
# Processed on: 2026-03-18T09:18:36.302026
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Simple MDP (Fully Observable)

# GNN Version: 1.0

# A Markov Decision Process with full observability (identity observation model)

# This model tests the MDP special case where A is identity — agent always knows its state

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1

## ModelName

Simple MDP Agent

## ModelAnnotation

This model describes a fully observable Markov Decision Process (MDP):

- 4 hidden states representing grid positions (corners of a 2x2 grid).
- Observations are identical to states (A = identity matrix).
- 4 actions: stay, move-north, move-south, move-east.
- Preferences strongly favor state/observation 3 (goal location).
- Tests the degenerate POMDP case where partial observability is absent.

## StateSpaceBlock

# Likelihood matrix: A[observation_outcomes, hidden_states] = Identity

A[4,4,type=float]    # Observation model: identity (fully observable)

# Transition matrix: B[states_next, states_previous, actions]

B[4,4,4,type=float]  # State transitions given state and action

# Preference vector: C[observation_outcomes]

C[4,type=float]      # Log-preferences over observations

# Prior vector: D[states]

D[4,type=float]      # Prior over initial hidden states

# Hidden State

s[4,1,type=float]    # Current hidden state distribution
s_prime[4,1,type=float] # Next hidden state distribution

# Observation

o[4,1,type=int]      # Current observation (same as state in MDP)

# Policy and Control

π[4,type=float]      # Policy (distribution over actions)
u[1,type=int]        # Action taken
G[π,type=float]      # Expected Free Energy (per policy)

# Time

t[1,type=int]        # Discrete time step

## Connections

D>s
s-A
s>s_prime
A-o
s-B
C>G
G>π
π>u
B>u
u>s_prime

## InitialParameterization

# A: 4x4 Identity — each state maps deterministically to its own observation

A={
  (1.0, 0.0, 0.0, 0.0),
  (0.0, 1.0, 0.0, 0.0),
  (0.0, 0.0, 1.0, 0.0),
  (0.0, 0.0, 0.0, 1.0)
}

# B: 4 actions. Action 0=stay, 1=go-to-1, 2=go-to-2, 3=go-to-3

B={
  ( (0.9, 0.1, 0.0, 0.0), (0.1, 0.9, 0.0, 0.0), (0.0, 0.0, 0.9, 0.1), (0.0, 0.0, 0.1, 0.9) ),
  ( (0.1, 0.9, 0.0, 0.0), (0.9, 0.1, 0.0, 0.0), (0.0, 0.0, 0.1, 0.9), (0.0, 0.0, 0.9, 0.1) ),
  ( (0.0, 0.0, 0.9, 0.1), (0.0, 0.0, 0.1, 0.9), (0.9, 0.1, 0.0, 0.0), (0.1, 0.9, 0.0, 0.0) ),
  ( (0.0, 0.0, 0.1, 0.9), (0.0, 0.0, 0.9, 0.1), (0.1, 0.9, 0.0, 0.0), (0.9, 0.1, 0.0, 0.0) )
}

# C: Strong preference for observation/state 3 (goal)

C={(0.0, 0.0, 0.0, 3.0)}

# D: Uniform prior — agent starts uncertain about position

D={(0.25, 0.25, 0.25, 0.25)}

## Equations

# Standard Active Inference update equations

# State inference: qs = softmax(ln(A[o,:]) + ln(B[s_prev] @ pi))

# Policy inference: G(pi) = sum_t EFE(pi,t)

# Action selection: u ~ Categorical(softmax(-G))

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=Unbounded

## ActInfOntologyAnnotation

A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
G=ExpectedFreeEnergy
s=HiddenState
s_prime=NextHiddenState
o=Observation
π=PolicyVector
u=Action
t=Time

## ModelParameters

num_hidden_states: 4
num_obs: 4
num_actions: 4
num_timesteps: 25

## Footer

Simple MDP Agent v1 - GNN Representation.
Fully observable (identity A). Tests MDP as degenerate POMDP.

## Signature

Cryptographic signature goes here

