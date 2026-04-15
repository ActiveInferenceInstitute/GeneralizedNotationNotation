
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/discrete/markov_chain.md
# Processed on: 2026-04-15T12:24:33.760856
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Markov Chain (Passive Dynamics)

# GNN Version: 1.0

# A simple discrete-time Markov Chain with no actions and no observation model

# Tests the simplest possible dynamical system: just state transitions

## GNNSection

HiddenMarkovModel

## GNNVersionAndFlags

GNN v1

## ModelName

Simple Markov Chain

## ModelAnnotation

This model describes a minimal discrete-time Markov Chain:

- 3 states representing weather (sunny, cloudy, rainy).
- No actions — the system evolves passively.
- Observations = states directly (identity mapping for monitoring).
- Stationary transition matrix with realistic weather dynamics.
- Tests the simplest model structure: passive state evolution with no control.

## StateSpaceBlock

# Emission/Observation matrix: A[observations, states]

A[3,3,type=float]    # Observation model (identity for direct monitoring)

# Transition matrix: B[states_next, states_previous] (no action dimension)

B[3,3,type=float]    # Markov transition matrix

# Prior vector: D[states]

D[3,type=float]      # Prior over initial states

# Hidden State

s[3,1,type=float]    # Current state distribution
s_prime[3,1,type=float] # Next state distribution

# Observation

o[3,1,type=int]      # Current observation

# Time

t[1,type=int]        # Discrete time step

## Connections

D>s
s-A
A-o
s>s_prime
B>s_prime
s-B

## InitialParameterization

# A: 3x3 Identity — states are directly observed

A={
  (1.0, 0.0, 0.0),
  (0.0, 1.0, 0.0),
  (0.0, 0.0, 1.0)
}

# B: 3x3 weather transition matrix (no action dimension)

# Sunny tends to stay sunny. Cloudy can go anywhere. Rainy tends to stay rainy

B={
  (0.7, 0.3, 0.1),
  (0.2, 0.4, 0.3),
  (0.1, 0.3, 0.6)
}

# D: Start slightly favoring sunny

D={(0.5, 0.3, 0.2)}

## Equations

# Markov Chain evolution: s(t+1) ~ Categorical(B @ s(t))

# Observation: o(t) = s(t) (identity)

# No inference or action selection needed — passive system

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=Unbounded

## ActInfOntologyAnnotation

A=EmissionMatrix
B=TransitionMatrix
D=InitialStateDistribution
s=HiddenState
s_prime=NextHiddenState
o=Observation
t=Time

## ModelParameters

num_hidden_states: 3
num_obs: 3
num_actions: 1
num_timesteps: 40

## Footer

Simple Markov Chain v1 - GNN Representation.
Passive dynamics, no actions. Tests minimal Markov model structure.

## Signature

Cryptographic signature goes here

