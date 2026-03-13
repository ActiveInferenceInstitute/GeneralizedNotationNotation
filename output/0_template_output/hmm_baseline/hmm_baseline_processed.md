
# Processed by GNN Pipeline Template
# Original file: /Users/4d/Documents/GitHub/generalizednotationnotation/input/gnn_files/discrete/hmm_baseline.md
# Processed on: 2026-03-13T11:27:59.214994
# Options: {'verbose': True, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Hidden Markov Model Baseline
# GNN Version: 1.0
# Simple HMM for comparison with Active Inference POMDP variants.

## GNNSection
HiddenMarkovModel

## GNNVersionAndFlags
GNN v1

## ModelName
Hidden Markov Model Baseline

## ModelAnnotation
A standard discrete Hidden Markov Model with:
- 4 hidden states with Markovian dynamics
- 6 observation symbols
- Fixed transition and emission matrices
- No action selection (passive inference only)
- Suitable for sequence modeling and state estimation tasks

## StateSpaceBlock
# HMM parameters
A[6,4,type=float]      # Emission matrix: observations x hidden states
B[4,4,type=float]      # Transition matrix (no action dependence)
D[4,type=float]        # Initial state distribution (prior)

# State and observations
s[4,1,type=float]      # Hidden state belief (posterior)
s_prime[4,1,type=float] # Next hidden state
o[6,1,type=int]        # Current observation (one-hot)

# Inference quantities
F[1,type=float]        # Variational Free Energy (negative ELBO)
alpha[4,1,type=float]  # Forward variable (belief propagation)
beta[4,1,type=float]   # Backward variable

# Time
t[1,type=int]          # Discrete time step

## Connections
D>s
s-A
s>s_prime
A-o
B>s_prime
s-B
s-F
o-F
s-alpha
o-alpha
alpha>s_prime
s_prime-beta

## InitialParameterization
# Emission: 6 observations, 4 states
A={
  (0.7, 0.1, 0.1, 0.1),
  (0.1, 0.7, 0.1, 0.1),
  (0.1, 0.1, 0.7, 0.1),
  (0.1, 0.1, 0.1, 0.7),
  (0.1, 0.1, 0.4, 0.4),
  (0.4, 0.4, 0.1, 0.1)
}

# Transition: 4x4 column stochastic
B={
  (0.7, 0.1, 0.1, 0.1),
  (0.1, 0.7, 0.2, 0.1),
  (0.1, 0.1, 0.6, 0.2),
  (0.1, 0.1, 0.1, 0.6)
}

D={(0.25, 0.25, 0.25, 0.25)}

## Equations
# Forward algorithm: alpha_t(s) = sum_{s'} P(o_t|s) * P(s|s') * alpha_{t-1}(s')
# Backward algorithm: beta_t(s) = sum_{s'} P(o_{t+1}|s') * B(s'|s) * beta_{t+1}(s')
# State posterior: Q(s_t) = alpha_t(s) * beta_t(s) / Z
# Free Energy: F = -log P(o_{1:T})

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
F=VariationalFreeEnergy
alpha=ForwardVariable
beta=BackwardVariable
t=Time

## ModelParameters
num_hidden_states: 4
num_observations: 6
num_timesteps: 50

## Footer
Hidden Markov Model Baseline v1 - GNN Representation.
No action selection — passive observer only.
Use as baseline comparison for POMDP Active Inference variants.

## Signature
Cryptographic signature goes here

