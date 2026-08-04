# GNN Example: Stochastic Continuous Dynamics

# GNN Version: 1.0

# Demonstrates explicit stochastic noise terms in the Equations — the model
# is a stochastic dynamical system. Rendered as a discrete POMDP equivalent
# (with noise handled via the observation model) for POMDP/pymdp and general
# framework compatibility.

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1

## ModelName

Stochastic Continuous Dynamics Agent

## ModelAnnotation

A continuous-state Active Inference agent whose dynamics include explicit
process and observation noise, represented as a discrete POMDP equivalent:

- 3 discrete hidden states (e.g., position + velocity regimes)
- 2 observation outcomes (noisy position readouts)
- 2 actions (thrust +, thrust -)
- Observation noise is captured in the likelihood A (the original SDE noise
  formulation is preserved in the Equations prose)

## StateSpaceBlock

# Likelihood matrix: A[observation_outcomes, hidden_states]

A[2,3,type=float]    # Observation model (noisy position readout)

# Transition matrix: B[states_next, states_previous, actions]

B[3,3,2,type=float]  # State transitions given state and action

# Preference vector: C[observation_outcomes]

C[2,type=float]      # Log-preferences over observations

# Prior vector: D[states]

D[3,type=float]      # Prior over initial hidden states

# Hidden state

s[3,1,type=float]    # Current hidden state distribution

# Observation

o[2,1,type=int]      # Current observation (one-hot encoded)

# Action

u[1,type=int]        # Action taken (0=thrust +, 1=thrust -)

# Noise / precision parameters (scalars)

gamma_state[1,type=float] # Process noise precision
gamma_obs[1,type=float]   # Observation noise precision

# Time

t[1,type=float]      # Discrete time step

## Connections

D>s
s-A
A-o
s-B
B>u
u>s
gamma_obs-A
gamma_state-B

## InitialParameterization

# A: noisy position readout — states 0 and 1 map to their own readings; state 2
# is ambiguous (half/half), consistent with the A[2,3] declaration (2 obs × 3 states)

A={
  (0.9, 0.1, 0.5),
  (0.1, 0.9, 0.5)
}

# B: 2 actions. Action 0=thrust +, 1=thrust - (state shifts with noise)

B={
  ( (0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9) ),
  ( (0.05, 0.9, 0.05), (0.9, 0.05, 0.05), (0.05, 0.05, 0.9) )
}

# C: neutral preferences

C={(0.0, 0.0)}

# D: start at state 0

D={(0.8, 0.1, 0.1)}

# Precisions (high precision = low noise)

gamma_state={(10.0)}
gamma_obs={(5.0)}

## Equations

# Original SDE formulation (preserved):
# dx/dt = F * x + G * u + ε_state,  ε_state ~ N(0, γ_state^-1 * I)
# o = H * x + ε_obs,                ε_obs ~ N(0, γ_obs^-1 * I)
# F = 0.5 * γ_state * ||x_{t+1} - F*x_t - G*u_t||^2 + 0.5 * γ_obs * ||o - H*x||^2
#
# Discrete POMDP equivalent used for rendering/execution:
# State inference: qs = softmax(ln(A[o,:]) + ln(B[s_prev] @ pi))
# Action selection: u ~ Categorical(softmax(-G))

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=15

## ActInfOntologyAnnotation

A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
s=HiddenState
o=Observation
u=Action
gamma_state=ProcessNoisePrecision
gamma_obs=ObservationNoisePrecision
t=Time

## ModelParameters

num_hidden_states: 3
num_obs: 2
num_actions: 2
num_timesteps: 15

## Footer

Stochastic Continuous Dynamics Agent v1.0 — discrete POMDP equivalent of the
linear-Gaussian SDE model with explicit process and observation noise.

## Signature

Cryptographic signature goes here
