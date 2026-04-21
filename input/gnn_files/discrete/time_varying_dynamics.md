# GNN Example: Time-Varying Transition Dynamics

# GNN Version: 1.0

# Demonstrates a model whose transition matrix B changes over time
# (e.g., a non-stationary environment where dynamics evolve).

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1

## ModelName

Time-Varying Transition Dynamics Agent

## ModelAnnotation

A POMDP agent operating in a non-stationary environment. The key feature
is that the transition matrix `B` is indexed by time (`B_t`), capturing
dynamics that evolve across the planning horizon — e.g., shifting wind
patterns for a sailing agent, or changing opponent strategy in a
sequential game.

- 3 hidden states, 3 observations, 2 actions
- B_t: 3D transition tensor per timestep (shape: next_state × current_state × action)
- Agent must adapt belief updates each step to the current B_t
- Exercises time-varying matrix handling in renderers

This sample pushes the language extensions around time-indexed tensors
and tests downstream code generation when matrix literals are
timestep-dependent.

## StateSpaceBlock

# Generative model with time-varying dynamics

A[3,3,type=float]         # Observation model (time-invariant)
B_t[3,3,2,type=float]     # Transition model, indexed by time t
C[3,1,type=float]         # Preference vector
D[3,1,type=float]         # Initial state prior

# Hidden state trajectory

s_t[3,1,type=float]       # Hidden state at time t
s_t+1[3,1,type=float]     # Hidden state at time t+1

# Observation and action

o_t[3,1,type=int]         # Observation at time t
u_t[2,1,type=int]         # Action at time t

## Connections

D>s_t
(s_t, u_t)>B_t
B_t>s_t+1
s_t-A
A-o_t
C-o_t

## InitialParameterization

# Time-invariant observation model (identity-like, mild noise)

A={
  (0.85, 0.10, 0.05),
  (0.10, 0.80, 0.10),
  (0.05, 0.10, 0.85)
}

# B_t for t=0: exploration-biased dynamics

B_t={
  (
    (0.6, 0.3, 0.1),
    (0.3, 0.6, 0.1),
    (0.1, 0.1, 0.8)
  ),
  (
    (0.1, 0.1, 0.8),
    (0.1, 0.6, 0.3),
    (0.8, 0.3, 0.1)
  )
}

# Preferences: goal state is state 2

C={(0.0, 0.0, 1.0)}

# Uniform prior

D={(0.33, 0.33, 0.34)}

## Equations

# Belief update under time-varying dynamics:
# Q(s_{t+1}) = softmax(ln(B_t[:, :, u_t] * Q(s_t)) + ln(A^T * o_{t+1}))
#
# Policy posterior accounts for B_t's evolution:
# Q(π) = softmax(-G(π))
# G(π) = Σ_t [KL(Q(s_t|π) || P(s_t|B_{t-1})) - E[ln(P(o_t|C))]]

## Time

Dynamic
DiscreteTime=t
ModelTimeHorizon=10

## ActInfOntologyAnnotation

A=LikelihoodMatrix
B_t=TimeVaryingTransitionMatrix
C=PreferenceVector
D=Prior
s_t=HiddenState
o_t=Observation
u_t=Action

## ModelParameters

num_hidden_states: 3
num_obs: 3
num_actions: 2
num_timesteps: 10

## Footer

Time-Varying Transition Dynamics Agent v1.0 — demonstrates non-stationary
B_t tensor in a 3-state POMDP. Tests time-indexed matrix handling.

## Signature

Cryptographic signature goes here
