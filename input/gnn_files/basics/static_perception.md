# GNN Example: Static Perception Model

# GNN Version: 1.0

# Simplest possible GNN model: perception with a minimal action/transition component
# (action/transition added for POMDP/pymdp rendering compatibility)

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1

## ModelName

Static Perception Model

## ModelAnnotation

The simplest Active Inference model demonstrating pure perception:

- 2 hidden states mapped to 2 observations via a recognition matrix A
- Prior D encodes initial beliefs over hidden states
- Minimal 2-action transition component B so the model is a complete POMDP
  (renderable and executable by pymdp and the general simulation frameworks)
- Suitable as a minimal baseline and for testing perception-only inference

## StateSpaceBlock

# Generative model parameters

A[2,2,type=float]    # Recognition/likelihood matrix: P(observation | hidden state)
B[2,2,2,type=float]  # Transition matrix: B[next_state, previous_state, actions]
C[2,type=float]      # Preference vector over observations
D[2,1,type=float]    # Prior belief over hidden states

# Hidden state

s[2,1,type=float]    # Hidden state (posterior belief)

# Observation

o[2,1,type=int]      # Observation (one-hot encoded)

# Action

u[1,type=int]        # Action taken

## Connections

D>s
s-A
A-o
s-B
B>u
u>s

## InitialParameterization

# Near-identity observation mapping with mild noise

A={
  (0.9, 0.1),
  (0.2, 0.8)
}

# Minimal transitions: action 0 = stay, action 1 = flip state. The transition tensor B is stored as (next_state, previous_state, action); per-action slices are column-stochastic: rows are next states, columns are previous states, and each column sums to 1 over next states.

B={
  ( (0.95, 0.05), (0.05, 0.95) ),
  ( (0.05, 0.95), (0.95, 0.05) )
}

# Neutral preference over observations

C={(0.0, 0.0)}

# Uniform prior over hidden states

D={(0.5, 0.5)}

## Equations

# Bayesian perception via softmax

# Q(s) = softmax(ln(D) + ln(A^T * o))

# Minimal temporal/action component: s_{t+1} = B[s_t, u_t]

## Time

Static

## ActInfOntologyAnnotation

A=RecognitionMatrix
B=TransitionMatrix
C=PreferenceVector
D=Prior
s=HiddenState
o=Observation
u=Action

## ModelParameters

num_hidden_states: 2
num_obs: 2
num_actions: 2
num_timesteps: 5

## Footer

Static Perception Model v1 - GNN Representation.
Simplest possible Active Inference model.
Minimal action/transition component added for POMDP/pymdp compatibility.

## Signature

Cryptographic signature goes here
