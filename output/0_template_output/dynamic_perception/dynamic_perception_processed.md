
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/basics/dynamic_perception.md
# Processed on: 2026-04-15T12:24:33.758662
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Dynamic Perception Model

# GNN Version: 1.0

# Passive temporal inference without action selection

## GNNSection

ActiveInferencePerception

## GNNVersionAndFlags

GNN v1

## ModelName

Dynamic Perception Model

## ModelAnnotation

A dynamic perception model extending the static model with temporal dynamics:

- 2 hidden states evolving over discrete time via transition matrix B
- 2 observations generated from states via recognition matrix A
- Prior D constrains the initial hidden state
- No action selection — the agent passively observes a changing world
- Demonstrates belief updating (state inference) across time steps
- Suitable for tracking hidden sources from noisy observations

## StateSpaceBlock

# Generative model parameters

A[2,2,type=float]        # Recognition matrix: P(observation | hidden state)
B[2,2,type=float]        # Transition matrix: P(s_{t+1} | s_t) — no action dependence
D[2,1,type=float]        # Prior over initial hidden states

# Hidden states

s_t[2,1,type=float]      # Hidden state belief at time t
s_prime[2,1,type=float]  # Hidden state belief at time t+1

# Observation

o_t[2,1,type=int]        # Observation at time t

# Inference quantities

F[1,type=float]          # Variational Free Energy (negative ELBO)

# Time

t[1,type=int]            # Discrete time index

## Connections

D>s_t
s_t-A
A-o_t
s_t-B
B>s_prime
s_t-F
o_t-F

## InitialParameterization

# Near-identity observation mapping

A={
  (0.9, 0.1),
  (0.2, 0.8)
}

# Mildly persistent transitions (states tend to persist)

B={
  (0.7, 0.3),
  (0.3, 0.7)
}

# Uniform prior

D={(0.5, 0.5)}

## Equations

# State inference at first timestep

# Q(s_{tau=1}) = softmax((1/2)(ln(D) + ln(B^T *s_{tau+1}) + ln(A^T* o_tau)))

# State inference at subsequent timesteps

# Q(s_{tau>1}) = softmax((1/2)(ln(B *s_{tau-1}) + ln(B^T* s_{tau+1}) + ln(A^T * o_tau)))

# Variational Free Energy

# F = D_KL[Q(s)||D] - E_Q[ln P(o|s)]

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=10

## ActInfOntologyAnnotation

A=RecognitionMatrix
B=TransitionMatrix
D=Prior
s_t=HiddenState
s_prime=NextHiddenState
o_t=Observation
F=VariationalFreeEnergy
t=Time

## ModelParameters

num_hidden_states: 2
num_obs: 2
num_timesteps: 10

## Footer

Dynamic Perception Model v1 - GNN Representation.
Passive observer — no actions, no policies.
Demonstrates temporal belief updating via variational inference.

## Signature

Cryptographic signature goes here

