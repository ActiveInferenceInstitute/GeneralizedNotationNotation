# EXPLAIN_MODEL

Here is a detailed description of the GNN (Generalized Notation Notation) specification:

## GNNSection

ActiveInferencePerception

# GNNExample: Dynamic Perception Model

GNNv1

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

s_f0={
  (0.9, 0.1),
  (0.2, 0.8)
}

# Hidden state belief at time t

o_t={
  (0.7, 0.3),
  (0.3, 0.7)
}


## Connection

D>s_f1=d(B^T * s_{tau+1}) + d(A^T* o_tau) = -ln(B *S_(t-1)*o_t) + ln(A * S_(t-1)*O_t)


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


--- INJECTED ACTIVE INFERENCE ONTOLOGY META ---
{
  "processed_files": 10,
  "reports": [
    "output/10_ontology_output/