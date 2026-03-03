
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/basics/static_perception.md
# Processed on: 2026-03-03T08:15:07.769427
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Static Perception Model

# GNN Version: 1.0

# Simplest possible GNN model: perception without temporal dynamics

## GNNSection

ActiveInferencePerception

## GNNVersionAndFlags

GNN v1

## ModelName

Static Perception Model

## ModelAnnotation

The simplest Active Inference model demonstrating pure perception:

- 2 hidden states mapped to 2 observations via a recognition matrix A
- Prior D encodes initial beliefs over hidden states
- No temporal dynamics — single-shot inference
- Demonstrates the core observation model: P(o|s) = A
- Suitable as a minimal baseline and for testing perception-only inference

## StateSpaceBlock

# Generative model parameters

A[2,2,type=float]    # Recognition/likelihood matrix: P(observation | hidden state)
D[2,1,type=float]    # Prior belief over hidden states

# Hidden state

s[2,1,type=float]    # Hidden state (posterior belief)

# Observation

o[2,1,type=int]      # Observation (one-hot encoded)

## Connections

D>s
s-A
A-o

## InitialParameterization

# Near-identity observation mapping with mild noise

A={
  (0.9, 0.1),
  (0.2, 0.8)
}

# Uniform prior over hidden states

D={(0.5, 0.5)}

## Equations

# Bayesian perception via softmax

# Q(s) = softmax(ln(D) + ln(A^T * o))

# No temporal or action components — pure state estimation

## Time

Static

## ActInfOntologyAnnotation

A=RecognitionMatrix
D=Prior
s=HiddenState
o=Observation

## ModelParameters

num_hidden_states: 2
num_obs: 2

## Footer

Static Perception Model v1 - GNN Representation.
Simplest possible Active Inference model.
No temporal dynamics, no action — perception only.

## Signature

Cryptographic signature goes here

