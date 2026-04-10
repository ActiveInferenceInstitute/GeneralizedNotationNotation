# GNN Example: bnlearn Causal Discovery Model

# GNN Version: 1.0

# Active Inference structural layout mapped for Bayesian Network causal discovery

## GNNSection

ActiveInferencePerception

## GNNVersionAndFlags

GNN v1

## ModelName

Bnlearn Causal Model

## ModelAnnotation

A Bayesian Network model mapping Active Inference structure:
- S: Hidden State
- A: Action
- S_prev: Previous State
- O: Observation

## StateSpaceBlock

# Generative model parameters
# Matrix mappings translated to CPTs natively in bnlearn generator
A[2,2,type=float]    # P(O | S)
B[2,2,2,type=float]  # P(S | S_prev, A)

# Dimensions
s[2,1,type=float]
o[2,1,type=int]
a[2,1,type=int]

## Connections

S_prev>S
A>S
S>O

## InitialParameterization

# Observation mapping
A={
  (0.9, 0.1),
  (0.1, 0.9)
}

# Uniform action distribution
D={(0.5, 0.5)}

## Equations

# Structure Learning
# DAG = make_DAG([('S_prev', 'S'), ('A', 'S'), ('S', 'O')])

## Time

Dynamic

## ActInfOntologyAnnotation

A=ObservationModel
B=TransitionModel
s=HiddenState
o=Observation
a=Action

## ModelParameters

num_timesteps: 30
num_hidden_states: 2
num_obs: 2
num_actions: 2

## Footer

Bnlearn Causal Discovery Model mapping POMDP structure to a Bayesian Network.

## Signature

Cryptographic signature goes here
