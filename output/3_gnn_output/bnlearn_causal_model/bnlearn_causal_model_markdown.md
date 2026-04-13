## GNNVersionAndFlags
Version: 1.0

## ModelName
Bnlearn Causal Model

## ModelAnnotation
A Bayesian Network model mapping Active Inference structure:
- S: Hidden State
- A: Action
- S_prev: Previous State
- O: Observation

## StateSpaceBlock
A[2,2],float
B[2,2,2],float
s[2,1],float
s_prev[2,1],float
o[2,1],integer
a[2,1],integer

## Connections
s_prev>s
a>s
s>o

## InitialParameterization
A = [[0.9, 0.1], [0.1, 0.9]]
B = [[[0.7, 0.3], [0.3, 0.7]], [[0.3, 0.7], [0.7, 0.3]]]
C = [[0.0, 1.0]]
D = [[0.5, 0.5]]

## Time
Dynamic

## ActInfOntologyAnnotation
A = ObservationModel
B = TransitionModel
s = HiddenState
s_prev = PreviousState
o = Observation
a = Action

## Footer
Generated: 2026-04-12T17:23:01.372686

## Signature
