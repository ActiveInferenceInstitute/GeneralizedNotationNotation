## GNNVersionAndFlags
Version: 1.0

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
A[2,2],float
B[2,2,2],float
C[2],float
D[2,1],float
s[2,1],float
o[2,1],integer
u[1],integer

## Connections
D>s
s-A
A-o
s-B
B>u
u>s

## InitialParameterization
A = [[0.9, 0.1], [0.2, 0.8]]
B = [[[0.95, 0.05], [0.05, 0.95]], [[0.05, 0.95], [0.95, 0.05]]]
C = [[0.0, 0.0]]
D = [[0.5, 0.5]]
num_hidden_states = 2
num_obs = 2
num_actions = 2
num_timesteps = 5

## Time
Static

## ActInfOntologyAnnotation
A = RecognitionMatrix
B = TransitionMatrix
C = PreferenceVector
D = Prior
s = HiddenState
o = Observation
u = Action

## Footer
Generated: 2026-09-05T20:30:45.902398

## Signature
