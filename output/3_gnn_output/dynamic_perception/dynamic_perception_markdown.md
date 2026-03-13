## GNNVersionAndFlags
Version: 1.0

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
A[2,2],float
B[2,2],float
D[2,1],float
s_t[2,1],float
s_prime[2,1],float
o_t[2,1],integer
F[1],float
t[1],integer

## Connections
D>s_t
s_t-A
A-o_t
s_t-B
B>s_prime
s_t-F
o_t-F

## InitialParameterization
A = [[0.9, 0.1], [0.2, 0.8]]
B = [[0.7, 0.3], [0.3, 0.7]]
D = [[0.5, 0.5]]

## Time
Dynamic
ModelTimeHorizon = 10

## ActInfOntologyAnnotation
A = RecognitionMatrix
B = TransitionMatrix
D = Prior
s_t = HiddenState
s_prime = NextHiddenState
o_t = Observation
F = VariationalFreeEnergy
t = Time

## Footer
Generated: 2026-03-13T14:15:02.721355

## Signature
