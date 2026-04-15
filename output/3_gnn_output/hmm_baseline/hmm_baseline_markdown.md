## GNNVersionAndFlags
Version: 1.0

## ModelName
Hidden Markov Model Baseline

## ModelAnnotation
A standard discrete Hidden Markov Model with:
- 4 hidden states with Markovian dynamics
- 6 observation symbols
- Fixed transition and emission matrices
- No action selection (passive inference only)
- Suitable for sequence modeling and state estimation tasks

## StateSpaceBlock
A[6,4],float
B[4,4],float
D[4],float
s[4,1],float
s_prime[4,1],float
o[6,1],integer
F[1],float
alpha[4,1],float
beta[4,1],float
t[1],integer

## Connections
D>s
s-A
s>s_prime
A-o
B>s_prime
s-B
s-F
o-F
s-alpha
o-alpha
alpha>s_prime
s_prime-beta

## InitialParameterization
A = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7], [0.1, 0.1, 0.4, 0.4], [0.4, 0.4, 0.1, 0.1]]
B = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.2, 0.1], [0.1, 0.1, 0.6, 0.2], [0.1, 0.1, 0.1, 0.6]]
D = [[0.25, 0.25, 0.25, 0.25]]

## Time
Dynamic
ModelTimeHorizon = Unbounded

## ActInfOntologyAnnotation
A = EmissionMatrix
B = TransitionMatrix
D = InitialStateDistribution
s = HiddenState
s_prime = NextHiddenState
o = Observation
F = VariationalFreeEnergy
alpha = ForwardVariable
beta = BackwardVariable
t = Time

## Footer
Generated: 2026-04-15T12:25:54.012829

## Signature
