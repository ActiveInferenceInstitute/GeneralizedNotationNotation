## GNNVersionAndFlags
Version: 1.0

## ModelName
Simple Markov Chain

## ModelAnnotation
This model describes a minimal discrete-time Markov Chain:

- 3 states representing weather (sunny, cloudy, rainy).
- No actions — the system evolves passively.
- Observations = states directly (identity mapping for monitoring).
- Stationary transition matrix with realistic weather dynamics.
- Tests the simplest model structure: passive state evolution with no control.

## StateSpaceBlock
A[3,3],float
B[3,3],float
D[3],float
s[3,1],float
s_prime[3,1],float
o[3,1],integer
t[1],integer

## Connections
D>s
s-A
A-o
s>s_prime
B>s_prime
s-B

## InitialParameterization
A = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
B = [[0.7, 0.3, 0.1], [0.2, 0.4, 0.3], [0.1, 0.3, 0.6]]
D = [[0.5, 0.3, 0.2]]

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
t = Time

## Footer
Generated: 2026-04-10T10:24:32.892217

## Signature
