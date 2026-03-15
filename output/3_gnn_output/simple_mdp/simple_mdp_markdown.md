## GNNVersionAndFlags
Version: 1.0

## ModelName
Simple MDP Agent

## ModelAnnotation
This model describes a fully observable Markov Decision Process (MDP):

- 4 hidden states representing grid positions (corners of a 2x2 grid).
- Observations are identical to states (A = identity matrix).
- 4 actions: stay, move-north, move-south, move-east.
- Preferences strongly favor state/observation 3 (goal location).
- Tests the degenerate POMDP case where partial observability is absent.

## StateSpaceBlock
A[4,4],float
B[4,4,4],float
C[4],float
D[4],float
s[4,1],float
s_prime[4,1],float
o[4,1],integer
π[4],float
u[1],integer
G[1],float
t[1],integer

## Connections
D>s
s-A
s>s_prime
A-o
s-B
C>G
G>π
π>u
B>u
u>s_prime

## InitialParameterization
A = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
B = [[[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0], [0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.1, 0.9]], [[0.1, 0.9, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9], [0.0, 0.0, 0.9, 0.1]], [[0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.1, 0.9], [0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]], [[0.0, 0.0, 0.1, 0.9], [0.0, 0.0, 0.9, 0.1], [0.1, 0.9, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]]
C = [[0.0, 0.0, 0.0, 3.0]]
D = [[0.25, 0.25, 0.25, 0.25]]

## Time
Dynamic
ModelTimeHorizon = Unbounded

## ActInfOntologyAnnotation
A = LikelihoodMatrix
B = TransitionMatrix
C = LogPreferenceVector
D = PriorOverHiddenStates
G = ExpectedFreeEnergy
s = HiddenState
s_prime = NextHiddenState
o = Observation
π = PolicyVector
u = Action
t = Time

## Footer
Generated: 2026-03-15T13:53:23.867054

## Signature
