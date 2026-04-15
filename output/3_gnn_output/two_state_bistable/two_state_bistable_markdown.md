## GNNVersionAndFlags
Version: 1.0

## ModelName
Two State Bistable POMDP

## ModelAnnotation
This model describes a minimal 2-state bistable POMDP:

- 2 hidden states: "left" and "right" in a symmetric bistable potential.
- 2 noisy observations: the agent gets a noisy readout of which side it is on.
- 2 actions: push-left or push-right.
- The agent prefers observation 1 ("right") over observation 0 ("left").
- Tests the absolute smallest POMDP with full active inference structure.

## StateSpaceBlock
A[2,2],float
B[2,2,2],float
C[2],float
D[2],float
E[2],float
s[2,1],float
s_prime[2,1],float
o[2,1],integer
π[2],float
u[1],integer
G[1],float
t[1],integer

## Connections
D>s
s-A
A-o
s>s_prime
s-B
C>G
E>π
G>π
π>u
B>u
u>s_prime

## InitialParameterization
A = [[0.8, 0.2], [0.2, 0.8]]
B = [[[0.8, 0.3], [0.2, 0.7]], [[0.3, 0.8], [0.7, 0.2]]]
C = [[0.0, 2.0]]
D = [[0.5, 0.5]]
E = [[0.5, 0.5]]

## Time
Dynamic
ModelTimeHorizon = Unbounded

## ActInfOntologyAnnotation
A = LikelihoodMatrix
B = TransitionMatrix
C = LogPreferenceVector
D = PriorOverHiddenStates
E = Habit
G = ExpectedFreeEnergy
s = HiddenState
s_prime = NextHiddenState
o = Observation
π = PolicyVector
u = Action
t = Time

## Footer
Generated: 2026-04-15T12:25:54.031422

## Signature
