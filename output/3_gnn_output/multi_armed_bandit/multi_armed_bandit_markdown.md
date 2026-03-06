## GNNVersionAndFlags
Version: 1.0

## ModelName
Multi Armed Bandit Agent

## ModelAnnotation
This model describes a 3-armed bandit as a degenerate POMDP:

- 3 hidden states representing the "reward context" (which arm is currently best).
- 3 observations representing reward signals (no-reward, small-reward, big-reward).
- 3 actions: pull arm 0, pull arm 1, or pull arm 2.
- Context switches slowly (sticky transitions), testing exploration vs exploitation.
- The agent prefers big-reward observations (observation 2).
- Tests the bandit structure: meaningful actions despite nearly-static state dynamics.

## StateSpaceBlock
A[3,3],float
B[3,3,3],float
C[3],float
D[3],float
s[3,1],float
s_prime[3,1],float
o[3,1],integer
π[3],float
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
G>π
π>u
B>u
u>s_prime

## InitialParameterization
A = [[0.1, 0.5, 0.5], [0.3, 0.4, 0.3], [0.6, 0.1, 0.2]]
B = [[[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]]
C = [[0.0, 1.0, 3.0]]
D = [[0.33333, 0.33333, 0.33333]]

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
Generated: 2026-03-06T15:00:16.221572

## Signature
