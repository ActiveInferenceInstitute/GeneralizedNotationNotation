## GNNVersionAndFlags
Version: 1.0

## ModelName
Hierarchical Active Inference POMDP

## ModelAnnotation
A two-level hierarchical POMDP where:
- Level 1 (fast): 4 observations, 4 hidden states, 3 actions
- Level 2 (slow): 2 contextual states that modulate Level 1 likelihood
- Higher-level beliefs are updated at a slower timescale
- Top-down predictions constrain bottom-up inference at Level 1

## StateSpaceBlock
A1[4,4],float
B1[4,4,3],float
C1[4],float
D1[4],float
s1[4,1],float
s1_prime[4,1],float
o1[4,1],integer
π1[3],float
u1[1],integer
G1[1],float
A2[4,2],float
B2[2,2,1],float
C2[2],float
D2[2],float
s2[2,1],float
o2[4,1],float
G2[1],float
t1[1],integer
t2[1],integer

## Connections
D1>s1
s1-A1
s1>s1_prime
A1-o1
C1>G1
G1>π1
π1>u1
B1>u1
u1>s1_prime
s1>o2
D2>s2
s2-A2
A2>D1
s2-B2
C2>G2
G2>s2

## InitialParameterization
A1 = [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85]]
B1 = [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
C1 = [[0.1, 0.1, 0.1, 1.0]]
D1 = [[0.25, 0.25, 0.25, 0.25]]
A2 = [[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0], [0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.1, 0.9]]
B2 = [[[0.9, 0.1], [0.1, 0.9]]]
C2 = [[0.1, 1.0]]
D2 = [[0.5, 0.5]]

## Time
Dynamic
ModelTimeHorizon = Unbounded

## ActInfOntologyAnnotation
A1 = LikelihoodMatrix
B1 = TransitionMatrix
C1 = LogPreferenceVector
D1 = PriorOverHiddenStates
s1 = HiddenState
o1 = Observation
π1 = PolicyVector
u1 = Action
G1 = ExpectedFreeEnergy
A2 = HigherLevelLikelihoodMatrix
B2 = ContextTransitionMatrix
s2 = ContextualHiddenState
o2 = HigherLevelObservation
G2 = HigherLevelExpectedFreeEnergy

## Footer
Generated: 2026-03-13T18:17:22.641347

## Signature
