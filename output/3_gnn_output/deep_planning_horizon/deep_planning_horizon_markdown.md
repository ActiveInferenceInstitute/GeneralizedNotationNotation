## GNNVersionAndFlags
Version: 1.0

## ModelName
Deep Planning Horizon POMDP

## ModelAnnotation
An Active Inference POMDP with deep (T=5) planning horizon:
- Evaluates policies over 5 future timesteps before acting
- Uses rollout Expected Free Energy accumulation
- 4 hidden states, 4 observations, 4 actions
- Each action policy is a sequence of T actions: π = [a_1, a_2, ..., a_T]
- Enables sophisticated multi-step reasoning and delayed reward attribution

## StateSpaceBlock
A[4,4],float
B[4,4,4],float
C[4],float
D[4],float
E[64],float
s[4,1],float
o[4,1],integer
π[64],float
u[1],integer
s_tau1[4,1],float
s_tau2[4,1],float
s_tau3[4,1],float
s_tau4[4,1],float
s_tau5[4,1],float
G_tau1[64],float
G_tau2[64],float
G_tau3[64],float
G_tau4[64],float
G_tau5[64],float
G[64],float
F[1],float
t[1],integer

## Connections
D>s
s-A
A-o
s-F
o-F
E>π
G>π
s>s_tau1
B>s_tau1
s_tau1>s_tau2
B>s_tau2
s_tau2>s_tau3
B>s_tau3
s_tau3>s_tau4
B>s_tau4
s_tau4>s_tau5
A-s_tau1
A-s_tau2
A-s_tau3
A-s_tau4
A-s_tau5
C>G_tau1
C>G_tau2
C>G_tau3
C>G_tau4
C>G_tau5
G_tau1>G
G_tau2>G
G_tau3>G
G_tau4>G
G_tau5>G
G>π
π>u

## InitialParameterization
A = [[0.9, 0.05, 0.025, 0.025], [0.05, 0.9, 0.025, 0.025], [0.025, 0.025, 0.9, 0.05], [0.025, 0.025, 0.05, 0.9]]
B = [[[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1], [0.1, 0.0, 0.0, 0.9]], [[0.9, 0.0, 0.0, 0.1], [0.1, 0.9, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0], [0.0, 0.0, 0.1, 0.9]], [[0.8, 0.1, 0.1, 0.0], [0.0, 0.8, 0.1, 0.1], [0.1, 0.0, 0.8, 0.1], [0.1, 0.1, 0.0, 0.8]], [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]]
C = [[-1.0, -0.5, -0.5, 2.0]]
D = [[0.25, 0.25, 0.25, 0.25]]

## Time
Dynamic
ModelTimeHorizon = Unbounded

## ActInfOntologyAnnotation
A = LikelihoodMatrix
B = TransitionMatrix
C = LogPreferenceVector
D = PriorOverHiddenStates
E = PolicyPrior
s = HiddenState
o = Observation
π = PolicySequenceDistribution
u = Action
G = CumulativeExpectedFreeEnergy
F = VariationalFreeEnergy
t = Time

## Footer
Generated: 2026-03-15T13:53:23.883290

## Signature
