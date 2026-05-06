## GNNVersionAndFlags
Version: 1.0

## ModelName
PyMDP Scaling N4 T10

## ModelAnnotation
PyMDP runtime scaling sweep with noisy observation and stochastic transitions.

## StateSpaceBlock
s[4,1],float
o[4,1],integer
A[4,4],float
B[4,4,4],float
C[4],float
D[4],float
π[4],float
u[1],integer
G[1],float
F[1],float
t[1],integer

## Connections
D>s
s-A
A-o
s-B
C>G
G>π
π>u
B>u
s-F
o-F

## InitialParameterization
A = [[0.925, 0.025, 0.025, 0.025], [0.025, 0.925, 0.025, 0.025], [0.025, 0.025, 0.925, 0.025], [0.025, 0.025, 0.025, 0.925]]
B = [[[0.85, 0.05, 0.05, 0.05], [0.85, 0.05, 0.05, 0.05], [0.85, 0.05, 0.05, 0.05], [0.85, 0.05, 0.05, 0.05]], [[0.05, 0.85, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05]], [[0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.85, 0.05]], [[0.05, 0.05, 0.05, 0.85], [0.05, 0.05, 0.05, 0.85], [0.05, 0.05, 0.05, 0.85], [0.05, 0.05, 0.05, 0.85]]]
C = [[0.0, 0.0, 0.0, 3.0]]
D = [[0.25, 0.25, 0.25, 0.25]]
num_hidden_states = 4
num_obs = 4
num_actions = 4
num_timesteps = 10

## Time
Dynamic
ModelTimeHorizon = 10

## ActInfOntologyAnnotation
A = LikelihoodMatrix
B = TransitionMatrix
C = LogPreferenceVector
D = PriorOverHiddenStates
G = ExpectedFreeEnergy
F = VariationalFreeEnergy
s = HiddenState
o = Observation
π = PolicyVector
u = Action
t = Time

## Footer
Generated: 2026-05-06T07:03:47.145277

## Signature
