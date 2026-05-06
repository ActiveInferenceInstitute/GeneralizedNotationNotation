## GNNVersionAndFlags
Version: 1.0

## ModelName
PyMDP Scaling N2 T100

## ModelAnnotation
PyMDP runtime scaling sweep with noisy observation and stochastic transitions.

## StateSpaceBlock
s[2,1],float
o[2,1],integer
A[2,2],float
B[2,2,2],float
C[2],float
D[2],float
π[2],float
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
A = [[0.95, 0.05], [0.05, 0.95]]
B = [[[0.9, 0.1], [0.9, 0.1]], [[0.1, 0.9], [0.1, 0.9]]]
C = [[0.0, 3.0]]
D = [[0.5, 0.5]]
num_hidden_states = 2
num_obs = 2
num_actions = 2
num_timesteps = 100

## Time
Dynamic
ModelTimeHorizon = 100

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
Generated: 2026-05-06T07:03:55.250260

## Signature
