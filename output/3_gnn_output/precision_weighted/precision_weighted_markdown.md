## GNNVersionAndFlags
Version: 1.0

## ModelName
Precision-Weighted Active Inference Agent

## ModelAnnotation
An Active Inference agent with explicit precision parameters:
- ω (omega): sensory precision weighting likelihood confidence
- γ (gamma): policy precision controlling action randomness
- β (beta): inverse temperature for policy selection (softmax)
- 3 hidden states, 3 observations, 3 actions (same topology as base POMDP)
- Precision parameters enable modeling of attention and confidence

## StateSpaceBlock
A[3,3],float
B[3,3,3],float
C[3],float
D[3],float
E[3],float
s[3,1],float
s_prime[3,1],float
o[3,1],integer
F[1],float
ω[1],float
γ[1],float
β[1],float
π[3],float
u[1],integer
G[1],float

## Connections
D>s
s-A
A-o
ω>A
s>s_prime
C>G
G>π
γ>π
β>π
E>π
π>u
B>u
u>s_prime
s-F
o-F
ω-F

## InitialParameterization
A = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
B = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]
C = [[0.1, 0.1, 1.0]]
D = [[0.333, 0.333, 0.333]]
E = [[0.333, 0.333, 0.333]]
ω = [[4.0]]
γ = [[2.0]]
β = [[0.5]]

## Time
Dynamic
ModelTimeHorizon = Unbounded

## ActInfOntologyAnnotation
A = LikelihoodMatrix
B = TransitionMatrix
C = LogPreferenceVector
D = PriorOverHiddenStates
E = Habit
s = HiddenState
s_prime = NextHiddenState
o = Observation
π = PolicyVector
u = Action
G = ExpectedFreeEnergy
F = VariationalFreeEnergy
ω = SensoryPrecision
γ = PolicyPrecision
β = InverseTemperature
t = Time

## Footer
Generated: 2026-04-15T12:25:55.003534

## Signature
