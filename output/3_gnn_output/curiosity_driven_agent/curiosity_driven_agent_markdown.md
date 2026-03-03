## GNNVersionAndFlags
Version: 1.0

## ModelName
Curiosity-Driven Active Inference Agent

## ModelAnnotation
An Active Inference agent with:
- Explicit epistemic value (information gain / Bayesian surprise) component in G
- Separate instrumental value (preference satisfaction) component
- Precision parameter γ weighting epistemic vs instrumental contributions
- 5 hidden states, 5 observations, 4 actions in a navigation context
- Agent is rewarded for reducing posterior uncertainty

## StateSpaceBlock
A[5,5],float
B[5,5,4],float
C[5],float
D[5],float
E[4],float
s[5,1],float
s_prime[5,1],float
o[5,1],integer
π[4],float
u[1],integer
G[1],float
G_epi[1],float
G_ins[1],float
γ[1],float
F[1],float
t[1],integer

## Connections
D>s
s-A
s>s_prime
A-o
C>G_ins
G_epi>G
G_ins>G
γ>G
E>π
G>π
π>u
B>u
u>s_prime
s-F
o-F

## InitialParameterization
A = [[0.9, 0.025, 0.025, 0.025, 0.025], [0.025, 0.9, 0.025, 0.025, 0.025], [0.025, 0.025, 0.9, 0.025, 0.025], [0.025, 0.025, 0.025, 0.9, 0.025], [0.025, 0.025, 0.025, 0.025, 0.9]]
B = [[[0.9, 0.1, 0.0, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0, 0.0], [0.0, 0.1, 0.8, 0.1, 0.0], [0.0, 0.0, 0.1, 0.8, 0.1], [0.0, 0.0, 0.0, 0.1, 0.9]], [[0.9, 0.1, 0.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9, 0.0], [0.0, 0.0, 0.0, 0.1, 0.9]], [[0.9, 0.0, 0.0, 0.0, 0.1], [0.0, 0.9, 0.0, 0.0, 0.1], [0.0, 0.0, 0.9, 0.0, 0.1], [0.0, 0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.0, 0.0, 1.0]]]
C = [[-2.0, -2.0, -2.0, -2.0, 2.0]]
D = [[0.2, 0.2, 0.2, 0.2, 0.2]]
E = [[0.25, 0.25, 0.25, 0.25]]
γ = [[1.0]]

## Time
Dynamic
ModelTimeHorizon = 30

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
G_epi = EpistemicValue
G_ins = InstrumentalValue
γ = PrecisionParameter
F = VariationalFreeEnergy
t = Time

## Footer
Generated: 2026-03-03T08:22:03.763004

## Signature
