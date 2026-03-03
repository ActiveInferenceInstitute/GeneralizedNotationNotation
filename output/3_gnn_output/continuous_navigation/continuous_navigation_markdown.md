## GNNVersionAndFlags
Version: 1.0

## ModelName
Continuous State Navigation Agent

## ModelAnnotation
A continuous-state Active Inference agent navigating a 2D environment:
- Hidden states: 2D position (x, y) as Gaussian belief
- Observations: noisy position measurements with Gaussian noise
- Actions: 2D velocity commands (dx, dy)
- Uses Laplace approximation for Gaussian belief updating
- Generalized coordinates of motion for smooth trajectories

## StateSpaceBlock
μ[2,1],float
Σ[2,2],float
μ_prime[2,1],float
μ_dot[2,1],float
μ_ddot[2,1],float
A_μ[2,2],float
A_Σ[2,2],float
o[2,1],float
B_f[2,2],float
B_u[2,2],float
C_μ[2,1],float
C_Σ[2,2],float
u[2,1],float
F[1],float
G[1],float
ε_o[2,1],float
ε_x[2,1],float
Π_o[2,2],float
Π_x[2,2],float
t[1],float

## Connections
μ-A_μ
A_μ-o
A_Σ-ε_o
o-ε_o
ε_o-F
Π_o-F
μ-B_f
B_f-μ_prime
B_u-μ_prime
u-B_u
ε_x-F
Π_x-F
C_μ-G
C_Σ-G
μ_prime-G
G>u
μ-Σ
Σ-Π_o

## InitialParameterization
A_μ = [[1.0, 0.0], [0.0, 1.0]]
A_Σ = [[0.1, 0.0], [0.0, 0.1]]
B_f = [[1.0, 0.0], [0.0, 1.0]]
B_u = [[0.1, 0.0], [0.0, 0.1]]
C_μ = [[1.0], [1.0]]
C_Σ = [[0.05, 0.0], [0.0, 0.05]]
μ = [[0.0], [0.0]]
Σ = [[0.5, 0.0], [0.0, 0.5]]
Π_o = [[10.0, 0.0], [0.0, 10.0]]
Π_x = [[20.0, 0.0], [0.0, 20.0]]

## Time
Dynamic
ModelTimeHorizon = 10.0

## ActInfOntologyAnnotation
μ = BeliefMean
Σ = BeliefCovariance
A_μ = ObservationMeanMapping
A_Σ = ObservationNoise
o = Observation
B_f = DynamicsMatrix
B_u = ActionEffectMatrix
C_μ = PreferenceMean
C_Σ = PreferenceCovariance
u = ContinuousAction
F = VariationalFreeEnergy
G = ExpectedFreeEnergy
ε_o = SensoryPredictionError
ε_x = DynamicPredictionError
Π_o = SensoryPrecision
Π_x = DynamicPrecision
t = ContinuousTime

## Footer
Generated: 2026-03-03T08:22:04.301536

## Signature
