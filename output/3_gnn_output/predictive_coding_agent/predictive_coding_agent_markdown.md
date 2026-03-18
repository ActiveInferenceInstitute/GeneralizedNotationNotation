## GNNVersionAndFlags
Version: 1.0

## ModelName
Predictive Coding Active Inference Agent

## ModelAnnotation
A continuous-state Active Inference agent implementing predictive coding:

- Two-level predictive hierarchy: sensory predictions and dynamics predictions
- Prediction errors drive belief updating via gradient descent on free energy
- Precision-weighted prediction errors enable attentional modulation
- Sensory level: predicts observations from hidden causes
- Dynamics level: predicts state evolution from generative dynamics
- Action minimizes expected free energy by changing sensory input
- Uses generalized coordinates of motion (position, velocity, acceleration)
- Demonstrates the core predictive processing framework underlying Active Inference

## StateSpaceBlock
mu[3,1],float
mu_dot[3,1],float
Sigma[3,3],float
e_s[4,1],float
e_d[3,1],float
g_params[12],float
f_params[9],float
Pi_s[4,4],float
Pi_d[3,3],float
o[4,1],float
u[2,1],float
F[1],float
F_s[1],float
F_d[1],float
mu_star[3,1],float
t[1],float

## Connections
mu-g_params
g_params-o
o-e_s
mu-e_s
Pi_s-e_s
e_s-F_s
mu-f_params
f_params-mu_dot
mu_dot-e_d
Pi_d-e_d
e_d-F_d
F_s>F
F_d>F
F>mu
F>u
mu_star-F
mu-Sigma
Sigma-Pi_s

## InitialParameterization
mu = [[0.0], [0.0], [0.0]]
mu_dot = [[0.0], [0.0], [0.0]]
Sigma = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
Pi_s = [[8.0, 0.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0], [0.0, 0.0, 8.0, 0.0], [0.0, 0.0, 0.0, 8.0]]
Pi_d = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
mu_star = [[1.0], [1.0], [0.5]]

## Time
Dynamic
ModelTimeHorizon = 5.0

## ActInfOntologyAnnotation
mu = BeliefMean
mu_dot = BeliefVelocity
Sigma = BeliefCovariance
e_s = SensoryPredictionError
e_d = DynamicPredictionError
g_params = SensoryMappingParameters
f_params = DynamicsParameters
Pi_s = SensoryPrecision
Pi_d = DynamicPrecision
o = Observation
u = ContinuousAction
F = VariationalFreeEnergy
F_s = SensoryFreeEnergy
F_d = DynamicFreeEnergy
mu_star = PriorExpectation
t = ContinuousTime

## Footer
Generated: 2026-03-17T16:46:48.640961

## Signature
