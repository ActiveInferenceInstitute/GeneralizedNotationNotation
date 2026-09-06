## GNNVersionAndFlags
Version: 1.0

## ModelName
Stochastic Continuous Dynamics Agent

## ModelAnnotation
A continuous-state Active Inference agent whose dynamics carry explicit process
and observation noise, rendered as a native linear-Gaussian state-space model
(LGSSM). The agent runs passively — it has no control input:
- Hidden state x = (position, velocity): the Euler-discretized (dt = 0.1) SDE.
- Observation y: two noisy readouts, both reading the position.
- Q is the process-noise covariance (inverse process precision); R is the
  observation-noise covariance (inverse observation precision).

## StateSpaceBlock
x[2,1],float
y[2,1],float
F[2,2],float
H[2,2],float
Q[2,2],float
R[2,2],float
prior_mean[2],float
prior_cov[2,2],float
t[1],integer

## Connections
prior_mean>x
F>x
x>y
H>y
Q>x
R>y

## InitialParameterization
F = [[1.0, 0.1], [0.0, 0.9]]
H = [[1.0, 0.0], [1.0, 0.0]]
Q = [[0.1, 0.0], [0.0, 0.1]]
R = [[0.2, 0.0], [0.0, 0.2]]
prior_mean = [[0.0, 0.0]]
prior_cov = [[0.5, 0.0], [0.0, 1.0]]
num_timesteps = 15
dt = 0.1
random_seed = 42
num_states = 2
num_observations = 2

## Time
Dynamic
ModelTimeHorizon = 15

## ActInfOntologyAnnotation
F = StateTransitionMatrix
H = ObservationMatrix
Q = ProcessNoiseCovariance
R = ObservationNoiseCovariance
prior_mean = PriorMean
prior_cov = PriorCovariance
x = ContinuousHiddenState
y = ContinuousObservation
t = Time

## Footer
Generated: 2026-09-05T20:30:54.992082

## Signature
