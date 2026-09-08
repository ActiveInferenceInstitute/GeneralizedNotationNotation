## GNNVersionAndFlags
Version: 1.0

## ModelName
Predictive Coding Active Inference Agent

## ModelAnnotation
A continuous predictive-coding Active Inference agent rendered as a native
linear-Gaussian state-space model (LGSSM). The agent runs passively — it has no
control input:
- Hidden state mu = (mu, mu_dot): the generalized coordinates of the belief.
- Observation y: an identity readout of both generalized coordinates.
- F encodes the linearized dynamics f(mu): mu integrates mu_dot (dt = 0.1) and
  mu_dot leaks toward the flow.
- Q and R are the dynamics- and sensory-error covariances (the inverse
  precisions of the predictive-coding formulation).

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
F = [[1.0, 0.1], [0.0, 0.8]]
H = [[1.0, 0.0], [0.0, 1.0]]
Q = [[0.1, 0.0], [0.0, 0.1]]
R = [[0.25, 0.0], [0.0, 0.25]]
prior_mean = [[0.0, 0.0]]
prior_cov = [[1.0, 0.0], [0.0, 1.0]]
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
Generated: 2026-09-08T06:53:47.220692

## Signature
