# GNN Example: Stochastic Continuous Dynamics
# GNN Version: 1.0
# Continuous linear-Gaussian SDE model with explicit process and observation noise

## GNNSection
ActInfContinuous

## GNNVersionAndFlags
GNN v1

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
# Continuous latent state x = (position, velocity)
x[2,1,type=float]        # continuous latent state
# Continuous observation (two noisy position readouts)
y[2,1,type=float]        # continuous observation
# State transition matrix (Euler-discretized SDE with velocity damping)
F[2,2,type=float]        # state transition
# Observation matrix (both channels read position)
H[2,2,type=float]        # observation matrix
# Process-noise covariance
Q[2,2,type=float]        # process-noise covariance
# Observation-noise covariance
R[2,2,type=float]        # observation-noise covariance
# Gaussian prior over the initial state
prior_mean[2,type=float] # prior mean over the initial latent state
prior_cov[2,2,type=float]# prior covariance over the initial latent state
# Time index
t[1,type=int]            # discrete time step

## Connections
prior_mean>x
F>x
x>y
H>y
Q>x
R>y

## InitialParameterization
# F: Euler discretization (dt = 0.1) of dx/dt = F x with mild velocity damping.
F={
  (1.0, 0.1),
  (0.0, 0.9)
}

# H: two noisy position readouts (both observation channels read position).
H={
  (1.0, 0.0),
  (1.0, 0.0)
}

# Q: process-noise covariance = gamma_state^-1 * I (gamma_state = 10.0).
Q={
  (0.1, 0.0),
  (0.0, 0.1)
}

# R: observation-noise covariance = gamma_obs^-1 * I (gamma_obs = 5.0).
R={
  (0.2, 0.0),
  (0.0, 0.2)
}

# Gaussian prior over the initial state: position fairly certain, velocity less so.
prior_mean={(0.0, 0.0)}
prior_cov={
  (0.5, 0.0),
  (0.0, 1.0)
}

## Equations
# Generative model (linear-Gaussian state-space):
#   x_1 ~ N(prior_mean, prior_cov)
#   x_t = F x_{t-1} + N(0, Q)      (passive: no control input)
#   y_t = H x_t + N(0, R)
# SDE reading: dx/dt = F x + eps_state, eps_state ~ N(0, gamma_state^-1 I);
# y = H x + eps_obs, eps_obs ~ N(0, gamma_obs^-1 I).

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=15

## ActInfOntologyAnnotation
F=StateTransitionMatrix
H=ObservationMatrix
Q=ProcessNoiseCovariance
R=ObservationNoiseCovariance
prior_mean=PriorMean
prior_cov=PriorCovariance
x=ContinuousHiddenState
y=ContinuousObservation
t=Time

## ModelParameters
num_timesteps: 15
dt: 0.1
random_seed: 42
num_states: 2
num_observations: 2

## Footer
Stochastic Continuous Dynamics Agent v1.0 - native linear-Gaussian (LGSSM) GNN model.
Passive linear-Gaussian SDE with explicit process and observation noise.

## Signature
Cryptographic signature goes here
