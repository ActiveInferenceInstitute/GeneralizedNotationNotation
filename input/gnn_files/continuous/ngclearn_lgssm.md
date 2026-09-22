# GNN Example: ngc-learn Linear-Gaussian State-Space Model
# GNN Version: 1.0
# Passive 2-state/2-observation linear-Gaussian model for the ngc-learn (ngclearn) backend

## GNNSection
ActInfContinuous

## GNNVersionAndFlags
GNN v1

## ModelName
ngc-learn Linear-Gaussian State-Space Agent

## ModelAnnotation
A continuous-state Active Inference agent with damped rotational latent
dynamics, observed through direct and mixed readouts. This file is the
ngc-learn (ngclearn) backend exemplar of the continuous family: it is rendered
via the shared linear-Gaussian generator, whose Kalman numerics are
byte-identical to the JAX backend. The agent runs passively — it has no
control input:
- Hidden state x: rotated and damped each step (dt = 0.1) with process noise.
- Observation y: two noisy readouts, one direct and one mixed.
- Q is the process-noise covariance (inverse process precision); R is the
  observation-noise covariance (inverse observation precision).

## StateSpaceBlock
# Continuous latent state x
x[2,1,type=float]        # continuous latent state
# Continuous observation (direct + mixed readouts)
y[2,1,type=float]        # continuous observation
# State transition matrix (damped rotation per step)
F[2,2,type=float]        # state transition
# Observation matrix (one direct and one mixed readout)
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
# F: damped rotation — each step shrinks the state slightly while rotating it.
F={
  (0.95, -0.2),
  (0.2, 0.95)
}

# H: direct + mixed readouts (channel 1 reads the state directly, channel 2
# mixes both coordinates).
H={
  (1.0, 0.0),
  (0.1, 0.9)
}

# Q: process-noise covariance = 0.05 * I.
Q={
  (0.05, 0.0),
  (0.0, 0.05)
}

# R: observation-noise covariance = diag(0.15, 0.25).
R={
  (0.15, 0.0),
  (0.0, 0.25)
}

# Gaussian prior over the initial state: mean at the origin, wider along the
# second coordinate.
prior_mean={(0.0, 0.0)}
prior_cov={
  (0.4, 0.0),
  (0.0, 0.8)
}

## Equations
# Generative model (linear-Gaussian state-space):
#   x_1 ~ N(prior_mean, prior_cov)
#   x_t = F x_{t-1} + N(0, Q)      (passive: no control input)
#   y_t = H x_t + N(0, R)
# F is a damped rotation: x_t is rotated and scaled toward the origin each
# step; y mixes the coordinates through H before adding observation noise.

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=20

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
num_timesteps: 20
dt: 0.1
random_seed: 42
num_states: 2
num_observations: 2

## Footer
ngc-learn Linear-Gaussian State-Space Agent v1.0 - native linear-Gaussian (LGSSM) GNN model.
Passive damped-rotation linear-Gaussian model; ngc-learn (ngclearn) backend exemplar of the continuous family.

## Signature
Cryptographic signature goes here
