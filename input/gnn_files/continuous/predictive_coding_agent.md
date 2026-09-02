# GNN Example: Predictive Coding Active Inference Agent
# GNN Version: 1.0
# Continuous predictive-coding agent as a native linear-Gaussian state-space model

## GNNSection
ActInfContinuous

## GNNVersionAndFlags
GNN v1

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
# Continuous latent state mu = (mu, mu_dot) generalized coordinates
x[2,1,type=float]        # continuous latent state
# Continuous observation (identity readout of both coordinates)
y[2,1,type=float]        # continuous observation
# State transition matrix (linearized dynamics)
F[2,2,type=float]        # state transition
# Observation matrix (linearized sensory mapping)
H[2,2,type=float]        # observation matrix
# Process-noise (dynamics-error) covariance
Q[2,2,type=float]        # process-noise covariance
# Observation-noise (sensory-error) covariance
R[2,2,type=float]        # observation-noise covariance
# Gaussian prior over the initial belief
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
# F: linearized dynamics f(mu) — mu integrates mu_dot (dt = 0.1); mu_dot leaks toward the flow.
F={
  (1.0, 0.1),
  (0.0, 0.8)
}

# H: linearized sensory mapping g(mu) — identity readout of both generalized coordinates.
H={
  (1.0, 0.0),
  (0.0, 1.0)
}

# Q: dynamics-error covariance Pi_d^-1 (dynamics precision Pi_d = 10).
Q={
  (0.1, 0.0),
  (0.0, 0.1)
}

# R: sensory-error covariance Pi_s^-1 (sensory precision Pi_s = 4).
R={
  (0.25, 0.0),
  (0.0, 0.25)
}

# Gaussian prior over the initial belief: broad / uninformative.
prior_mean={(0.0, 0.0)}
prior_cov={
  (1.0, 0.0),
  (0.0, 1.0)
}

## Equations
# Generative model (linear-Gaussian state-space):
#   x_1 ~ N(prior_mean, prior_cov)
#   x_t = F x_{t-1} + N(0, Q)      (passive: no control input)
#   y_t = H x_t + N(0, R)
# Predictive-coding reading: e_s = y - H mu (sensory error), e_d = mu_dot - f(mu)
# (dynamics error); F = (1/2) e_s^T Pi_s e_s + (1/2) e_d^T Pi_d e_d.

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
Predictive Coding Active Inference Agent v1 - native linear-Gaussian (LGSSM) GNN model.
Passive continuous predictive-coding agent (no control input).

## Signature
Cryptographic signature goes here
