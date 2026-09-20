# GNN Example: Damped Oscillator with Sensor Bias
# GNN Version: 1.0
# Continuous linear-Gaussian LGSSM with rectangular readout, correlated noise, and an AR(1) bias state

## GNNSection
ActInfContinuous

## GNNVersionAndFlags
GNN v1

## ModelName
Damped Oscillator Bias Agent

## ModelAnnotation
A continuous-state Active Inference agent rendered as a native linear-Gaussian
state-space model (LGSSM). The agent runs passively — it has no control input:
- Hidden state x = (position, velocity, bias): the Euler-discretized (dt = 0.1)
  damped harmonic oscillator drives position and velocity as a stable complex
  pole pair, while the third coordinate is a slowly mean-reverting first-order
  autoregressive sensor bias.
- Observation y: two noisy readouts — position plus bias, and velocity plus
  half the bias. Both channels observe the bias, the readout is rectangular
  (2 observations over 3 states), and the bias is never observed directly, so
  the filter must infer it from the correlated channel dynamics.
- Q is the process-noise covariance (position and velocity noise correlated,
  covariance 0.002); R is the observation-noise covariance (both channels
  correlated, covariance 0.008). Q, R, and the prior covariance are strictly
  positive definite.

## StateSpaceBlock
# Continuous latent state x = (position, velocity, bias)
x[3,1,type=float]        # continuous latent state
# Continuous observation (rectangular readout: two channels, three states)
y[2,1,type=float]        # continuous observation
# State transition matrix (Euler-discretized damped oscillator + AR(1) bias)
F[3,3,type=float]        # state transition
# Observation matrix (partial rectangular readout; both channels see the bias)
H[2,3,type=float]        # observation matrix
# Process-noise covariance (correlated position/velocity block)
Q[3,3,type=float]        # process-noise covariance
# Observation-noise covariance (correlated across the two channels)
R[2,2,type=float]        # observation-noise covariance
# Gaussian prior over the initial state
prior_mean[3,type=float] # prior mean over the initial latent state
prior_cov[3,3,type=float]# prior covariance over the initial latent state
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
# F: Euler discretization (dt = 0.1) of the damped oscillator
#   position' = velocity; velocity' = -0.9 position - 0.1 velocity
#   so the 2x2 block carries a stable complex pole pair (underdamped ring),
#   plus an independent AR(1) bias coordinate: bias+ = 0.95 bias.
F={
  (1.0, 0.1, 0.0),
  (-0.09, 0.99, 0.0),
  (0.0, 0.0, 0.95)
}

# H: two observation channels reading (position + bias) and (velocity + 0.5 bias).
H={
  (1.0, 0.0, 1.0),
  (0.0, 1.0, 0.5)
}

# Q: process-noise covariance; position and velocity noise are correlated
# (covariance 0.002), the bias walks with its own variance.
Q={
  (0.01, 0.002, 0.0),
  (0.002, 0.01, 0.0),
  (0.0, 0.0, 0.004)
}

# R: observation-noise covariance; both channels share correlated noise
# (covariance 0.008).
R={
  (0.04, 0.008),
  (0.008, 0.06)
}

# Gaussian prior over the initial state: the bias starts away from zero
# (mean 0.2) so the filter must actively track and discount it; the position
# and velocity uncertainties are correlated (covariance 0.2).
prior_mean={(0.0, 0.0, 0.2)}
prior_cov={
  (1.0, 0.2, 0.0),
  (0.2, 1.0, 0.0),
  (0.0, 0.0, 0.5)
}

## Equations
# Generative model (linear-Gaussian state-space):
#   x_1 ~ N(prior_mean, prior_cov)
#   x_t = F x_{t-1} + N(0, Q)      (passive: no control input)
#   y_t = H x_t + N(0, R)
# Dynamics reading: the (position, velocity) block is the Euler-discretized
# damped oscillator dx/dt = (velocity, -0.9 position - 0.1 velocity); the bias
# follows an AR(1) process bias_t = 0.95 bias_{t-1} + eps_bias.
# Observation reading: y1 = position + bias and y2 = velocity + 0.5 bias, with
# correlated observation noise eps_obs ~ N(0, R).

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=12

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
num_timesteps: 12
dt: 0.1
random_seed: 43
num_states: 3
num_observations: 2

## Footer
Damped Oscillator with Sensor Bias v1.0 - native linear-Gaussian (LGSSM) GNN model.
Passive linear-Gaussian state-space model with a rectangular (2x3) observation
matrix, correlated process and observation noise, and an autoregressive sensor bias.

## Signature
Cryptographic signature goes here