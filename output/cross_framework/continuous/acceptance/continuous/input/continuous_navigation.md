# GNN Example: Continuous State Navigation Agent
# GNN Version: 1.0
# Closed-loop continuous (linear-Gaussian) 2D navigator with a goal-seeking control input

## GNNSection
ActInfContinuous

## GNNVersionAndFlags
GNN v1

## ModelName
Continuous State Navigation Agent

## ModelAnnotation
A continuous-state Active Inference navigation agent rendered as a native
linear-Gaussian state-space model (LGSSM):
- Hidden state x = (x, y): the continuous 2D position of the navigator.
- Observation y: noisy readings of the 2D position (identity readout).
- Control input u: a goal-seeking command added to the state each step.
- The controller closes the loop on beliefs — it pushes the filtered posterior
  mean toward the preferred position goal_mean = (2.0, 2.0) with proportional
  gain control_gain = 0.3, i.e. u_t = control_gain * (goal_mean - mu_t).

## StateSpaceBlock
# Continuous latent state x = (x, y) position
x[2,1,type=float]        # continuous latent state
# Continuous observation y = noisy position readings
y[2,1,type=float]        # continuous observation
# Control input added to the state each step
u[2,1,type=float]        # control input
# State transition matrix (position persists between steps)
F[2,2,type=float]        # state transition
# Observation matrix (identity readout of the position)
H[2,2,type=float]        # observation matrix
# Process-noise covariance
Q[2,2,type=float]        # process-noise covariance
# Observation-noise covariance
R[2,2,type=float]        # observation-noise covariance
# Gaussian prior over the initial position
prior_mean[2,type=float] # prior mean over the initial latent state
prior_cov[2,2,type=float]# prior covariance over the initial latent state
# Preferred state the controller steers toward
goal_mean[2,type=float]  # preferred state (goal)
# Scalar proportional control gain
control_gain[1,type=float] # scalar proportional gain
# Time index
t[1,type=int]            # discrete time step

## Connections
prior_mean>x
F>x
x>y
H>y
Q>x
R>y
u>x
goal_mean>u
control_gain>u

## InitialParameterization
# F: position persists between steps; movement enters through the control input u.
F={
  (1.0, 0.0),
  (0.0, 1.0)
}

# H: identity readout — the observation reads the 2D position directly.
H={
  (1.0, 0.0),
  (0.0, 1.0)
}

# Q: process (motion) noise — modest slip per step.
Q={
  (0.05, 0.0),
  (0.0, 0.05)
}

# R: observation noise on the position readings.
R={
  (0.1, 0.0),
  (0.0, 0.1)
}

# Gaussian prior over the initial position: start at the origin, fairly certain.
prior_mean={(0.0, 0.0)}
prior_cov={
  (0.5, 0.0),
  (0.0, 0.5)
}

# goal_mean: the preferred position the closed-loop controller steers toward.
goal_mean={(2.0, 2.0)}

# control_gain: scalar proportional gain on (goal_mean - posterior_mean).
control_gain={(0.3)}

## Equations
# Generative model (linear-Gaussian state-space):
#   x_1 ~ N(prior_mean, prior_cov)
#   x_t = F x_{t-1} + u_{t-1} + N(0, Q)
#   y_t = H x_t + N(0, R)
# Closed-loop control on beliefs:
#   u_t = control_gain * (goal_mean - mu_t),  mu_t = current filtered posterior mean.

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
goal_mean=PreferredState
control_gain=ControlGain
x=ContinuousHiddenState
y=ContinuousObservation
u=ControlInput
t=Time

## ModelParameters
num_timesteps: 15
dt: 0.1
random_seed: 42
num_states: 2
num_observations: 2

## Footer
Continuous State Navigation Agent v1 - native linear-Gaussian (LGSSM) GNN model.
Closed-loop 2D navigator: the control input steers the belief toward the goal.

## Signature
Cryptographic signature goes here
