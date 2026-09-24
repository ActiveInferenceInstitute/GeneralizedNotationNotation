# GNN Example: Factored Continuous LGSSM (Two Independent Factors)
# GNN Version: 1.0
# Composed spec: two independent 2-dim linear-Gaussian factors declared with
# per-factor keys F_f1/... and F_f2/... — the exemplar of the composed
# factored × continuous kind set

## GNNSection
ActInfContinuous

## GNNVersionAndFlags
GNN v1

## ModelName
Factored Continuous LGSSM (Two Independent Factors)

## ModelAnnotation
A composed model: two INDEPENDENT 2-dim linear-Gaussian (LGSSM) factors are
declared side by side with per-factor keys (factor 1: F_f1/H_f1/Q_f1/R_f1
with a Gaussian prior and closed-loop goal control; factor 2: F_f2/H_f2/
Q_f2/R_f2 with a Gaussian prior, passive), so the spec classifies as the
composed kind set {FACTORED, CONTINUOUS}:

- Factor 1 (damped drift, closed-loop): x_f1 = (position, velocity) with
  control input u_f1; u_f1,t = control_gain_f1 * (goal_mean_f1 - mu_f1,t)
  closes the loop on beliefs.
- Factor 2 (rotation, passive): x_f2 = (position, velocity) evolving under a
  small-angle rotation matrix F_f2; no control input is declared.
- The factors share no state, observation, or control variables — the
  dynamics are factored, not coupled; num_factors: 2 in ModelParameters.

Factored-continuous specs classify under BOTH families — `detect_model_kinds`
returns {FACTORED, CONTINUOUS} — and the render step is family-aware rather
than compositional-refusal: the JAX backend renders one native LGSSM block
per factor (`gnn.render.continuous_script` receives a FactoredContinuousSpec
and emits per-factor state-space blocks), while every other continuous
backend receipts the spec `unsupported-factored-continuous` — never silently
rendered flat as a single joint LGSSM.

## StateSpaceBlock
# Factor 1: continuous latent state x_f1 = (position, velocity)
x_f1[2,1,type=float]     # factor 1 continuous latent state
# Factor 1: continuous observation (noisy position-velocity readouts)
y_f1[2,1,type=float]     # factor 1 continuous observation
# Factor 1: control input added to the state each step
u_f1[2,1,type=float]     # factor 1 control input
# Factor 1: state transition matrix (Euler damped drift, dt = 0.1)
F_f1[2,2,type=float]     # factor 1 state transition
# Factor 1: observation matrix (identity readout)
H_f1[2,2,type=float]     # factor 1 observation matrix
# Factor 1: process-noise covariance
Q_f1[2,2,type=float]     # factor 1 process-noise covariance
# Factor 1: observation-noise covariance
R_f1[2,2,type=float]     # factor 1 observation-noise covariance
# Factor 1: Gaussian prior over the initial state
prior_mean_f1[2,type=float] # factor 1 prior mean
prior_cov_f1[2,2,type=float]# factor 1 prior covariance
# Factor 1: preferred state the controller steers toward
goal_mean_f1[2,type=float] # factor 1 preferred state (goal)
# Factor 1: scalar proportional control gain
control_gain_f1[1,type=float] # factor 1 scalar proportional gain
# Factor 2: continuous latent state x_f2 = (position, velocity)
x_f2[2,1,type=float]     # factor 2 continuous latent state
# Factor 2: continuous observation (noisy position-velocity readouts)
y_f2[2,1,type=float]     # factor 2 continuous observation
# Factor 2: state transition matrix (small-angle rotation)
F_f2[2,2,type=float]     # factor 2 state transition
# Factor 2: observation matrix (identity readout)
H_f2[2,2,type=float]     # factor 2 observation matrix
# Factor 2: process-noise covariance
Q_f2[2,2,type=float]     # factor 2 process-noise covariance
# Factor 2: observation-noise covariance
R_f2[2,2,type=float]     # factor 2 observation-noise covariance
# Factor 2: Gaussian prior over the initial state
prior_mean_f2[2,type=float] # factor 2 prior mean
prior_cov_f2[2,2,type=float]# factor 2 prior covariance
# Time index
t[1,type=int]            # discrete time step

## Connections
prior_mean_f1>x_f1
F_f1>x_f1
x_f1>y_f1
H_f1>y_f1
Q_f1>x_f1
R_f1>y_f1
u_f1>x_f1
goal_mean_f1>u_f1
control_gain_f1>u_f1
prior_mean_f2>x_f2
F_f2>x_f2
x_f2>y_f2
H_f2>y_f2
Q_f2>x_f2
R_f2>y_f2

## InitialParameterization
# F_f1: Euler discretization (dt = 0.1) of dx/dt = F x with mild velocity damping.
F_f1={
  (1.0, 0.1),
  (0.0, 0.9)
}

# H_f1: identity readout — the observation reads the factor-1 state directly.
H_f1={
  (1.0, 0.0),
  (0.0, 1.0)
}

# Q_f1: process (motion) noise — modest slip per step.
Q_f1={
  (0.05, 0.0),
  (0.0, 0.05)
}

# R_f1: observation noise on the factor-1 readings.
R_f1={
  (0.1, 0.0),
  (0.0, 0.1)
}

# Gaussian prior over the factor-1 initial state: start at the origin, fairly certain.
prior_mean_f1={(0.0, 0.0)}
prior_cov_f1={
  (0.5, 0.0),
  (0.0, 0.5)
}

# goal_mean_f1: the preferred state the closed-loop controller steers toward.
goal_mean_f1={(1.5, 0.0)}

# control_gain_f1: scalar proportional gain on (goal_mean_f1 - posterior_mean_f1).
control_gain_f1={(0.3)}

# F_f2: small-angle rotation (theta = 0.1 rad per step) — factor 2 orbits.
F_f2={
  (0.995, -0.100),
  (0.100, 0.995)
}

# H_f2: identity readout — the observation reads the factor-2 state directly.
H_f2={
  (1.0, 0.0),
  (0.0, 1.0)
}

# Q_f2: process (motion) noise — tight orbit per step.
Q_f2={
  (0.02, 0.0),
  (0.0, 0.02)
}

# R_f2: observation noise on the factor-2 readings.
R_f2={
  (0.1, 0.0),
  (0.0, 0.1)
}

# Gaussian prior over the factor-2 initial state: start on the unit circle.
prior_mean_f2={(1.0, 0.0)}
prior_cov_f2={
  (1.0, 0.0),
  (0.0, 1.0)
}

## Equations
# Generative model, per factor (linear-Gaussian state-space):
#   Factor 1 (damped drift, closed-loop):
#     x_f1,1 ~ N(prior_mean_f1, prior_cov_f1)
#     x_f1,t = F_f1 x_f1,t-1 + u_f1,t-1 + N(0, Q_f1)
#     y_f1,t = H_f1 x_f1,t + N(0, R_f1)
#     u_f1,t = control_gain_f1 * (goal_mean_f1 - mu_f1,t),
#       mu_f1,t = current filtered posterior mean of factor 1.
#   Factor 2 (rotation, passive: no control input):
#     x_f2,1 ~ N(prior_mean_f2, prior_cov_f2)
#     x_f2,t = F_f2 x_f2,t-1 + N(0, Q_f2)
#     y_f2,t = H_f2 x_f2,t + N(0, R_f2)
# The factors are independent: no cross-factor state, observation, or
# control coupling is declared.

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=15

## ActInfOntologyAnnotation
F_f1=StateTransitionMatrix
H_f1=ObservationMatrix
Q_f1=ProcessNoiseCovariance
R_f1=ObservationNoiseCovariance
prior_mean_f1=PriorMean
prior_cov_f1=PriorCovariance
goal_mean_f1=PreferredState
control_gain_f1=ControlGain
x_f1=ContinuousHiddenState
y_f1=ContinuousObservation
u_f1=ControlInput
F_f2=StateTransitionMatrix
H_f2=ObservationMatrix
Q_f2=ProcessNoiseCovariance
R_f2=ObservationNoiseCovariance
prior_mean_f2=PriorMean
prior_cov_f2=PriorCovariance
x_f2=ContinuousHiddenState
y_f2=ContinuousObservation
t=Time

## ModelParameters
num_factors: 2
num_timesteps: 15
dt: 0.1
random_seed: 42
num_states: 2
num_observations: 2

## Footer
Factored Continuous LGSSM v1 - a composed factored x continuous GNN model:
two independent 2-dim linear-Gaussian factors are declared with per-factor
keys (F_f1/... and F_f2/..., num_factors: 2). detect_model_kinds classifies
it {FACTORED, CONTINUOUS}; the JAX backend renders one native LGSSM block
per factor (FactoredContinuousSpec), while every other continuous backend
receipts it unsupported-factored-continuous instead of silently rendering
the factors flat as one joint LGSSM.

## Signature
Cryptographic signature goes here
