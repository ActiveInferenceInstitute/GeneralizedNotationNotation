## GNNVersionAndFlags
Version: 1.0

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
x[2,1],float
y[2,1],float
u[2,1],float
F[2,2],float
H[2,2],float
Q[2,2],float
R[2,2],float
prior_mean[2],float
prior_cov[2,2],float
goal_mean[2],float
control_gain[1],float
t[1],integer

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
F = [[1.0, 0.0], [0.0, 1.0]]
H = [[1.0, 0.0], [0.0, 1.0]]
Q = [[0.05, 0.0], [0.0, 0.05]]
R = [[0.1, 0.0], [0.0, 0.1]]
prior_mean = [[0.0, 0.0]]
prior_cov = [[0.5, 0.0], [0.0, 0.5]]
goal_mean = [[2.0, 2.0]]
control_gain = [[0.3]]
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
goal_mean = PreferredState
control_gain = ControlGain
x = ContinuousHiddenState
y = ContinuousObservation
u = ControlInput
t = Time

## Footer
Generated: 2026-09-05T20:30:55.005914

## Signature
