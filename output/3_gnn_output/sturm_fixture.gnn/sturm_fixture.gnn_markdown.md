## GNNVersionAndFlags
Version: 1.0

## ModelName
NEST D03 Synthetic 2-State LGSSM

## ModelAnnotation
Passive linear-Gaussian state-space model (no control input) generating the
summary indices of the D03 STURM wrapper blanket over T coupling iterations.
- Hidden state x = (renovation_rate_dev, price_pressure), dimensionless deviations.
- Observation y = (price_index, demand_index, stock_index, turnover_index): relative indices of fuel prices p
  (sensory, into STURM), final energy demand d (active, out of STURM), the
  archetype stock aggregate and the turnover aggregate against fixed reference
  values (see the fixture JSON, `blanket.references`).
- q_agg/r_agg are internal aggregates carried in the fixture for posterior
  checks only; they never cross the blanket.
- There is NO counterpart to D02's cap: the cap is the Q-13 HIERARCHY imposed
  constraint of the MESSAGEix side; STURM receives none.
Deliverable D03 (AII); mirror of the D04 fixture under the D03 blanket.

## StateSpaceBlock
x[2,1],float
y[4,1],float
F[2,2],float
H[4,2],float
Q[2,2],float
R[4,4],float
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
F = [[0.8, -0.15], [0.05, 0.9]]
H = [[-0.4, 0.7], [0.5, -0.3], [0.6, 0.0], [0.8, 0.0]]
Q = [[0.0036, -0.0003], [-0.0003, 0.0049]]
R = [[0.0016, 0.0, 0.0, 0.0], [0.0, 0.0025, 0.0, 0.0], [0.0, 0.0, 0.0009, 0.0], [0.0, 0.0, 0.0, 0.0012]]
prior_mean = [[0.02, 0.0]]
prior_cov = [[0.02, 0.0], [0.0, 0.02]]
num_timesteps = 24
random_seed = 20260911
num_states = 2
num_observations = 4
num_regions = 1
num_fuel_types = 2
num_time_slices = 12
num_archetypes = 4

## Time
Dynamic
ModelTimeHorizon = 24

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
Generated: 2026-09-11T18:29:40.240459

## Signature
