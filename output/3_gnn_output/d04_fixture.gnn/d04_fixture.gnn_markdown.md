## GNNVersionAndFlags
Version: 1.0

## ModelName
NEST D04 Synthetic 2-State LGSSM

## ModelAnnotation
Passive linear-Gaussian state-space model (no control input) generating the
summary indices of the D02 MESSAGEix wrapper blanket over T coupling iterations.
- Hidden state x = (decarb_rate_dev, demand_pressure), dimensionless deviations.
- Observation y = (emissions_index, price_index, objective_index, demand_index): relative indices of total
  emissions e, mean commodity price p, objective J and mean demand d against
  fixed reference values (see the fixture JSON, `blanket.references`).
- cap (emissions cap) is a declared exogenous schedule in the fixture, not a
  variable of this model.
Deliverable D04 (AII); pattern for D07 (3–5 states) and D14 (GNN → RxInfer render).

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
F = [[0.85, -0.1], [0.05, 0.9]]
H = [[-0.8, 0.5], [0.6, 0.4], [0.3, 0.7], [0.0, 1.0]]
Q = [[0.004, 0.0005], [0.0005, 0.006]]
R = [[0.0025, 0.0, 0.0, 0.0], [0.0, 0.0036, 0.0, 0.0], [0.0, 0.0, 0.0016, 0.0], [0.0, 0.0, 0.0, 0.0009]]
prior_mean = [[0.05, 0.0]]
prior_cov = [[0.02, 0.0], [0.0, 0.02]]
num_timesteps = 24
random_seed = 20260910
num_states = 2
num_observations = 4
num_regions = 1
num_commodities = 2
num_time_slices = 12

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
Generated: 2026-09-11T18:29:39.540417

## Signature
