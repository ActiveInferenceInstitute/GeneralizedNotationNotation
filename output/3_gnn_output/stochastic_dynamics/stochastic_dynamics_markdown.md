## GNNVersionAndFlags
Version: 1.0

## ModelName
Stochastic Continuous Dynamics Agent

## ModelAnnotation
A continuous-state Active Inference agent whose dynamics include
explicit process and observation noise. This sample exercises the
continuous-time path of the language: Time=Continuous, state as a
multi-dimensional vector, and Equations containing noise terms ε_state
and ε_obs that downstream renderers must handle.

- 4-dimensional continuous hidden state (e.g., position + velocity in 2D)
- 2-dimensional continuous observation (noisy position readout)
- Linear Gaussian dynamics: ẋ = Fx + Gu + ε_state
- Linear Gaussian observation: o = Hx + ε_obs
- Precision parameters γ_state (process) and γ_obs (observation)

Used to validate that downstream backends either generate SDE solvers
(Stan via latent-variable sampling, NumPyro) or raise a clean
"unsupported feature" warning (PyMDP, which is discrete-only).

## StateSpaceBlock
x[4,1],float
o[2,1],float
u[2,1],float
F[4,4],float
G[4,2],float
H[2,4],float
gamma_state[1],float
gamma_obs[1],float
epsilon_state[4,1],float
epsilon_obs[2,1],float

## Connections
x>F
u>F
epsilon_state>F
F>x
x>H
epsilon_obs>H
H>o
gamma_state-epsilon_state
gamma_obs-epsilon_obs

## InitialParameterization
F = [[1.0, 0.0, 0.1, 0.0], [0.0, 1.0, 0.0, 0.1], [0.0, 0.0, 0.95, 0.0], [0.0, 0.0, 0.0, 0.95]]
G = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
H = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
gamma_state = [[10.0]]
gamma_obs = [[5.0]]
state_dim = 4
obs_dim = 2
control_dim = 2
num_timesteps = 50
time_horizon = 5.0
integration_step = 0.1

## Time
Static
ContinuousTime
ModelTimeHorizon = 5.0

## ActInfOntologyAnnotation
x = ContinuousHiddenState
o = ContinuousObservation
u = ContinuousAction
F = DriftMatrix
G = ControlMatrix
H = ObservationMatrix
gamma_state = ProcessNoisePrecision
gamma_obs = ObservationNoisePrecision
epsilon_state = ProcessNoise
epsilon_obs = ObservationNoise

## Footer
Generated: 2026-05-22T06:17:19.369826

## Signature
