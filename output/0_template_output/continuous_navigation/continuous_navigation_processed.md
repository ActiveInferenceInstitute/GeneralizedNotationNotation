
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/continuous/continuous_navigation.md
# Processed on: 2026-04-14T11:51:42.086846
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Continuous State Navigation Agent
# GNN Version: 1.0
# Active Inference agent with continuous (Gaussian) state space.

## GNNSection
ActInfContinuous

## GNNVersionAndFlags
GNN v1

## ModelName
Continuous State Navigation Agent

## ModelAnnotation
A continuous-state Active Inference agent navigating a 2D environment:
- Hidden states: 2D position (x, y) as Gaussian belief
- Observations: noisy position measurements with Gaussian noise
- Actions: 2D velocity commands (dx, dy)
- Uses Laplace approximation for Gaussian belief updating
- Generalized coordinates of motion for smooth trajectories

## StateSpaceBlock
# Continuous state (2D position belief as Gaussian)
μ[2,1,type=float]      # Mean of position belief (x, y)
Σ[2,2,type=float]      # Covariance of position belief
μ_prime[2,1,type=float] # Next position mean (predicted)

# Generalized coordinates
μ_dot[2,1,type=float]  # Velocity (first motion derivative)
μ_ddot[2,1,type=float] # Acceleration (second motion derivative)

# Observation model (Gaussian)
A_μ[2,2,type=float]    # Observation mean mapping (identity + noise)
A_Σ[2,2,type=float]    # Observation noise covariance
o[2,1,type=float]      # Observation (noisy position measurement)

# Dynamics model (linear Gaussian)
B_f[2,2,type=float]    # State transition matrix (flow)
B_u[2,2,type=float]    # Action effect matrix

# Preferences (target position as Gaussian)
C_μ[2,1,type=float]    # Target position (preference mean)
C_Σ[2,2,type=float]    # Preference uncertainty

# Action
u[2,1,type=float]      # Continuous velocity command

# Free Energy quantities
F[1,type=float]        # Variational Free Energy (scalar)
G[1,type=float]        # Expected Free Energy (scalar)
ε_o[2,1,type=float]    # Sensory prediction error
ε_x[2,1,type=float]    # Dynamic prediction error

# Precision
Π_o[2,2,type=float]    # Sensory precision matrix (inverse noise cov)
Π_x[2,2,type=float]    # Dynamic precision matrix

# Time
t[1,type=float]        # Continuous time

## Connections
μ-A_μ
A_μ-o
A_Σ-ε_o
o-ε_o
ε_o-F
Π_o-F
μ-B_f
B_f-μ_prime
B_u-μ_prime
u-B_u
ε_x-F
Π_x-F
C_μ-G
C_Σ-G
μ_prime-G
G>u
μ-Σ
Σ-Π_o

## InitialParameterization
# Identity observation mapping
A_μ={(1.0, 0.0), (0.0, 1.0)}

# Observation noise (moderate uncertainty)
A_Σ={(0.1, 0.0), (0.0, 0.1)}

# State transition (identity: position persists without action)
B_f={(1.0, 0.0), (0.0, 1.0)}
B_u={(0.1, 0.0), (0.0, 0.1)}

# Target: goal position at (1.0, 1.0)
C_μ={(1.0), (1.0)}
C_Σ={(0.05, 0.0), (0.0, 0.05)}

# Initial belief: starting at origin with moderate uncertainty
μ={(0.0), (0.0)}
Σ={(0.5, 0.0), (0.0, 0.5)}

# Precision matrices (inverse of covariance)
Π_o={(10.0, 0.0), (0.0, 10.0)}
Π_x={(20.0, 0.0), (0.0, 20.0)}

## Equations
# Sensory prediction error: ε_o = o - A_μ * μ
# Dynamic prediction error: ε_x = μ_dot - (B_f * μ + B_u * u)
# Free Energy: F = 0.5 * (ε_o^T Π_o ε_o + ε_x^T Π_x ε_x)
# Belief update: dμ/dt = -∂F/∂μ (gradient descent on VFE)
# Action: u = -∂G/∂u (minimize Expected Free Energy w.r.t. action)

## Time
Time=t
Dynamic
Continuous
ModelTimeHorizon=10.0

## ActInfOntologyAnnotation
μ=BeliefMean
Σ=BeliefCovariance
A_μ=ObservationMeanMapping
A_Σ=ObservationNoise
o=Observation
B_f=DynamicsMatrix
B_u=ActionEffectMatrix
C_μ=PreferenceMean
C_Σ=PreferenceCovariance
u=ContinuousAction
F=VariationalFreeEnergy
G=ExpectedFreeEnergy
ε_o=SensoryPredictionError
ε_x=DynamicPredictionError
Π_o=SensoryPrecision
Π_x=DynamicPrecision
t=ContinuousTime

## ModelParameters
state_dim: 2
obs_dim: 2
action_dim: 2
dt: 0.01
simulation_time: 10.0
goal_x: 1.0
goal_y: 1.0

## Footer
Continuous State Navigation Agent v1 - GNN Representation.
Uses Laplace approximation for Gaussian belief updating.
Generalized coordinates enable smooth predictive control.

## Signature
Cryptographic signature goes here

