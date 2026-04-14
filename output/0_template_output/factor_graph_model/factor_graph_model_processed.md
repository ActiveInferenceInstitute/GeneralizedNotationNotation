
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/structured/factor_graph_model.md
# Processed on: 2026-04-14T10:56:40.579010
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Factor Graph Active Inference Model
# GNN Version: 1.0
# Factor graph decomposition for tractable inference in structured models.

## GNNSection
ActInfFactorGraph

## GNNVersionAndFlags
GNN v1

## ModelName
Factor Graph Active Inference Model

## ModelAnnotation
A factor graph decomposition of an Active Inference generative model with:
- Two independent observation modalities (visual and proprioceptive)
- Two independent hidden state factors (position and velocity)
- Factored joint distribution: P(o,s) = P(o_vis|s_pos) * P(o_prop|s_vel) * P(s_pos|s_vel) * P(s_vel)
- Variable nodes: observation and state variables
- Factor nodes: likelihood and transition factors
- Enables modality-specific processing and efficient belief propagation

## StateSpaceBlock
# Visual modality
o_vis[6,1,type=int]    # Visual observations (6 possible)
A_vis[6,3,type=float]  # Visual likelihood factor: P(o_vis | s_pos)

# Proprioceptive modality
o_prop[4,1,type=float] # Proprioceptive observations (4D)
A_prop[4,2,type=float] # Proprioceptive likelihood: P(o_prop | s_vel)

# State factor 1: position
s_pos[3,1,type=float]  # Position state (3 discrete locations)
B_pos[3,3,2,type=float] # Position transition factor

# State factor 2: velocity
s_vel[2,1,type=float]  # Velocity state (2 levels: slow/fast)
B_vel[2,2,1,type=float] # Velocity transition (action-independent)

# Priors
D_pos[3,type=float]    # Prior over position
D_vel[2,type=float]    # Prior over velocity

# Preferences
C_vis[6,type=float]    # Visual preferences (goal location)
C_prop[4,type=float]   # Proprioceptive preferences (comfort)

# Action and policy
π[2,type=float]        # Policy (2 actions: move/stay)
u[1,type=int]          # Selected action
G[π,type=float]        # Expected Free Energy

# Factor graph messages
m_pos_to_vis[3,1,type=float]  # Message: position→visual factor
m_vel_to_prop[2,1,type=float] # Message: velocity→proprioceptive factor
m_vis_to_pos[3,1,type=float]  # Message: visual factor→position
m_prop_to_vel[2,1,type=float] # Message: proprioceptive factor→velocity

# Free Energy
F[1,type=float]        # Total Variational Free Energy (sum of factors)

# Time
t[1,type=int]

## Connections
D_pos>s_pos
D_vel>s_vel
s_pos-A_vis
A_vis-o_vis
s_vel-A_prop
A_prop-o_prop
s_pos-B_pos
s_vel-B_vel
B_pos>s_pos
B_vel>s_vel
s_pos-m_pos_to_vis
m_pos_to_vis-A_vis
o_vis-m_vis_to_pos
m_vis_to_pos-s_pos
s_vel-m_vel_to_prop
m_vel_to_prop-A_prop
o_prop-m_prop_to_vel
m_prop_to_vel-s_vel
C_vis>G
C_prop>G
G>π
π>u
B_pos>u
s_pos-F
s_vel-F
o_vis-F
o_prop-F

## InitialParameterization
A_vis={
  (0.8, 0.1, 0.1),
  (0.1, 0.8, 0.1),
  (0.1, 0.1, 0.8),
  (0.05, 0.45, 0.5),
  (0.45, 0.05, 0.5),
  (0.5, 0.5, 0.0)
}

A_prop={
  (0.9, 0.1),
  (0.1, 0.9),
  (0.5, 0.5),
  (0.5, 0.5)
}

B_pos={
  ( (0.9,0.1,0.0), (0.0,0.9,0.1), (0.1,0.0,0.9) ),
  ( (0.5,0.5,0.0), (0.0,0.5,0.5), (0.5,0.0,0.5) )
}

B_vel={
  ( (0.8, 0.2), (0.2, 0.8) )
}

D_pos={(0.333, 0.333, 0.333)}
D_vel={(0.5, 0.5)}

# Goal: observe visual feature 1 (index 0) and comfortable proprioception
C_vis={(2.0, -0.5, -0.5, -0.5, -0.5, -0.5)}
C_prop={(1.0, -1.0, 0.0, 0.0)}

## Equations
# Factored VFE: F = F_vis + F_prop + F_pos + F_vel
# F_vis = D_KL[Q(s_pos)||P(s_pos)] - E_Q[log P(o_vis|s_pos)]
# Message passing: belief propagation on factor graph
# Variable nodes aggregate incoming messages: Q(s) ∝ product of incoming messages
# Factor nodes compute messages as marginals of factor * incoming messages

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=Unbounded

## ActInfOntologyAnnotation
A_vis=VisualLikelihoodMatrix
A_prop=ProprioceptiveLikelihoodMatrix
B_pos=PositionTransitionMatrix
B_vel=VelocityTransitionMatrix
D_pos=PositionPrior
D_vel=VelocityPrior
C_vis=VisualPreferenceVector
C_prop=ProprioceptivePreferenceVector
s_pos=PositionHiddenState
s_vel=VelocityHiddenState
o_vis=VisualObservation
o_prop=ProprioceptiveObservation
π=PolicyVector
u=Action
G=ExpectedFreeEnergy
F=VariationalFreeEnergy
t=Time

## ModelParameters
num_positions: 3
num_velocities: 2
num_visual_obs: 6
num_proprio_obs: 4
num_actions: 2
num_timesteps: 25
num_modalities: 2
num_state_factors: 2

## Footer
Factor Graph Active Inference Model v1 - GNN Representation.
Factored model enables modality-specific processing.
Belief propagation on factor graph is more efficient than joint inference.

## Signature
Cryptographic signature goes here

