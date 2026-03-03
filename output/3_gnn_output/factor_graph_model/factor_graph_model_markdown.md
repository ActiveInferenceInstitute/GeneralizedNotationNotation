## GNNVersionAndFlags
Version: 1.0

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
o_vis[6,1],integer
A_vis[6,3],float
o_prop[4,1],float
A_prop[4,2],float
s_pos[3,1],float
B_pos[3,3,2],float
s_vel[2,1],float
B_vel[2,2,1],float
D_pos[3],float
D_vel[2],float
C_vis[6],float
C_prop[4],float
π[2],float
u[1],integer
G[1],float
m_pos_to_vis[3,1],float
m_vel_to_prop[2,1],float
m_vis_to_pos[3,1],float
m_prop_to_vel[2,1],float
F[1],float
t[1],integer

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
A_vis = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.05, 0.45, 0.5], [0.45, 0.05, 0.5], [0.5, 0.5, 0.0]]
A_prop = [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.5, 0.5]]
B_pos = [[[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]], [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]]]
B_vel = [[[0.8, 0.2], [0.2, 0.8]]]
D_pos = [[0.333, 0.333, 0.333]]
D_vel = [[0.5, 0.5]]
C_vis = [[2.0, -0.5, -0.5, -0.5, -0.5, -0.5]]
C_prop = [[1.0, -1.0, 0.0, 0.0]]

## Time
Dynamic
ModelTimeHorizon = Unbounded

## ActInfOntologyAnnotation
A_vis = VisualLikelihoodMatrix
A_prop = ProprioceptiveLikelihoodMatrix
B_pos = PositionTransitionMatrix
B_vel = VelocityTransitionMatrix
D_pos = PositionPrior
D_vel = VelocityPrior
C_vis = VisualPreferenceVector
C_prop = ProprioceptivePreferenceVector
s_pos = PositionHiddenState
s_vel = VelocityHiddenState
o_vis = VisualObservation
o_prop = ProprioceptiveObservation
π = PolicyVector
u = Action
G = ExpectedFreeEnergy
F = VariationalFreeEnergy
t = Time

## Footer
Generated: 2026-03-03T08:22:03.397248

## Signature
