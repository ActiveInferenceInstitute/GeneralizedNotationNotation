## GNNVersionAndFlags
Version: 1.0

## ModelName
Three-Level Temporal Hierarchy Agent

## ModelAnnotation
A three-level hierarchical Active Inference agent with distinct temporal scales:

- Level 0 (fast, 100ms): Sensorimotor control — immediate reflexive responses
- Level 1 (medium, 1s): Tactical planning — goal-directed behavior sequences
- Level 2 (slow, 10s): Strategic planning — long-term objective management
- Top-down flow: Strategy sets tactical goals, tactics set sensorimotor preferences
- Bottom-up flow: Sensorimotor observations inform tactical beliefs, tactical outcomes inform strategy
- Each level maintains its own generative model with A, B, C, D matrices
- Timescale separation encoded via update ratios (Level 2 updates every 10 Level 0 steps)
- Demonstrates deep temporal models from Friston et al. hierarchical Active Inference

## StateSpaceBlock
A_level0[3,4],float
B_level0[4,4,3],float
C_level0[3],float
D_level0[4],float
s_level0[4,1],float
o_level0[3,1],integer
pi0[3],float
u_level0[1],integer
G0[1],float
A_level1[4,3],float
B_level1[3,3,3],float
C_level1[4],float
D_level1[3],float
s_level1[3,1],float
o_level1[4,1],float
pi1[3],float
u_level1[1],integer
G1[1],float
A_level2[3,2],float
B_level2[2,2,2],float
C_level2[3],float
D_level2[2],float
s_level2[2,1],float
o_level2[3,1],float
pi2[2],float
u_level2[1],integer
G2[1],float
tau_level0[1],float
tau_level1[1],float
tau_level2[1],float
t[1],integer

## Connections
D_level0>s_level0
s_level0-A_level0
A_level0-o_level0
C_level0>G0
G0>pi0
pi0>u_level0
B_level0>u_level0
D_level1>s_level1
s_level1-A_level1
A_level1-o_level1
C_level1>G1
G1>pi1
pi1>u_level1
B_level1>u_level1
D_level2>s_level2
s_level2-A_level2
A_level2-o_level2
C_level2>G2
G2>pi2
pi2>u_level2
B_level2>u_level2
s_level2>C_level1
s_level1>C_level0
s_level2>D_level1
s_level0>o_level1
s_level1>o_level2

## InitialParameterization
A_level0 = [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05]]
C_level0 = [[0.0, -1.0, 1.0]]
D_level0 = [[0.25, 0.25, 0.25, 0.25]]
A_level1 = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.1]]
C_level1 = [[-0.5, 1.0, 1.5, -1.0]]
D_level1 = [[0.33, 0.33, 0.34]]
A_level2 = [[0.9, 0.1], [0.1, 0.9], [0.1, 0.1]]
C_level2 = [[-1.0, 2.0, 0.5]]
D_level2 = [[0.5, 0.5]]
tau_level0 = [[0.1]]
tau_level1 = [[1.0]]
tau_level2 = [[10.0]]
B_level0 = [[[0.9, 0.05, 0.05, 0.0], [0.05, 0.9, 0.05, 0.0], [0.05, 0.05, 0.9, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.05, 0.9, 0.05, 0.0], [0.9, 0.05, 0.05, 0.0], [0.05, 0.05, 0.9, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.9, 0.05, 0.05, 0.0], [0.05, 0.9, 0.05, 0.0], [0.05, 0.05, 0.9, 0.0], [0.0, 0.0, 0.0, 1.0]]]
B_level1 = [[[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], [[0.05, 0.9, 0.05], [0.9, 0.05, 0.05], [0.05, 0.05, 0.9]], [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]]
B_level2 = [[[0.95, 0.05], [0.05, 0.95]]]
num_hidden_states = 24
num_obs = 36
num_actions = 3
num_levels = 3
num_states_l0 = 4
num_obs_l0 = 3
num_actions_l0 = 3
num_states_l1 = 3
num_obs_l1 = 4
num_actions_l1 = 3
num_states_l2 = 2
num_obs_l2 = 3
num_actions_l2 = 2
timescale_ratio_1_0 = 10
timescale_ratio_2_1 = 10
num_timesteps = 100

## Time
Dynamic
ModelTimeHorizon = 100

## ActInfOntologyAnnotation
A_level0 = FastLikelihoodMatrix
B_level0 = FastTransitionMatrix
C_level0 = FastPreferenceVector
D_level0 = FastPrior
s_level0 = FastHiddenState
o_level0 = FastObservation
pi0 = FastPolicyVector
u_level0 = FastAction
G0 = FastExpectedFreeEnergy
A_level1 = TacticalLikelihoodMatrix
B_level1 = TacticalTransitionMatrix
C_level1 = TacticalPreferenceVector
D_level1 = TacticalPrior
s_level1 = TacticalHiddenState
o_level1 = TacticalObservation
pi1 = TacticalPolicyVector
u_level1 = TacticalAction
G1 = TacticalExpectedFreeEnergy
A_level2 = StrategicLikelihoodMatrix
B_level2 = StrategicTransitionMatrix
C_level2 = StrategicPreferenceVector
D_level2 = StrategicPrior
s_level2 = StrategicHiddenState
o_level2 = StrategicObservation
pi2 = StrategicPolicyVector
u_level2 = StrategicAction
G2 = StrategicExpectedFreeEnergy
tau_level0 = FastTimeConstant
tau_level1 = TacticalTimeConstant
tau_level2 = StrategicTimeConstant
t = Time

## Footer
Generated: 2026-09-06T11:17:18.497065

## Signature
