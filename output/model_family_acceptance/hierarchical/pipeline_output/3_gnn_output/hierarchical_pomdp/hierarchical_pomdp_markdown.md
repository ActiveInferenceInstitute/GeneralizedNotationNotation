## GNNVersionAndFlags
Version: 1.0

## ModelName
Hierarchical Active Inference POMDP

## ModelAnnotation
A two-level hierarchical POMDP where:
- Level 1 (fast): 4 observations, 4 hidden states, 3 actions
- Level 2 (slow): 2 contextual states that modulate Level 1 likelihood
- Higher-level beliefs are updated at a slower timescale
- Top-down predictions constrain bottom-up inference at Level 1

## StateSpaceBlock
A_level1[4,4],float
B_level1[4,4,3],float
C_level1[4],float
D_level1[4],float
s_level1[4,1],float
x_next1[4,1],float
o_level1[4,1],integer
π1[3],float
u_level1[1],integer
G1[1],float
A_level2[4,2],float
B_level2[2,2,1],float
C_level2[2],float
D_level2[2],float
s_level2[2,1],float
o_level2[4,1],float
G2[1],float
t1[1],integer
t2[1],integer

## Connections
D_level1>s_level1
s_level1-A_level1
s_level1>x_next1
A_level1-o_level1
C_level1>G1
G1>π1
π1>u_level1
B_level1>u_level1
u_level1>x_next1
s_level1>o_level2
D_level2>s_level2
s_level2-A_level2
A_level2>D_level1
s_level2-B_level2
C_level2>G2
G2>s_level2

## InitialParameterization
A_level1 = [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85]]
B_level1 = [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
C_level1 = [[0.1, 0.1, 0.1, 1.0]]
D_level1 = [[0.25, 0.25, 0.25, 0.25]]
A_level2 = [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.5, 0.5]]
B_level2 = [[[0.9, 0.1], [0.1, 0.9]]]
C_level2 = [[0.0, 0.5, 0.0, 0.5]]
D_level2 = [[0.5, 0.5]]
num_hidden_states = 8
num_obs = 16
num_actions = 3
num_timesteps = 20
num_hidden_states_l1 = 4
num_obs_l1 = 4
num_actions_l1 = 3
num_context_states_l2 = 2
timescale_ratio = 5

## Time
Dynamic
ModelTimeHorizon = Unbounded

## ActInfOntologyAnnotation
A_level1 = LikelihoodMatrix
B_level1 = TransitionMatrix
C_level1 = LogPreferenceVector
D_level1 = PriorOverHiddenStates
s_level1 = HiddenState
o_level1 = Observation
π1 = PolicyVector
u_level1 = Action
G1 = ExpectedFreeEnergy
A_level2 = HigherLevelLikelihoodMatrix
B_level2 = ContextTransitionMatrix
s_level2 = ContextualHiddenState
o_level2 = HigherLevelObservation
G2 = HigherLevelExpectedFreeEnergy

## Footer
Generated: 2026-09-06T11:17:18.591564

## Signature
