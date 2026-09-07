## GNNVersionAndFlags
Version: 1.0

## ModelName
Multi-Agent Cooperative Active Inference

## ModelAnnotation
Two Active Inference agents cooperating on a joint task:

- Agent 1 and Agent 2 each maintain independent beliefs
- Shared observation space: agents observe each other's actions
- Joint task state includes both agents' positions (4x4 = 16 joint states)
- Cooperative preferences: both agents prefer the same goal configuration
- Models social cognition and coordination without explicit communication

## StateSpaceBlock
A_agent1[4,4],float
B_agent1[4,4,3],float
C_agent1[4],float
D_agent1[4],float
s_agent1[4,1],float
x_next1[4,1],float
o_agent1[4,1],integer
π1[3],float
u1[1],integer
G1[1],float
A_agent2[4,4],float
B_agent2[4,4,3],float
C_agent2[4],float
D_agent2[4],float
s_agent2[4,1],float
x_next2[4,1],float
o_agent2[4,1],integer
π2[3],float
u2[1],integer
G2[1],float
s_joint[16,1],float
o_joint[4,1],integer
t[1],integer

## Connections
D_agent1>s_agent1
s_agent1-A_agent1
A_agent1-o_agent1
s_agent1>x_next1
C_agent1>G1
G1>π1
π1>u1
B_agent1>u1
D_agent2>s_agent2
s_agent2-A_agent2
A_agent2-o_agent2
s_agent2>x_next2
C_agent2>G2
G2>π2
π2>u2
B_agent2>u2
u1>s_joint
u2>s_joint
s_joint-o_joint
o_agent1-s_joint
o_agent2-s_joint

## InitialParameterization
A_agent1 = [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85]]
A_agent2 = [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85]]
C_agent1 = [[-1.0, -1.0, -1.0, 2.0]]
C_agent2 = [[-1.0, -1.0, -1.0, 2.0]]
D_agent1 = [[0.25, 0.25, 0.25, 0.25]]
D_agent2 = [[0.25, 0.25, 0.25, 0.25]]
B_agent1 = [[[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1], [0.1, 0.0, 0.0, 0.9]], [[0.9, 0.0, 0.0, 0.1], [0.1, 0.9, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0], [0.0, 0.0, 0.1, 0.9]], [[0.8, 0.1, 0.1, 0.0], [0.1, 0.8, 0.0, 0.1], [0.1, 0.0, 0.8, 0.1], [0.0, 0.1, 0.1, 0.8]]]
B_agent2 = [[[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1], [0.1, 0.0, 0.0, 0.9]], [[0.9, 0.0, 0.0, 0.1], [0.1, 0.9, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0], [0.0, 0.0, 0.1, 0.9]], [[0.8, 0.1, 0.1, 0.0], [0.1, 0.8, 0.0, 0.1], [0.1, 0.0, 0.8, 0.1], [0.0, 0.1, 0.1, 0.8]]]
num_hidden_states = 16
num_obs = 16
num_actions = 3
num_agents = 2
num_hidden_states_per_agent = 4
num_obs_per_agent = 4
num_actions_per_agent = 3
num_timesteps = 20

## Time
Dynamic
ModelTimeHorizon = 20

## ActInfOntologyAnnotation
A_agent1 = LikelihoodMatrix
B_agent1 = TransitionMatrix
C_agent1 = LogPreferenceVector
D_agent1 = PriorOverHiddenStates
s_agent1 = Agent1HiddenState
x_next1 = Agent1NextHiddenState
o_agent1 = Agent1Observation
π1 = Agent1PolicyVector
u1 = Agent1Action
G1 = Agent1ExpectedFreeEnergy
A_agent2 = LikelihoodMatrix
B_agent2 = TransitionMatrix
C_agent2 = LogPreferenceVector
D_agent2 = PriorOverHiddenStates
s_agent2 = Agent2HiddenState
x_next2 = Agent2NextHiddenState
o_agent2 = Agent2Observation
π2 = Agent2PolicyVector
u2 = Agent2Action
G2 = Agent2ExpectedFreeEnergy
s_joint = JointState
o_joint = JointObservation
t = Time

## Footer
Generated: 2026-09-06T11:26:43.399913

## Signature
