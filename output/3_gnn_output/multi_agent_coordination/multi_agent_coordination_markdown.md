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
A1[4,4],float
B1[4,4,3],float
C1[4],float
D1[4],float
s1[4,1],float
s1_prime[4,1],float
o1[4,1],integer
π1[3],float
u1[1],integer
G1[1],float
A2[4,4],float
B2[4,4,3],float
C2[4],float
D2[4],float
s2[4,1],float
s2_prime[4,1],float
o2[4,1],integer
π2[3],float
u2[1],integer
G2[1],float
s_joint[16,1],float
o_joint[4,1],integer
t[1],integer

## Connections
D1>s1
s1-A1
A1-o1
s1>s1_prime
C1>G1
G1>π1
π1>u1
B1>u1
D2>s2
s2-A2
A2-o2
s2>s2_prime
C2>G2
G2>π2
π2>u2
B2>u2
u1>s_joint
u2>s_joint
s_joint-o_joint
o1-s_joint
o2-s_joint

## InitialParameterization
A1 = [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85]]
A2 = [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85]]
C1 = [[-1.0, -1.0, -1.0, 2.0]]
C2 = [[-1.0, -1.0, -1.0, 2.0]]
D1 = [[0.25, 0.25, 0.25, 0.25]]
D2 = [[0.25, 0.25, 0.25, 0.25]]
B1 = [[[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1], [0.1, 0.0, 0.0, 0.9]], [[0.9, 0.0, 0.0, 0.1], [0.1, 0.9, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0], [0.0, 0.0, 0.1, 0.9]], [[0.8, 0.1, 0.1, 0.0], [0.1, 0.8, 0.0, 0.1], [0.1, 0.0, 0.8, 0.1], [0.0, 0.1, 0.1, 0.8]]]
B2 = [[[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1], [0.1, 0.0, 0.0, 0.9]], [[0.9, 0.0, 0.0, 0.1], [0.1, 0.9, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0], [0.0, 0.0, 0.1, 0.9]], [[0.8, 0.1, 0.1, 0.0], [0.1, 0.8, 0.0, 0.1], [0.1, 0.0, 0.8, 0.1], [0.0, 0.1, 0.1, 0.8]]]

## Time
Dynamic
ModelTimeHorizon = 20

## ActInfOntologyAnnotation
A1 = LikelihoodMatrix
B1 = TransitionMatrix
C1 = LogPreferenceVector
D1 = PriorOverHiddenStates
s1 = Agent1HiddenState
s1_prime = Agent1NextHiddenState
o1 = Agent1Observation
π1 = Agent1PolicyVector
u1 = Agent1Action
G1 = Agent1ExpectedFreeEnergy
A2 = LikelihoodMatrix
B2 = TransitionMatrix
C2 = LogPreferenceVector
D2 = PriorOverHiddenStates
s2 = Agent2HiddenState
s2_prime = Agent2NextHiddenState
o2 = Agent2Observation
π2 = Agent2PolicyVector
u2 = Agent2Action
G2 = Agent2ExpectedFreeEnergy
s_joint = JointState
o_joint = JointObservation
t = Time

## Footer
Generated: 2026-04-12T17:23:00.877411

## Signature
