
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/multiagent/multi_agent_coordination.md
# Processed on: 2026-04-14T11:51:42.083261
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Multi-Agent Cooperative Active Inference

# GNN Version: 1.0

# Two agents cooperating via shared observation space

## GNNSection

ActInfPOMDP_MultiAgent

## GNNVersionAndFlags

GNN v1

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

# Agent 1

A1[4,4,type=float]     # Agent 1 likelihood
B1[4,4,3,type=float]   # Agent 1 transitions (3 actions)
C1[4,type=float]       # Agent 1 preferences
D1[4,type=float]       # Agent 1 prior
s1[4,1,type=float]     # Agent 1 hidden state
s1_prime[4,1,type=float] # Agent 1 next hidden state
o1[4,1,type=int]       # Agent 1 observations (includes Agent 2 obs)
π1[3,type=float]       # Agent 1 policy
u1[1,type=int]         # Agent 1 action
G1[π1,type=float]      # Agent 1 EFE

# Agent 2

A2[4,4,type=float]     # Agent 2 likelihood
B2[4,4,3,type=float]   # Agent 2 transitions (3 actions)
C2[4,type=float]       # Agent 2 preferences
D2[4,type=float]       # Agent 2 prior
s2[4,1,type=float]     # Agent 2 hidden state
s2_prime[4,1,type=float] # Agent 2 next hidden state
o2[4,1,type=int]       # Agent 2 observations (includes Agent 1 obs)
π2[3,type=float]       # Agent 2 policy
u2[1,type=int]         # Agent 2 action
G2[π2,type=float]      # Agent 2 EFE

# Shared environment state

s_joint[16,1,type=float]  # Joint state (Agent1_pos x Agent2_pos)
o_joint[4,1,type=int]     # Joint observation (goal achievement)

# Time

t[1,type=int]

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

A1={
  (0.85, 0.05, 0.05, 0.05),
  (0.05, 0.85, 0.05, 0.05),
  (0.05, 0.05, 0.85, 0.05),
  (0.05, 0.05, 0.05, 0.85)
}

A2={
  (0.85, 0.05, 0.05, 0.05),
  (0.05, 0.85, 0.05, 0.05),
  (0.05, 0.05, 0.85, 0.05),
  (0.05, 0.05, 0.05, 0.85)
}

# Shared cooperative preference: goal = state 4 (index 3)

C1={(-1.0, -1.0, -1.0, 2.0)}
C2={(-1.0, -1.0, -1.0, 2.0)}
D1={(0.25, 0.25, 0.25, 0.25)}
D2={(0.25, 0.25, 0.25, 0.25)}

B1={
  ( (0.9,0.1,0.0,0.0), (0.0,0.9,0.1,0.0), (0.0,0.0,0.9,0.1), (0.1,0.0,0.0,0.9) ),
  ( (0.9,0.0,0.0,0.1), (0.1,0.9,0.0,0.0), (0.0,0.1,0.9,0.0), (0.0,0.0,0.1,0.9) ),
  ( (0.8,0.1,0.1,0.0), (0.1,0.8,0.0,0.1), (0.1,0.0,0.8,0.1), (0.0,0.1,0.1,0.8) )
}

B2={
  ( (0.9,0.1,0.0,0.0), (0.0,0.9,0.1,0.0), (0.0,0.0,0.9,0.1), (0.1,0.0,0.0,0.9) ),
  ( (0.9,0.0,0.0,0.1), (0.1,0.9,0.0,0.0), (0.0,0.1,0.9,0.0), (0.0,0.0,0.1,0.9) ),
  ( (0.8,0.1,0.1,0.0), (0.1,0.8,0.0,0.1), (0.1,0.0,0.8,0.1), (0.0,0.1,0.1,0.8) )
}

## Equations

# Each agent independently minimizes their own VFE

# Coordination emerges from shared observation space and aligned preferences

# Agent 1 observes both own state and Agent 2's last action

# No explicit communication channel — implicit coordination only

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=20

## ActInfOntologyAnnotation

A1=LikelihoodMatrix
B1=TransitionMatrix
C1=LogPreferenceVector
D1=PriorOverHiddenStates
s1=Agent1HiddenState
s1_prime=Agent1NextHiddenState
o1=Agent1Observation
π1=Agent1PolicyVector
u1=Agent1Action
G1=Agent1ExpectedFreeEnergy
A2=LikelihoodMatrix
B2=TransitionMatrix
C2=LogPreferenceVector
D2=PriorOverHiddenStates
s2=Agent2HiddenState
s2_prime=Agent2NextHiddenState
o2=Agent2Observation
π2=Agent2PolicyVector
u2=Agent2Action
G2=Agent2ExpectedFreeEnergy
s_joint=JointState
o_joint=JointObservation
t=Time

## ModelParameters

num_agents: 2
num_hidden_states_per_agent: 4
num_obs_per_agent: 4
num_actions_per_agent: 3
num_timesteps: 20

## Footer

Multi-Agent Cooperative Active Inference v1 - GNN Representation.
Implicit coordination via shared observation space.
No explicit communication — emergent cooperation from aligned preferences.

## Signature

Cryptographic signature goes here

