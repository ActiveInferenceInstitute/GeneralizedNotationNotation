
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/multiagent/stigmergic_swarm.md
# Processed on: 2026-04-10T10:23:34.159346
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Stigmergic Swarm Active Inference

# GNN Version: 1.0

# Three agents coordinating via environmental traces (stigmergy)

## GNNSection

ActInfPOMDP_MultiAgent

## GNNVersionAndFlags

GNN v1

## ModelName

Stigmergic Swarm Active Inference

## ModelAnnotation

Three Active Inference agents coordinating via stigmergy (environmental traces):

- No direct communication between agents — coordination emerges from environment
- Agents deposit and sense environmental signals (pheromone analogy)
- Shared 3x3 grid environment with signal intensity at each cell
- Each agent navigates independently while responding to accumulated signals
- Signal deposition: actions leave traces that other agents can observe
- Signal decay: environmental signals decay over time (volatility)
- Demonstrates emergent collective behavior from individual free energy minimization
- Models ant colony foraging, distributed robotics, and decentralized coordination

## StateSpaceBlock

# Agent 1

A1[4,9,type=float]         # Agent 1 likelihood: P(obs | position on 3x3 grid)
B1[9,9,4,type=float]       # Agent 1 transitions: (9 positions × 4 actions: N/S/E/W)
C1[4,type=float]           # Agent 1 preferences over observations
D1[9,type=float]           # Agent 1 position prior
s1[9,1,type=float]         # Agent 1 position belief (9 grid cells)
o1[4,1,type=int]           # Agent 1 observation: (empty, signal_low, signal_high, goal)
pi1[4,type=float]          # Agent 1 policy
u1[1,type=int]             # Agent 1 action
G1[pi1,type=float]         # Agent 1 EFE

# Agent 2

A2[4,9,type=float]         # Agent 2 likelihood
B2[9,9,4,type=float]       # Agent 2 transitions
C2[4,type=float]           # Agent 2 preferences
D2[9,type=float]           # Agent 2 position prior
s2[9,1,type=float]         # Agent 2 position belief
o2[4,1,type=int]           # Agent 2 observation
pi2[4,type=float]          # Agent 2 policy
u2[1,type=int]             # Agent 2 action
G2[pi2,type=float]         # Agent 2 EFE

# Agent 3

A3[4,9,type=float]         # Agent 3 likelihood
B3[9,9,4,type=float]       # Agent 3 transitions
C3[4,type=float]           # Agent 3 preferences
D3[9,type=float]           # Agent 3 position prior
s3[9,1,type=float]         # Agent 3 position belief
o3[4,1,type=int]           # Agent 3 observation
pi3[4,type=float]          # Agent 3 policy
u3[1,type=int]             # Agent 3 action
G3[pi3,type=float]         # Agent 3 EFE

# Shared environment (stigmergic signals)

env_signal[9,1,type=float]     # Signal intensity at each grid cell (0.0 to 1.0)
signal_decay[1,type=float]     # Signal decay rate per timestep

# Time

t[1,type=int]                  # Discrete time step

## Connections

# Agent 1

D1>s1
s1-A1
A1-o1
C1>G1
G1>pi1
pi1>u1
B1>u1

# Agent 2

D2>s2
s2-A2
A2-o2
C2>G2
G2>pi2
pi2>u2
B2>u2

# Agent 3

D3>s3
s3-A3
A3-o3
C3>G3
G3>pi3
pi3>u3
B3>u3

# Stigmergic coupling via environment

u1>env_signal
u2>env_signal
u3>env_signal
env_signal-A1
env_signal-A2
env_signal-A3
signal_decay>env_signal

## InitialParameterization

# All agents have identical generative models (homogeneous swarm)

# Likelihood: observation depends on position and environmental signal

# At each cell, agent observes: empty (no signal), signal_low, signal_high, or goal

A1={
  (0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1),
  (0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1),
  (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
  (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7)
}

A2={
  (0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1),
  (0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1),
  (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
  (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7)
}

A3={
  (0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1),
  (0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1),
  (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
  (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7)
}

# Shared preferences: follow signals, seek goal

C1={(-0.5, 0.5, 1.5, 3.0)}
C2={(-0.5, 0.5, 1.5, 3.0)}
C3={(-0.5, 0.5, 1.5, 3.0)}

# Starting positions: agents start at different corners

D1={(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
D2={(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
D3={(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)}

# Initial environment: no signals

env_signal={(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}

# Signal decay: 10% per timestep

signal_decay={(0.9)}

## Equations

# Each agent independently minimizes VFE and selects actions via EFE

# Stigmergic coupling mechanism

# env_signal[cell] += deposit_rate when agent visits cell

# env_signal[cell] *= signal_decay each timestep

# Agent likelihood A is modulated by env_signal

# P(signal_obs | cell) increases with env_signal[cell]

# No direct communication — coordination emerges from shared environment

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=30

## ActInfOntologyAnnotation

A1=Agent1LikelihoodMatrix
B1=Agent1TransitionMatrix
C1=Agent1PreferenceVector
D1=Agent1PositionPrior
s1=Agent1PositionState
o1=Agent1Observation
pi1=Agent1PolicyVector
u1=Agent1Action
G1=Agent1ExpectedFreeEnergy
A2=Agent2LikelihoodMatrix
B2=Agent2TransitionMatrix
C2=Agent2PreferenceVector
D2=Agent2PositionPrior
s2=Agent2PositionState
o2=Agent2Observation
pi2=Agent2PolicyVector
u2=Agent2Action
G2=Agent2ExpectedFreeEnergy
A3=Agent3LikelihoodMatrix
B3=Agent3TransitionMatrix
C3=Agent3PreferenceVector
D3=Agent3PositionPrior
s3=Agent3PositionState
o3=Agent3Observation
pi3=Agent3PolicyVector
u3=Agent3Action
G3=Agent3ExpectedFreeEnergy
env_signal=EnvironmentalSignal
signal_decay=SignalDecayRate
t=Time

## ModelParameters

num_agents: 3
grid_size: 9
num_obs: 4
num_actions: 4
signal_decay_rate: 0.9
signal_deposit_rate: 0.3
num_timesteps: 30

## Footer

Stigmergic Swarm Active Inference v1 - GNN Representation.
3 agents coordinating via environmental traces only.
No direct communication — emergent coordination from shared signals.
Models ant colony foraging and decentralized robotic coordination.

## Signature

Cryptographic signature goes here

