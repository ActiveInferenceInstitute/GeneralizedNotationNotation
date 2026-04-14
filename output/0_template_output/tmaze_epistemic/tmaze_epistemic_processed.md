
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/discrete/tmaze_epistemic.md
# Processed on: 2026-04-14T10:56:40.578827
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: T-Maze Epistemic Foraging Agent

# GNN Version: 1.0

# Classic Active Inference T-maze demonstrating epistemic foraging behavior

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1

## ModelName

T-Maze Epistemic Foraging Agent

## ModelAnnotation

The classic T-maze task from Active Inference literature (Friston et al.):

- Agent navigates a T-shaped maze with 4 locations: center, left arm, right arm, cue location
- Two observation modalities: location (where am I?) and reward/cue (what do I see?)
- Reward is hidden behind one of the two arms (left or right), determined by context
- Cue location provides partial information about which arm holds the reward
- Agent must decide: go directly to an arm (exploit) or visit cue location first (explore)
- Demonstrates epistemic foraging: Active Inference naturally balances exploration vs exploitation
- The Expected Free Energy decomposes into epistemic (information gain) + instrumental (reward) value

## StateSpaceBlock

# Hidden state factors

s_loc[4,1,type=float]        # Location state: (0:center, 1:left_arm, 2:right_arm, 3:cue_location)
s_ctx[2,1,type=float]        # Context state: (0:reward_left, 1:reward_right)

# Observation modalities

o_loc[4,1,type=int]          # Location observation: (0:center, 1:left, 2:right, 3:cue)
o_rew[3,1,type=int]          # Reward/cue observation: (0:no_reward, 1:reward, 2:cue_left)

# Generative model matrices

A_loc[4,4,type=float]        # Location likelihood: P(o_loc | s_loc) — identity
A_rew[3,4,2,type=float]      # Reward likelihood: P(o_rew | s_loc, s_ctx) — context-dependent

# Transition matrices

B_loc[4,4,4,type=float]      # Location transitions: P(s_loc' | s_loc, action)
B_ctx[2,2,1,type=float]      # Context transitions: identity (context doesn't change)

# Preferences

C_loc[4,type=float]          # No location preference (agent doesn't prefer a location per se)
C_rew[3,type=float]          # Reward preference: strongly prefers reward observation

# Priors

D_loc[4,type=float]          # Prior: agent starts at center
D_ctx[2,type=float]          # Prior: uncertain about which arm has reward

# Policy and action

pi[4,type=float]             # Policy over 4 actions: (go_left, go_right, go_cue, stay)
u[1,type=int]                # Selected action
G[pi,type=float]             # Expected Free Energy per policy
G_epi[pi,type=float]         # Epistemic value (information gain about context)
G_ins[pi,type=float]         # Instrumental value (expected reward)

# Inference

F[1,type=float]              # Variational Free Energy

# Time

t[1,type=int]                # Discrete time step

## Connections

D_loc>s_loc
D_ctx>s_ctx
s_loc-A_loc
A_loc-o_loc
s_loc-A_rew
s_ctx-A_rew
A_rew-o_rew
s_loc-B_loc
s_ctx-B_ctx
C_rew>G_ins
G_epi>G
G_ins>G
G>pi
pi>u
B_loc>u
s_loc-F
s_ctx-F
o_loc-F
o_rew-F

## InitialParameterization

# Location likelihood: identity mapping (agent knows its location)

A_loc={
  (1.0, 0.0, 0.0, 0.0),
  (0.0, 1.0, 0.0, 0.0),
  (0.0, 0.0, 1.0, 0.0),
  (0.0, 0.0, 0.0, 1.0)
}

# Reward likelihood depends on location × context

# At center or cue: no reward visible

# At left arm: reward if context=reward_left

# At right arm: reward if context=reward_right

# At cue location: cue_left shown if context=reward_left

# Shape: [3 reward_obs, 4 locations, 2 contexts]

# Slice for context=reward_left

# center: (1.0, 0.0, 0.0), left: (0.0, 1.0, 0.0), right: (1.0, 0.0, 0.0), cue: (0.0, 0.0, 1.0)

# Slice for context=reward_right

# center: (1.0, 0.0, 0.0), left: (1.0, 0.0, 0.0), right: (0.0, 1.0, 0.0), cue: (1.0, 0.0, 0.0)

# Transition: deterministic moves

B_loc={
  ( (0.0,0.0,0.0,0.0), (1.0,1.0,0.0,0.0), (0.0,0.0,1.0,0.0), (0.0,0.0,0.0,1.0) ),
  ( (0.0,0.0,0.0,0.0), (0.0,0.0,0.0,0.0), (1.0,0.0,1.0,0.0), (0.0,0.0,0.0,1.0) ),
  ( (0.0,0.0,0.0,0.0), (0.0,0.0,0.0,0.0), (0.0,0.0,0.0,0.0), (1.0,0.0,0.0,1.0) ),
  ( (1.0,0.0,0.0,0.0), (0.0,0.0,0.0,0.0), (0.0,0.0,0.0,0.0), (0.0,1.0,1.0,0.0) )
}

# Context is static

B_ctx={
  ( (1.0, 0.0), (0.0, 1.0) )
}

# Preferences: strongly prefer reward, indifferent to no-reward or cue

C_loc={(0.0, 0.0, 0.0, 0.0)}
C_rew={(-1.0, 3.0, 0.0)}

# Prior: starts at center, uncertain about context

D_loc={(1.0, 0.0, 0.0, 0.0)}
D_ctx={(0.5, 0.5)}

## Equations

# G(pi) = G_epi(pi) + G_ins(pi)

# G_epi(pi) = -E_Q[H[P(o|s)]] = expected information gain about hidden context

# G_ins(pi) = -E_Q[log P(C|o)] = expected reward under policy

# The epistemic component drives the agent to visit the cue location

# to reduce uncertainty about context before committing to an arm

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=3

## ActInfOntologyAnnotation

A_loc=LocationLikelihoodMatrix
A_rew=RewardLikelihoodMatrix
B_loc=LocationTransitionMatrix
B_ctx=ContextTransitionMatrix
C_loc=LocationPreferenceVector
C_rew=RewardPreferenceVector
D_loc=LocationPrior
D_ctx=ContextPrior
s_loc=LocationHiddenState
s_ctx=ContextHiddenState
o_loc=LocationObservation
o_rew=RewardObservation
pi=PolicyVector
u=Action
G=ExpectedFreeEnergy
G_epi=EpistemicValue
G_ins=InstrumentalValue
F=VariationalFreeEnergy
t=Time

## ModelParameters

num_locations: 4
num_contexts: 2
num_location_obs: 4
num_reward_obs: 3
num_actions: 4
num_timesteps: 3
num_modalities: 2
num_state_factors: 2

## Footer

T-Maze Epistemic Foraging Agent v1 - GNN Representation.
Demonstrates the natural exploration-exploitation trade-off in Active Inference.
Agent visits cue location to reduce uncertainty before committing to reward arm.
Classic paradigm from Friston et al. Active Inference literature.

## Signature

Cryptographic signature goes here

