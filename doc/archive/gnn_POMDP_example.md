# GNN Standard Example: Comprehensive POMDP Agent
# Format: Markdown representation of a complex POMDP model for testing GNN features.
# Version: 1.0
# This file is machine-readable and aims to be a standard for the GNN pipeline.

## GNNSection
ComprehensivePOMDPAgent

## GNNVersionAndFlags
GNN v1

## ModelName
Standard POMDP Agent v1.0

## ModelAnnotation
This model represents a comprehensive Partially Observable Markov Decision Process (POMDP) agent.
It includes:
- Two hidden state factors: Location (3 states), ResourceLevel (2 states).
- Two observation modalities: VisualCue (4 outcomes), AuditorySignal (2 outcomes).
- Two control factors (actions): Movement (3 actions), Interaction (2 actions).
This example is designed to test various GNN parsing and rendering capabilities, especially for PyMDP.

## StateSpaceBlock
# Hidden States (s_factorIndex[num_states_factor, 1, type=dataType])
s_f0[3,1,type=int]   # Hidden State Factor 0: Location (e.g., RoomA, RoomB, Corridor)
s_f1[2,1,type=int]   # Hidden State Factor 1: ResourceLevel (e.g., Low, High)

# Observations (o_modalityIndex[num_outcomes_modality, 1, type=dataType])
o_m0[4,1,type=int]   # Observation Modality 0: VisualCue (e.g., Door, Window, Food, Empty)
o_m1[2,1,type=int]   # Observation Modality 1: AuditorySignal (e.g., Silence, Beep)

# Control Factors / Policies (pi_factorIndex[num_actions_factor, type=dataType])
# These represent the distribution over actions for each controllable factor.
pi_c0[3,type=float]  # Policy for Control Factor 0: Movement (e.g., Stay, MoveClockwise, MoveCounterClockwise)
pi_c1[2,type=float]  # Policy for Control Factor 1: Interaction (e.g., Wait, InteractWithResource)

# Actions (u_factorIndex[1, type=dataType]) - chosen actions
u_c0[1,type=int]     # Chosen action for Movement
u_c1[1,type=int]     # Chosen action for Interaction

# Likelihood Mapping (A_modalityIndex[outcomes, factor0_states, factor1_states, ..., type=dataType])
A_m0[4,3,2,type=float] # VisualCue likelihood given Location and ResourceLevel
A_m1[2,3,2,type=float] # AuditorySignal likelihood

# Transition Dynamics (B_factorIndex[next_states, prev_states, control0_actions, control1_actions, ..., type=dataType])
B_f0[3,3,3,2,type=float] # Location transitions, depends on current Location, Movement action, and Interaction action
B_f1[2,2,3,2,type=float] # ResourceLevel transitions, depends on current ResourceLevel, Movement action, and Interaction action

# Preferences (C_modalityIndex[outcomes, type=dataType]) - Log preferences over outcomes
C_m0[4,type=float]   # Preferences for VisualCues
C_m1[2,type=float]   # Preferences for AuditorySignals

# Priors over Initial Hidden States (D_factorIndex[num_states_factor, type=dataType])
D_f0[3,type=float]   # Prior for Location
D_f1[2,type=float]   # Prior for ResourceLevel

# Expected Free Energy (G[num_policies_controlFactor0 * num_policies_controlFactor1, ... , type=dataType])
# For this example, G would be complex if policies are combinations of actions.
# Simpler: G_c0[3,type=float] for EFE of movement, G_c1[2,type=float] for EFE of interaction.
# Or a single G if policies are evaluated jointly. Let's use a single G for overall policy EFE.
# Assuming policies are evaluated over a fixed horizon and combined.
G[1,type=float]      # Overall Expected Free Energy of chosen combined policy

# Time
t[1,type=int]

## Connections
# Priors to initial states
(D_f0, D_f1) -> (s_f0, s_f1)

# States to likelihoods to observations
(s_f0, s_f1) -> (A_m0, A_m1)
(A_m0) -> (o_m0)
(A_m1) -> (o_m1)

# States and actions to transitions to next states
(s_f0, s_f1, u_c0, u_c1) -> (B_f0, B_f1)
(B_f0) -> s_f0_next # Implied next state for factor 0
(B_f1) -> s_f1_next # Implied next state for factor 1

# Preferences and states/observations to EFE
(C_m0, C_m1, s_f0, s_f1, A_m0, A_m1) > G # Simplified EFE dependency

# EFE to policies
G > (pi_c0, pi_c1)

# Policies to chosen actions
(pi_c0) -> u_c0
(pi_c1) -> u_c1

## InitialParameterization
# D_f0: Prior for Location (3 states: RoomA, RoomB, Corridor) - e.g., start in RoomA
D_f0={(1.0, 0.0, 0.0)}
# D_f1: Prior for ResourceLevel (2 states: Low, High) - e.g., start Low
D_f1={(1.0, 0.0)}

# A_m0: VisualCue[4] given Location[3] and ResourceLevel[2]. (4 x 3 x 2)
# A_m0[cue_idx, loc_idx, res_idx]
# Example: If in RoomA (loc=0) with Low Resource (res=0), high chance of seeing Door (cue=0)
A_m0={ # For res=0 (Low)
      ( (0.8,0.1,0.1), (0.1,0.8,0.1), (0.1,0.1,0.8) ), # cue=0 (Door): high if loc matches expected
      ( (0.1,0.7,0.2), (0.7,0.1,0.2), (0.2,0.7,0.1) ), # cue=1 (Window)
      ( (0.05,0.1,0.7), (0.1,0.05,0.7), (0.7,0.1,0.05) ),# cue=2 (Food) - higher if res high, but this is for res=low
      ( (0.05,0.1,0.1), (0.1,0.05,0.1), (0.1,0.1,0.05) ) # cue=3 (Empty)
      # TODO: This format for multi-dim A is tricky for GNN eval(). Needs to be a list of lists of lists.
      # PyMDP expects A as a list of arrays. Each array is [num_outcomes, num_states_factor_0, num_states_factor_1, ...]
      # A_m0_data = np.zeros((4,3,2)) -> This is what pymdp.py would build from.
      # For now, this InitialParameterization might need manual JSON override or parser enhancement for >2D arrays.
      # Placeholder until matrix parsing is more robust for >2D.
     }
# A_m1: AuditorySignal[2] given Location[3] and ResourceLevel[2]. (2 x 3 x 2)
A_m1={ # For res=0 (Low)
      ( (0.9,0.7,0.8), (0.7,0.9,0.8), (0.8,0.7,0.9) ), # cue=0 (Silence)
      ( (0.1,0.3,0.2), (0.3,0.1,0.2), (0.2,0.3,0.1) )  # cue=1 (Beep)
     }

# B_f0: Location[3] transition given Location[3], Movement[3], Interaction[2]. (3 x 3 x 3 x 2)
# B_f0[loc_next, loc_prev, move_action, interact_action]
# Simplified: Interaction action doesn't affect location.
# B_f0[loc_next, loc_prev, move_action] -> list of 2 identical (3x3x3) arrays.
B_f0={ # Placeholder due to >2D complexity for GNN eval()
     }
# B_f1: ResourceLevel[2] transition given ResourceLevel[2], Movement[3], Interaction[2]. (2 x 2 x 3 x 2)
B_f1={ # Placeholder
     }

# C_m0: Preferences for VisualCues[4] (e.g., prefer Food)
C_m0={(0.0, 0.0, 1.0, -1.0)} # (Door, Window, Food, Empty)
# C_m1: Preferences for AuditorySignals[2] (e.g., prefer Beep)
C_m1={(-1.0, 1.0)} # (Silence, Beep)

## Equations
# Standard POMDP / Active Inference equations for:
# 1. State estimation (approximate posterior over hidden states)
#    q(s_t) = σ( ln(A^T o_t) + ln(D) ) for t=0
#    q(s_t) = σ( ln(A^T o_t) + sum_{s_{t-1}} B(s_t | s_{t-1}, u_{t-1}) q(s_{t-1}) ) for t>0
# 2. Policy evaluation (Expected Free Energy)
#    G(π) = sum_t E_q(o_t, s_t | π) [ ln q(s_t | o_t, π) - ln q(s_t, o_t | π) - ln C(o_t) ]
# 3. Action selection (Softmax over -G)
#    P(u_t | π) = σ(-G(π))

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=50 # Example planning horizon for simulations

## ActInfOntologyAnnotation
# Hidden States
s_f0=HiddenStateFactor0
s_f1=HiddenStateFactor1

# Observations
o_m0=ObservationModality0
o_m1=ObservationModality1

# Control/Policy Related
pi_c0=PolicyVector      # Or PolicyVectorFactor0
pi_c1=PolicyVectorFactor1      # Or PolicyVectorFactor1
u_c0=Action # Or ActionFactor0
u_c1=ActionFactor1 # Or ActionFactor1

# Likelihoods
A_m0=LikelihoodMatrixModality0
A_m1=LikelihoodMatrixModality1

# Transitions
B_f0=TransitionMatrixFactor0
B_f1=TransitionMatrixFactor1

# Preferences
C_m0=LogPreferenceVectorModality0 # Or LogPreferenceVectorModality0
C_m1=LogPreferenceVectorModality1 # Or LogPreferenceVectorModality1

# Priors
D_f0=PriorOverHiddenStatesFactor0
D_f1=PriorOverHiddenStatesFactor1

# Other
G=ExpectedFreeEnergy
t=Time

## ModelParameters
num_hidden_states_factors: [3, 2]  # s_f0[3], s_f1[2]
num_obs_modalities: [4, 2]     # o_m0[4], o_m1[2]
num_control_factors: [3, 2]    # pi_c0[3], pi_c1[2]

## Footer
Standard POMDP Agent v1.0 - End of Specification

## Signature
Creator: AI Assistant for GNN
Date: 2024-07-26
Status: Example for testing and demonstration. InitialParameterization for A & B matrices is placeholder due to parsing complexity of >2D arrays from GNN string format. 