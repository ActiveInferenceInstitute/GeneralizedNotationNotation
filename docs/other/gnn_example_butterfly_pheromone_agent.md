# GNN Example: Butterfly Pheromone Detection PyMDP Agent
# Format: Markdown representation of a Butterfly Pheromone Detection PyMDP model in Active Inference format
# Version: 1.0
# This file is machine-readable and represents a conceptual PyMDP agent for a butterfly detecting pheromones.

## GNNSection
ButterflyPheromonePyMDPAgent

## GNNVersionAndFlags
GNN v1

## ModelName
Butterfly Pheromone Detection PyMDP Agent v1

## ModelAnnotation
This model represents a PyMDP agent for a butterfly navigating based on pheromone cues, wind direction, and visual information.
- Observation modalities:
  - "pheromone_concentration" (3 outcomes: Low, Medium, High)
  - "wind_direction" (3 outcomes: Upwind, Still, Downwind)
  - "visual_cue" (2 outcomes: No Flower, Flower)
- Hidden state factors:
  - "location" (3 states: Far, Mid, Close to pheromone source)
  - "mating_readiness" (2 states: Not Ready, Ready)
- Control:
  - "location" factor is controllable with 3 actions (Fly Upwind, Fly Crosswind, Fly Downwind).
The parameterization is illustrative of plausible butterfly behavior.

## StateSpaceBlock
# A_matrices: A_m[observation_outcomes, state_factor0_states, state_factor1_states]
A_m0[3,3,2,type=float]   # Likelihood for modality 0 ("pheromone_concentration") vs Location (f0) & Readiness (f1)
A_m1[3,3,2,type=float]   # Likelihood for modality 1 ("wind_direction") vs Location (f0) & Readiness (f1)
A_m2[2,3,2,type=float]   # Likelihood for modality 2 ("visual_cue") vs Location (f0) & Readiness (f1)

# B_matrices: B_f[states_next, states_previous, actions]
B_f0[3,3,3,type=float]   # Transitions for factor 0 ("location"), 3 actions
B_f1[2,2,1,type=float]   # Transitions for factor 1 ("mating_readiness"), 1 implicit action (uncontrolled)

# C_vectors: C_m[observation_outcomes]
C_m0[3,type=float]       # Preferences for modality 0 ("pheromone_concentration")
C_m1[3,type=float]       # Preferences for modality 1 ("wind_direction")
C_m2[2,type=float]       # Preferences for modality 2 ("visual_cue")

# D_vectors: D_f[states]
D_f0[3,type=float]       # Prior for factor 0 ("location")
D_f1[2,type=float]       # Prior for factor 1 ("mating_readiness")

# Hidden States
s_f0[3,1,type=float]     # Hidden state for factor 0 ("location")
s_f1[2,1,type=float]     # Hidden state for factor 1 ("mating_readiness")
s_prime_f0[3,1,type=float] # Next hidden state for factor 0
s_prime_f1[2,1,type=float] # Next hidden state for factor 1

# Observations
o_m0[3,1,type=float]     # Observation for modality 0 ("pheromone_concentration")
o_m1[3,1,type=float]     # Observation for modality 1 ("wind_direction")
o_m2[2,1,type=float]     # Observation for modality 2 ("visual_cue")

# Policy and Control
π_f0[3,type=float]       # Policy (distribution over actions) for controllable factor 0 ("location")
u_f0[1,type=int]         # Action taken for controllable factor 0
G[1,type=float]          # Expected Free Energy
t[1,type=int]            # Time step

## Connections
(D_f0,D_f1)-(s_f0,s_f1)
(s_f0,s_f1)-(A_m0,A_m1,A_m2)
(A_m0,A_m1,A_m2)-(o_m0,o_m1,o_m2)
(s_f0,s_f1,u_f0)-(B_f0,B_f1) # u_f0 primarily affects B_f0; B_f1 is uncontrolled by u_f0
(B_f0,B_f1)-(s_prime_f0,s_prime_f1)
(C_m0,C_m1,C_m2)>G
G>π_f0
π_f0-u_f0
G=ExpectedFreeEnergy
t=Time

## InitialParameterization
# States for s_f0 (Location): 0:Far, 1:Mid, 2:Close
# States for s_f1 (Readiness): 0:NotReady, 1:Ready
# Obs for o_m0 (Pheromone): 0:Low, 1:Medium, 2:High
# Obs for o_m1 (Wind): 0:Upwind, 1:Still, 2:Downwind
# Obs for o_m2 (Visual): 0:NoFlower, 1:Flower
# Actions for u_f0 (Flight affecting Location): 0:FlyUpwind, 1:FlyCrosswind, 2:FlyDownwind

# A_m0[obs_pheromone, loc, ready]: Pheromone likelihood
# If Close(2) & Ready(1), expect High(2) Pheromone. If Far(0), expect Low(0) Pheromone.
A_m0={
  # Pheromone=Low(0)
  ( ((0.8,0.7), (0.3,0.2), (0.1,0.05)) ), # P_ObsLow[loc][ready] = ( (P_ObsL|Far,NR), (P_ObsL|Far,R) ), ( (P_ObsL|Mid,NR), (P_ObsL|Mid,R) ), ...
  # Pheromone=Medium(1)
  ( ((0.15,0.2), (0.6,0.5), (0.3,0.25)) ),
  # Pheromone=High(2)
  ( ((0.05,0.1), (0.1,0.3), (0.6,0.7)) )
}

# A_m1[obs_wind, loc, ready]: Wind likelihood (assuming somewhat independent of internal state for simplicity)
# General probabilities for observing Upwind, Still, Downwind.
A_m1={
  # Wind=Upwind(0)
  ( ((0.4,0.4), (0.4,0.4), (0.4,0.4)) ), # P_ObsUpwind[loc][ready]
  # Wind=Still(1)
  ( ((0.3,0.3), (0.3,0.3), (0.3,0.3)) ), # P_ObsStill[loc][ready]
  # Wind=Downwind(2)
  ( ((0.3,0.3), (0.3,0.3), (0.3,0.3)) )  # P_ObsDownwind[loc][ready]
}

# A_m2[obs_visual, loc, ready]: Visual likelihood
# If Close(2) to source (assume source often near flowers), expect Flower(1)
A_m2={
  # Visual=NoFlower(0)
  ( ((0.8,0.7), (0.6,0.5), (0.2,0.1)) ), # P_ObsNoFlower[loc][ready]
  # Visual=Flower(1)
  ( ((0.2,0.3), (0.4,0.5), (0.8,0.9)) )  # P_ObsFlower[loc][ready]
}

# B_f0[s_next_loc, s_prev_loc, action_flight]: Location transitions
# Action 0 (FlyUpwind): tends to move from Far->Mid, Mid->Close
# Action 1 (FlyCrosswind): tends to stay in current region or diffuse
# Action 2 (FlyDownwind): tends to move from Close->Mid, Mid->Far
B_f0={
  # s_next_loc = Far(0)
  ( ((0.7,0.2,0.1), (0.4,0.1,0.8), (0.1,0.1,0.9)) ), # P_NextFar[prev_loc][action] = ( (P(F|F,AU),P(F|F,AC),P(F|F,AD)), (P(F|M,AU),...), (P(F|C,AU),...) )
  # s_next_loc = Mid(1)
  ( ((0.25,0.7,0.1), (0.5,0.8,0.15), (0.3,0.7,0.05)) ),
  # s_next_loc = Close(2)
  ( ((0.05,0.1,0.8), (0.1,0.1,0.05), (0.6,0.2,0.05)) )
}
# Simplified B_f0 (deterministic for illustration, real would be stochastic)
# FlyUpwind (act=0): Far->Mid, Mid->Close, Close->Close
# FlyCross (act=1): Far->Far, Mid->Mid, Close->Close
# FlyDown  (act=2): Far->Far, Mid->Far, Close->Mid
B_f0={
  # s_next_loc = Far(0)
  ( ((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (0.0, 0.0, 0.0)) ), # prev_loc=(F,M,C) for actions(FU,FC,FD)
  # s_next_loc = Mid(1)
  ( ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 1.0)) ),
  # s_next_loc = Close(2)
  ( ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)) )
}
# More plausible stochastic B_f0
B_f0={ # B[s_next, s_prev, action]
  # s_next_loc = Far(0)
  ( ((0.2, 0.9, 0.9), (0.7, 0.1, 0.1), (0.1, 0.0, 0.0)) ), # Action: FlyUpwind(0), FlyCrosswind(1), FlyDownwind(2) for s_prev_loc = Far(0)
                                                           # P(next=Far | prev=Far, action), P(next=Far | prev=Mid, action), P(next=Far | prev=Close, action)
  # s_next_loc = Mid(1)
  ( ((0.7, 0.1, 0.1), (0.2, 0.8, 0.8), (0.1, 0.1, 0.0)) ),
  # s_next_loc = Close(2)
  ( ((0.1, 0.0, 0.0), (0.1, 0.1, 0.1), (0.8, 0.9, 1.0)) )
}


# B_f1[s_next_ready, s_prev_ready, action_implicit=0]: Readiness transitions (slowly becomes Ready)
# B_f1 = [[0.9, 0.0], [0.1, 1.0]] (NotReady stays NotReady w.p. 0.9, Ready stays Ready w.p. 1.0)
B_f1={
  ( ((0.9),(0.0)) ), # s_next=NotReady(0); (val for s_prev=NotReady), (val for s_prev=Ready)
  ( ((0.1),(1.0)) )  # s_next=Ready(1)
}

# C_m0 (Pheromone): Prefers High(2), dislikes Low(0)
C_m0={(-2.0, 0.0, 3.0)} # (Low, Medium, High)

# C_m1 (Wind): Mild preference for Upwind(0) if seeking, otherwise neutral.
C_m1={(0.5, 0.0, -0.5)} # (Upwind, Still, Downwind)

# C_m2 (Visual): Prefers Flower(1) if Ready and pheromone is high (not directly encoded here, but influences policy)
C_m2={(0.0, 1.0)} # (NoFlower, Flower)

# D_f0 (Location): Start Far(0)
D_f0={(1.0, 0.0, 0.0)} # (Far, Mid, Close)

# D_f1 (Readiness): Start NotReady(0)
D_f1={(1.0, 0.0)} # (NotReady, Ready)

## Equations
# Standard PyMDP agent equations:
# qs_f0, qs_f1 = infer_states(o_m0, o_m1, o_m2)
# q_pi_f0, G = infer_policies()
# u_f0 = sample_action()

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=Unbounded

## ActInfOntologyAnnotation
A_m0=LikelihoodPheromoneModalityDistribution
A_m1=LikelihoodWindModalityDistribution
A_m2=LikelihoodVisualModalityDistribution
B_f0=TransitionMatrixLocationFactorDistribution
B_f1=TransitionMatrixReadinessFactorDistribution
C_m0=LogPreferencePheromoneModalityVector
C_m1=LogPreferenceWindModalityVector
C_m2=LogPreferenceVisualModalityVector
D_f0=PriorBeliefOverLocationFactorStates
D_f1=PriorBeliefOverReadinessFactorStates
s_f0=HiddenStateLocationFactor
s_f1=HiddenStateReadinessFactor
s_prime_f0=NextHiddenStateLocationFactor
s_prime_f1=NextHiddenStateReadinessFactor
o_m0=ObservationPheromoneModality
o_m1=ObservationWindModality
o_m2=ObservationVisualModality
π_f0=PolicyDistributionOverLocationActions # Policy for Location factor
u_f0=ActionLocationFactor                 # Chosen action for Location factor
G=ExpectedFreeEnergyScalar
t=TimeStepInteger

## ModelParameters
num_hidden_states_factors: [3, 2]  # s_f0[3] (Location), s_f1[2] (Readiness)
num_obs_modalities: [3, 3, 2]     # o_m0[3] (Pheromone), o_m1[3] (Wind), o_m2[2] (Visual)
num_control_factors_actions: [3, 1] # B_f0 controllable with 3 actions, B_f1 has 1 implicit (uncontrolled by agent's policy)

## Footer
Butterfly Pheromone Detection PyMDP Agent v1 - GNN Representation

## Signature
NA 