# GNN Example: 5 Hungry Dogs Multi-Agent System
# Format: Markdown representation of a 5 Hungry Dogs multi-agent system in Active Inference format
# Version: 1.0
# This file is machine-readable and represents a multi-agent system for 5 hungry dogs competing for food

## GNNSection
FiveHungryDogsMultiAgent

## GNNVersionAndFlags
GNN v1

## ModelName
5 Hungry Dogs: Multi-Agent Food Competition Model v1

## ModelAnnotation
This model represents a multi-agent Active Inference system with 5 hungry dogs competing for limited food resources in a shared environment.
- Each dog is an independent agent with its own beliefs, preferences, and policies
- Dogs observe food locations, other dogs' positions, and their own hunger levels
- Hidden states include food availability, dog positions, and hunger states
- Control actions: Move North, South, East, West, Stay, or Compete for food
- The system demonstrates emergent behavior through competitive and cooperative dynamics
- Based on principles of multi-agent Active Inference and resource competition

## StateSpaceBlock
# Multi-Agent State Variables (5 dogs, indexed 0-4)
# Each dog has its own state variables with index i

# Dog i's position on 5x5 grid (25 possible locations)
s_dog_i_position[25,1,type=int]     # Position of dog i (0-24, representing 5x5 grid)

# Dog i's hunger level (3 states: Low, Medium, High)
s_dog_i_hunger[3,1,type=int]        # Hunger state of dog i

# Dog i's energy level (4 states: Exhausted, Low, Medium, High)
s_dog_i_energy[4,1,type=int]        # Energy state of dog i

# Dog i's social state (2 states: Cooperative, Competitive)
s_dog_i_social[2,1,type=int]        # Social behavior tendency of dog i

# Environment States (shared across all dogs)
s_food_available[3,1,type=int]      # Food availability: None (0), Low (1), High (2)
s_food_location[25,1,type=int]      # Food location on grid (0-24, 0 if no food)
s_competition_level[3,1,type=int]   # Competition intensity: Low (0), Medium (1), High (2)

# Observations for each dog i
o_dog_i_visible_food[3,1,type=int]  # Visible food: None (0), Some (1), Much (2)
o_dog_i_nearby_dogs[4,1,type=int]   # Nearby dogs: 0, 1, 2, 3+ dogs
o_dog_i_own_hunger[3,1,type=int]    # Dog i's perceived hunger level
o_dog_i_own_energy[4,1,type=int]    # Dog i's perceived energy level
o_dog_i_own_position[25,1,type=int] # Dog i's perceived position

# Control and Policy for each dog i
pi_dog_i[6,type=float]              # Policy for dog i: Move N/S/E/W, Stay, Compete
u_dog_i[1,type=int]                 # Action chosen by dog i

# Expected Free Energy for each dog
G_dog_i[1,type=float]               # Expected Free Energy for dog i

# Likelihood Matrices (shared across dogs)
A_visible_food[3,3,type=float]      # P(o_visible_food | s_food_available)
A_nearby_dogs[4,25,type=float]      # P(o_nearby_dogs | s_dog_positions) - simplified
A_hunger_obs[3,3,type=float]        # P(o_hunger | s_hunger)
A_energy_obs[4,4,type=float]        # P(o_energy | s_energy)
A_position_obs[25,25,type=float]    # P(o_position | s_position) - identity for simplicity

# Transition Matrices
B_position[25,25,6,type=float]      # P(s_position' | s_position, action) for movement
B_hunger[3,3,type=float]            # P(s_hunger' | s_hunger) - increases over time
B_energy[4,4,6,type=float]          # P(s_energy' | s_energy, action) - decreases with movement
B_social[2,2,type=float]            # P(s_social' | s_social) - can change based on competition
B_food[3,3,type=float]              # P(s_food_available' | s_food_available) - food gets consumed
B_competition[3,3,type=float]       # P(s_competition_level' | s_competition_level, actions)

# Preferences for each dog i
C_dog_i_food[3,type=float]          # Dog i's preferences over food availability
C_dog_i_energy[4,type=float]        # Dog i's preferences over energy levels
C_dog_i_hunger[3,type=float]        # Dog i's preferences over hunger levels

# Priors
D_initial_position[25,type=float]   # Prior over initial positions
D_initial_hunger[3,type=float]      # Prior over initial hunger levels
D_initial_energy[4,type=float]      # Prior over initial energy levels
D_initial_social[2,type=float]      # Prior over initial social states
D_food_available[3,type=float]      # Prior over food availability
D_competition_level[3,type=float]   # Prior over competition level

# Time
t[1,type=int]                       # Time step

## Connections
# For each dog i (0-4), establish connections:

# Prior connections
(D_initial_position) -> (s_dog_i_position)
(D_initial_hunger) -> (s_dog_i_hunger)
(D_initial_energy) -> (s_dog_i_energy)
(D_initial_social) -> (s_dog_i_social)
(D_food_available) -> (s_food_available)
(D_competition_level) -> (s_competition_level)

# Observation connections
(s_food_available) -> (A_visible_food) -> (o_dog_i_visible_food)
(s_dog_0_position, s_dog_1_position, s_dog_2_position, s_dog_3_position, s_dog_4_position) -> (A_nearby_dogs) -> (o_dog_i_nearby_dogs)
(s_dog_i_hunger) -> (A_hunger_obs) -> (o_dog_i_own_hunger)
(s_dog_i_energy) -> (A_energy_obs) -> (o_dog_i_own_energy)
(s_dog_i_position) -> (A_position_obs) -> (o_dog_i_own_position)

# Control and transition connections
(s_dog_i_position, u_dog_i) -> (B_position) -> (s_dog_i_position_next)
(s_dog_i_hunger) -> (B_hunger) -> (s_dog_i_hunger_next)
(s_dog_i_energy, u_dog_i) -> (B_energy) -> (s_dog_i_energy_next)
(s_dog_i_social) -> (B_social) -> (s_dog_i_social_next)
(s_food_available) -> (B_food) -> (s_food_available_next)
(s_competition_level, u_dog_0, u_dog_1, u_dog_2, u_dog_3, u_dog_4) -> (B_competition) -> (s_competition_level_next)

# Policy and action connections
(C_dog_i_food, C_dog_i_energy, C_dog_i_hunger, s_dog_i_hunger, s_dog_i_energy, s_food_available) > (G_dog_i)
(G_dog_i) > (pi_dog_i)
(pi_dog_i) -> (u_dog_i)

# Cross-dog interactions (competition effects)
(s_dog_i_position, s_dog_j_position, s_food_location) > (s_competition_level)  # For all i≠j

## InitialParameterization
# A matrices - Likelihood mappings
A_visible_food={
  # P(visible_food | food_available)
  ((0.9, 0.1, 0.0),   # visible=None | food=None
   (0.3, 0.6, 0.1),   # visible=None | food=Low
   (0.1, 0.3, 0.6))   # visible=None | food=High
}

A_nearby_dogs={
  # Simplified: P(nearby_dogs | positions) - assumes uniform distribution
  "description": "Complex mapping from 5 dog positions to nearby dog count. Simplified as uniform distribution."
}

A_hunger_obs={
  ((0.8, 0.2, 0.0),   # obs=Low | hunger=Low
   (0.2, 0.6, 0.2),   # obs=Low | hunger=Medium
   (0.0, 0.2, 0.8))   # obs=Low | hunger=High
}

A_energy_obs={
  ((0.8, 0.2, 0.0, 0.0),   # obs=Exhausted | energy=Exhausted
   (0.2, 0.6, 0.2, 0.0),   # obs=Exhausted | energy=Low
   (0.0, 0.2, 0.6, 0.2),   # obs=Exhausted | energy=Medium
   (0.0, 0.0, 0.2, 0.8))   # obs=Exhausted | energy=High
}

A_position_obs={
  "description": "Identity matrix - dogs know their positions accurately."
}

# B matrices - Transition dynamics
B_position={
  # Movement transitions (simplified for 5x5 grid)
  # Action 0: North, 1: South, 2: East, 3: West, 4: Stay, 5: Compete
  "description": "Movement transitions on 5x5 grid. North/South/East/West move one step in direction. Stay remains. Compete attempts to move toward food."
}

B_hunger={
  ((0.7, 0.3, 0.0),   # hunger stays Low
   (0.0, 0.6, 0.4),   # hunger increases from Medium
   (0.0, 0.0, 0.8))   # hunger stays High
}

B_energy={
  # Energy decreases with movement, increases slightly with rest
  ((0.8, 0.2, 0.0, 0.0),   # Exhausted stays exhausted
   (0.3, 0.5, 0.2, 0.0),   # Low energy transitions
   (0.0, 0.3, 0.5, 0.2),   # Medium energy transitions
   (0.0, 0.0, 0.3, 0.7))   # High energy transitions
}

B_social={
  ((0.8, 0.2),   # Cooperative stays cooperative
   (0.3, 0.7))   # Competitive can become cooperative
}

B_food={
  ((0.9, 0.1, 0.0),   # No food stays none
   (0.2, 0.6, 0.2),   # Low food can decrease or increase
   (0.0, 0.2, 0.8))   # High food tends to decrease
}

B_competition={
  ((0.7, 0.3, 0.0),   # Low competition stays low
   (0.2, 0.6, 0.2),   # Medium competition transitions
   (0.0, 0.3, 0.7))   # High competition stays high
}

# C vectors - Preferences for each dog
# Dog 0: Very hungry, aggressive
C_dog_0_food={(-2.0, 0.0, 3.0)}      # Strong preference for high food
C_dog_0_energy={(-1.0, 0.0, 1.0, 2.0)} # Prefers high energy
C_dog_0_hunger={(-3.0, -1.0, 0.0)}    # Strong aversion to hunger

# Dog 1: Moderately hungry, cooperative
C_dog_1_food={(-1.0, 0.5, 2.0)}
C_dog_1_energy={(-0.5, 0.0, 1.0, 1.5)}
C_dog_1_hunger={(-2.0, -0.5, 0.0)}

# Dog 2: Slightly hungry, cautious
C_dog_2_food={(-0.5, 0.0, 1.5)}
C_dog_2_energy={(-0.5, 0.0, 0.5, 1.0)}
C_dog_2_hunger={(-1.5, -0.5, 0.0)}

# Dog 3: Not very hungry, social
C_dog_3_food={(-0.5, 0.0, 1.0)}
C_dog_3_energy={(-0.5, 0.0, 0.5, 1.0)}
C_dog_3_hunger={(-1.0, -0.5, 0.0)}

# Dog 4: Least hungry, very cooperative
C_dog_4_food={(-0.5, 0.0, 0.5)}
C_dog_4_energy={(-0.5, 0.0, 0.5, 1.0)}
C_dog_4_hunger={(-0.5, -0.5, 0.0)}

# D vectors - Priors
D_initial_position={
  # Dogs start at corners and center
  (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # Dog 0 at (0,0)
  (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # Dog 1 at (0,4)
  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),  # Dog 2 at (4,4)
  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),  # Dog 3 at (4,0)
  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # Dog 4 at center (2,2)
}

D_initial_hunger={(0.1, 0.3, 0.6)}      # Most dogs start hungry
D_initial_energy={(0.1, 0.2, 0.4, 0.3)} # Most dogs start with medium energy
D_initial_social={(0.7, 0.3)}            # Most dogs start cooperative
D_food_available={(0.2, 0.5, 0.3)}      # Moderate food availability
D_competition_level={(0.3, 0.5, 0.2)}   # Moderate initial competition

## Equations
# For each dog i:
# 1. State inference using variational message passing
# q(s_dog_i) = softmax(ln(D) + ln(B^T * s_next) + ln(A^T * o))

# 2. Expected Free Energy for each dog
# G_dog_i(π) = E_q(s|π)[ln q(s|π) - ln P(s)] + E_q(s|π)[ln P(o|s) - ln P(o)]

# 3. Policy selection
# π_dog_i = softmax(-G_dog_i)

# 4. Action selection
# u_dog_i = sample(π_dog_i)

# 5. Multi-agent interactions
# Competition level increases when multiple dogs are near food
# Food availability decreases when dogs compete successfully

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=50  # 50 time steps for simulation

## ActInfOntologyAnnotation
s_dog_i_position=HiddenStateDogIPosition
s_dog_i_hunger=HiddenStateDogIHungerLevel
s_dog_i_energy=HiddenStateDogIEnergyLevel
s_dog_i_social=HiddenStateDogISocialBehavior
s_food_available=HiddenStateFoodAvailability
s_food_location=HiddenStateFoodLocation
s_competition_level=HiddenStateCompetitionLevel
o_dog_i_visible_food=ObservationDogIVisibleFood
o_dog_i_nearby_dogs=ObservationDogINearbyDogs
o_dog_i_own_hunger=ObservationDogIOwnHunger
o_dog_i_own_energy=ObservationDogIOwnEnergy
o_dog_i_own_position=ObservationDogIOwnPosition
pi_dog_i=PolicyDogIActionDistribution
u_dog_i=ActionDogIChosenAction
G_dog_i=ExpectedFreeEnergyDogI
A_visible_food=LikelihoodMatrixVisibleFoodGivenAvailability
A_nearby_dogs=LikelihoodMatrixNearbyDogsGivenPositions
A_hunger_obs=LikelihoodMatrixHungerObservation
A_energy_obs=LikelihoodMatrixEnergyObservation
A_position_obs=LikelihoodMatrixPositionObservation
B_position=TransitionMatrixPositionGivenAction
B_hunger=TransitionMatrixHungerLevel
B_energy=TransitionMatrixEnergyGivenAction
B_social=TransitionMatrixSocialBehavior
B_food=TransitionMatrixFoodAvailability
B_competition=TransitionMatrixCompetitionLevel
C_dog_i_food=LogPreferenceDogIFoodAvailability
C_dog_i_energy=LogPreferenceDogIEnergyLevel
C_dog_i_hunger=LogPreferenceDogIHungerLevel
D_initial_position=PriorBeliefOverInitialPositions
D_initial_hunger=PriorBeliefOverInitialHunger
D_initial_energy=PriorBeliefOverInitialEnergy
D_initial_social=PriorBeliefOverInitialSocial
D_food_available=PriorBeliefOverFoodAvailability
D_competition_level=PriorBeliefOverCompetitionLevel
t=TimeStepInteger

## ModelParameters
num_dogs: 5
num_hidden_states_per_dog: [25, 3, 4, 2]  # position, hunger, energy, social
num_shared_hidden_states: [3, 25, 3]      # food_available, food_location, competition_level
num_obs_modalities_per_dog: [3, 4, 3, 4, 25]  # visible_food, nearby_dogs, hunger, energy, position
num_actions_per_dog: 6  # N, S, E, W, Stay, Compete
grid_size: [5, 5]  # 5x5 grid environment

## Footer
5 Hungry Dogs: Multi-Agent Food Competition Model v1 - GNN Representation
This model demonstrates emergent behavior through competitive and cooperative dynamics
among 5 agents with different hunger levels, energy states, and social tendencies.

## Signature
Creator: AI Assistant for GNN
Date: 2024-07-26
Status: Example demonstrating multi-agent Active Inference with resource competition.
Context: Shows how individual agent preferences and social dynamics interact in a shared environment. 