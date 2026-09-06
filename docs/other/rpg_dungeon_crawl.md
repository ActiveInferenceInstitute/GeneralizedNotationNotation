# GNN Example: Role Playing Game Dungeon Crawl
# Format: Markdown representation of a Multi-agent RPG Active Inference model
# Version: 1.0
# This file represents a comprehensive D&D-style role playing game with team coordination, environmental exploration, and DM interaction.

## GNNSection
RPGDungeonCrawl

## GNNVersionAndFlags
GNN v1

## ModelName
Collaborative Role Playing Game with Active Inference Agents

## ModelAnnotation
This model represents a comprehensive Role Playing Game scenario with:
- Multiple player characters (Fighter, Mage, Rogue, Cleric) each with distinct capabilities and goals
- Dungeon Master as an adaptive environmental controller and narrative agent
- Dynamic labyrinth environment with rooms, corridors, traps, monsters, and treasures
- Multi-modal observations (visual, auditory, tactical, social, magical)
- Complex action spaces (movement, combat, magic, social interaction, exploration)
- Collaborative decision-making with turn-based coordination
- Uncertainty from dice mechanics and hidden information
- Hierarchical goal structures (immediate survival, quest objectives, character development)
- Emergent narrative through agent interactions and environmental responses

The model enables rich collaborative gameplay with adaptive AI agents that can:
- Plan coordinated strategies while maintaining individual character motivations
- Adapt to unexpected environmental changes and DM narrative decisions
- Learn from experience and develop character-specific expertise
- Balance competition and cooperation in dynamic social situations
- Generate emergent storytelling through believable character interactions

## StateSpaceBlock
# === PLAYER CHARACTER STATES ===
# Fighter - Tank/Melee specialist
fighter_health[100,type=int]              # Current hit points
fighter_position[2,type=float]            # X,Y coordinates in dungeon
fighter_orientation[1,type=float]         # Facing direction (0-2π radians)
fighter_stamina[50,type=int]              # Action points/endurance
fighter_equipment[10,type=discrete]       # Weapon/armor state vector
fighter_abilities[6,type=int]             # STR, DEX, CON, INT, WIS, CHA
fighter_experience[1,type=int]            # Experience points/level
fighter_morale[1,type=float]              # Confidence/fear state
fighter_tactical_knowledge[20,type=float] # Known enemy/trap locations

# Mage - Spellcaster/Ranged specialist  
mage_health[60,type=int]                  # Current hit points
mage_position[2,type=float]               # X,Y coordinates in dungeon
mage_orientation[1,type=float]            # Facing direction
mage_mana[80,type=int]                    # Spell casting resource
mage_spells_known[15,type=discrete]       # Available spells
mage_abilities[6,type=int]                # Ability scores
mage_experience[1,type=int]               # Experience points/level
mage_focus[1,type=float]                  # Concentration state
mage_arcane_knowledge[25,type=float]      # Magical environment awareness

# Rogue - Stealth/Skills specialist
rogue_health[70,type=int]                 # Current hit points
rogue_position[2,type=float]              # X,Y coordinates in dungeon
rogue_orientation[1,type=float]           # Facing direction
rogue_stealth[1,type=float]               # Current stealth level
rogue_skills[12,type=int]                 # Lockpicking, trapfinding, etc.
rogue_abilities[6,type=int]               # Ability scores
rogue_experience[1,type=int]              # Experience points/level
rogue_alertness[1,type=float]             # Perception/awareness state
rogue_secret_knowledge[30,type=float]     # Hidden passages, traps discovered

# Cleric - Healer/Support specialist
cleric_health[80,type=int]                # Current hit points
cleric_position[2,type=float]             # X,Y coordinates in dungeon
cleric_orientation[1,type=float]          # Facing direction
cleric_divine_power[60,type=int]          # Healing/divine magic resource
cleric_blessings[8,type=discrete]         # Active divine effects
cleric_abilities[6,type=int]              # Ability scores
cleric_experience[1,type=int]             # Experience points/level
cleric_faith[1,type=float]                # Divine connection strength
cleric_party_awareness[4,type=float]      # Team member status knowledge

# === DUNGEON MASTER STATE ===
dm_narrative_state[10,type=float]         # Current story arc progression
dm_difficulty_scaling[1,type=float]       # Dynamic challenge adjustment
dm_environmental_control[50,type=discrete] # Room/trap/monster activation
dm_npc_states[20,type=float]              # Non-player character conditions
dm_plot_advancement[5,type=float]         # Main quest progression markers
dm_improvisation_mode[1,type=float]       # Adaptive storytelling state
dm_party_observation[4,type=float]        # Player behavior assessment

# === ENVIRONMENT STATE ===
dungeon_layout[100,100,type=discrete]     # 2D grid of room/corridor/wall states
room_contents[50,15,type=discrete]        # Items/monsters/features per room
trap_states[30,type=discrete]             # Armed/disarmed/triggered traps
lighting_conditions[100,100,type=float]   # Visibility map
atmospheric_effects[10,type=float]        # Weather/magic/ambiance
treasure_locations[25,type=discrete]      # Hidden/revealed/collected treasures
monster_positions[15,3,type=float]        # Active creature locations and states
door_states[40,type=discrete]             # Open/closed/locked/secret doors

# === OBSERVATION MODALITIES ===
# Visual observations
visual_fighter[20,type=float]             # What fighter can see
visual_mage[20,type=float]               # What mage can see
visual_rogue[25,type=float]              # What rogue can see (enhanced perception)
visual_cleric[20,type=float]             # What cleric can see

# Auditory observations
audio_all[15,type=float]                 # Sounds audible to all players
audio_rogue[20,type=float]               # Enhanced rogue hearing

# Tactical observations (game mechanics visible to players)
tactical_health[4,type=float]            # All party member health states
tactical_resources[4,type=float]         # Mana/stamina/divine power levels
tactical_positions[8,type=float]         # Relative party positions
tactical_threats[10,type=float]          # Known enemy positions/capabilities

# Social observations (NPC interactions, DM descriptions)
social_dm_descriptions[20,type=float]    # DM narrative and environmental descriptions
social_npc_interactions[15,type=float]   # NPC dialogue and behavior
social_party_communication[12,type=float] # Inter-party member communication

# Magical observations (arcane/divine environmental effects)
magical_auras[10,type=float]             # Magical effects visible to mage/cleric
magical_traps[8,type=float]              # Magical trap signatures
magical_items[12,type=float]             # Enchanted item detection

# === ACTION SPACES ===
# Fighter actions
fighter_movement[2,type=float]           # Movement vector
fighter_combat[8,type=discrete]          # Attack types/targets
fighter_tactics[5,type=discrete]         # Defensive/supportive actions
fighter_social[4,type=discrete]          # Social interaction choices

# Mage actions  
mage_movement[2,type=float]              # Movement vector
mage_spells[15,type=discrete]            # Spell casting choices
mage_positioning[4,type=discrete]        # Strategic positioning
mage_social[4,type=discrete]             # Social interaction choices

# Rogue actions
rogue_movement[2,type=float]             # Movement vector (includes stealth)
rogue_skills[12,type=discrete]           # Skill use (lockpicking, etc.)
rogue_sneak_attack[6,type=discrete]      # Stealth combat options
rogue_exploration[8,type=discrete]       # Secret finding actions
rogue_social[4,type=discrete]            # Social interaction choices

# Cleric actions
cleric_movement[2,type=float]            # Movement vector
cleric_healing[8,type=discrete]          # Healing spell/ability targets
cleric_divine_magic[10,type=discrete]    # Divine spell choices
cleric_support[6,type=discrete]          # Buff/protection actions
cleric_social[4,type=discrete]           # Social interaction choices

# DM actions
dm_environmental[20,type=discrete]       # Environmental changes/activations
dm_narrative[15,type=discrete]           # Story progression choices
dm_npc_control[25,type=discrete]         # NPC behavior directions
dm_challenge_scaling[5,type=discrete]    # Difficulty adjustments

# === POLICY AND COORDINATION ===
party_coordination[16,type=float]        # Joint action policy distribution
individual_policies[4,12,type=float]     # Each character's action preferences
turn_order[4,type=discrete]              # Initiative/turn sequence
communication_policy[6,type=discrete]    # Inter-party communication strategy

# === FREE ENERGY AND PLANNING ===
expected_free_energy[4,type=float]       # EFE for each party member
collaborative_efe[1,type=float]          # Joint expected free energy
planning_horizon[4,type=int]             # Forward planning depth per character
uncertainty_estimates[10,type=float]     # Environmental/DM unpredictability

# === TEMPORAL DYNAMICS ===
game_turn[1,type=int]                    # Current turn number
session_time[1,type=float]               # Real-time session progress
narrative_pacing[1,type=float]           # Story progression rate

## Connections
# Character state dynamics
(fighter_abilities, fighter_equipment) > fighter_combat
(fighter_health, fighter_stamina) > fighter_tactics
(mage_abilities, mage_spells_known) > mage_spells
(mage_mana, mage_focus) > mage_positioning
(rogue_abilities, rogue_skills) > rogue_exploration
(rogue_stealth, rogue_alertness) > rogue_sneak_attack
(cleric_abilities, cleric_blessings) > cleric_divine_magic
(cleric_divine_power, cleric_faith) > cleric_healing

# Environmental observation generation
(fighter_position, fighter_orientation, lighting_conditions) > visual_fighter
(mage_position, mage_orientation, lighting_conditions) > visual_mage
(rogue_position, rogue_orientation, lighting_conditions, rogue_alertness) > visual_rogue
(cleric_position, cleric_orientation, lighting_conditions) > visual_cleric

# Tactical information flow
(fighter_health, mage_health, rogue_health, cleric_health) > tactical_health
(fighter_stamina, mage_mana, cleric_divine_power) > tactical_resources
(fighter_position, mage_position, rogue_position, cleric_position) > tactical_positions
(monster_positions, trap_states) > tactical_threats

# Social and narrative dynamics
(dm_narrative_state, dm_npc_states) > social_dm_descriptions
(dm_npc_states, social_party_communication) > social_npc_interactions
(party_coordination, communication_policy) > social_party_communication

# Magical environment awareness
(mage_arcane_knowledge, atmospheric_effects) > magical_auras
(magical_traps, rogue_secret_knowledge) > magical_traps
(treasure_locations, magical_items) > magical_items

# Environmental dynamics
(dm_environmental, dungeon_layout) > room_contents
(rogue_exploration, trap_states) > door_states
(party_coordination, monster_positions) > lighting_conditions

# DM adaptive control
(tactical_health, tactical_resources, dm_difficulty_scaling) > dm_environmental
(social_party_communication, dm_party_observation) > dm_narrative
(dm_plot_advancement, narrative_pacing) > dm_npc_control

# Coordination and planning
(fighter_tactics, mage_positioning, rogue_exploration, cleric_support) > party_coordination
(expected_free_energy, collaborative_efe) > individual_policies
(communication_policy, social_party_communication) > turn_order
(planning_horizon, uncertainty_estimates) > expected_free_energy

# Temporal progression
(game_turn, session_time) > narrative_pacing

## InitialParameterization
# Character starting conditions
fighter_health=100
fighter_position={(5.0, 5.0)}
fighter_orientation=1.57
fighter_stamina=50
fighter_equipment={(1,1,1,0,0,0,0,0,0,0)}
fighter_abilities={(16,12,15,10,13,14)}
fighter_experience=0
fighter_morale=0.8

mage_health=60
mage_position={(3.0, 5.0)}
mage_orientation=1.57
mage_mana=80
mage_spells_known={(1,1,1,0,1,0,0,0,0,0,0,0,0,0,0)}
mage_abilities={(8,14,12,17,15,11)}
mage_experience=0
mage_focus=0.9

rogue_health=70
rogue_position={(7.0, 5.0)}
rogue_orientation=1.57
rogue_stealth=0.5
rogue_skills={(5,4,6,3,5,4,3,5,4,6,5,4)}
rogue_abilities={(12,17,13,14,16,10)}
rogue_experience=0
rogue_alertness=0.95

cleric_health=80
cleric_position={(5.0, 3.0)}
cleric_orientation=1.57
cleric_divine_power=60
cleric_blessings={(1,0,0,0,0,0,0,0)}
cleric_abilities={(14,10,14,12,17,16)}
cleric_experience=0
cleric_faith=0.85
cleric_party_awareness={(1.0,0.9,0.95,1.0)}

# DM initialization
dm_narrative_state={(0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)}
dm_difficulty_scaling=0.5
dm_improvisation_mode=0.3
dm_party_observation={(0.5,0.5,0.5,0.5)}

# Environment initialization
lighting_conditions=0.3
atmospheric_effects={(0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)}
trap_states={(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)}
treasure_locations={(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)}
door_states={(2,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)}

# Coordination initialization
party_coordination={(0.25,0.25,0.25,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)}
turn_order={(0,1,2,3)}
communication_policy={(1,1,1,0,0,0)}

# Planning parameters
expected_free_energy={(2.5,2.8,2.2,2.6)}
collaborative_efe=10.1
planning_horizon={(3,4,5,3)}
uncertainty_estimates={(0.8,0.7,0.9,0.6,0.8,0.7,0.5,0.9,0.8,0.7)}

# Temporal initialization
game_turn=1
session_time=0.0
narrative_pacing=0.5

## Equations
# Character state inference:
# q(s_character) = σ(ln(P(o|s)) + ln(P(s|s_prev,a)) + EFE_character)

# Collaborative policy inference:
# q(π_joint) = σ(-G_collaborative - Σ_i(G_individual_i))

# Party coordination dynamics:
# coordination_t+1 = coordination_t + α(successful_actions) - β(conflicting_actions)

# Character development:
# experience_t+1 = experience_t + reward(actions, outcomes)

# DM adaptive difficulty:
# difficulty_scaling_t+1 = difficulty_scaling_t + ε(party_performance - target_challenge)

# Expected Free Energy for coordination:
# G_collaborative = Σ_t[KL(q(s_joint|π)||p(s_joint)) + KL(q(o_joint|π)||p(o_joint|C_joint))]

## Time
Dynamic
DiscreteTime=game_turn
ModelTimeHorizon=Unbounded
TemporalDepth=5

## ActInfOntologyAnnotation
fighter_health=CharacterHealthFighter
mage_mana=SpellcastingResource
rogue_stealth=StealthState
cleric_divine_power=DivineResource
dungeon_layout=EnvironmentalState
lighting_conditions=PerceptualContext
trap_states=HiddenEnvironmentalThreats
monster_positions=DynamicEnvironmentalActors
visual_fighter=VisualObservationModalityFighter
tactical_health=TacticalInformationModality
social_dm_descriptions=NarrativeObservationModality
magical_auras=ArcaneSensoryModality
fighter_combat=CombatActionSpace
mage_spells=SpellcastingActionSpace
rogue_exploration=ExplorationActionSpace
cleric_healing=HealingActionSpace
dm_environmental=EnvironmentalControlActionSpace
party_coordination=CollaborativePolicySpace
individual_policies=IndividualPolicyBeliefs
expected_free_energy=ExpectedFreeEnergyEstimates
collaborative_efe=JointExpectedFreeEnergy
game_turn=DiscreteTimeStep
narrative_pacing=StoryProgressionRate

## ModelParameters
num_players: 4
num_dm_agents: 1
dungeon_size: [100, 100]
max_monsters: 15
max_rooms: 50
observation_modalities: 6
action_modalities: 5
planning_depth: 5
collaboration_enabled: true
adaptive_dm: true
narrative_generation: true
character_development: true
session_persistence: true

## Footer
Collaborative Role Playing Game - Comprehensive Multi-Agent Active Inference Model
Supports dynamic D&D-style gameplay with adaptive AI agents and emergent narrative

## Signature
Creator: AI Assistant for GNN  
Date: 2024-12-28
Source: Comprehensive RPG Active Inference Architecture
Status: Research implementation for collaborative AI gaming and emergent narrative systems 