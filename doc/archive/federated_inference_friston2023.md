# GNN Example: Federated Inference and Belief Sharing
# Format: Comprehensive GNN representation of the federated inference model from Friston et al. (2023)
# Version: 1.0
# Based on: "Federated inference and belief sharing" (Neuroscience & Biobehavioral Reviews, 2024)

## GNNSection
FederatedInferenceFriston2023

## GNNVersionAndFlags
GNN v1

## ModelName
Federated Inference Multi-Agent Belief Sharing Model v1

## ModelAnnotation
This model represents the federated inference system described in Friston et al. (2023), featuring three sentinel agents 
that share beliefs about their environment through communication. The agents have complementary perspectives of a shared 
world and can broadcast their posterior beliefs to minimize joint free energy across the collective.

Key features:
- Three agents with restricted fields of view (120° each, back-to-back configuration)
- Shared generative model with conserved likelihood mappings for communication
- Hidden state factors: allocentric location (9 positions), proximity (2 levels), pose (2 states), gaze direction (3 options)
- Observation modalities: visual (4 types) + auditory communication (3 channels) + proprioceptive
- Belief sharing through identity likelihood mappings from posterior beliefs to communicative expressions
- Active inference with expected free energy minimization for epistemic foraging and communication

The model demonstrates emergence of distributed cognition, language acquisition, and collective intelligence
through nested free energy minimization processes across inference, learning, and selection timescales.

## StateSpaceBlock
### Hidden State Factors (B matrices as shown in diagram)
# B{1} Location - 9 states (0-8 in circular arrangement)
s_location[9,1,type=float]          # Allocentric position: {0,1,2,3,4,5,6,7,8} in radial layout

# B{2} Proximity - 2 states  
s_proximity[2,1,type=float]         # Distance: {close=0, near=1}

# B{3} Pose - 2 states
s_pose[2,1,type=float]              # Disposition: {foe=0, friend=1}

# B{4} Gaze - 3 states per agent (each agent has own gaze factor)
s_gaze_a1[3,1,type=float]          # Agent 1 gaze: {left=0, center=1, right=2}  
s_gaze_a2[3,1,type=float]          # Agent 2 gaze: {left=0, center=1, right=2}
s_gaze_a3[3,1,type=float]          # Agent 3 gaze: {left=0, center=1, right=2}

### Next Hidden States (for temporal dynamics)
s_prime_location[9,1,type=float]    # Next time step location
s_prime_proximity[2,1,type=float]   # Next time step proximity
s_prime_pose[2,1,type=float]        # Next time step pose
s_prime_gaze_a1[3,1,type=float]    # Next gaze agent 1
s_prime_gaze_a2[3,1,type=float]    # Next gaze agent 2
s_prime_gaze_a3[3,1,type=float]    # Next gaze agent 3

### Observation Likelihood Matrices (A matrices as shown in diagram)
# A{1} Vision - Subject detection with line of sight (LOS) constraints
A_vis_subject_a1[6,9,2,2,3,type=float]    # Subject attributes: {LOS=-4, proximity_foe=-2, proximity_friend=1, else_nothing=0, else_close_foe=2, else_far_person=-4}

# A{2}, A{3}, A{4} Vision - Contrast energy detection  
A_vis_center_a1[3,9,2,2,3,type=float]     # Central contrast: {nothing=0, near=1, close=2}
A_vis_left_a1[3,9,2,2,3,type=float]       # Left peripheral: {nothing=0, near=1, close=2}  
A_vis_right_a1[3,9,2,2,3,type=float]      # Right peripheral: {nothing=0, near=1, close=2}

# A{5} Proprioception - Gaze direction
A_proprioceptive_a1[3,3,type=float]       # Gaze direction: {left=0, center=1, right=2} (identity mapping)

# Agent 2 - Same structure as Agent 1
A_vis_subject_a2[6,9,2,2,3,type=float]    # Subject detection with field-of-view constraints
A_vis_center_a2[3,9,2,2,3,type=float]     # Central contrast energy
A_vis_left_a2[3,9,2,2,3,type=float]       # Left peripheral contrast  
A_vis_right_a2[3,9,2,2,3,type=float]      # Right peripheral contrast
A_proprioceptive_a2[3,3,type=float]       # Gaze proprioception

# Agent 3 - Same structure as Agent 1  
A_vis_subject_a3[6,9,2,2,3,type=float]    # Subject detection with field-of-view constraints
A_vis_center_a3[3,9,2,2,3,type=float]     # Central contrast energy
A_vis_left_a3[3,9,2,2,3,type=float]       # Left peripheral contrast
A_vis_right_a3[3,9,2,2,3,type=float]      # Right peripheral contrast 
A_proprioceptive_a3[3,3,type=float]       # Gaze proprioception

# A{6}, A{7}, A{8} Auditory - Communication channels (shared/conserved mappings)
A_comm_location[9,9,type=float]      # A{6} Location communication (identity matrix)
A_comm_proximity[2,2,type=float]    # A{7} Proximity communication (identity matrix)
A_comm_pose[2,2,type=float]         # A{8} Pose communication (identity matrix)

### Transition Matrices (B matrices as shown in diagram)
# B{1} Location - 9 states with 3 movement patterns
B_location[9,9,3,type=float]        # B{1}: {still=0, left=1, right=2} movement around 9 locations

# B{2} Proximity - 2 states (uncontrolled)  
B_proximity[2,2,1,type=float]       # B{2}: {close, near} transitions (environmental)

# B{3} Pose - 2 states (uncontrolled)
B_pose[2,2,1,type=float]            # B{3}: {friend, foe} transitions (environmental)

# B{4} Gaze - 3 states per agent (controllable)
B_gaze_a1[3,3,3,type=float]        # B{4a}: Agent 1 gaze {left, center, right} 
B_gaze_a2[3,3,3,type=float]        # B{4b}: Agent 2 gaze {left, center, right}
B_gaze_a3[3,3,3,type=float]        # B{4c}: Agent 3 gaze {left, center, right}

### Prior Preference Vectors
C_vis_foveal[3,type=float]          # Preferences over foveal observations
C_vis_contrast[3,type=float]        # Preferences over contrast observations
C_comm_location[9,type=float]       # Preferences over location communications
C_comm_proximity[2,type=float]      # Preferences over proximity communications
C_comm_pose[2,type=float]           # Preferences over pose communications
C_proprioceptive[3,type=float]      # Preferences over proprioceptive observations

### Prior State Distributions
D_location[9,type=float]            # Prior over initial locations (uniform)
D_proximity[2,type=float]           # Prior over initial proximity (uniform)
D_pose[2,type=float]               # Prior over initial pose (uniform)
D_gaze_a1[3,type=float]           # Prior over initial gaze agent 1
D_gaze_a2[3,type=float]           # Prior over initial gaze agent 2
D_gaze_a3[3,type=float]           # Prior over initial gaze agent 3

### Observations (matching A matrix structure from diagram)
# Agent 1 Observations  
o_vis_subject_a1[6,1,type=float]    # A{1} Subject detection observation  
o_vis_center_a1[3,1,type=float]     # A{2} Central contrast observation
o_vis_left_a1[3,1,type=float]       # A{3} Left peripheral observation
o_vis_right_a1[3,1,type=float]      # A{4} Right peripheral observation
o_proprioceptive_a1[3,1,type=float] # A{5} Gaze proprioception

# Agent 2 Observations
o_vis_subject_a2[6,1,type=float]    # A{1} Subject detection
o_vis_center_a2[3,1,type=float]     # A{2} Central contrast
o_vis_left_a2[3,1,type=float]       # A{3} Left peripheral 
o_vis_right_a2[3,1,type=float]      # A{4} Right peripheral
o_proprioceptive_a2[3,1,type=float] # A{5} Gaze proprioception

# Agent 3 Observations
o_vis_subject_a3[6,1,type=float]    # A{1} Subject detection
o_vis_center_a3[3,1,type=float]     # A{2} Central contrast
o_vis_left_a3[3,1,type=float]       # A{3} Left peripheral
o_vis_right_a3[3,1,type=float]      # A{4} Right peripheral
o_proprioceptive_a3[3,1,type=float] # A{5} Gaze proprioception

# Auditory Communication Observations (A{6}, A{7}, A{8} from diagram)
o_comm_location_a1[9,1,type=float]  # A{6} Location broadcast from agent 1
o_comm_location_a2[9,1,type=float]  # A{6} Location broadcast from agent 2  
o_comm_location_a3[9,1,type=float]  # A{6} Location broadcast from agent 3

o_comm_proximity_a1[2,1,type=float] # A{7} Proximity broadcast from agent 1
o_comm_proximity_a2[2,1,type=float] # A{7} Proximity broadcast from agent 2
o_comm_proximity_a3[2,1,type=float] # A{7} Proximity broadcast from agent 3

o_comm_pose_a1[2,1,type=float]     # A{8} Pose broadcast from agent 1
o_comm_pose_a2[2,1,type=float]     # A{8} Pose broadcast from agent 2
o_comm_pose_a3[2,1,type=float]     # A{8} Pose broadcast from agent 3

### Policy and Control Variables
π_gaze_a1[3,type=float]           # Policy distribution over gaze actions agent 1
π_gaze_a2[3,type=float]           # Policy distribution over gaze actions agent 2
π_gaze_a3[3,type=float]           # Policy distribution over gaze actions agent 3

u_gaze_a1[1,type=int]             # Chosen gaze action agent 1
u_gaze_a2[1,type=int]             # Chosen gaze action agent 2
u_gaze_a3[1,type=int]             # Chosen gaze action agent 3

### Free Energy Terms
G_a1[1,type=float]                # Expected free energy agent 1
G_a2[1,type=float]                # Expected free energy agent 2
G_a3[1,type=float]                # Expected free energy agent 3
G_joint[1,type=float]             # Joint free energy across agents

F_a1[1,type=float]                # Variational free energy agent 1
F_a2[1,type=float]                # Variational free energy agent 2
F_a3[1,type=float]                # Variational free energy agent 3
F_joint[1,type=float]             # Joint variational free energy

### Posterior Beliefs (sufficient statistics)
q_location[9,type=float]          # Posterior beliefs over location
q_proximity[2,type=float]         # Posterior beliefs over proximity
q_pose[2,type=float]              # Posterior beliefs over pose
q_gaze_a1[3,type=float]          # Posterior beliefs over gaze agent 1
q_gaze_a2[3,type=float]          # Posterior beliefs over gaze agent 2
q_gaze_a3[3,type=float]          # Posterior beliefs over gaze agent 3

### Time and Learning Parameters
t[1,type=int]                     # Current time step
τ_learning[1,type=float]          # Learning timescale parameter
η_precision[1,type=float]         # Precision parameter for belief sharing
γ_attention[1,type=float]         # Attentional precision parameter

## Connections
### State Dependencies
(D_location,D_proximity,D_pose)-(s_location,s_proximity,s_pose)
(D_gaze_a1,D_gaze_a2,D_gaze_a3)-(s_gaze_a1,s_gaze_a2,s_gaze_a3)

### Visual Observations for Agent 1
(s_location,s_proximity,s_pose,s_gaze_a1)-(A_vis_foveal_a1,A_vis_center_a1,A_vis_left_a1,A_vis_right_a1)
(A_vis_foveal_a1)-(o_vis_foveal_a1)
(A_vis_center_a1)-(o_vis_center_a1)
(A_vis_left_a1)-(o_vis_left_a1)
(A_vis_right_a1)-(o_vis_right_a1)

### Visual Observations for Agent 2
(s_location,s_proximity,s_pose,s_gaze_a2)-(A_vis_foveal_a2,A_vis_center_a2,A_vis_left_a2,A_vis_right_a2)
(A_vis_foveal_a2)-(o_vis_foveal_a2)
(A_vis_center_a2)-(o_vis_center_a2)
(A_vis_left_a2)-(o_vis_left_a2)
(A_vis_right_a2)-(o_vis_right_a2)

### Visual Observations for Agent 3
(s_location,s_proximity,s_pose,s_gaze_a3)-(A_vis_foveal_a3,A_vis_center_a3,A_vis_left_a3,A_vis_right_a3)
(A_vis_foveal_a3)-(o_vis_foveal_a3)
(A_vis_center_a3)-(o_vis_center_a3)
(A_vis_left_a3)-(o_vis_left_a3)
(A_vis_right_a3)-(o_vis_right_a3)

### Proprioceptive Observations
(s_gaze_a1)-(A_proprioceptive_a1)-(o_proprioceptive_a1)
(s_gaze_a2)-(A_proprioceptive_a2)-(o_proprioceptive_a2)
(s_gaze_a3)-(A_proprioceptive_a3)-(o_proprioceptive_a3)

### Communication Dependencies (Belief Broadcasting)
(q_location)-(A_comm_location)-(o_comm_location_a1,o_comm_location_a2,o_comm_location_a3)
(q_proximity)-(A_comm_proximity)-(o_comm_proximity_a1,o_comm_proximity_a2,o_comm_proximity_a3)
(q_pose)-(A_comm_pose)-(o_comm_pose_a1,o_comm_pose_a2,o_comm_pose_a3)

### Temporal Transitions
(s_location,u_subject)-(B_location)-(s_prime_location)  ### u_subject is implicit environmental control
(s_proximity)-(B_proximity)-(s_prime_proximity)
(s_pose)-(B_pose)-(s_prime_pose)
(s_gaze_a1,u_gaze_a1)-(B_gaze_a1)-(s_prime_gaze_a1)
(s_gaze_a2,u_gaze_a2)-(B_gaze_a2)-(s_prime_gaze_a2)
(s_gaze_a3,u_gaze_a3)-(B_gaze_a3)-(s_prime_gaze_a3)

### Policy and Action Selection
(C_vis_foveal,C_vis_contrast,C_comm_location,C_comm_proximity,C_comm_pose)>G_a1
(C_vis_foveal,C_vis_contrast,C_comm_location,C_comm_proximity,C_comm_pose)>G_a2
(C_vis_foveal,C_vis_contrast,C_comm_location,C_comm_proximity,C_comm_pose)>G_a3
G_a1>π_gaze_a1
G_a2>π_gaze_a2
G_a3>π_gaze_a3
π_gaze_a1-u_gaze_a1
π_gaze_a2-u_gaze_a2
π_gaze_a3-u_gaze_a3

### Joint Free Energy
(G_a1,G_a2,G_a3)>G_joint
(F_a1,F_a2,F_a3)>F_joint

### Belief Updates
(o_vis_foveal_a1,o_vis_center_a1,o_vis_left_a1,o_vis_right_a1,o_proprioceptive_a1,o_comm_location_a2,o_comm_location_a3,o_comm_proximity_a2,o_comm_proximity_a3,o_comm_pose_a2,o_comm_pose_a3)>(q_location,q_proximity,q_pose,q_gaze_a1)
(o_vis_foveal_a2,o_vis_center_a2,o_vis_left_a2,o_vis_right_a2,o_proprioceptive_a2,o_comm_location_a1,o_comm_location_a3,o_comm_proximity_a1,o_comm_proximity_a3,o_comm_pose_a1,o_comm_pose_a3)>(q_location,q_proximity,q_pose,q_gaze_a2)
(o_vis_foveal_a3,o_vis_center_a3,o_vis_left_a3,o_vis_right_a3,o_proprioceptive_a3,o_comm_location_a1,o_comm_location_a2,o_comm_proximity_a1,o_comm_proximity_a2,o_comm_pose_a1,o_comm_pose_a2)>(q_location,q_proximity,q_pose,q_gaze_a3)

### Temporal progression
t=Time

## InitialParameterization
### Transition Matrix Parameterization (from diagram)
# B{1} Location - 9 locations in circular arrangement
# d = [0-1:1]; % still, left, or right  
# for u = 1:3
# B{1}(:,:,u) = eye(9,9);
B_location={
  still: eye(9),           # u=0: No movement
  left: circ_shift(eye(9), -1),   # u=1: Counter-clockwise 
  right: circ_shift(eye(9), 1)    # u=2: Clockwise
}

# B{2} Proximity transitions (uncontrolled environmental dynamics)
# B{2} = eye(2,2) + 1/4;
B_proximity={((0.9,0.1),(0.1,0.9))}  # Close/near transitions with persistence

# B{3} Pose transitions (uncontrolled environmental dynamics)  
# B{3} = eye(2,2);
B_pose={((0.95,0.05),(0.05,0.95))}   # Friend/foe transitions with high persistence

# B{4} Gaze Transition Matrices (controllable actions)
# for u = 1:3 % center, left, or right  
# B{4}(:,:,u) = zeros(3,3);
# B{4}(u,u,u) = 1;
B_gaze_a1={
  center: ((1,0,0),(0,0,0),(0,0,0)),   # u=1: Look center
  left: ((0,0,0),(0,1,0),(0,0,0)),     # u=2: Look left  
  right: ((0,0,0),(0,0,0),(0,0,1))     # u=3: Look right
}
B_gaze_a2=B_gaze_a1  # Same gaze control structure for all agents
B_gaze_a3=B_gaze_a1

### Visual Likelihood Matrix Specifications (from diagram)
# A{1} Subject attributes with line of sight (LOS) constraints
# % subject attributes
# A{1}(1,:,1,2,s4) = 1; % if there is someone in the line of sight (LOS)
# A{1}(1,:,1,1,s4) = -4; % LOS = -4
# % proximity s{2} & pose s{3}
# if s{2} = 1 && s{3} = 1 % close foe
#   elseif s{2}(s2) = 1 && s{3}(s3) = 1  
#   o = 2; % close foe
# if s{2} = 1  
#   o = 2; % close foe  
# o = 3; % for person
# o = 4; % nothing to see
# else
A_vis_subject_a1={
  # Complex LOS mapping based on agent field of view, proximity, and pose
  # This would require detailed field-of-view calculations per agent
  subject_detection_mapping: "agent_specific_visual_field_implementation"
}

# A{2}, A{3}, A{4} Contrast energy detection
# % contrast energy: close, near or nothing
A_vis_contrast={
  contrast_energy_mapping: "location_proximity_gaze_dependent_contrast_detection"
}

# A{6}, A{7}, A{8} Communication Likelihood Matrices (identity mappings)
A_comm_location=eye(9)    # A{6}: Perfect identity for location communication
A_comm_proximity=eye(2)   # A{7}: Perfect identity for proximity communication  
A_comm_pose=eye(2)       # A{8}: Perfect identity for pose communication

### Prior Preferences (epistemic drive and risk aversion)
C_vis_foveal={(0.0, 1.0, -2.0)}     # Prefer seeing friends, avoid foes
C_vis_contrast={(0.0, 0.5, 1.0)}    # Prefer detecting something vs nothing
C_comm_location={(-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1)}  # Slight cost for uncertainty
C_comm_proximity={(0.0, -0.5)}       # Prefer knowing proximity is far (safety)
C_comm_pose={(1.0, 0.0)}            # Prefer friend communications
C_proprioceptive={(0.0, 0.0, 0.0)}  # No preference over gaze direction per se

### Prior State Distributions (uniform uncertainty)
D_location={(0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111)}  # Uniform over 9 locations
D_proximity={(0.5, 0.5)}            # Uniform over near/far
D_pose={(0.5, 0.5)}                 # Uniform over friend/foe
D_gaze_a1={(0.33, 0.34, 0.33)}     # Slight preference for center gaze
D_gaze_a2={(0.33, 0.34, 0.33)}
D_gaze_a3={(0.33, 0.34, 0.33)}

### Learning and Precision Parameters
τ_learning=32.0         # Forgetting timescale (32 experiences retained)
η_precision=4.0         # Precision of belief sharing communication
γ_attention=2.0         # Attentional precision for epistemic foraging

## Equations
### Variational Free Energy (per agent)
# F = E_q[ln q(s) - ln p(o,s)] = D_KL[q(s)||p(s|o)] - ln p(o)
F_a1 = sum_s[q_s * ln(q_s)] - sum_s[q_s * ln(p_s_given_o_a1)]

### Expected Free Energy (policy evaluation)  
# G = E_q[ln q(π) - ln p(o_future|π)] - risk - ambiguity
G_a1 = risk_term_a1 + ambiguity_term_a1 - information_gain_a1

### Belief Sharing Update (federated inference)
# Modified likelihood message includes communications from other agents
# ln p(o_i|s) + sum_j≠i[η * ln q_j(s)]  where j indexes other agents
q_location_a1 ∝ exp[ln p(o_vis_a1|s_location) + η * (ln q_location_a2 + ln q_location_a3)]

### Joint Free Energy Minimization
# F_joint = sum_i[F_i] - η * sum_i≠j[D_KL[q_i(s)||q_j(s)]]
F_joint = F_a1 + F_a2 + F_a3 - η_precision * belief_alignment_term

### Communication Generation
# o_comm = A_comm * q(s) + noise
o_comm_location_a1 = A_comm_location @ q_location + communication_noise

### Active Learning (Dirichlet count updates)
# a_new = a_old + learning_rate * (observation - expected_observation)
# With forgetting: a_new = decay_factor * a_old + learning_increment
dirichlet_update: "accumulate_experience_with_forgetting"

### Bayesian Model Reduction (structure learning)
# ΔF = ln B(a) - ln B(a') + (a-a')^T [ψ(a) - ψ(sum(a))]
structure_learning_criterion: "sparsity_inducing_model_comparison"

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=Unbounded  # Continuous operation, specific episodes have finite horizon

## ActInfOntologyAnnotation
### Generative Model Components
A_vis_foveal_a1=LikelihoodMatrixVisualFovealAgent1
A_vis_center_a1=LikelihoodMatrixVisualCentralAgent1
A_vis_left_a1=LikelihoodMatrixVisualLeftAgent1
A_vis_right_a1=LikelihoodMatrixVisualRightAgent1
A_proprioceptive_a1=LikelihoodMatrixProprioceptiveAgent1
A_comm_location=LikelihoodMatrixCommunicationLocation
A_comm_proximity=LikelihoodMatrixCommunicationProximity
A_comm_pose=LikelihoodMatrixCommunicationPose

B_location=TransitionMatrixLocation
B_proximity=TransitionMatrixProximity  
B_pose=TransitionMatrixPose
B_gaze_a1=TransitionMatrixGazeAgent1
B_gaze_a2=TransitionMatrixGazeAgent2
B_gaze_a3=TransitionMatrixGazeAgent3

C_vis_foveal=LogPreferenceVectorVisualFoveal
C_vis_contrast=LogPreferenceVectorVisualContrast
C_comm_location=LogPreferenceVectorCommunicationLocation
C_comm_proximity=LogPreferenceVectorCommunicationProximity
C_comm_pose=LogPreferenceVectorCommunicationPose

D_location=PriorOverLocationStates
D_proximity=PriorOverProximityStates
D_pose=PriorOverPoseStates
D_gaze_a1=PriorOverGazeStatesAgent1
D_gaze_a2=PriorOverGazeStatesAgent2
D_gaze_a3=PriorOverGazeStatesAgent3

### State Variables
s_location=HiddenStateLocation
s_proximity=HiddenStateProximity
s_pose=HiddenStatePose
s_gaze_a1=HiddenStateGazeAgent1
s_gaze_a2=HiddenStateGazeAgent2  
s_gaze_a3=HiddenStateGazeAgent3

### Observations
o_vis_foveal_a1=ObservationVisualFovealAgent1
o_vis_center_a1=ObservationVisualCentralAgent1
o_vis_left_a1=ObservationVisualLeftAgent1
o_vis_right_a1=ObservationVisualRightAgent1
o_proprioceptive_a1=ObservationProprioceptiveAgent1
o_comm_location_a1=ObservationCommunicationLocationAgent1
o_comm_proximity_a1=ObservationCommunicationProximityAgent1
o_comm_pose_a1=ObservationCommunicationPoseAgent1

### Policy and Control
π_gaze_a1=PolicyVectorGazeAgent1
π_gaze_a2=PolicyVectorGazeAgent2
π_gaze_a3=PolicyVectorGazeAgent3
u_gaze_a1=ActionGazeAgent1
u_gaze_a2=ActionGazeAgent2
u_gaze_a3=ActionGazeAgent3

### Free Energy Terms
G_a1=ExpectedFreeEnergyAgent1
G_a2=ExpectedFreeEnergyAgent2
G_a3=ExpectedFreeEnergyAgent3
G_joint=ExpectedFreeEnergyJoint
F_a1=VariationalFreeEnergyAgent1
F_a2=VariationalFreeEnergyAgent2
F_a3=VariationalFreeEnergyAgent3
F_joint=VariationalFreeEnergyJoint

### Posterior Beliefs
q_location=PosteriorBeliefLocation
q_proximity=PosteriorBeliefProximity
q_pose=PosteriorBeliefPose
q_gaze_a1=PosteriorBeliefGazeAgent1
q_gaze_a2=PosteriorBeliefGazeAgent2
q_gaze_a3=PosteriorBeliefGazeAgent3

### Meta-Parameters
τ_learning=LearningTimescale
η_precision=BeliefSharingPrecision
γ_attention=AttentionalPrecision
t=TimeStep

## ModelParameters
### Multi-Agent Configuration
num_agents: 3
agent_configuration: "back_to_back_sentinel_array"
communication_topology: "all_to_all_belief_broadcasting"

### State Space Dimensions
num_locations: 9            # Allocentric radial positions around agents
num_proximity_levels: 2     # {near, far}
num_pose_states: 2         # {foe, friend}
num_gaze_directions: 3     # {left, center, right} for each agent

### Observation Modality Dimensions
num_visual_foveal_outcomes: 3      # {distant, friend_close, foe_close}
num_visual_contrast_outcomes: 3    # {none, near, close}
num_proprioceptive_outcomes: 3     # {left, center, right}
num_communication_outcomes: [9, 2, 2]  # [location, proximity, pose]

### Control Dimensions
num_gaze_actions: 3        # {look_left, look_center, look_right}
controllable_factors: ["gaze_a1", "gaze_a2", "gaze_a3"]
uncontrollable_factors: ["location", "proximity", "pose"]  # Environmental dynamics

### Learning Parameters
learning_rate: 0.1
forgetting_timescale: 32
belief_sharing_precision: 4.0
attention_precision: 2.0
model_selection_threshold: 1.0  # Natural units for Bayesian model reduction

### Communication Protocol
communication_modalities: ["location_broadcast", "proximity_broadcast", "pose_broadcast"]
likelihood_mapping_type: "identity_matrix"  # Perfect communication assumption
sensory_attenuation: true  # Agents don't hear themselves
communication_synchrony: "simultaneous_with_observation"

### Performance Metrics
convergence_criterion: "joint_free_energy_minimization"
epistemic_drive_strength: "expected_information_gain"
pragmatic_drive_strength: "expected_cost_minimization"

## Footer
Federated Inference Multi-Agent Belief Sharing Model v1 - GNN Representation
Based on Friston, K.J., Parr, T., Heins, C., Constant, A., Friedman, D., Isomura, T., 
Fields, C., Verbelen, T., Ramstead, M., Clippinger, J., & Frith, C.D. (2023).
"Federated inference and belief sharing." Neuroscience & Biobehavioral Reviews, 156, 105500.

This model demonstrates:
- Distributed cognition through belief sharing
- Emergence of communication protocols via active inference
- Joint free energy minimization across multiple agents
- Active learning and structure learning at nested timescales
- Cultural transmission and language acquisition
- Collective intelligence and self-evidencing

## Signature
GNN_FederatedInference_v1.0_Friston2023