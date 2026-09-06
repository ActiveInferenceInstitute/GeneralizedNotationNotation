# Spatial Attention Model

## GNNVersionAndFlags
GNN_v1.0
ProcessingFlags: spatial_attention, precision_weighting, attentional_control

## ModelName
SpatialAttentionModel

## ModelAnnotation
Models spatial attention through precision-weighted prediction error minimization and attentional control.
The model captures bottom-up capture, top-down control, and inhibition of return in visual spatial attention.
Attention emerges from precision allocation that optimizes information processing efficiency.

## StateSpaceBlock
# Spatial attention states
s_f0[16,1,type=categorical]              ### Spatial locations: 4x4 grid positions 0-15
s_f1[4,1,type=categorical]               ### Attention focus: Focused=0, Scanning=1, Switching=2, Disengaged=3
s_f2[3,1,type=categorical]               ### Attention scope: Narrow=0, Medium=1, Broad=2

# Target and distractor states
s_f3[5,1,type=categorical]               ### Target presence: Absent=0, Target_A=1, Target_B=2, Distractor=3, Noise=4
s_f4[4,1,type=categorical]               ### Target detection: Undetected=0, Detected=1, False_Alarm=2, Miss=3

# Attentional control states
s_f5[3,1,type=categorical]               ### Control mode: Bottom_Up=0, Top_Down=1, Mixed=2
s_f6[16,1,type=categorical]              ### Attention map: precision weights for each location

# Working memory and goals
s_f7[8,1,type=categorical]               ### Current goal: Find_A=0, Find_B=1, Scan_All=2, etc.
s_f8[4,1,type=categorical]               ### Attention history: location sequence buffer

## Observations
o_m0[5,1,type=categorical]               ### Visual input: matches s_f3 target/distractor states
o_m1[16,1,type=categorical]              ### Spatial saliency: bottom-up salience at each location
o_m2[3,1,type=categorical]               ### Task cues: instruction signals
o_m3[2,1,type=categorical]               ### Feedback: Correct=0, Incorrect=1

## Actions
u_c0[16,1,type=categorical]              ### Overt attention: eye movement to location 0-15
u_c1[4,1,type=categorical]               ### Covert attention: shift without eye movement
u_c2[3,1,type=categorical]               ### Attention control: Focus=0, Broaden=1, Switch=2
u_c3[2,1,type=categorical]               ### Response: Detect=0, Reject=1

## Connections
# Attention-perception coupling
s_f0, s_f6 > o_m0                        ### Attended location and precision determine perception
s_f1, s_f2 > s_f6                        ### Attention focus and scope determine precision map

# Bottom-up and top-down control
o_m1 > s_f5                              ### Saliency influences control mode
s_f7 > s_f5                              ### Goals influence control mode
s_f5 > s_f1                              ### Control mode determines attention focus

# Spatial attention dynamics
u_c0 > s_f0                              ### Overt actions change attended location
u_c1 > s_f6                              ### Covert actions change precision allocation
s_f0 > s_f8                              ### Attended locations update history

# Target detection
s_f3, s_f6 > s_f4                        ### Target presence and attention determine detection

## InitialParameterization
# Attention precision parameters
base_precision = 1.0                     ### Baseline precision for unattended locations
focused_precision = 4.0                  ### High precision for focused attention
scanning_precision = 2.0                 ### Medium precision for scanning mode

# Spatial attention parameters
attention_scope_narrow = 1               ### Number of locations for narrow focus
attention_scope_medium = 4               ### Number of locations for medium focus  
attention_scope_broad = 9                ### Number of locations for broad focus

# Control parameters
bottom_up_strength = 1.5                 ### Strength of saliency-driven attention
top_down_strength = 2.0                  ### Strength of goal-driven attention
inhibition_of_return = 0.3               ### Reduced attention to previously attended locations

# Learning and adaptation
attention_learning_rate = 0.1            ### Rate of attention map updates
precision_adaptation_rate = 0.05         ### Rate of precision optimization

# A matrices (attention-modulated observation likelihoods)
A_m0_attended = [
    [0.95, 0.02, 0.02, 0.005, 0.005],   # High precision: accurate detection
    [0.05, 0.90, 0.03, 0.01, 0.01],     # Target A detection
    [0.05, 0.03, 0.90, 0.01, 0.01],     # Target B detection  
    [0.10, 0.05, 0.05, 0.75, 0.05],     # Distractor discrimination
    [0.20, 0.10, 0.10, 0.10, 0.50]      # Noise handling
]

A_m0_unattended = [
    [0.70, 0.10, 0.10, 0.05, 0.05],     # Low precision: poor discrimination
    [0.20, 0.50, 0.15, 0.10, 0.05],     # Reduced target detection
    [0.20, 0.15, 0.50, 0.10, 0.05],     # Reduced target detection
    [0.25, 0.15, 0.15, 0.35, 0.10],     # Poor distractor discrimination
    [0.30, 0.15, 0.15, 0.15, 0.25]      # Poor noise handling
]

# B matrices (attention transition dynamics)
B_attention_focus = [
    [0.8, 0.1, 0.05, 0.05],             # Focused -> stay focused (high persistence)
    [0.2, 0.6, 0.15, 0.05],             # Scanning -> moderate persistence
    [0.3, 0.3, 0.3, 0.1],               # Switching -> unstable
    [0.1, 0.4, 0.2, 0.3]                # Disengaged -> moderate re-engagement
]

# C matrices (preferences)
C_m3 = [2.0, -2.0]                      ### Strong preference for correct responses

# D matrices (initial state priors)
D_f0 = np.ones(16) / 16                  ### Uniform spatial prior
D_f1 = [0.4, 0.3, 0.2, 0.1]             ### Prior: likely to be focused or scanning
D_f5 = [0.4, 0.5, 0.1]                  ### Prior: balanced bottom-up/top-down

## Equations
# Attention-modulated precision
precision(location, t) = base_precision + attention_weight(location, t) * (focused_precision - base_precision)

# Attention weight function  
attention_weight(location, t) = f(goal_relevance(location), saliency(location), inhibition_of_return(location, t))

# Inhibition of return
inhibition_of_return(location, t) = exp(-decay_rate * time_since_attended(location, t))

# Bottom-up attention capture
bottom_up_attention(t) = argmax(saliency_map(t) * (1 - inhibition_map(t)))

# Top-down attention control
top_down_attention(t) = argmax(goal_relevance_map(t) * current_precision_map(t))

# Competition for attention
attention_competition(t) = softmax(bottom_up_strength * bottom_up_signal(t) + top_down_strength * top_down_signal(t))

## Time
Dynamic
DiscreteTime = t
ModelTimeHorizon = 100

## ActInfOntologyAnnotation
AttentionMechanism: precision_optimization
SpatialProcessing: location_based_attention
ControlMechanism: goal_directed_attention
PerceptualProcess: attention_modulated_perception

## Footer
This model demonstrates spatial attention through Active Inference mechanisms.
Attention emerges from precision allocation optimizing information processing efficiency.
Model captures key phenomena: selective attention, attention capture, inhibition of return.

## Signature
Model: SpatialAttentionModel
Framework: Active Inference  
Domain: Cognitive Phenomena
Created: 2025
Validated: Posner cueing task, visual search paradigms 