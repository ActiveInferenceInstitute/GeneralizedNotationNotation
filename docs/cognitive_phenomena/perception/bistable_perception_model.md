# Bistable Perception Model

## GNNVersionAndFlags
GNN_v1.0
ProcessingFlags: bistable_perception, perceptual_switching, predictive_coding

## ModelName
BistablePerceptionModel

## ModelAnnotation
Models bistable perception phenomena like the Necker cube through competing generative models
and stochastic switching between perceptual hypotheses. Switching emerges from adaptation
of precision weights and accumulation of prediction error over time.

## StateSpaceBlock
# Perceptual interpretation states
s_f0[2,1,type=categorical]               ### Perceptual interpretation: Interpretation_A=0, Interpretation_B=1
s_f1[3,1,type=categorical]               ### Perceptual confidence: Low=0, Medium=1, High=2
s_f2[4,1,type=categorical]               ### Adaptation level: None=0, Mild=1, Moderate=2, Strong=3

# Attention and control states
s_f3[3,1,type=categorical]               ### Attention state: Passive=0, Active=1, Focused=2
s_f4[2,1,type=categorical]               ### Control strategy: Automatic=0, Voluntary=1

# Neural adaptation states
s_f5[4,1,type=categorical]               ### Neural fatigue: Fresh=0, Mild=1, Moderate=2, Fatigued=3
s_f6[3,1,type=categorical]               ### Precision dynamics: Stable=0, Increasing=1, Decreasing=2

# Temporal dynamics
s_f7[5,1,type=categorical]               ### Dwell time: VeryShort=0, Short=1, Medium=2, Long=3, VeryLong=4
s_f8[2,1,type=categorical]               ### Switch tendency: Stable=0, Switching=1

## Observations
o_m0[6,1,type=categorical]               ### Visual features: Ambiguous stimulus features
o_m1[3,1,type=categorical]               ### Depth cues: Cue_A=0, Neutral=1, Cue_B=2
o_m2[2,1,type=categorical]               ### Switch event: NoSwitch=0, Switch=1
o_m3[4,1,type=categorical]               ### Internal signals: Confidence, adaptation, fatigue indicators

## Actions
u_c0[2,1,type=categorical]               ### Perceptual report: Report_A=0, Report_B=1
u_c1[3,1,type=categorical]               ### Attention deployment: Maintain=0, Switch=1, Explore=2
u_c2[2,1,type=categorical]               ### Control effort: Passive=0, Active=1

## Connections
# Core perceptual inference
s_f0, s_f1 > o_m0                        ### Interpretation and confidence determine perceived features
s_f2 > s_f1                              ### Adaptation reduces confidence
s_f5 > s_f6                              ### Neural fatigue affects precision dynamics

# Switching dynamics
s_f6, s_f2 > s_f8                        ### Precision changes and adaptation trigger switching
s_f8, s_f7 > s_f0                        ### Switch tendency and dwell time determine new interpretation

# Attention and control
s_f3, u_c1 > s_f6                        ### Attention state and deployment affect precision
s_f4, u_c2 > s_f3                        ### Control strategy determines attention state

# Temporal adaptation
s_f0, s_f7 > s_f2                        ### Current interpretation and time cause adaptation
s_f0, s_f7 > s_f5                        ### Neural populations adapt over time

## InitialParameterization
# Baseline perception parameters
interpretation_bias = 0.5                ### No inherent bias toward either interpretation
baseline_confidence = 0.7                ### Moderate initial confidence
adaptation_rate = 0.05                   ### Rate of perceptual adaptation

# Switching dynamics
switch_threshold = 0.3                   ### Threshold for perceptual switching
precision_decay_rate = 0.02              ### Rate of precision decay due to adaptation
spontaneous_switch_rate = 0.01           ### Background switching probability

# Attention and control
attention_boost = 1.5                    ### Attention increases precision
voluntary_control_strength = 0.8         ### Effectiveness of voluntary control
fatigue_accumulation_rate = 0.03         ### Rate of neural fatigue buildup

# Temporal parameters
min_dwell_time = 3                       ### Minimum time before switching possible
max_dwell_time = 20                      ### Maximum stable perception duration
adaptation_time_constant = 10            ### Time scale for adaptation effects

# A matrices - interpretation-dependent observation likelihoods
A_m0_interpretation_A = [
    [0.8, 0.1, 0.05, 0.03, 0.015, 0.005],  # Features consistent with interpretation A
    [0.1, 0.6, 0.15, 0.1, 0.04, 0.01],
    [0.05, 0.15, 0.6, 0.15, 0.04, 0.01],
    [0.03, 0.1, 0.15, 0.6, 0.1, 0.02],
    [0.015, 0.04, 0.04, 0.1, 0.6, 0.205],
    [0.005, 0.01, 0.01, 0.02, 0.205, 0.75]
]

A_m0_interpretation_B = [
    [0.005, 0.01, 0.01, 0.02, 0.205, 0.75],  # Features consistent with interpretation B
    [0.015, 0.04, 0.04, 0.1, 0.6, 0.205],
    [0.03, 0.1, 0.15, 0.6, 0.1, 0.02],
    [0.05, 0.15, 0.6, 0.15, 0.04, 0.01],
    [0.1, 0.6, 0.15, 0.1, 0.04, 0.01],
    [0.8, 0.1, 0.05, 0.03, 0.015, 0.005]
]

# B matrices - switching transition probabilities
B_interpretation = [
    [0.95, 0.05],                        # Interpretation A -> stay or switch
    [0.05, 0.95]                         # Interpretation B -> stay or switch
]

# Adaptation-dependent switching probabilities
B_switching_adapted = [
    [0.7, 0.3],                          # Higher switching when adapted
    [0.3, 0.7]
]

B_switching_fresh = [
    [0.98, 0.02],                        # Lower switching when fresh
    [0.02, 0.98]
]

# C matrices - no strong preferences for either interpretation
C_m2 = [0.0, 0.0]                       ### Neutral preferences for switching

# D matrices - initial state priors
D_f0 = [0.5, 0.5]                       ### Equal prior for both interpretations
D_f1 = [0.2, 0.6, 0.2]                  ### Prior: medium confidence most likely
D_f2 = [0.7, 0.2, 0.08, 0.02]           ### Prior: start with no adaptation
D_f5 = [0.8, 0.15, 0.04, 0.01]          ### Prior: start fresh

## Equations
# Perceptual switching probability
switch_probability(t) = base_rate + adaptation_factor(t) + fatigue_factor(t) + noise(t)

# Adaptation buildup
adaptation(t+1) = adaptation(t) + adaptation_rate * perception_strength(t) - decay_rate

# Neural fatigue accumulation
fatigue(t+1) = fatigue(t) + fatigue_rate * neural_activity(t) - recovery_rate

# Precision modulation
precision(interpretation, t) = base_precision * (1 - adaptation(interpretation, t)) * attention_weight(t)

# Dwell time distribution (gamma-like)
dwell_time_probability(t) = gamma_pdf(t, shape=2, scale=mean_dwell_time/2)

# Confidence based on prediction error
confidence(t) = 1 / (1 + prediction_error_magnitude(t))

# Voluntary control effectiveness
control_effect(t) = control_strength * attention_level(t) * (1 - fatigue(t))

# Switching dynamics with hysteresis
switch_decision(t) = sigmoid(adaptation_pressure(t) - switch_threshold + noise(t))

## Time
Dynamic
DiscreteTime = t
ModelTimeHorizon = 200

## ActInfOntologyAnnotation
BistablePerception: competing_interpretations
PerceptualSwitching: adaptation_driven_transitions
NeuralAdaptation: precision_modulation
VoluntaryControl: attention_mediated_switching

## Footer
This model demonstrates bistable perception through Active Inference mechanisms.
Perceptual switching emerges from neural adaptation and precision dynamics.
Model captures key phenomena: dwell times, adaptation effects, voluntary control.

## Signature
Model: BistablePerceptionModel
Framework: Active Inference
Domain: Cognitive Phenomena - Perception
Created: 2025
Validated: Necker cube, Rubin's vase, binocular rivalry paradigms 