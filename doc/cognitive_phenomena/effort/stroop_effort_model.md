# Stroop Task Effort Model

## GNNVersionAndFlags
GNN_v1.0
ProcessingFlags: cognitive_effort, stroop_task, mental_action, inhibitory_control

## ModelName
StroopTaskEffortModel

## ModelAnnotation
Models cognitive effort in the Stroop task through the divergence between habitual reading responses 
and goal-directed color naming. Effort emerges from overcoming mental habits when they conflict with task demands.
Based on Parr, Holmes, Friston, & Pezzulo (2023) Active Inference formulation of cognitive effort.

## StateSpaceBlock
# Slow level - narrative and instruction context
s_slow_f0[2,1,type=categorical]          ### Narrative state: Instruction=0, Response=1
s_slow_f1[2,1,type=categorical]          ### Instruction type: Read_Word=0, Name_Color=1
s_slow_f2[2,1,type=categorical]          ### Response modality: Word_Reading=0, Color_Naming=1

# Fast level - trial-specific states
s_fast_f0[4,1,type=categorical]          ### Font color: Red=0, Blue=1, Green=2, Yellow=3
s_fast_f1[4,1,type=categorical]          ### Written word: Red=0, Blue=1, Green=2, Yellow=3
s_fast_f2[3,1,type=categorical]          ### Task sequence: Instruction=0, Viewing=1, Response=2
s_fast_f3[2,1,type=categorical]          ### Correctness: Incorrect=0, Correct=1

# Effort and control states
s_effort_f0[3,1,type=categorical]        ### Effort level: Low=0, Medium=1, High=2
s_effort_f1[4,1,type=categorical]        ### Control mode: Habitual=0, Controlled=1, Conflict=2, Override=3
s_effort_f2[3,1,type=categorical]        ### Cognitive demand: Low=0, Medium=1, High=2

## Observations
o_m0[2,1,type=categorical]               ### Instruction: Read=0, Color=1
o_m1[4,1,type=categorical]               ### Visual word: Red=0, Blue=1, Green=2, Yellow=3
o_m2[4,1,type=categorical]               ### Visual color: Red=0, Blue=1, Green=2, Yellow=3
o_m3[4,1,type=categorical]               ### Verbal response: Red=0, Blue=1, Green=2, Yellow=3

## Actions
u_c0[2,1,type=categorical]               ### Mental action - response modality: Word=0, Color=1
u_c1[4,1,type=categorical]               ### Overt response: Red=0, Blue=1, Green=2, Yellow=3
u_c2[2,1,type=categorical]               ### Effort deployment: Deploy=0, Conserve=1

## Connections
# Hierarchical structure
s_slow_f0 > s_fast_f2                    ### Narrative determines task sequence
s_slow_f1 > s_fast_f3                    ### Instruction determines correctness evaluation
s_slow_f2 > u_c1                        ### Response modality influences overt response

# Stimulus processing
s_fast_f0 > o_m2                        ### Font color generates color observation
s_fast_f1 > o_m1                        ### Written word generates word observation

# Effort and control
s_slow_f1, s_slow_f2 > s_effort_f2       ### Instruction-modality mismatch creates demand
s_effort_f2 > s_effort_f0                ### Cognitive demand determines effort level
s_effort_f1, u_c0 > s_slow_f2            ### Control mode and mental action determine modality

# Response generation
s_fast_f0, s_fast_f1, s_slow_f2 > u_c1   ### Stimuli and modality determine response
s_fast_f3 > o_m3                        ### Correctness observable through somatic marker

## InitialParameterization
# Habitual biases - strong prior for word reading
habitual_word_reading = 0.85             ### 85% prior probability for word reading
habitual_color_naming = 0.15             ### 15% prior probability for color naming

# Effort parameters
cognitive_demand_word = -0.5             ### Low demand for word reading (habitual)
cognitive_demand_color = 1.5             ### High demand for color naming (effortful)
motivation_strength = 1.0                ### Preference for correct responses

# Precision parameters
precision_habitual = 2.0                 ### High precision for habitual responses
precision_controlled = 4.0               ### Higher precision for controlled responses
precision_conflict = 1.0                 ### Reduced precision during conflict

# Learning rates
effort_learning_rate = 0.05              ### Rate of effort adaptation
habit_strength_decay = 0.01              ### Slow decay of habitual strength

# A matrices - observation likelihoods
A_m1_word_modality = [
    [0.95, 0.02, 0.02, 0.01],           # High precision word reading
    [0.02, 0.95, 0.02, 0.01],
    [0.02, 0.02, 0.95, 0.01],
    [0.01, 0.02, 0.02, 0.95]
]

A_m2_color_modality = [
    [0.90, 0.03, 0.04, 0.03],           # Lower precision color naming
    [0.03, 0.90, 0.04, 0.03],
    [0.04, 0.03, 0.90, 0.03],
    [0.03, 0.04, 0.03, 0.90]
]

# B matrices - transition dynamics for effort states
B_effort_level = [
    [0.7, 0.2, 0.1],                    # Low effort tends to persist
    [0.3, 0.5, 0.2],                    # Medium effort is transitional
    [0.1, 0.3, 0.6]                     # High effort is costly to maintain
]

# C matrices - preferences for correctness
C_m3_correctness = [-2.0, 2.0]          ### Strong preference for correct responses

# D matrices - initial state priors
D_slow_f2 = [0.85, 0.15]                ### Strong prior for word reading modality
D_effort_f0 = [0.6, 0.3, 0.1]           ### Prior: likely to start with low effort

# E matrices - habitual policy priors
E_mental_action = [1.5, -1.5]           ### Strong bias toward word reading mental action

## Equations
# Cognitive effort calculation (KL divergence)
cognitive_effort(t) = gamma * KL_divergence(posterior_policy(t), habitual_prior)

# Posterior policy beliefs
posterior_policy(t) = softmax(gamma * (-expected_free_energy(t) + habitual_prior))

# Expected free energy components
expected_free_energy(t) = expected_cost(t) + epistemic_value(t)

# Cognitive demand based on instruction-habit mismatch
cognitive_demand(t) = mismatch_strength * |instruction(t) - habitual_preference|

# Effort deployment decision
effort_deployment(t) = sigmoid(motivation * expected_benefit(t) - effort_cost(t))

# Habit strength modulation
habit_strength(t+1) = habit_strength(t) * (1 - decay_rate) + reinforcement(t)

# Performance prediction based on effort
performance(t) = base_accuracy + effort_benefit * effort_deployed(t) - interference(t)

# Reaction time model based on confidence
reaction_time(t) = base_RT + RT_scaling / confidence(t)

# Confidence based on precision-weighted beliefs
confidence(t) = 1 / entropy(posterior_beliefs(t))

## Time
Dynamic
DiscreteTime = t
ModelTimeHorizon = 100

## ActInfOntologyAnnotation
CognitiveEffort: divergence_from_habits
MentalAction: covert_cognitive_operation
InhibitoryControl: habit_override
ConflictMonitoring: demand_assessment
ExecutiveControl: effort_deployment

## Footer
This model demonstrates cognitive effort through Active Inference mechanisms in the Stroop task.
Effort emerges from the information-theoretic cost of overcoming habitual mental actions.
Model captures key phenomena: Stroop interference, individual differences, effort-performance trade-offs.

## Signature
Model: StroopTaskEffortModel
Framework: Active Inference
Domain: Cognitive Phenomena - Effort
Created: 2025
Validated: Stroop task, flanker task, inhibitory control paradigms
Reference: Parr, Holmes, Friston, & Pezzulo (2023) 