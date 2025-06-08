# GNNVersionAndFlags

GNN Version: 1.0
Processing Flags: ParseMath=True, ValidateTypes=True, GenerateCode=True, CreateDiagrams=True

# ModelName

Hierarchical Learning and Adaptation Model

# ModelAnnotation

This model implements hierarchical learning and adaptation across multiple timescales in Active Inference. Learning emerges from prediction error minimization and precision optimization at fast (perceptual), medium (associative), and slow (conceptual) temporal scales.

Key features:
- Multi-timescale learning from fast perceptual adaptation to slow conceptual change
- Skill acquisition with automatization and expertise development
- Curiosity-driven exploration and uncertainty reduction
- Metacognitive learning regulation and strategy selection
- Social learning through observation and imitation

The model captures how organisms adaptively learn from experience, balancing exploration and exploitation while optimizing their generative models across different hierarchical levels and temporal scales.

# StateSpaceBlock

### Hidden States
s_fast[20,1,type=categorical] ### Fast learning: perceptual features and motor patterns
s_medium[15,1,type=categorical] ### Medium learning: associative links and habits
s_slow[10,1,type=categorical] ### Slow learning: concepts and schemas
s_skill[5,1,type=categorical] ### Skill level: {0=novice, 1=beginner, 2=competent, 3=proficient, 4=expert}
s_mode[4,1,type=categorical] ### Learning mode: {0=passive, 1=active, 2=strategic, 3=metacognitive}
s_attention[6,1,type=categorical] ### Attention allocation across learning levels
s_curiosity[3,1,type=categorical] ### Curiosity state: {0=low, 1=moderate, 2=high}

### Observations
o_sensory[20,1,type=categorical] ### Environmental sensory input
o_feedback[5,1,type=categorical] ### Performance feedback and error signals
o_social[8,1,type=categorical] ### Social learning cues from others
o_context[6,1,type=categorical] ### Learning context and situational factors

### Actions
u_learn[6,1,type=categorical] ### Learning actions: attend, practice, explore, test, imitate, reflect
u_allocate[4,1,type=categorical] ### Resource allocation: focus distribution across levels

# Connections

### Hierarchical Learning Flow
s_fast > s_medium ### Fast learning feeds into associative learning
s_medium > s_slow ### Associative patterns contribute to concept formation
s_slow > s_fast ### Conceptual knowledge guides perceptual learning

### Skill Development
s_fast > s_skill ### Perceptual learning contributes to skill development
s_medium > s_skill ### Habit formation enhances skill automation
s_skill > s_mode ### Skill level influences learning mode selection

### Attention and Control
s_mode > s_attention ### Learning mode determines attention allocation
s_attention > s_fast ### Attention modulates fast learning
s_attention > s_medium ### Attention influences associative learning
s_attention > s_slow ### Attention affects conceptual processing

### Curiosity and Exploration
s_curiosity > u_learn ### Curiosity drives learning actions
s_curiosity > u_allocate ### Curiosity affects resource allocation
o_feedback > s_curiosity ### Unexpected feedback triggers curiosity

### Social Learning
o_social > s_fast ### Social cues influence perceptual learning
o_social > s_medium ### Social information affects associative learning
o_social > u_learn ### Social context influences learning actions

### Contextual Modulation
o_context > s_mode ### Context influences learning mode
o_context > s_attention ### Context shapes attention allocation

### Sensorimotor Learning
o_sensory > s_fast ### Sensory input drives fast learning
s_fast > u_learn ### Fast learning states influence learning actions
u_learn > o_feedback ### Learning actions generate feedback

# InitialParameterization

### Observation Model (A matrices)
A_sensory = eye(20) * 0.9 + 0.005 ### Noisy sensory observation
A_feedback = [[0.8, 0.1, 0.05, 0.03, 0.02], [0.05, 0.8, 0.1, 0.04, 0.01], [0.02, 0.05, 0.8, 0.1, 0.03], [0.01, 0.02, 0.05, 0.8, 0.12], [0.01, 0.01, 0.02, 0.06, 0.9]] ### Performance feedback likelihood
A_social = eye(8) * 0.85 + 0.02 ### Social observation model
A_context = eye(6) * 0.9 + 0.017 ### Context observation

### Transition Dynamics (B matrices)
B_fast = eye(20) * 0.7 + 0.015 ### Fast learning with moderate persistence
B_medium = eye(15) * 0.85 + 0.01 ### Medium learning with higher persistence
B_slow = eye(10) * 0.95 + 0.005 ### Slow learning with high persistence
B_skill = [[0.9, 0.08, 0.02, 0.0, 0.0], [0.1, 0.8, 0.08, 0.02, 0.0], [0.0, 0.1, 0.8, 0.08, 0.02], [0.0, 0.0, 0.1, 0.8, 0.1], [0.0, 0.0, 0.0, 0.05, 0.95]] ### Skill progression
B_mode = [[0.7, 0.2, 0.08, 0.02], [0.15, 0.7, 0.12, 0.03], [0.05, 0.15, 0.7, 0.1], [0.02, 0.08, 0.2, 0.7]] ### Learning mode transitions
B_attention = eye(6) * 0.8 + 0.033 ### Attention allocation dynamics
B_curiosity = [[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.1, 0.25, 0.65]] ### Curiosity state transitions

### Preferences (C matrices)
C_feedback = [3.0, 1.0, 0.0, -1.0, -2.0] ### Strong preference for positive feedback
C_sensory = zeros(20) ### No specific sensory preferences
C_social = [0.5] * 8 ### Mild preference for social learning opportunities
C_context = zeros(6) ### Context neutral

### Initial Beliefs (D vectors)
D_fast = normalize(ones(20)) ### Uniform initial fast learning
D_medium = normalize(ones(15)) ### Uniform initial associative states
D_slow = normalize(ones(10)) ### Uniform initial conceptual states
D_skill = [0.6, 0.25, 0.1, 0.04, 0.01] ### Start as novice with some variability
D_mode = [0.4, 0.35, 0.2, 0.05] ### Initially passive-active learning
D_attention = normalize(ones(6)) ### Distributed initial attention
D_curiosity = [0.5, 0.35, 0.15] ### Moderate initial curiosity

### Precision Parameters
γ_fast = 2.5 ### High precision for fast learning
γ_medium = 1.8 ### Moderate precision for associative learning
γ_slow = 1.2 ### Lower precision for conceptual flexibility
γ_skill = 2.0 ### Moderate precision for skill tracking
γ_mode = 1.5 ### Learning mode precision
γ_attention = 3.0 ### High attention precision
γ_curiosity = 1.8 ### Curiosity precision
α = 12.0 ### Action precision

# Equations

### Hierarchical Learning Update
\\[ \\Delta s^{level}_{t+1} = \\eta_{level} \\cdot \\gamma_{level} \\cdot \\epsilon^{level}_t \\cdot f(s^{attention}_t) \\]

where \\( \\eta_{level} \\) is the learning rate, \\( \\gamma_{level} \\) is precision, and \\( \\epsilon^{level}_t \\) is prediction error.

### Multi-timescale Learning Rates
\\[ \\eta_{fast} = 0.3, \\quad \\eta_{medium} = 0.1, \\quad \\eta_{slow} = 0.02 \\]

### Skill Acquisition Dynamics
\\[ P(s^{skill}_{t+1}|s^{skill}_t) = \\text{Cat}(\\text{softmax}(\\beta \\cdot (\\text{practice}_t + \\text{feedback}_t))) \\]

### Curiosity-Driven Exploration
\\[ P(u^{learn}_t|s^{curiosity}_t) = \\text{Cat}(\\text{softmax}(\\gamma_{curiosity} \\cdot (I(s) + \\mathbb{E}[G]))) \\]

where \\( I(s) \\) is state information and \\( \\mathbb{E}[G] \\) is expected free energy.

### Attention Allocation
\\[ P(s^{attention}_t|s^{mode}_t, s^{curiosity}_t) = \\text{Cat}(\\text{softmax}(\\mathbf{W}_{mode} s^{mode}_t + \\mathbf{W}_{curiosity} s^{curiosity}_t)) \\]

### Learning Mode Selection
\\[ P(s^{mode}_t|s^{skill}_t, o^{context}_t) = \\text{Cat}(\\text{softmax}(\\mathbf{W}_{skill} s^{skill}_t + \\mathbf{W}_{context} o^{context}_t)) \\]

### Social Learning Integration
\\[ s^{social}_{t+1} = s^{social}_t + \\eta_{social} \\cdot (o^{social}_t - s^{social}_t) \\cdot \\text{credibility}(o^{social}_t) \\]

### Forgetting and Consolidation
\\[ s^{level}_{t+1} = s^{level}_t \\cdot (1 - \\delta_{level}) + \\text{update}^{level}_t \\]

where \\( \\delta_{fast} = 0.1, \\delta_{medium} = 0.02, \\delta_{slow} = 0.005 \\).

# Time

Dynamic: True
DiscreteTime: True
ModelTimeHorizon: 500

# ActInfOntologyAnnotation

- Fast Learning (s_fast): Maps to "Perceptual Learning" and "Motor Adaptation" in Active Inference Ontology
- Medium Learning (s_medium): Corresponds to "Associative Learning" and "Habit Formation" concepts
- Slow Learning (s_slow): Maps to "Conceptual Learning" and "Schema Formation" mechanisms
- Skill Development (s_skill): Relates to "Expertise Acquisition" and "Automatization" processes
- Learning Modes (s_mode): Corresponds to "Metacognitive Control" and "Learning Strategy Selection"
- Curiosity (s_curiosity): Maps to "Information Seeking" and "Exploration Drive" concepts
- Hierarchical Learning: Implemented through multi-timescale precision optimization and belief updating

# Footer

Model Type: Hierarchical Learning and Adaptation
Domain: Cognitive Science, Educational Psychology, Developmental Neuroscience
Compatible Backends: PyMDP, RxInfer.jl
Complexity: High (7 hidden states, 4 observations, 2 actions)

# Signature

Generated by: Active Inference Institute - GNN Learning Sciences Working Group
Date: 2025-01-27
Version: 1.0
Contact: info@activeinference.org
License: Creative Commons Attribution 4.0 International 