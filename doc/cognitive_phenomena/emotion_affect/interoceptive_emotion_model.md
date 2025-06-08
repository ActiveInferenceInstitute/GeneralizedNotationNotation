# GNNVersionAndFlags

GNN Version: 1.0
Processing Flags: ParseMath=True, ValidateTypes=True, GenerateCode=True, CreateDiagrams=True

# ModelName

Interoceptive Emotion and Affect Model

# ModelAnnotation

This model implements emotion and affect through interoceptive inference, allostatic regulation, and precision-weighted prediction error minimization. Emotions emerge from predictions about bodily states, cognitive appraisals, and their homeostatic implications for survival and wellbeing.

Key features:
- Interoceptive prediction and physiological homeostasis (allostasis)
- Cognitive appraisal processes generating distinct emotions
- Emotion regulation strategies with varying effectiveness and costs
- Cultural and developmental modulation of emotional responses
- Clinical applications for mood and anxiety disorders

The model captures how emotions serve as policies for maintaining adaptive responses to environmental challenges while optimizing long-term biological and psychological flourishing.

# StateSpaceBlock

### Hidden States
s_physio[8,1,type=continuous] ### Physiological states: HR, BP, GSR, temp, cortisol, breathing, tension, autonomic
s_intero[6,1,type=categorical] ### Interoceptive signals: {0=hunger, 1=thirst, 2=pain, 3=fatigue, 4=arousal, 5=visceral}
s_appraisal[6,1,type=categorical] ### Cognitive appraisal: {0=threat, 1=challenge, 2=benefit, 3=loss, 4=irrelevant, 5=novel}
s_emotion[8,1,type=categorical] ### Basic emotions: {0=joy, 1=sadness, 2=anger, 3=fear, 4=disgust, 5=surprise, 6=interest, 7=shame}
s_intensity[5,1,type=categorical] ### Emotion intensity: {0=very_low, 1=low, 2=medium, 3=high, 4=very_high}
s_regulation[6,1,type=categorical] ### Regulation strategy: {0=reappraisal, 1=suppression, 2=distraction, 3=acceptance, 4=problem_solving, 5=social_support}
s_arousal[4,1,type=categorical] ### Arousal level: {0=low, 1=medium, 2=high, 3=very_high}
s_valence[3,1,type=categorical] ### Emotional valence: {0=negative, 1=neutral, 2=positive}

### Observations
o_physio[8,1,type=continuous] ### Physiological measurements from body sensors
o_environment[10,1,type=categorical] ### Environmental stimuli: threats, rewards, social cues, novelty
o_context[6,1,type=categorical] ### Situational context: social, achievement, relationship, survival, safety, stress
o_social[4,1,type=categorical] ### Social emotional cues: support, threat, acceptance, rejection

### Actions
u_expression[8,1,type=categorical] ### Emotional expression: matches s_emotion categories
u_regulation[6,1,type=categorical] ### Regulation choice: matches s_regulation strategies
u_behavior[5,1,type=categorical] ### Behavioral response: approach, avoid, freeze, fight, seek_support

# Connections

### Bottom-up Interoceptive Processing
o_physio > s_physio ### Physiological sensors update internal states
s_physio > s_intero ### Physiological states generate interoceptive signals
s_intero > s_arousal ### Interoceptive signals influence arousal
s_intero > s_emotion ### Interoceptive signals contribute to emotions

### Cognitive Appraisal Process
o_environment > s_appraisal ### Environmental stimuli trigger appraisals
o_context > s_appraisal ### Context influences appraisal patterns
s_appraisal > s_emotion ### Appraisals generate specific emotions
s_appraisal > s_valence ### Appraisals determine emotional valence

### Emotion Generation and Modulation
s_arousal > s_intensity ### Arousal level affects emotion intensity
s_valence > s_emotion ### Valence constrains emotion type
s_emotion > s_intensity ### Emotion type influences intensity range

### Emotion Regulation
s_emotion > s_regulation ### Current emotion triggers regulation choice
s_intensity > s_regulation ### High intensity increases regulation likelihood
s_regulation > s_emotion ### Regulation modulates emotional experience
s_regulation > s_intensity ### Regulation reduces emotion intensity
s_regulation > s_physio ### Regulation affects physiological arousal

### Social and Environmental Feedback
o_social > s_appraisal ### Social cues influence appraisals
o_social > s_regulation ### Social context affects regulation strategy choice
s_emotion > u_expression ### Emotions drive expressive behaviors
u_expression > o_social ### Expressions influence social environment

### Behavioral Output
s_emotion > u_behavior ### Emotions motivate specific behaviors
s_appraisal > u_behavior ### Appraisals influence behavioral choices
s_regulation > u_behavior ### Regulation strategies affect behavior

### Homeostatic Regulation
s_physio > s_physio ### Physiological self-regulation dynamics
u_regulation > s_physio ### Regulation actions influence physiology

# InitialParameterization

### Observation Model (A matrices)
A_physio = eye(8) * 0.9 + 0.0125 ### Physiological observation with moderate noise
A_environment = eye(10) * 0.85 + 0.015 ### Environmental observation with uncertainty
A_context = eye(6) * 0.8 + 0.033 ### Context observation with ambiguity
A_social = eye(4) * 0.75 + 0.083 ### Social cue observation with interpretation variability

### Transition Dynamics (B matrices)
B_physio = eye(8) * 0.85 + 0.019 ### Physiological dynamics with homeostatic drift
B_intero = eye(6) * 0.7 + 0.05 ### Interoceptive signals with fast dynamics
B_appraisal = eye(6) * 0.6 + 0.067 ### Appraisal with moderate persistence
B_emotion = eye(8) * 0.75 + 0.031 ### Emotion persistence with decay
B_intensity = [[0.6, 0.3, 0.08, 0.02, 0.0], [0.2, 0.5, 0.2, 0.08, 0.02], [0.05, 0.2, 0.5, 0.2, 0.05], [0.02, 0.08, 0.2, 0.5, 0.2], [0.0, 0.02, 0.08, 0.3, 0.6]] ### Intensity regulation
B_regulation = eye(6) * 0.8 + 0.033 ### Regulation strategy persistence
B_arousal = [[0.7, 0.2, 0.08, 0.02], [0.25, 0.5, 0.2, 0.05], [0.1, 0.3, 0.5, 0.1], [0.05, 0.15, 0.3, 0.5]] ### Arousal transitions
B_valence = [[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]] ### Valence persistence

### Preferences (C matrices)
C_physio = [0.0, 0.0, -0.5, 0.0, -1.0, 0.0, -0.3, 0.5] ### Prefer homeostatic balance, avoid stress
C_environment = [−2.0, 2.0, 1.0, −1.5, 0.0, 0.5, 1.5, 0.0, 0.5, 0.0] ### Avoid threats, approach rewards
C_context = [0.5, 1.0, 1.5, 0.0, 1.0, -1.0] ### Prefer positive social/achievement contexts
C_social = [2.0, -2.0, 1.5, -1.5] ### Strong preference for support and acceptance

### Initial Beliefs (D vectors)
D_physio = [70, 120, 0.5, 37, 0.1, 0.3, 0.2, 0.5] ### Homeostatic set points (normalized)
D_intero = [0.4, 0.3, 0.1, 0.1, 0.05, 0.05] ### Initially low interoceptive activation
D_appraisal = [0.1, 0.2, 0.3, 0.1, 0.25, 0.05] ### Neutral to positive appraisal bias
D_emotion = [0.15, 0.05, 0.05, 0.1, 0.05, 0.1, 0.4, 0.1] ### Initially neutral with some interest
D_intensity = [0.4, 0.35, 0.2, 0.04, 0.01] ### Initially low intensity
D_regulation = [0.2, 0.1, 0.2, 0.3, 0.15, 0.05] ### Preference for acceptance and reappraisal
D_arousal = [0.5, 0.35, 0.12, 0.03] ### Initially low arousal
D_valence = [0.2, 0.6, 0.2] ### Initially neutral valence

### Precision Parameters
γ_physio = 2.5 ### High precision for physiological states
γ_intero = 1.5 ### Moderate interoceptive precision
γ_appraisal = 1.8 ### High appraisal precision
γ_emotion = 2.0 ### High emotion precision
γ_intensity = 1.6 ### Moderate intensity precision
γ_regulation = 1.4 ### Moderate regulation precision
γ_arousal = 2.2 ### High arousal precision
γ_valence = 2.0 ### High valence precision
α = 14.0 ### Action precision for emotional expression and regulation

# Equations

### Interoceptive Prediction Error
\\[ \\epsilon^{intero}_t = \\gamma_{intero} \\cdot (o^{physio}_t - \\mu^{physio}_t) \\]

### Allostatic Regulation
\\[ \\Delta s^{physio}_{t+1} = -\\eta_{allostatic} \\cdot (s^{physio}_t - \\text{homeostatic\\_targets}) + \\text{regulation\\_input}_t \\]

### Emotion Generation from Appraisal
\\[ P(s^{emotion}_t|s^{appraisal}_t, s^{arousal}_t, s^{valence}_t) = \\text{Cat}(\\text{softmax}(\\mathbf{W}_{appraisal} s^{appraisal}_t + \\mathbf{W}_{arousal} s^{arousal}_t + \\mathbf{W}_{valence} s^{valence}_t)) \\]

### Emotion Regulation Effectiveness
\\[ s^{emotion}_{regulated} = s^{emotion}_t \\cdot (1 - \\eta_{regulation} \\cdot \\text{effectiveness}(s^{regulation}_t)) \\]

where effectiveness varies by strategy:
\\[ \\text{effectiveness}_{reappraisal} = 0.8, \\quad \\text{effectiveness}_{suppression} = 0.4 \\]

### Arousal-Valence Dynamics
\\[ s^{arousal}_t = f(||\\epsilon^{intero}_t||, s^{appraisal}_t) \\]
\\[ s^{valence}_t = g(s^{appraisal}_t, o^{environment}_t) \\]

### Social Emotion Regulation
\\[ P(s^{regulation}_t|o^{social}_t, s^{emotion}_t) = \\text{Cat}(\\text{softmax}(\\gamma_{social} \\cdot \\mathbf{W}_{social} o^{social}_t + \\mathbf{W}_{emotion} s^{emotion}_t)) \\]

### Emotional Expression
\\[ P(u^{expression}_t|s^{emotion}_t, s^{intensity}_t, o^{social}_t) = \\text{Cat}(\\text{softmax}(\\alpha \\cdot \\mathbf{W}_{expr} [s^{emotion}_t; s^{intensity}_t; \\text{display\\_rules}(o^{social}_t)])) \\]

### Mood Integration
\\[ \\text{mood}_t = \\text{mood}_{t-1} + \\eta_{mood} \\cdot (\\text{valence}_t \\cdot \\text{intensity}_t - \\text{mood}_{t-1}) \\]

### Cultural Display Rules
\\[ \\text{display\\_rules}(context) = \\text{culture\\_matrix} \\cdot \\text{context\\_vector} \\]

# Time

Dynamic: True
DiscreteTime: True
ModelTimeHorizon: 200

# ActInfOntologyAnnotation

- Physiological States (s_physio): Maps to "Interoceptive Signals" and "Allostatic Regulation" in Active Inference Ontology
- Interoceptive Processing (s_intero): Corresponds to "Bodily Awareness" and "Homeostatic Monitoring" concepts
- Cognitive Appraisal (s_appraisal): Maps to "Situational Evaluation" and "Threat Assessment" mechanisms
- Emotion Generation (s_emotion): Relates to "Affective States" and "Adaptive Responses" processes
- Emotion Regulation (s_regulation): Corresponds to "Affective Control" and "Coping Strategies" concepts
- Arousal and Valence: Map to "Dimensional Emotion" and "Core Affect" frameworks
- Emotional Expression: Implemented through action policies and social communication

# Footer

Model Type: Emotion and Affect through Interoceptive Inference
Domain: Affective Neuroscience, Clinical Psychology, Computational Psychiatry
Compatible Backends: PyMDP, RxInfer.jl
Complexity: Very High (8 hidden states, 4 observations, 3 actions)

# Signature

Generated by: Active Inference Institute - GNN Affective Sciences Working Group
Date: 2025-01-27
Version: 1.0
Contact: info@activeinference.org
License: Creative Commons Attribution 4.0 International 