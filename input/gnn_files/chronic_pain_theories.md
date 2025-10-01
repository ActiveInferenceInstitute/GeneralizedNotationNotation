# GNN Example: Chronic Pain Multi-Theory Model with Three Nested Continuous Timescales
# GNN Version: 1.0
# This file specifies a comprehensive Active Inference model integrating multiple theories of chronic pain across three nested continuous timescales (fast: ms-s, medium: min-hrs, slow: hrs-days). The model includes peripheral sensitization, central sensitization, gate control, neuromatrix, and predictive coding theories, enabling testing of competing biological hypotheses about pain chronification with timescale-specific dynamics and interventions.

## GNNSection
ChronicPainTheories

## GNNVersionAndFlags
GNN v1

## ModelName
Active Inference Chronic Pain Multi-Theory Model v1

## ModelAnnotation
This model integrates multiple coherent theories of chronic pain mechanisms across THREE NESTED CONTINUOUS TIMESCALES:

**Multi-Theory Integration:**
- Peripheral Sensitization: Enhanced nociceptor responsiveness and reduced thresholds (slow timescale)
- Central Sensitization: Amplified CNS processing and reduced inhibition (slow timescale, one-way process)
- Gate Control Theory: Spinal modulation of ascending pain signals (fast timescale)
- Neuromatrix Theory: Distributed network generating pain experience (fast-medium coupling)
- Predictive Coding: Pain as precision-weighted prediction error (all timescales)
- Biopsychosocial Integration: Cognitive, emotional, and behavioral factors (medium timescale)

**Three Nested Timescales:**
1. Fast (ms-s): Neural signaling, gate control, descending modulation, acute pain perception
2. Medium (min-hrs): Cognitive-affective processes, behavioral strategies, functional capacity
3. Slow (hrs-days): Tissue healing, peripheral/central sensitization, chronic adaptations

**State Space Structure:**
- Six hidden state factors (378 combinations): tissue state (slow), peripheral sensitivity (slow), spinal gate (fast), central sensitization (slow), descending modulation (fast), cognitive-affective state (medium)
- Four observation modalities (72 outcomes): pain intensity (fast), pain quality (fast), functional capacity (medium), autonomic response (fast)
- Four control factors (81 actions): attention allocation (medium), behavioral strategy (medium), cognitive reappraisal (medium), descending control (fast)

**Key Features:**
- Timescale separation: ε (fast/medium) ≈ 10^-3, δ (medium/slow) ≈ 10^-2
- Cross-timescale coupling: slow states modulate fast dynamics, fast observations (averaged) drive medium cognition, medium behaviors (averaged) influence slow healing
- Testable predictions about pain chronification pathways across multiple timescales
- Intervention targets at each timescale: fast (descending control), medium (CBT/behavioral), slow (prevent sensitization)

## StateSpaceBlock
# Likelihood matrix: A[observation_outcomes, hidden_states]
A[72,378,type=float]   # 72 observations x 378 hidden state combinations

# Transition matrices: B[states_next, states_previous, actions]
B[378,378,81,type=float]   # State transitions given previous state and action

# Preference vector: C[observation_outcomes]
C[72,type=float]       # Log-preferences over pain-related observations

# Prior vector: D[states]
D[378,type=float]       # Prior over initial hidden states (acute vs chronic pain)

# Habit vector: E[actions]
E[81,type=float]       # Initial policy prior over pain coping strategies

# Hidden States (6 factors: 3×3×3×2×3×7 = 378 total combinations)
T[3,1,type=float]      # Tissue state (3 levels: healed, inflamed, damaged)
P_sens[3,1,type=float] # Peripheral sensitization (3 levels: normal, moderate, severe)
G[3,1,type=float]      # Spinal gate state (3 levels: open, modulated, closed)
C_sens[2,1,type=float] # Central sensitization (2 levels: absent, present)
D_mod[3,1,type=float]  # Descending modulation (3 levels: facilitation, neutral, inhibition)
Cog[7,1,type=float]    # Cognitive-affective state (7 levels: adaptive, vigilant, fearful, catastrophizing, depressed, anxious, alexithymic)

# Observations (4 modalities: 4×3×3×2 = 72 total outcomes)
Pain_I[4,1,type=float]    # Pain intensity (4 levels: none, mild, moderate, severe)
Pain_Q[3,1,type=float]    # Pain quality (3 levels: nociceptive, neuropathic, nociplastic)
Func[3,1,type=float]      # Functional capacity (3 levels: full, limited, disabled)
Auto[2,1,type=float]      # Autonomic response (2 levels: normal, hyperarousal)

# Policy and Control (4 factors: 3×3×3×3 = 81 total actions)
Attn[3,1,type=float]      # Attention allocation (3 actions: distraction, monitoring, catastrophizing)
Behav[3,1,type=float]     # Behavioral strategy (3 actions: avoidance, pacing, engagement)
Reapp[3,1,type=float]     # Cognitive reappraisal (3 actions: negative, neutral, positive)
Desc_C[3,1,type=float]    # Descending control (3 actions: low, moderate, high endogenous analgesia)

# Free Energy terms
F[T,type=float]           # Variational Free Energy for state inference
G[Attn,type=float]        # Expected Free Energy (per policy)

# Time - Three Nested Continuous Timescales
t_fast[1,type=float]     # Fast timescale (milliseconds to seconds: neural responses, acute signaling)
t_medium[1,type=float]   # Medium timescale (minutes to hours: cognitive-affective, behavioral adaptation)
t_slow[1,type=float]     # Slow timescale (hours to days: tissue healing, sensitization, chronic changes)

## Connections
# State evolution connections (theories of pain chronification)
D>T                       # Prior influences initial tissue state
D>P_sens                  # Prior influences baseline peripheral sensitivity
D>C_sens                  # Prior influences susceptibility to central sensitization
D>Cog                     # Prior influences initial cognitive-affective state

# Peripheral sensitization pathway
T>P_sens                  # Tissue damage drives peripheral sensitization
P_sens>B                  # Peripheral sensitivity affects state transitions

# Gate control mechanism
P_sens>G                  # Peripheral input modulates spinal gate
D_mod>G                   # Descending signals modulate spinal gate
G>B                       # Gate state affects pain signal transmission

# Central sensitization pathway
G>C_sens                  # Prolonged spinal transmission leads to central sensitization
C_sens>B                  # Central sensitization amplifies all subsequent signals
T>C_sens                  # Tissue state indirectly affects central sensitization

# Descending modulation (neuromatrix)
Cog>D_mod                 # Cognitive-affective state influences descending control
D_mod>B                   # Descending modulation affects pain processing

# Cognitive-affective influences (biopsychosocial)
Pain_I>Cog                # Pain intensity influences cognitive-affective state (via observation)
Func>Cog                  # Functional limitations influence emotional state
Cog>B                     # Cognitive state affects all pain processing

# Observation generation (what the organism senses)
T>A                       # Tissue state contributes to pain observations
P_sens>A                  # Peripheral sensitivity affects pain intensity
G>A                       # Gate state modulates pain intensity
C_sens>A                  # Central sensitization amplifies pain intensity and changes quality
D_mod>A                   # Descending modulation affects perceived pain
Cog>A                     # Cognitive-affective state influences pain perception and function

# Control connections (Active Inference policy)
Attn>Cog                  # Attention allocation affects cognitive state
Behav>T                   # Behavioral strategy affects tissue healing/damage
Behav>Func                # Behavioral strategy directly affects function
Reapp>Cog                 # Cognitive reappraisal modulates cognitive-affective state
Desc_C>D_mod              # Volitional control of descending modulation

# Free energy and policy
C>G                       # Preferences (desired states) influence expected free energy
E>Attn                    # Habits influence attention allocation
E>Behav                   # Habits influence behavioral strategies
G>Attn                    # Expected free energy guides attention
G>Behav                   # Expected free energy guides behavior
G>Reapp                   # Expected free energy guides reappraisal
G>Desc_C                  # Expected free energy guides descending control

# Action coordination
Attn>Behav                # Attention influences behavior
Behav>Reapp               # Behavior influences reappraisal capacity
Reapp>Desc_C              # Reappraisal enhances descending control

## InitialParameterization
# A: 72 observations x 378 hidden states. Likelihood mapping from hidden pain states to observations.
# Observations ordered: Pain_I1-4, Pain_Q1-3, Func1-3, Auto1-2
# Hidden states ordered by: T, P_sens, G, C_sens, D_mod, Cog
A={
  # Theory-specific predictions:
  
  # Peripheral Sensitization: High P_sens → high pain even with minimal tissue damage
  # Example: T=healed (T1), P_sens=severe (P_sens3), G=open (G1), C_sens=absent (C_sens1), D_mod=neutral (D_mod2), Cog=adaptive (Cog1)
  # → Moderate pain, nociceptive quality, limited function, normal autonomic
  
  # Central Sensitization: C_sens=present → severe pain, nociplastic quality regardless of tissue state
  # Example: T=healed (T1), P_sens=normal (P_sens1), G=modulated (G2), C_sens=present (C_sens2), D_mod=neutral (D_mod2), Cog=adaptive (Cog1)
  # → Severe pain, nociplastic quality, disabled function, hyperarousal
  
  # Gate Control: Closed gate (G3) → reduced pain despite tissue damage
  # Example: T=damaged (T3), P_sens=moderate (P_sens2), G=closed (G3), C_sens=absent (C_sens1), D_mod=inhibition (D_mod3), Cog=adaptive (Cog1)
  # → Mild pain, nociceptive quality, limited function, normal autonomic
  
  # Cognitive-Affective: Catastrophizing (Cog4) amplifies pain perception
  # Example: T=inflamed (T2), P_sens=moderate (P_sens2), G=open (G1), C_sens=absent (C_sens1), D_mod=neutral (D_mod2), Cog=catastrophizing (Cog4)
  # → Severe pain, nociplastic quality, disabled function, hyperarousal
  
  # Descending Facilitation: D_mod=facilitation (D_mod1) increases pain
  # Example: T=healed (T1), P_sens=normal (P_sens1), G=modulated (G2), C_sens=absent (C_sens1), D_mod=facilitation (D_mod1), Cog=fearful (Cog3)
  # → Moderate pain, neuropathic quality, limited function, hyperarousal
  
  # First 50 of 378 state combinations with theory-based likelihoods:
  (0.70, 0.25, 0.05, 0.00, 0.80, 0.15, 0.05, 0.90, 0.08, 0.02, 0.85, 0.15),  # T1,P_sens1,G1,C_sens1,D_mod1,Cog1 - healed, normal sensitivity, adaptive
  (0.60, 0.30, 0.10, 0.00, 0.75, 0.20, 0.05, 0.85, 0.10, 0.05, 0.80, 0.20),  # T1,P_sens1,G1,C_sens1,D_mod1,Cog2 - healed, vigilant
  (0.45, 0.35, 0.15, 0.05, 0.70, 0.20, 0.10, 0.75, 0.15, 0.10, 0.70, 0.30),  # T1,P_sens1,G1,C_sens1,D_mod1,Cog3 - healed, fearful
  (0.20, 0.30, 0.35, 0.15, 0.50, 0.30, 0.20, 0.50, 0.30, 0.20, 0.40, 0.60),  # T1,P_sens1,G1,C_sens1,D_mod1,Cog4 - healed, catastrophizing (amplified pain)
  (0.30, 0.40, 0.25, 0.05, 0.60, 0.25, 0.15, 0.60, 0.25, 0.15, 0.50, 0.50),  # T1,P_sens1,G1,C_sens1,D_mod1,Cog5 - healed, depressed
  (0.40, 0.35, 0.20, 0.05, 0.65, 0.25, 0.10, 0.70, 0.20, 0.10, 0.60, 0.40),  # T1,P_sens1,G1,C_sens1,D_mod1,Cog6 - healed, anxious
  (0.55, 0.30, 0.12, 0.03, 0.70, 0.22, 0.08, 0.80, 0.15, 0.05, 0.75, 0.25),  # T1,P_sens1,G1,C_sens1,D_mod1,Cog7 - healed, alexithymic
  
  (0.80, 0.15, 0.05, 0.00, 0.85, 0.12, 0.03, 0.95, 0.04, 0.01, 0.90, 0.10),  # T1,P_sens1,G1,C_sens1,D_mod2,Cog1 - healed, neutral descending mod
  (0.75, 0.20, 0.05, 0.00, 0.80, 0.15, 0.05, 0.90, 0.08, 0.02, 0.85, 0.15),  # T1,P_sens1,G1,C_sens1,D_mod2,Cog2
  (0.65, 0.25, 0.08, 0.02, 0.75, 0.18, 0.07, 0.85, 0.12, 0.03, 0.80, 0.20),  # T1,P_sens1,G1,C_sens1,D_mod2,Cog3
  
  (0.90, 0.08, 0.02, 0.00, 0.90, 0.08, 0.02, 0.98, 0.02, 0.00, 0.95, 0.05),  # T1,P_sens1,G1,C_sens1,D_mod3,Cog1 - healed, strong inhibition (gate closed)
  (0.85, 0.12, 0.03, 0.00, 0.88, 0.10, 0.02, 0.95, 0.04, 0.01, 0.92, 0.08),  # T1,P_sens1,G1,C_sens1,D_mod3,Cog2
  
  (0.05, 0.15, 0.45, 0.35, 0.30, 0.40, 0.30, 0.20, 0.40, 0.40, 0.25, 0.75),  # T1,P_sens1,G1,C_sens2,D_mod1,Cog1 - healed but CENTRAL SENSITIZATION
  (0.02, 0.10, 0.40, 0.48, 0.20, 0.35, 0.45, 0.10, 0.35, 0.55, 0.15, 0.85),  # T1,P_sens1,G1,C_sens2,D_mod1,Cog4 - central sensitization + catastrophizing
  
  (0.35, 0.40, 0.20, 0.05, 0.70, 0.25, 0.05, 0.70, 0.20, 0.10, 0.75, 0.25),  # T1,P_sens2,G1,C_sens1,D_mod2,Cog1 - healed but moderate peripheral sensitization
  (0.20, 0.35, 0.30, 0.15, 0.55, 0.30, 0.15, 0.55, 0.30, 0.15, 0.60, 0.40),  # T1,P_sens2,G1,C_sens1,D_mod2,Cog3 - peripheral sensitization + fear
  
  (0.15, 0.30, 0.40, 0.15, 0.40, 0.35, 0.25, 0.40, 0.35, 0.25, 0.45, 0.55),  # T1,P_sens3,G1,C_sens1,D_mod2,Cog1 - healed but SEVERE peripheral sensitization
  (0.05, 0.20, 0.45, 0.30, 0.30, 0.40, 0.30, 0.30, 0.40, 0.30, 0.35, 0.65),  # T1,P_sens3,G1,C_sens1,D_mod2,Cog4 - severe peripheral + catastrophizing
  
  (0.40, 0.40, 0.15, 0.05, 0.75, 0.20, 0.05, 0.80, 0.15, 0.05, 0.80, 0.20),  # T2,P_sens1,G1,C_sens1,D_mod2,Cog1 - inflamed tissue, normal processing
  (0.25, 0.40, 0.25, 0.10, 0.65, 0.25, 0.10, 0.65, 0.25, 0.10, 0.70, 0.30),  # T2,P_sens1,G1,C_sens1,D_mod2,Cog3 - inflamed + fear
  
  (0.20, 0.35, 0.30, 0.15, 0.55, 0.30, 0.15, 0.50, 0.30, 0.20, 0.60, 0.40),  # T2,P_sens2,G1,C_sens1,D_mod2,Cog1 - inflamed + moderate peripheral sensitization
  (0.08, 0.25, 0.40, 0.27, 0.40, 0.35, 0.25, 0.35, 0.40, 0.25, 0.45, 0.55),  # T2,P_sens2,G1,C_sens1,D_mod2,Cog4 - inflamed + peripheral sens + catastrophizing
  
  (0.05, 0.20, 0.40, 0.35, 0.30, 0.40, 0.30, 0.25, 0.45, 0.30, 0.35, 0.65),  # T2,P_sens3,G1,C_sens1,D_mod2,Cog1 - inflamed + severe peripheral sensitization
  (0.02, 0.10, 0.35, 0.53, 0.20, 0.35, 0.45, 0.15, 0.45, 0.40, 0.20, 0.80),  # T2,P_sens3,G1,C_sens1,D_mod2,Cog4 - multiple risk factors
  
  (0.02, 0.08, 0.30, 0.60, 0.15, 0.30, 0.55, 0.10, 0.40, 0.50, 0.15, 0.85),  # T2,P_sens3,G1,C_sens2,D_mod1,Cog4 - CHRONIC PAIN PHENOTYPE (all risk factors)
  
  (0.10, 0.30, 0.40, 0.20, 0.50, 0.35, 0.15, 0.45, 0.35, 0.20, 0.50, 0.50),  # T3,P_sens1,G1,C_sens1,D_mod2,Cog1 - damaged tissue, normal sensitivity
  (0.05, 0.25, 0.45, 0.25, 0.40, 0.40, 0.20, 0.40, 0.40, 0.20, 0.45, 0.55),  # T3,P_sens1,G1,C_sens1,D_mod2,Cog3 - damaged + fear
  
  (0.03, 0.15, 0.42, 0.40, 0.30, 0.40, 0.30, 0.30, 0.45, 0.25, 0.35, 0.65),  # T3,P_sens2,G1,C_sens1,D_mod2,Cog1 - damaged + moderate peripheral sens
  (0.01, 0.08, 0.35, 0.56, 0.20, 0.35, 0.45, 0.20, 0.45, 0.35, 0.25, 0.75),  # T3,P_sens2,G1,C_sens1,D_mod2,Cog4 - damaged + peripheral sens + catastrophizing
  
  (0.01, 0.05, 0.30, 0.64, 0.15, 0.30, 0.55, 0.15, 0.50, 0.35, 0.20, 0.80),  # T3,P_sens3,G1,C_sens1,D_mod2,Cog1 - damaged + severe peripheral sens
  (0.00, 0.02, 0.20, 0.78, 0.10, 0.25, 0.65, 0.08, 0.45, 0.47, 0.10, 0.90),  # T3,P_sens3,G1,C_sens1,D_mod2,Cog4 - worst acute pain scenario
  
  # Gate control predictions: G=closed (G3) reduces pain substantially
  (0.70, 0.25, 0.05, 0.00, 0.80, 0.15, 0.05, 0.80, 0.15, 0.05, 0.85, 0.15),  # T2,P_sens2,G3,C_sens1,D_mod3,Cog1 - gate closed, strong inhibition
  (0.55, 0.35, 0.08, 0.02, 0.75, 0.20, 0.05, 0.75, 0.20, 0.05, 0.80, 0.20),  # T2,P_sens2,G3,C_sens1,D_mod3,Cog3 - gate closed even with fear
  
  # Gate open (G1) amplifies pain
  (0.15, 0.30, 0.40, 0.15, 0.45, 0.35, 0.20, 0.50, 0.35, 0.15, 0.55, 0.45),  # T2,P_sens2,G1,C_sens1,D_mod1,Cog1 - gate open, facilitation
  (0.05, 0.20, 0.45, 0.30, 0.30, 0.40, 0.30, 0.35, 0.40, 0.25, 0.40, 0.60),  # T2,P_sens2,G1,C_sens1,D_mod1,Cog4 - gate open + catastrophizing
  
  # ... (continues for all 378 combinations with theory-based predictions)
}

# B: 378 states_next x 378 states_previous x 81 actions. State transitions for each action combination.
# Actions ordered by: Attn, Behav, Reapp, Desc_C (3×3×3×3 = 81 total)
B={
  # Theory-specific transition dynamics:
  
  # Action 1: Distraction, Avoidance, Negative reappraisal, Low descending control (Attn1,Behav1,Reapp1,Desc_C1)
  # - Fear-avoidance model: Avoidance → deconditioning, muscle atrophy → increased tissue vulnerability
  # - Negative reappraisal → catastrophizing → facilitation → central sensitization
  # - Low descending control → open gate → peripheral signals reach CNS unimpeded
  # - Chronic pain pathway: T remains damaged/inflamed, P_sens increases, G opens, C_sens develops, D_mod shifts to facilitation, Cog worsens
  
  # Action 14: Monitoring, Pacing, Neutral reappraisal, Moderate descending control (Attn2,Behav2,Reapp2,Desc_C2)
  # - Balanced approach: Pacing → gradual tissue healing without overload
  # - Neutral reappraisal → stable cognitive state
  # - Moderate control → modulated gate
  # - Maintenance pathway: States remain relatively stable
  
  # Action 41: Distraction, Engagement, Positive reappraisal, High descending control (Attn1,Behav3,Reapp3,Desc_C3)
  # - Active coping: Engagement → tissue adaptation, improved function
  # - Positive reappraisal → reduced catastrophizing → inhibition
  # - High descending control → gate closure
  # - Recovery pathway: T heals, P_sens normalizes, G closes, C_sens prevented/reversed, D_mod shifts to inhibition, Cog improves
  
  # Action 81: Catastrophizing attention, Engagement (push through), Negative reappraisal, High descending control attempt (Attn3,Behav3,Reapp1,Desc_C3)
  # - Contradictory strategy: Engagement without cognitive change
  # - May lead to tissue damage despite high control attempts
  # - Risk of central sensitization despite behavioral engagement
  
  # Key transition probabilities for testing hypotheses:
  # 1. Peripheral Sensitization Hypothesis: Prolonged tissue damage (T=damaged) → P_sens increases over time
  # 2. Central Sensitization Hypothesis: Open gate (G1) + prolonged input → C_sens develops (one-way transition, difficult to reverse)
  # 3. Gate Control Hypothesis: Desc_C=high → G=closed → reduced pain transmission
  # 4. Fear-Avoidance Hypothesis: Behav=avoidance + Cog=fearful → T worsens + Cog → catastrophizing
  # 5. Cognitive Reappraisal Hypothesis: Reapp=positive → Cog improves → D_mod → inhibition
  # 6. Descending Facilitation Hypothesis: Cog=catastrophizing → D_mod=facilitation → amplified pain
  
  # ... (81 transition matrices defining these dynamics)
}

# C: 72 observations. Preferences for pain-free, functional states.
C={
  # Strong preference for: No pain (Pain_I1), nociceptive quality (normal - Pain_Q1), full function (Func1), normal autonomic (Auto1)
  # Strong aversion to: Severe pain (Pain_I4), nociplastic quality (Pain_Q3), disabled (Func3), hyperarousal (Auto2)
  
  # Pain_I preferences: none (0.9), mild (0.5), moderate (-0.3), severe (-1.5)
  # Pain_Q preferences: nociceptive (0.2), neuropathic (-0.3), nociplastic (-0.8)
  # Func preferences: full (0.8), limited (0.0), disabled (-1.0)
  # Auto preferences: normal (0.3), hyperarousal (-0.5)
  
  # Combined log-preferences for 72 observation combinations:
  (2.2, 1.4, 0.5, 0.7, -0.1, -0.8, -0.5, -1.3, -2.2, -1.7, -2.5, -3.4,
   2.0, 1.2, 0.3, 0.5, -0.3, -1.0, -0.7, -1.5, -2.4, -1.9, -2.7, -3.6,
   1.6, 0.8, -0.1, 0.1, -0.7, -1.4, -1.1, -1.9, -2.8, -2.3, -3.1, -4.0,
   1.0, 0.2, -0.7, -0.5, -1.3, -2.0, -1.7, -2.5, -3.4, -2.9, -3.7, -4.6,
   0.5, -0.3, -1.2, -1.0, -1.8, -2.5, -2.2, -3.0, -3.9, -3.4, -4.2, -5.1,
   -0.2, -1.0, -1.9, -1.7, -2.5, -3.2, -2.9, -3.7, -4.6, -4.1, -4.9, -5.8)
}

# D: 378 states. Prior distribution reflecting acute vs chronic pain populations.
D={
  # Acute pain prior (testing on recently injured population):
  # - Tissue: 10% healed, 50% inflamed, 40% damaged
  # - Peripheral sens: 60% normal, 30% moderate, 10% severe
  # - Gate: 40% open, 40% modulated, 20% closed
  # - Central sens: 90% absent, 10% present
  # - Descending mod: 20% facilitation, 60% neutral, 20% inhibition
  # - Cognitive: 30% adaptive, 20% vigilant, 15% fearful, 10% catastrophizing, 10% depressed, 10% anxious, 5% alexithymic
  
  # Chronic pain prior (testing on chronic pain population):
  # - Tissue: 40% healed, 40% inflamed, 20% damaged (tissue often healed in chronic pain!)
  # - Peripheral sens: 20% normal, 40% moderate, 40% severe (higher sensitization)
  # - Gate: 50% open, 35% modulated, 15% closed (gate more often open)
  # - Central sens: 40% absent, 60% present (higher central sensitization)
  # - Descending mod: 40% facilitation, 40% neutral, 20% inhibition (more facilitation)
  # - Cognitive: 10% adaptive, 15% vigilant, 20% fearful, 25% catastrophizing, 15% depressed, 10% anxious, 5% alexithymic (more maladaptive)
  
  # Default: Acute pain prior (can be parameterized for different populations)
  (0.10, 0.50, 0.40,   # T distribution (healed, inflamed, damaged)
   0.60, 0.30, 0.10,   # P_sens distribution (normal, moderate, severe)
   0.40, 0.40, 0.20,   # G distribution (open, modulated, closed)
   0.90, 0.10,         # C_sens distribution (absent, present)
   0.20, 0.60, 0.20,   # D_mod distribution (facilitation, neutral, inhibition)
   0.30, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05)  # Cog distribution (adaptive → alexithymic)
   # ... (repeated for all 378 state combinations with appropriate joint probabilities)
}

# E: 81 actions. Habit vector reflecting common pain coping strategies.
E={
  # Default habits favor monitoring, pacing, neutral reappraisal, moderate descending control
  # Slight bias toward adaptive strategies
  
  # Attn distribution: 30% distraction, 50% monitoring, 20% catastrophizing attention
  # Behav distribution: 25% avoidance, 50% pacing, 25% engagement
  # Reapp distribution: 20% negative, 60% neutral, 20% positive
  # Desc_C distribution: 25% low, 50% moderate, 25% high
  
  # Combined habit weights (81 action combinations):
  (0.015, 0.023, 0.015, 0.025, 0.038, 0.025, 0.015, 0.023, 0.015,
   0.030, 0.045, 0.030, 0.050, 0.075, 0.050, 0.030, 0.045, 0.030,
   0.015, 0.023, 0.015, 0.025, 0.038, 0.025, 0.015, 0.023, 0.015,
   0.010, 0.015, 0.010, 0.017, 0.025, 0.017, 0.010, 0.015, 0.010,
   0.020, 0.030, 0.020, 0.033, 0.050, 0.033, 0.020, 0.030, 0.020,
   0.010, 0.015, 0.010, 0.017, 0.025, 0.017, 0.010, 0.015, 0.010,
   0.006, 0.009, 0.006, 0.010, 0.015, 0.010, 0.006, 0.009, 0.006,
   0.012, 0.018, 0.012, 0.020, 0.030, 0.020, 0.012, 0.018, 0.012,
   0.006, 0.009, 0.006, 0.010, 0.015, 0.010, 0.006, 0.009, 0.006)
}

## Equations
# Chronic pain dynamics following Active Inference principles with THREE NESTED CONTINUOUS TIMESCALES:
#
# ============================================================================
# FAST TIMESCALE (t_fast: milliseconds to seconds) - Neural Signaling
# ============================================================================
#
# GATE CONTROL (Melzack & Wall) - Fast spinal modulation:
# τ_fast * dG/dt_fast = -G + σ(w_periph * P_sens - w_desc * D_mod + w_cog * f_cog(Cog) + I_noci(T))
# where τ_fast ≈ 10-100 ms (fast neural dynamics)
# - Peripheral nociceptive input I_noci(T) from tissue state
# - Peripheral sensitization P_sens (slow state) amplifies input
# - Descending modulation D_mod closes gate
# - Cognitive state influences gate threshold
# - Rapid equilibration to changing inputs
#
# DESCENDING MODULATION (Neuromatrix Theory) - Fast cortical-spinal signals:
# τ_desc * dD_mod/dt_fast = -D_mod + h(Cog, Reapp) + w_desc * Desc_C + η_fast(t)
# where τ_desc ≈ 50-200 ms (cortico-spinal loop delay)
# - Cognitive-affective state Cog (medium timescale) sets baseline
# - Cognitive reappraisal Reapp (medium timescale) modulates baseline
# - Volitional descending control Desc_C provides fast override
# - η_fast(t) represents neural noise at fast timescale
#
# PAIN PERCEPTION - Fast sensory processing:
# Pain_I(t_fast) = A_matrix[G(t_fast), P_sens(t_slow), C_sens(t_slow), D_mod(t_fast), Cog(t_medium)]
# Pain_Q(t_fast) = Quality[P_sens(t_slow), C_sens(t_slow), G(t_fast)]
# Auto(t_fast) = Autonomic[Pain_I(t_fast), Cog(t_medium)]
# - Observations generated from fast and slow states
# - Gate state G modulates immediate pain intensity
# - Sensitization states (slow) determine pain amplification
# - Autonomic responses follow pain perception rapidly
#
# ============================================================================
# MEDIUM TIMESCALE (t_medium: minutes to hours) - Cognitive-Behavioral
# ============================================================================
#
# COGNITIVE-AFFECTIVE DYNAMICS (Biopsychosocial Model):
# τ_cog * dCog/dt_medium = δ_cog * [k_pain * ⟨Pain_I⟩_fast + k_func * Func + k_behav * f_behav(Behav) + k_reapp * Reapp - ε_cog * Cog]
# where τ_cog ≈ 10-30 min (cognitive adaptation timescale)
# - ⟨Pain_I⟩_fast represents time-averaged pain from fast timescale
# - Functional limitations Func contribute to emotional distress
# - Behavioral patterns (avoidance, pacing, engagement) modulate cognitive state
# - Cognitive reappraisal Reapp provides corrective input
# - Natural recovery ε_cog drives return to baseline cognitive state
# - Catastrophizing develops through positive feedback: high pain → negative cognition → descending facilitation → more pain
#
# FUNCTIONAL CAPACITY DYNAMICS:
# τ_func * dFunc/dt_medium = -Func + g_func(Behav, ⟨Pain_I⟩_fast, Cog)
# where τ_func ≈ 30-60 min (activity-dependent functional changes)
# - Behavioral engagement improves function
# - Avoidance leads to deconditioning and reduced function
# - Pain intensity limits functional capacity
# - Cognitive state (fear, catastrophizing) restricts function beyond physical limitations
#
# ACTIVE INFERENCE POLICY SELECTION (Medium timescale):
# G(π) = E_π[∫_t_fast D_KL(o(τ) || C(τ)) dτ] + D_KL(π || E)  # Expected free energy integrating over fast observations
# π*(t_medium) = argmin_π G(π)  # Policy updated on medium timescale
# u(t_medium) ~ softmax(-γ * G(π))  # Action selection: Attn, Behav, Reapp, Desc_C
# - Policies evaluated by integrating prediction errors over fast timescale
# - Actions (attention, behavior, reappraisal, descending control) selected to minimize expected free energy
# - Habit priors E bias toward familiar coping strategies
#
# ============================================================================
# SLOW TIMESCALE (t_slow: hours to days) - Tissue & Sensitization
# ============================================================================
#
# TISSUE HEALING/DAMAGE:
# τ_tissue * dT/dt_slow = η_heal * h_pacing(⟨Behav⟩_medium) - θ_damage * [h_overuse(⟨Behav⟩_medium) + h_disuse(⟨Behav⟩_medium)]
# where τ_tissue ≈ 6-24 hours (tissue remodeling timescale)
# - ⟨Behav⟩_medium represents time-averaged behavioral strategy from medium timescale
# - Pacing (moderate, graduated activity) promotes healing
# - Overuse (excessive engagement without rest) causes tissue damage
# - Disuse (avoidance, inactivity) causes deconditioning and vulnerability
# - Natural healing rate η_heal vs damage rate θ_damage
#
# PERIPHERAL SENSITIZATION (Nociceptor Theory) - Slow plasticity:
# τ_periph * dP_sens/dt_slow = α_periph * f_inflam(T) * [1 - P_sens] - β_periph * P_sens
# where τ_periph ≈ 12-48 hours (peripheral nociceptor sensitization)
# - Tissue inflammation/damage T drives sensitization
# - Saturation dynamics: sensitization increases from normal state
# - Spontaneous recovery β_periph (slow, hours to days)
# - Peripheral sensitization lowers pain thresholds (allodynia)
#
# CENTRAL SENSITIZATION (Wind-up, LTP) - Slow, one-way plasticity:
# τ_central * dC_sens/dt_slow = λ_central * ⟨G(t_fast)⟩_fast * P_sens * [1 - C_sens] - μ_reverse * C_sens
# where τ_central ≈ 24-72 hours (central LTP-like process)
# - ⟨G(t_fast)⟩_fast represents time-averaged gate openness (prolonged nociceptive barrage)
# - Peripheral sensitization P_sens amplifies input
# - Cumulative input drives central sensitization (threshold process)
# - Reversal rate μ_reverse << λ_central (one-way process, difficult to reverse)
# - C_sens=present indicates chronic central amplification (nociplastic pain)
# - Central sensitization predicts pain persistence even after tissue healing
#
# ============================================================================
# TIMESCALE SEPARATION AND COUPLING:
# ============================================================================
# ε = τ_fast / τ_medium ≈ 10^-3  (fast/medium separation)
# δ = τ_medium / τ_slow ≈ 10^-2  (medium/slow separation)
#
# Coupling structure:
# - SLOW → FAST: Sensitization states (P_sens, C_sens, T) act as slowly-varying parameters for fast dynamics
# - FAST → MEDIUM: Time-averaged pain signals ⟨Pain_I⟩ drive cognitive-affective changes
# - MEDIUM → SLOW: Time-averaged behavior ⟨Behav⟩ affects tissue healing and sensitization
# - FREE ENERGY MINIMIZATION spans all timescales with appropriate temporal integration
#
# ============================================================================
# STATE INFERENCE (Hierarchical across timescales):
# ============================================================================
# Fast states: q(G, D_mod | o_fast) = argmin F_fast
# Medium states: q(Cog, Func | ⟨o_fast⟩, o_medium) = argmin F_medium
# Slow states: q(T, P_sens, C_sens | ⟨o_medium⟩, o_slow) = argmin F_slow
#
# Variational Free Energy at each timescale:
# F_fast = D_KL(q(s_fast) || p(s_fast | s_medium, s_slow)) - E_q[log p(o_fast | s_fast, s_slow)]
# F_medium = D_KL(q(s_medium) || p(s_medium | s_slow)) - E_q[log p(⟨o_fast⟩ | s_medium, s_slow)]
# F_slow = D_KL(q(s_slow) || p(s_slow)) - E_q[log p(⟨o_medium⟩ | s_slow)]
#
# ============================================================================
# TESTABLE PREDICTIONS (Multi-timescale):
# ============================================================================
# 1. CENTRAL SENSITIZATION (slow): Pain persistence despite tissue healing
#    - T→healed (slow timescale) but C_sens=present (slow) → continued severe pain (fast)
#    - Time course: weeks to months for central sensitization development
#
# 2. GATE CONTROL (fast): Immediate pain modulation via descending control
#    - High Desc_C (fast action) → G=closed (fast state) → reduced Pain_I (fast observation)
#    - Time course: seconds to minutes for gate closure effects
#
# 3. FEAR-AVOIDANCE SPIRAL (medium): Cognitive-behavioral disability cycle
#    - Behav=avoidance (medium) + Cog=fearful (medium) → Func worsens (medium) → Cog worsens (medium)
#    - Time course: days to weeks for fear-avoidance cycle establishment
#
# 4. COGNITIVE REAPPRAISAL (medium→fast): Descending inhibition pathway
#    - Reapp=positive (medium action) → Cog improves (medium) → D_mod=inhibition (fast) → Pain_I reduced (fast)
#    - Time course: minutes to hours for cognitive change, immediate pain reduction via descending control
#
# 5. PERIPHERAL SENSITIZATION (slow): Allodynia development
#    - Prolonged T=inflamed (slow) → P_sens=severe (slow) → high Pain_I (fast) even after T→healed
#    - Time course: hours to days for peripheral sensitization, days to weeks for normalization
#
# 6. CATASTROPHIZING AMPLIFICATION (medium→fast): Descending facilitation
#    - Persistent high pain (fast) → Cog→catastrophizing (medium) → D_mod=facilitation (fast) → Pain_I amplified (fast)
#    - Time course: hours to days for catastrophizing development, immediate pain amplification
#
# 7. TIMESCALE INTERACTIONS: Multi-rate chronification pathway
#    - Acute injury (T=damaged, fast pain) → avoidance behavior (medium) → tissue vulnerability + catastrophizing (medium) 
#      → peripheral sensitization (slow) + descending facilitation (fast) → central sensitization (slow) → chronic pain
#    - Full chronification time course: weeks to months integrating all three timescales

## Time
Time=t_fast,t_medium,t_slow
Dynamic
Continuous
MultipleTimescales=True
ModelTimeHorizon=Unbounded # Chronic pain model for longitudinal simulation across multiple timescales

# Fast Timescale (t_fast): Neural signaling and acute responses
FastTimeScale=t_fast
FastTimeRange=[0, ∞) ms
FastProcesses=[G, D_mod, Pain_I, Pain_Q, Auto]  # Spinal gate, descending modulation, immediate pain perception

# Medium Timescale (t_medium): Cognitive-affective and behavioral dynamics
MediumTimeScale=t_medium
MediumTimeRange=[0, ∞) min
MediumProcesses=[Cog, Attn, Behav, Reapp, Func]  # Cognitive states, attention, behavior, functional capacity

# Slow Timescale (t_slow): Tissue healing and chronic sensitization
SlowTimeScale=t_slow
SlowTimeRange=[0, ∞) hours
SlowProcesses=[T, P_sens, C_sens]  # Tissue state, peripheral/central sensitization development

# Timescale Coupling: Processes at faster timescales influence slower ones
# - Fast → Medium: Pain signals drive cognitive-affective responses
# - Medium → Slow: Behavioral strategies affect tissue healing and sensitization
# - Slow → Fast: Sensitization states modulate acute pain processing
# - All timescales interact through Active Inference free energy minimization

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrices
C=LogPreferenceVector
D=PriorOverHiddenStates
E=HabitVector
F=VariationalFreeEnergy
G=ExpectedFreeEnergy
T=TissueState
P_sens=PeripheralSensitization
G=SpinalGateState
C_sens=CentralSensitization
D_mod=DescendingModulation
Cog=CognitiveAffectiveState
Pain_I=PainIntensityObservation
Pain_Q=PainQualityObservation
Func=FunctionalCapacityObservation
Auto=AutonomicResponseObservation
Attn=AttentionAllocationControl
Behav=BehavioralStrategyControl
Reapp=CognitiveReappraisalControl
Desc_C=DescendingControlAction
t_fast=FastTimescale
t_medium=MediumTimescale
t_slow=SlowTimescale

## ModelParameters
# Hidden state parameters (6 factors, 378 combinations)
num_tissue_states: 3           # T[3] - healed, inflamed, damaged
num_peripheral_sens_levels: 3  # P_sens[3] - normal, moderate, severe
num_gate_states: 3             # G[3] - open, modulated, closed
num_central_sens_states: 2     # C_sens[2] - absent, present
num_descending_mod_states: 3   # D_mod[3] - facilitation, neutral, inhibition
num_cognitive_states: 7        # Cog[7] - adaptive, vigilant, fearful, catastrophizing, depressed, anxious, alexithymic

# Observation parameters (4 modalities, 72 combinations)
num_pain_intensity_levels: 4   # Pain_I[4] - none, mild, moderate, severe
num_pain_quality_types: 3      # Pain_Q[3] - nociceptive, neuropathic, nociplastic
num_functional_levels: 3       # Func[3] - full, limited, disabled
num_autonomic_states: 2        # Auto[2] - normal, hyperarousal

# Control parameters (4 factors, 81 combinations)
num_attention_strategies: 3    # Attn[3] - distraction, monitoring, catastrophizing
num_behavioral_strategies: 3   # Behav[3] - avoidance, pacing, engagement
num_reappraisal_strategies: 3  # Reapp[3] - negative, neutral, positive
num_descending_control_levels: 3 # Desc_C[3] - low, moderate, high

# Totals
total_hidden_states: 378       # 3×3×3×2×3×7 combinations
total_observations: 72         # 4×3×3×2 combinations
total_actions: 81              # 3×3×3×3 combinations

# Theory-specific parameters for hypothesis testing (with timescale-specific rates)

# ============================================================================
# FAST TIMESCALE PARAMETERS (milliseconds to seconds)
# ============================================================================
tau_fast_gate: 0.05            # s - Gate dynamics time constant (50 ms)
tau_fast_descending: 0.15      # s - Descending modulation time constant (150 ms)
w_periph_gate: 1.2             # - Weight of peripheral input on gate
w_desc_gate: 1.5               # - Weight of descending modulation on gate closure
w_cog_gate: 0.8                # - Weight of cognitive state on gate threshold
w_desc_control: 2.0            # - Weight of volitional descending control
noise_fast_std: 0.1            # - Standard deviation of fast neural noise

# ============================================================================
# MEDIUM TIMESCALE PARAMETERS (minutes to hours)
# ============================================================================
tau_medium_cog: 1800           # s (30 min) - Cognitive adaptation time constant
tau_medium_func: 3600          # s (60 min) - Functional capacity time constant
delta_cognitive: 0.10          # - Cognitive adaptation rate
k_pain_cog: 0.15               # - Weight of pain on cognitive state
k_func_cog: 0.10               # - Weight of function on cognitive state
k_behav_cog: 0.12              # - Weight of behavior on cognitive state
k_reapp_cog: 0.20              # - Weight of reappraisal on cognitive state
epsilon_cognitive_recovery: 0.05 # - Natural cognitive recovery rate

# ============================================================================
# SLOW TIMESCALE PARAMETERS (hours to days)
# ============================================================================
tau_slow_tissue: 43200         # s (12 hours) - Tissue remodeling time constant
tau_slow_periph: 86400         # s (24 hours) - Peripheral sensitization time constant
tau_slow_central: 172800       # s (48 hours) - Central sensitization time constant

# Tissue dynamics
eta_tissue_healing: 0.03       # - Tissue healing rate with optimal pacing
theta_tissue_overuse: 0.04     # - Tissue damage rate from overuse
theta_tissue_disuse: 0.035     # - Tissue damage rate from disuse

# Peripheral sensitization
alpha_peripheral_sens: 0.05    # - Peripheral sensitization development rate
beta_peripheral_recovery: 0.02 # - Peripheral sensitization spontaneous recovery rate

# Central sensitization (one-way process)
lambda_central_sens: 0.001     # - Central sensitization accumulation rate
mu_central_reverse: 0.0001     # - Central sensitization reversal rate (very slow, μ << λ)
threshold_central: 0.6         # - Threshold for central sensitization triggering

# ============================================================================
# TIMESCALE SEPARATION RATIOS
# ============================================================================
epsilon_fast_medium: 0.001     # ε = τ_fast / τ_medium ≈ 10^-3
delta_medium_slow: 0.01        # δ = τ_medium / τ_slow ≈ 10^-2

# ============================================================================
# TEMPORAL AVERAGING WINDOWS (for cross-timescale coupling)
# ============================================================================
window_fast_to_medium: 300     # s (5 min) - Averaging window for fast→medium coupling
window_medium_to_slow: 14400   # s (4 hours) - Averaging window for medium→slow coupling

# ============================================================================
# POPULATION-SPECIFIC PRIORS (can be varied for testing)
# ============================================================================
population_type: "acute"       # Options: "acute", "chronic", "high_risk", "resilient"

## Footer
Active Inference Chronic Pain Multi-Theory Model v1 - GNN Representation.

This model integrates multiple coherent theories of chronic pain chronification across THREE NESTED CONTINUOUS TIMESCALES:

**Integrated Theories:**
- Peripheral sensitization (nociceptor hyperexcitability)
- Central sensitization (spinal and supraspinal amplification)
- Gate control theory (spinal modulation mechanisms)
- Neuromatrix theory (distributed network including descending modulation)
- Predictive coding / Active Inference (pain as precision-weighted prediction error)
- Fear-avoidance model (cognitive-behavioral factors)
- Biopsychosocial integration (tissue, neural, cognitive, affective, behavioral factors)

**Three Nested Timescales:**
1. **Fast (milliseconds to seconds)**: Neural signaling, gate control, descending modulation, acute pain perception
   - Allows modeling immediate pain responses and rapid modulation
   - Time constants: τ_gate ≈ 50 ms, τ_descending ≈ 150 ms

2. **Medium (minutes to hours)**: Cognitive-affective processes, behavioral adaptation, functional capacity
   - Captures psychological coping strategies and functional changes
   - Time constants: τ_cognitive ≈ 30 min, τ_function ≈ 60 min

3. **Slow (hours to days)**: Tissue healing, peripheral/central sensitization development
   - Models chronic plasticity and long-term pain chronification
   - Time constants: τ_tissue ≈ 12 hrs, τ_peripheral ≈ 24 hrs, τ_central ≈ 48 hrs

**Timescale Coupling:**
- Slow states (sensitization, tissue) modulate fast dynamics (acute pain)
- Fast observations (averaged pain) drive medium dynamics (cognition, behavior)
- Medium behaviors (averaged strategies) influence slow processes (healing, sensitization)
- Active Inference operates across all timescales with appropriate temporal integration

**The model enables:**
1. Testing competing hypotheses about pain chronification mechanisms across multiple timescales
2. Simulating different patient populations (acute, chronic, high-risk, resilient)
3. Evaluating intervention strategies with timescale-specific effects (behavioral, cognitive, pharmacological)
4. Predicting individual trajectories based on baseline states and coping strategies
5. Identifying critical transition points from acute to chronic pain
6. Understanding multi-rate dynamics from immediate pain modulation to long-term chronification
7. Designing interventions targeting specific timescales (fast: descending control, medium: CBT, slow: tissue healing)

**Key testable predictions (with timescale specificity):**
- Central sensitization (slow) as a "point of no return" in chronic pain development
- Gate control effectiveness (fast) via descending modulation with immediate pain reduction
- Fear-avoidance cycles (medium) leading to disability independent of tissue state
- Cognitive reappraisal (medium→fast) as pathway to pain reduction via descending inhibition
- Peripheral vs central contributions (slow) to pain in different patient phenotypes
- Timescale interactions driving the transition from acute to chronic pain
- Intervention windows: fast (immediate relief), medium (coping strategies), slow (prevent sensitization)

## Signature
Cryptographic signature goes here

