# GNN Example: Active Inference Language Model (AILM)
# Format: Markdown representation of a comprehensive Active Inference model for language.
# Version: 0.1
# This file is machine-readable and outlines a multi-level agent for linguistic processing.

## GNNSection
ActiveInferenceLanguageModel

## GNNVersionAndFlags
GNN v1

## ModelName
Active Inference Language Model (AILM) v0.1

## ModelAnnotation
This model outlines a comprehensive Active Inference agent for language understanding and generation.
It attempts to capture nested and interacting levels of linguistic processing, from phonetics to discourse.
Key features demonstrated:
- Hierarchical state representation (phonetic, lexical, syntactic, semantic, discourse, contextual).
- Multiple observation modalities reflecting both external input and internal states.
- Control factors for linguistic actions (vocalization, lexical choice, intent formation).
- Complex interplay of likelihoods, transitions, preferences, and policies.
- Designed as a "one-shot" GNN-based blueprint, not reliant on pre-existing training datasets.

This AILM is an illustrative example due to the inherent vastness of real-world language.
State spaces and parameterizations are simplified for GNN demonstration.
It aims to test the full GNN pipeline's capacity to handle complex, multi-faceted models.
This model implies a dynamic system operating over discrete time steps.

## StateSpaceBlock
# --- Hidden State Factors (s_fX) ---
# Phonetic & Articulatory Level
s_f0[10,1,type=int]  # Hidden State Factor 0: PhoneticTarget (e.g., current target phoneme category; 10 categories)
s_f1[5,1,type=int]   # Hidden State Factor 1: ArticulatoryConfiguration (e.g., abstract vocal tract state; 5 configurations)

# Lexical & Morpho-Syntactic Level
s_f2[50,1,type=int]  # Hidden State Factor 2: ActiveLexicalConceptID (e.g., current word/morpheme being processed/generated; 50 abstract concepts)
s_f3[8,1,type=int]   # Hidden State Factor 3: CurrentSyntacticRole (e.g., Subject, Verb, Object, Adjunct; 8 roles)
s_f4[10,1,type=int]  # Hidden State Factor 4: MorphoSyntacticFeatures (e.g., tense, number, gender markers; 10 feature sets)

# Semantic Level
s_f5[100,1,type=int] # Hidden State Factor 5: SemanticPropositionID (e.g., core meaning/event being conveyed/understood; 100 abstract propositions)
s_f6[5,1,type=int]   # Hidden State Factor 6: SemanticValence (e.g., positive, negative, neutral sentiment of proposition; 5 valences)

# Situational & Narrative Context Level
s_f7[10,1,type=int]  # Hidden State Factor 7: SituationalContextKey (e.g., speaker, hearer, location, formality; 10 keys)
s_f8[20,1,type=int]  # Hidden State Factor 8: NarrativeFocusID (e.g., current topic/theme in discourse; 20 foci)
s_f9[5,1,type=int]   # Hidden State Factor 9: PartnerModelState (e.g., inferred understanding/attention of interlocutor; 5 states)

# Agent's Internal Goal & Prediction Level
s_f10[10,1,type=int] # Hidden State Factor 10: CommunicativeIntentID (agent's high-level goal, e.g., inform, query; 10 intents)
s_f11[20,1,type=int] # Hidden State Factor 11: PredictedNextLexicalConceptID (anticipation of next word; 20 concepts)

# --- Observation Modalities (o_mX) ---
o_m0[20,1,type=int]  # Observation Modality 0: AuditoryStreamSegment (discretized features of incoming/outgoing sound; 20 categories)
o_m1[5,1,type=int]   # Observation Modality 1: SemanticCoherenceSignal (internal assessment of understanding/expression clarity; 5 levels)
o_m2[5,1,type=int]   # Observation Modality 2: DiscourseProgressSignal (e.g., turn-taking cues, topic shift cues; 5 signals)
o_m3[5,1,type=int]   # Observation Modality 3: PartnerFeedbackCue (e.g., facial expression, backchannel; 5 cues)

# --- Control Factors / Policies (pi_cX) & Chosen Actions (u_cX) ---
pi_c0[5,type=float]  # Policy for Control Factor 0: VocalizationEffort (modulates articulatory precision/energy)
u_c0[1,type=int]     # Chosen action for VocalizationEffort

pi_c1[20,type=float] # Policy for Control Factor 1: LexicalEmphasis (choosing next lexical item to focus/generate from s_f11 related space)
u_c1[1,type=int]     # Chosen action for LexicalEmphasis

pi_c2[10,type=float] # Policy for Control Factor 2: IntentRefinement (adjusting/committing to a communicative intent from s_f10 space)
u_c2[1,type=int]     # Chosen action for IntentRefinement

# --- Likelihood Mappings (A_mX) ---
# A_mX[outcomes, s_fA_states, s_fB_states, ..., type=dataType]
A_m0[20,10,5,type=float] # P(o_m0 | s_f0, s_f1) - AuditoryStream likelihood given PhoneticTarget, ArticulatoryConfig
A_m1[5,100,5,type=float] # P(o_m1 | s_f5, s_f6) - SemanticCoherence likelihood given SemanticProposition, Valence
A_m2[5,20,5,type=float]  # P(o_m2 | s_f8, s_f9) - DiscourseProgress likelihood given NarrativeFocus, PartnerModel
A_m3[5,5,type=float]     # P(o_m3 | s_f9) - PartnerFeedback likelihood given PartnerModel

# --- Transition Dynamics (B_fX) ---
# B_fX[next_states, prev_states, u_c0_actions, u_c1_actions, ..., type=dataType]
# Phonetic/Articulatory Level Transitions (influenced by VocalizationEffort u_c0)
B_f0[10,10,5,type=float] # P(s_f0' | s_f0, u_c0)
B_f1[5,5,5,type=float]   # P(s_f1' | s_f1, u_c0)

# Lexical/Morpho-Syntactic Level Transitions (influenced by LexicalEmphasis u_c1, and other states)
B_f2[50,50,10,8,20,type=float] # P(s_f2' | s_f2, s_f0, s_f3, u_c1) - ActiveLexicalConcept influenced by prev lexical, phonetic target, syntactic role, lexical emphasis
B_f3[8,8,50,type=float]        # P(s_f3' | s_f3, s_f2) - SyntacticRole influenced by prev role, current lexical concept
B_f4[10,10,50,type=float]      # P(s_f4' | s_f4, s_f2) - MorphoSyntacticFeatures influenced by prev features, current lexical concept

# Semantic Level Transitions (influenced by lexical, syntactic, previous semantic states)
B_f5[100,100,50,8,type=float]  # P(s_f5' | s_f5, s_f2, s_f3) - SemanticProposition influenced by prev proposition, lexical concept, syntactic role
B_f6[5,5,100,type=float]       # P(s_f6' | s_f6, s_f5) - SemanticValence influenced by prev valence, current proposition

# Context/Narrative Level Transitions (influenced by semantics, partner model, discourse goals)
B_f7[10,10,5,type=float]       # P(s_f7' | s_f7, s_f9) - SituationalContext influenced by prev context, partner model
B_f8[20,20,100,10,type=float]  # P(s_f8' | s_f8, s_f5, s_f10) - NarrativeFocus influenced by prev focus, semantic proposition, communicative intent
B_f9[5,5,5,type=float]         # P(s_f9' | s_f9, o_m3) - PartnerModel influenced by prev state, partner feedback cues

# Goal/Prediction Level Transitions (influenced by high-level states and IntentRefinement u_c2)
B_f10[10,10,20,10,type=float]  # P(s_f10' | s_f10, s_f8, u_c2) - CommunicativeIntent influenced by prev intent, narrative focus, intent refinement action
B_f11[20,20,100,10,type=float] # P(s_f11' | s_f11, s_f5, s_f10) - PredictedLexicalConcept influenced by prev prediction, current semantic proposition, communicative intent

# --- Preferences (C_mX) ---
C_m0[20,type=float] # Preferences over AuditoryStreamSegments (e.g., prefer clear, expected sounds)
C_m1[5,type=float]  # Preferences over SemanticCoherenceLevels (e.g., prefer high clarity)
C_m2[5,type=float]  # Preferences over DiscourseProgressSignals (e.g., prefer smooth turn-taking)
C_m3[5,type=float]  # Preferences over PartnerFeedbackCues (e.g., prefer positive feedback)

# --- Priors over Initial Hidden States (D_fX) ---
D_f0[10,type=float]  # Prior for PhoneticTarget
D_f1[5,type=float]   # Prior for ArticulatoryConfiguration
D_f2[50,type=float]  # Prior for ActiveLexicalConceptID
D_f3[8,type=float]   # Prior for CurrentSyntacticRole
D_f4[10,type=float]  # Prior for MorphoSyntacticFeatures
D_f5[100,type=float] # Prior for SemanticPropositionID
D_f6[5,type=float]   # Prior for SemanticValence
D_f7[10,type=float]  # Prior for SituationalContextKey
D_f8[20,type=float]  # Prior for NarrativeFocusID
D_f9[5,type=float]   # Prior for PartnerModelState
D_f10[10,type=float] # Prior for CommunicativeIntentID
D_f11[20,type=float] # Prior for PredictedLexicalConceptID

# --- Expected Free Energy (G) ---
# G would be calculated over policies combining pi_c0, pi_c1, pi_c2.
# For GNN representation, a single G or G per policy factor might be used.
G[1,type=float]      # Overall Expected Free Energy of chosen combined policy

# --- Time ---
t[1,type=int]        # Current time step

## Connections
# Priors to initial states (example for a few factors)
(D_f0, D_f1, D_f2, D_f3, D_f4, D_f5, D_f6, D_f7, D_f8, D_f9, D_f10, D_f11) -> (s_f0, s_f1, s_f2, s_f3, s_f4, s_f5, s_f6, s_f7, s_f8, s_f9, s_f10, s_f11)

# State factors to Likelihoods (A_mX) to Observations (o_mX)
(s_f0, s_f1) -> A_m0 -> o_m0
(s_f5, s_f6) -> A_m1 -> o_m1
(s_f8, s_f9) -> A_m2 -> o_m2
(s_f9)       -> A_m3 -> o_m3

# States and Actions (u_cX) to Transitions (B_fX) to Next States (s_fX_next - implied)
# Phonetic/Articulatory
(s_f0, u_c0) -> B_f0 -> s_f0_next
(s_f1, u_c0) -> B_f1 -> s_f1_next

# Lexical/Morpho-Syntactic
(s_f2, s_f0, s_f3, u_c1) -> B_f2 -> s_f2_next
(s_f3, s_f2)             -> B_f3 -> s_f3_next
(s_f4, s_f2)             -> B_f4 -> s_f4_next

# Semantic
(s_f5, s_f2, s_f3) -> B_f5 -> s_f5_next
(s_f6, s_f5)       -> B_f6 -> s_f6_next

# Context/Narrative
(s_f7, s_f9)          -> B_f7 -> s_f7_next
(s_f8, s_f5, s_f10)   -> B_f8 -> s_f8_next
(s_f9, o_m3)          -> B_f9 -> s_f9_next # Partner model updated by their feedback

# Goal/Prediction
(s_f10, s_f8, u_c2) -> B_f10 -> s_f10_next
(s_f11, s_f5, s_f10) -> B_f11 -> s_f11_next

# Preferences, Expected Future States/Observations to Expected Free Energy (G)
(C_m0, C_m1, C_m2, C_m3, A_m0, A_m1, A_m2, A_m3, B_f0, B_f1, ..., B_f11, s_f0, ..., s_f11) > G # Highly simplified EFE dependency

# EFE to Policies (pi_cX)
G > (pi_c0, pi_c1, pi_c2)

# Policies to Chosen Actions (u_cX)
(pi_c0) -> u_c0
(pi_c1) -> u_c1
(pi_c2) -> u_c2

## InitialParameterization
# Due to the vastness, parameterizations are conceptual placeholders.
# Real values would require extensive research or learning.
# All B matrices need to be column-stochastic (sum to 1 over next_states for each prev_state/action combo).
# All A matrices need to be column-stochastic (sum to 1 over outcomes for each state combo).

# Priors (D_fX) - Example: Uniform for most, could be specific for some (e.g. initial intent)
D_f0={(0.1, ..., 0.1)} # Uniform over 10 phonetic targets
D_f5={(0.01, ..., 0.01)} # Uniform over 100 semantic propositions
D_f10={(0.5, 0.1, ..., 0.1)} # e.g., high prior on 'inform' intent initially

# Likelihoods (A_mX) - Example for A_m1 (SemanticCoherence)
# A_m1[clarity_level, proposition_ID, valence_ID]
A_m1={ # P(o_m1 | s_f5, s_f6) - Highly schematic
  # Clarity=0 (Very Low)
  ( ((0.8, ..., 0.8), ... ), ... ), # High prob of low clarity if prop/valence are 'incoherent' (not defined here)
  # Clarity=4 (Very High)
  ( ((0.1, ..., 0.1), ... ), ... ) # High prob of high clarity if prop/valence are 'coherent'
}

# Transitions (B_fX) - Example for B_f0 (PhoneticTarget)
# B_f0[next_target, prev_target, vocal_effort_action]
B_f0={ # P(s_f0' | s_f0, u_c0) - Schematic
  # next_target = 0
  ( ((0.9, 0.1, ...), (0.8, 0.2, ...), ... ), ...), # Depending on effort, might stay or shift phoneme
  # ...
}

# Preferences (C_mX) - Example for C_m1 (SemanticCoherence)
C_m1={(-2.0, -1.0, 0.0, 1.0, 2.0)} # Prefer high semantic coherence (0-4 scale)

## Equations
# Standard Active Inference equations apply:
# 1. State Estimation: Approximate posterior over hidden states q(s_t) based on observations o_t and priors.
#    q(s_t) ~ σ( ln(A^T o_t) + ln(P(s_t|s_{t-1}, u_{t-1})) )
# 2. Policy Evaluation: Expected Free Energy G(π) for each policy π.
#    G(π) = E_q(o_τ, s_τ | π) [ ln q(s_τ|o_τ,π) - ln q(s_τ,o_τ|π) - ln C(o_τ) ] for τ > t
# 3. Action Selection: Softmax over negative EFE to choose actions.
#    P(u_t|π) ~ σ(-G(π))

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=100 # Example: an interaction of 100 time steps

## ActInfOntologyAnnotation
# Hidden States
s_f0=PhoneticRepresentation
s_f1=ArticulatoryMotorState
s_f2=LexicalEntry
s_f3=SyntacticConstituentRole
s_f4=MorphoSyntacticMarkerSet
s_f5=SemanticProposition
s_f6=AffectiveValence
s_f7=SituationalModelParameter
s_f8=NarrativeState_TopicFocus
s_f9=TheoryOfMind_PartnerState
s_f10=Goal_CommunicativeIntent
s_f11=Prediction_NextLexicalUnit

# Observations
o_m0=AuditoryObservation
o_m1=InternalObservation_SemanticClarity
o_m2=InternalObservation_DiscourseFlow
o_m3=SocialObservation_PartnerFeedback

# Control/Policy
pi_c0=Policy_VocalizationControl
u_c0=Action_VocalizationModulation
pi_c1=Policy_LexicalSelectionProcess
u_c1=Action_SelectNextLexeme
pi_c2=Policy_IntentManagement
u_c2=Action_UpdateIntent

# Matrices
A_m0=LikelihoodMatrix_AuditoryStream
B_f0=TransitionMatrix_PhoneticTarget
C_m0=LogPreferenceVector_AuditoryStream
D_f0=PriorDistribution_PhoneticTarget
# ... (annotations for all A, B, C, D matrices)

# Other
G=ExpectedFreeEnergy
t=TimeStep

## ModelParameters
num_hidden_state_factors: 12 # s_f0 to s_f11
dimensions_hidden_state_factors: [10, 5, 50, 8, 10, 100, 5, 10, 20, 5, 10, 20]
num_observation_modalities: 4 # o_m0 to o_m3
dimensions_observation_modalities: [20, 5, 5, 5]
num_control_factors: 3 # pi_c0 to pi_c2
dimensions_control_factors_actions: [5, 20, 10] # Number of actions for each control factor

## Footer
Active Inference Language Model (AILM) v0.1 - End of Specification.
This GNN file provides a structural blueprint. Parameterization is illustrative and requires substantial further work for a functional model.

## Signature
Creator: GNN Example Contributor (AI)
Date: Current Date
Status: Ambitious conceptual example for testing GNN pipeline with complex, hierarchical Active Inference models. 