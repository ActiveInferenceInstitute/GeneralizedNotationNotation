# GNNVersionAndFlags
GNN v1.0 flags=latex_rendering,graphviz_compatible

## GNNSection
PTSDHierarchicalActiveInferenceAgent

## ModelName
PTSD Hierarchical Active Inference Agent

## ModelAnnotation
This hierarchical Active Inference agent consists of two levels:

- **Lower Level Agent**: Processes sensorimotor information including trustworthiness, card states, affect, choices, and game stages
- **Higher Level Agent**: Processes abstract safety concepts (self, world, other) based on lower-level posteriors

The agents communicate bidirectionally: lower-level posteriors become higher-level observations, and higher-level inferred states become lower-level priors.

## StateSpaceBlock
# Lower Level Agent Variables
Trustworthiness[2,type=float]        # Hidden state: trust, distrust
CorrectCard[2,type=float]           # Hidden state: blue, green
Affect[2,type=float]                # Hidden state: angry, calm
Choice[3,type=float]                # Hidden state: blue, green, null
Stage[3,type=float]                 # Hidden state: null, advice, decision

# Lower Level Observations
Advice[3,type=float]                # Observation: blue, green, null
Feedback[3,type=float]              # Observation: correct, incorrect, null
Arousal[2,type=float]               # Observation: high, low
ChoiceObs[3,type=float]             # Observation: blue, green, null

# Lower Level Actions
TrustActions[2,type=float]          # Actions: trust, distrust
CardActions[3,type=float]           # Actions: blue, green, null
NullActions[1,type=float]           # Actions: NULL

# Higher Level Agent Variables
SafetySelf[2,type=float]            # Hidden state: safe, danger
SafetyWorld[2,type=float]           # Hidden state: safe, danger
SafetyOther[2,type=float]           # Hidden state: safe, danger

# Higher Level Observations (from lower level posteriors)
TrustworthinessObs[2,type=float]    # Observation from lower level
CorrectCardObs[2,type=float]        # Observation from lower level
AffectObs[2,type=float]             # Observation from lower level
ChoiceObsHigher[3,type=float]       # Observation from lower level
StageObs[3,type=float]              # Observation from lower level

# Higher Level Actions
NullActionsHigher[1,type=float]     # Actions: NULL

## Connections
# Lower Level Connections
Trustworthiness>Advice               # Trust influences advice
CorrectCard>Feedback                 # Card state influences feedback
Affect>Arousal                       # Affect influences arousal
Choice>ChoiceObs                     # Choice influences choice observation
Stage>Advice                         # Stage influences advice availability

# Higher Level Connections
SafetySelf>TrustworthinessObs        # Safety influences trust observation
SafetyWorld>CorrectCardObs           # World safety influences card observation
SafetyOther>AffectObs                # Other safety influences affect observation

# Inter-level Connections
Trustworthiness>TrustworthinessObs   # Lower to higher level mapping
CorrectCard>CorrectCardObs           # Lower to higher level mapping
Affect>AffectObs                     # Lower to higher level mapping
Choice>ChoiceObsHigher               # Lower to higher level mapping
Stage>StageObs                       # Lower to higher level mapping

## InitialParameterization
# Lower Level A Matrix (Likelihood) - 4 modalities x 5 factors
A_lower_0=[[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]]  # Advice modality
A_lower_1=[[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]]  # Feedback modality  
A_lower_2=[[1.0, 0.0], [0.0, 1.0]]                               # Arousal modality
A_lower_3=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # Choice modality

# Lower Level B Matrix (Transitions) - 5 factors x 5 factors x actions
B_lower_0=[[0.9, 0.1], [0.1, 0.9]]                               # Trustworthiness transitions
B_lower_1=[[0.9, 0.1], [0.1, 0.9]]                               # Correct card transitions
B_lower_2=[[0.3333, 0.6667], [0.6667, 0.3333]]                   # Affect transitions
B_lower_3=[[0.95, 0.025, 0.025], [0.025, 0.95, 0.025], [0.025, 0.025, 0.95]]  # Choice transitions
B_lower_4=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]  # Stage transitions

# Lower Level C Matrix (Preferences) - 4 modalities
C_lower_0=[0.3333, 0.3333, 0.3333]                               # Advice preferences
C_lower_1=[0.5, -3.5, 0.0]                                        # Feedback preferences
C_lower_2=[0.65, 0.35]                                             # Arousal preferences
C_lower_3=[0.3333, 0.3333, 0.3333]                               # Choice preferences

# Lower Level D Matrix (Priors) - 5 factors
D_lower_0=[0.5, 0.5]                                               # Trustworthiness priors
D_lower_1=[0.5, 0.5]                                               # Correct card priors
D_lower_2=[0.5, 0.5]                                               # Affect priors
D_lower_3=[0.0, 0.0, 1.0]                                         # Choice priors
D_lower_4=[1.0, 0.0, 0.0]                                         # Stage priors

# Lower Level E Matrix (Policy Priors) - 6 policies
E_lower=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]                           # Policy priors

# Higher Level A2 Matrix (Likelihood) - 5 modalities x 3 factors
A2_higher_0=[[0.667, 0.333], [0.333, 0.667]]                      # Trustworthiness observation mapping
A2_higher_1=[[0.5, 0.5], [0.5, 0.5]]                              # Correct card observation mapping
A2_higher_2=[[0.333, 0.667], [0.667, 0.333]]                      # Affect observation mapping
A2_higher_3=[[0.333, 0.333, 0.333], [0.333, 0.333, 0.333]]       # Choice observation mapping
A2_higher_4=[[0.333, 0.333, 0.333], [0.333, 0.333, 0.333]]       # Stage observation mapping

# Higher Level B2 Matrix (Transitions) - 3 factors x 3 factors x 1 action
B2_higher_0=[[1.0, 0.0], [0.0, 1.0]]                              # Safety self transitions
B2_higher_1=[[1.0, 0.0], [0.0, 1.0]]                              # Safety world transitions
B2_higher_2=[[1.0, 0.0], [0.0, 1.0]]                              # Safety other transitions

# Higher Level C2 Matrix (Preferences) - 5 modalities
C2_higher_0=[0.5, 0.5]                                             # Trustworthiness preferences
C2_higher_1=[0.5, 0.5]                                             # Correct card preferences
C2_higher_2=[1.0, 0.0]                                             # Affect preferences
C2_higher_3=[0.333, 0.333, 0.333]                                 # Choice preferences
C2_higher_4=[0.333, 0.333, 0.333]                                 # Stage preferences

# Higher Level D2 Matrix (Priors) - 3 factors
D2_higher_0=[0.25, 0.75]                                           # Safety self priors
D2_higher_1=[0.25, 0.75]                                           # Safety world priors
D2_higher_2=[0.25, 0.75]                                           # Safety other priors

# Higher Level E2 Matrix (Policy Priors) - No policies for higher level
E2_higher=[0.0]                                                     # No policy priors

# Learning Parameters
pA_lower=1.0                                                        # A matrix learning rate
pB_lower=1.0                                                        # B matrix learning rate
pD_lower=1.0                                                        # D matrix learning rate
pA2_higher=1.0                                                      # A2 matrix learning rate
pB2_higher=1.0                                                      # B2 matrix learning rate
pD2_higher=1.0                                                      # D2 matrix learning rate

# Model Parameters
p_advice=0.9                                                        # Probability of trustworthy advice
alpha=0.9                                                           # Precision of feedback mapping
p_Btrust=0.9                                                        # Trust transition probability
p_Bcorrectcard=0.9                                                  # Card belief persistence
p_Bchoice=0.95                                                      # Choice belief precision
p_Bstage=1.0                                                        # Stage transition determinism
cc=0.5                                                              # Correct feedback preference
arousal_low_preference=0.35                                         # Low arousal preference
trust_safety_association=0.667                                      # Trust-safety mapping strength
affect_safety_association=0.667                                     # Affect-safety mapping strength
prior_on_danger=0.75                                                # Higher-level danger bias
gamma=16.0                                                          # Policy precision
alpha_policy=16.0                                                   # Action selection precision

## Equations
# Lower Level Agent
F = D_KL[Q(s)||P(s|o)] - ln P(o)                                   # Free Energy
Q(s) = softmax(ln A + ln B + ln C + ln D)                          # Variational Message Passing
π* = argmin_π F(π)                                                  # Policy Selection

# Higher Level Agent
F_high = D_KL[Q(s_high)||P(s_high|o_high)] - ln P(o_high)         # Free Energy
Q(s_high) = softmax(ln A_high + ln B_high + ln C_high + ln D_high) # Hierarchical Inference
o_high = f(Q_low(s_low))                                            # Inter-level Coupling

## Time
time_horizon=2                                                       # Time horizon
time_step=1.0                                                        # Time step
time_units="timesteps"                                               # Time units
trial_structure=[null, advice, decision]                             # Trial structure

## ActInfOntologyAnnotation
Trustworthiness:trust                                                # Trust variable mapping
CorrectCard:card_state                                               # Card state variable mapping
Affect:emotional_state                                               # Emotional state variable mapping
Choice:action_selection                                              # Action selection variable mapping
Stage:game_stage                                                     # Game stage variable mapping
SafetySelf:self_safety                                               # Self safety variable mapping
SafetyWorld:world_safety                                             # World safety variable mapping
SafetyOther:other_safety                                             # Other safety variable mapping

## ModelParameters
parameters:
  p_advice: 0.9                                                     # Probability of trustworthy advice
  alpha: 0.9                                                        # Precision of feedback mapping
  p_Btrust: 0.9                                                     # Trust transition probability
  p_Bcorrectcard: 0.9                                               # Card belief persistence
  p_Bchoice: 0.95                                                   # Choice belief precision
  p_Bstage: 1.0                                                     # Stage transition determinism
  cc: 0.5                                                           # Correct feedback preference
  arousal_low_preference: 0.35                                      # Low arousal preference
  trust_safety_association: 0.667                                   # Trust-safety mapping strength
  affect_safety_association: 0.667                                  # Affect-safety mapping strength
  prior_on_danger: 0.75                                             # Higher-level danger bias
  gamma: 16.0                                                       # Policy precision
  alpha_policy: 16.0                                                # Action selection precision

## Footer
This GNN specification defines a hierarchical Active Inference agent for PTSD modeling, with explicit inter-level communication and learning dynamics. The model captures the interaction between sensorimotor processing and abstract safety assessment in trauma-related decision making.

## Signature
signature:
  hash: "sha256:..."
  timestamp: "2024-12-20T00:00:00Z"
  version: "1.0.0"
  author: "Cooper"
  validation_status: "draft" 