## GNNVersionAndFlags
Version: 1.0

## ModelName
PTSD Hierarchical Active Inference Agent

## ModelAnnotation
This hierarchical Active Inference agent consists of two levels:

- **Lower Level Agent**: Processes sensorimotor information including trustworthiness, card states, affect, choices, and game stages
- **Higher Level Agent**: Processes abstract safety concepts (self, world, other) based on lower-level posteriors

The agents communicate bidirectionally: lower-level posteriors become higher-level observations, and higher-level inferred states become lower-level priors.

## StateSpaceBlock
Trustworthiness[2],float
CorrectCard[2],float
Affect[2],float
Choice[3],float
Stage[3],float
Advice[3],float
Feedback[3],float
Arousal[2],float
ChoiceObs[3],float
TrustActions[2],float
CardActions[3],float
NullActions[1],float
SafetySelf[2],float
SafetyWorld[2],float
SafetyOther[2],float
TrustworthinessObs[2],float
CorrectCardObs[2],float
AffectObs[2],float
ChoiceObsHigher[3],float
StageObs[3],float
NullActionsHigher[1],float

## Connections
Trustworthiness>Advice
CorrectCard>Feedback
Affect>Arousal
Choice>ChoiceObs
Stage>Advice
SafetySelf>TrustworthinessObs
SafetyWorld>CorrectCardObs
SafetyOther>AffectObs
Trustworthiness>TrustworthinessObs
CorrectCard>CorrectCardObs
Affect>AffectObs
Choice>ChoiceObsHigher
Stage>StageObs

## InitialParameterization
A_lower_0 = [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]]
A_lower_1 = [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]]
A_lower_2 = [[1.0, 0.0], [0.0, 1.0]]
A_lower_3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
B_lower_0 = [[0.9, 0.1], [0.1, 0.9]]
B_lower_1 = [[0.9, 0.1], [0.1, 0.9]]
B_lower_2 = [[0.3333, 0.6667], [0.6667, 0.3333]]
B_lower_3 = [[0.95, 0.025, 0.025], [0.025, 0.95, 0.025], [0.025, 0.025, 0.95]]
B_lower_4 = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
C_lower_0 = [0.3333, 0.3333, 0.3333]
C_lower_1 = [0.5, -3.5, 0.0]
C_lower_2 = [0.65, 0.35]
C_lower_3 = [0.3333, 0.3333, 0.3333]
D_lower_0 = [0.5, 0.5]
D_lower_1 = [0.5, 0.5]
D_lower_2 = [0.5, 0.5]
D_lower_3 = [0.0, 0.0, 1.0]
D_lower_4 = [1.0, 0.0, 0.0]
E_lower = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
A2_higher_0 = [[0.667, 0.333], [0.333, 0.667]]
A2_higher_1 = [[0.5, 0.5], [0.5, 0.5]]
A2_higher_2 = [[0.333, 0.667], [0.667, 0.333]]
A2_higher_3 = [[0.333, 0.333, 0.333], [0.333, 0.333, 0.333]]
A2_higher_4 = [[0.333, 0.333, 0.333], [0.333, 0.333, 0.333]]
B2_higher_0 = [[1.0, 0.0], [0.0, 1.0]]
B2_higher_1 = [[1.0, 0.0], [0.0, 1.0]]
B2_higher_2 = [[1.0, 0.0], [0.0, 1.0]]
C2_higher_0 = [0.5, 0.5]
C2_higher_1 = [0.5, 0.5]
C2_higher_2 = [1.0, 0.0]
C2_higher_3 = [0.333, 0.333, 0.333]
C2_higher_4 = [0.333, 0.333, 0.333]
D2_higher_0 = [0.25, 0.75]
D2_higher_1 = [0.25, 0.75]
D2_higher_2 = [0.25, 0.75]
E2_higher = [0.0]
pA_lower = 1.0
pB_lower = 1.0
pD_lower = 1.0
pA2_higher = 1.0
pB2_higher = 1.0
pD2_higher = 1.0
p_advice = 0.9
alpha = 0.9
p_Btrust = 0.9
p_Bcorrectcard = 0.9
p_Bchoice = 0.95
p_Bstage = 1.0
cc = 0.5
arousal_low_preference = 0.35
trust_safety_association = 0.667
affect_safety_association = 0.667
prior_on_danger = 0.75
gamma = 16.0
alpha_policy = 16.0

## Equations
$$F = D_KL[Q(s)||P(s|o)] - ln P(o)                                   # Free Energy Q(s) = softmax(ln A + ln B + ln C + ln D)                          # Variational Message Passing π* = argmin_π F(π)                                                  # Policy Selection$$

$$F_high = D_KL[Q(s_high)||P(s_high|o_high)] - ln P(o_high)         # Free Energy Q(s_high) = softmax(ln A_high + ln B_high + ln C_high + ln D_high) # Hierarchical Inference o_high = f(Q_low(s_low))                                            # Inter-level Coupling$$

## Time
Static

## Footer
Generated: 2025-07-25T22:44:10.760465

## Signature
