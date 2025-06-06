# GNN Example: Cognitive Effort Model
# Format: Markdown representation of a Cognitive Effort model using Active Inference formalism
# Version: 1.0
# This file is machine-readable and represents a generative model for cognitive effort in Active Inference

## GNNSection
CognitiveEffortModel

## GNNVersionAndFlags
GNN v1

## ModelName
Cognitive Effort: Stroop Task Model v1.0

## ModelAnnotation
This model represents a computational formulation of cognitive effort within the Active Inference framework.
Key features:
- Cognitive effort is defined as the divergence between context-sensitive beliefs about how to act (G) and context-insensitive prior beliefs (E)
- The model simulates the classic Stroop task with a two-level hierarchical structure (slow/fast timescales)
- Slow level variables: narrative state, instruction state, response modality (mental action)
- Fast level variables: task sequence, font color, written word, correctness state
- Parameters include prior bias (E) toward habitual responses and prior preferences (C) for being correct
- The strength of effort deployed depends on the relationship between preferences and habits
Based on: "Cognitive effort: A neurocomputational account" (Parr, Holmes, Friston & Pezzulo)

## StateSpaceBlock
# Slow Level (Higher level) Variables
s_sl_narrative[2,1,type=int]     # Narrative state: instruction (0) or response (1) context
s_sl_instruction[2,1,type=int]   # Instruction state: read word (0) or report color (1)
s_sl_response[2,1,type=int]      # Response modality: word-reading (0) or color-naming (1) - POLICY-DEPENDENT

# Fast Level (Lower level) Variables
s_fl_task[3,1,type=int]          # Task sequence: instruction (0), viewing (1), response (2) phases
s_fl_color[4,1,type=int]         # Font color: red (0), green (1), blue (2), yellow (3)
s_fl_word[4,1,type=int]          # Written word: "red" (0), "green" (1), "blue" (2), "yellow" (3)
s_fl_correct[2,1,type=int]       # Correctness state: incorrect (0), correct (1)

# Observations
o_instruction[2,1,type=int]      # Instruction: read word (0) or report color (1)
o_color[4,1,type=int]            # Visual observation of color: red (0), green (1), blue (2), yellow (3)
o_word[4,1,type=int]             # Visual observation of word: "red" (0), "green" (1), "blue" (2), "yellow" (3)
o_response[4,1,type=int]         # Verbal response: "red" (0), "green" (1), "blue" (2), "yellow" (3)

# Policy and Effort
pi[2,type=float]                 # Policy (distribution over response modality): word-reading (0) or color-naming (1)
G[2,type=float]                  # Expected Free Energy for each policy
E[2,type=float]                  # Prior bias/potential for each policy (context-insensitive)
xi[1,type=float]                 # Cognitive effort

# Likelihood Matrices
A_instruction[2,2,type=float]    # P(o_instruction | s_sl_instruction) 
A_color[4,4,type=float]          # P(o_color | s_fl_color)
A_word[4,4,type=float]           # P(o_word | s_fl_word)
A_response[4,4,4,2,type=float]   # P(o_response | s_fl_color, s_fl_word, s_sl_response)

# Transition Matrices
B_narrative[2,2,type=float]      # P(s_sl_narrative' | s_sl_narrative)
B_instruction[2,2,type=float]    # P(s_sl_instruction' | s_sl_instruction)
B_task[3,3,2,type=float]         # P(s_fl_task' | s_fl_task, s_sl_narrative)

# Preferences and Priors
C_correct[2,type=float]          # Preferences over correctness states: log P(s_fl_correct)
D_narrative[2,type=float]        # Prior over narrative states
D_instruction[2,type=float]      # Prior over instruction states
D_task[3,type=float]             # Prior over task sequence states
D_color[4,type=float]            # Prior over font colors
D_word[4,type=float]             # Prior over written words

# Time
t[1,type=int]                    # Time step

## Connections
# Slow level connections
(D_narrative) -> (s_sl_narrative)
(D_instruction) -> (s_sl_instruction)
(E) > (pi)                       # Prior bias influences policy selection
(G) > (pi)                       # Expected free energy influences policy selection 
(pi) -> (s_sl_response)          # Policy determines response modality (mental action)

# Fast level connections
(D_task) -> (s_fl_task)
(D_color) -> (s_fl_color)
(D_word) -> (s_fl_word)
(s_sl_instruction, s_sl_response) -> (s_fl_correct)  # Correctness depends on instruction/response match

# Observations connections
(s_sl_instruction) -> (A_instruction) -> (o_instruction)
(s_fl_color) -> (A_color) -> (o_color)
(s_fl_word) -> (A_word) -> (o_word)
(s_fl_color, s_fl_word, s_sl_response) -> (A_response) -> (o_response)

# Expected Free Energy and Cognitive Effort connections
(C_correct, s_fl_correct) > (G)  # Preferences influence expected free energy
(G, E) > (xi)                    # Cognitive effort is the divergence between G and E

## InitialParameterization
# Prior bias (E) - 85% bias toward word-reading (0), 15% toward color-naming (1)
E={(ln(0.85), ln(0.15))}  # Represented as log probabilities

# Preferences (C) - preference for being correct is e^2 (≈7.4) times more probable than being incorrect
C_correct={(ln(1/8.4), ln(7.4/8.4))}  # Normalized log probabilities for incorrect (0) vs correct (1)

# Likelihoods
A_instruction={
  ((1.0, 0.0), (0.0, 1.0))  # Identity mapping for instruction observation
}

A_color={
  ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),  # color=0 (red)
  ((0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),  # color=1 (green)
  ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 0.0)),  # color=2 (blue)
  ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))   # color=3 (yellow)
}

A_word={
  ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),  # word=0 ("red")
  ((0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),  # word=1 ("green")
  ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 0.0)),  # word=2 ("blue")
  ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))   # word=3 ("yellow")
}

# Response depends on color when response modality is color-naming (1), and on word when response modality is word-reading (0)
# A_response is complex (4x4x4x2) so we only specify the general mapping rules:
# If s_sl_response=0 (word-reading): o_response = s_fl_word (regardless of color)
# If s_sl_response=1 (color-naming): o_response = s_fl_color (regardless of word)

# Initial state priors
D_narrative={(1.0, 0.0)}         # Start in instruction context
D_instruction={(0.5, 0.5)}       # Equal probability for either instruction
D_task={(1.0, 0.0, 0.0)}         # Start in instruction phase
D_color={(0.25, 0.25, 0.25, 0.25)}  # Uniform prior over colors
D_word={(0.25, 0.25, 0.25, 0.25)}   # Uniform prior over words

## Equations
# 1. Policy selection (with temperature parameter λ = 0.25)
pi = σ(-G - E)

# 2. Action selection (response is based on predicted outcomes)
P(o_response) = σ(λ * E_π[ln P(o_response | s_fl_color, s_fl_word, s_sl_response)])

# 3. Cognitive effort (divergence between context-sensitive and context-insensitive priors)
xi = D_KL[Cat(σ(-G)) || Cat(σ(-E))]

# 4. Expected Free Energy
G(π) = E_q(s_fl_correct|π)[ln q(s_fl_correct|π) - ln P(s_fl_correct)]

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=10  # Example for a sequence of Stroop trials

## ActInfOntologyAnnotation
s_sl_narrative=HiddenStateSlowLevelNarrative
s_sl_instruction=HiddenStateSlowLevelInstruction
s_sl_response=HiddenStateSlowLevelResponseModality
s_fl_task=HiddenStateFastLevelTaskSequence
s_fl_color=HiddenStateFastLevelFontColor
s_fl_word=HiddenStateFastLevelWrittenWord
s_fl_correct=HiddenStateFastLevelCorrectness
o_instruction=ObservationInstruction
o_color=ObservationColor
o_word=ObservationWord
o_response=ObservationResponse
pi=PolicyVector
G=ExpectedFreeEnergy
E=PriorPotentialEnergy
xi=CognitiveEffort
A_instruction=LikelihoodMatrixInstruction
A_color=LikelihoodMatrixColor
A_word=LikelihoodMatrixWord
A_response=LikelihoodMatrixResponse
B_narrative=TransitionMatrixNarrative
B_instruction=TransitionMatrixInstruction
B_task=TransitionMatrixTask
C_correct=LogPreferenceVectorCorrectness
D_narrative=PriorOverNarrativeStates
D_instruction=PriorOverInstructionStates
D_task=PriorOverTaskSequenceStates
D_color=PriorOverColorStates
D_word=PriorOverWordStates
t=Time

## ModelParameters
num_hidden_states_slow: [2, 2, 2]  # narrative, instruction, response
num_hidden_states_fast: [3, 4, 4, 2]  # task, color, word, correct
num_obs_modalities: [2, 4, 4, 4]  # instruction, color, word, response
num_policies: 2  # word-reading, color-naming

## Footer
Cognitive Effort: Stroop Task Model v1.0 - End of Specification

## Signature
Creator: AI Assistant based on "Cognitive effort: A neurocomputational account" (Parr et al.)
Date: 2024-07-26
Status: Example for demonstration of cognitive effort modeling in the GNN framework. 