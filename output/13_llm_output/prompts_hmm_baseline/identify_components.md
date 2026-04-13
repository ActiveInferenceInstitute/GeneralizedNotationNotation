# IDENTIFY_COMPONENTS

Here's a systematic breakdown of the GNN specification:

1. **State Variables (Hidden States)**:
   - Variable names and dimensions
   - What each state represents conceptually
   - State space structure (discrete/continuous, finite/infinite)
   - Action/control variables are represented as vectors in the state variable space.

2. **Observation Variables**:
   - Observation modalities and their meanings
   - Sensor/measurement interpretations
   - Noise models or uncertainty characterization

**State Variable Names:**

   - Variable names:
    - Hidden states (represented by 4 matrices)
    - Observation variables (represented as vectors in state variable space)

3. **Observation Variables**:
   - Observation modalities and their meanings
   - Sensor/measurement interpretations
   - Noise models or uncertainty characterization

**Action/Control Variables:**

   - Available actions and their effects
   - Control policies and decision variables
   - Action space properties

4. **Model Matrices**:

   - A matrices: Observation models P(o|s)
    - B matrices: Transition dynamics P(s'|s,u)
    - C matrices: Preferences/goals
    - D matrices: Prior beliefs over initial states

**Parameters and Hyperparameters:**

1. **Initial Variable Parameters**:
   - Precision parameters (γ, α, etc.)
   - Learning rates and adaptation parameters
   - Fixed vs. learnable parameters

2. **Learning Rates and Adaptation Parameters**:
   - Precision parameter (γ)
    - Learning rate
    - Adaptive threshold
3. **Fixed vs. Learnable Parameters**:
   - Precision parameter (γ)
    - Learning rate
    - Adaptive threshold