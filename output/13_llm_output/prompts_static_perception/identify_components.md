# IDENTIFY_COMPONENTS

Based on the provided documentation, here is a systematic breakdown of the active inference capabilities:

1. **State Variables (Hidden States)**:
   - Variable names and dimensions
   - What each state represents conceptually
   - State space structure (discrete/continuous, finite/infinite)

2. **Observation Variables**:
   - Observation modalities and their meanings
   - Sensor/measurement interpretations
   - Noise models or uncertainty characterization

3. **Action/Control Variables**:
   - Available actions and their effects
   - Control policies and decision variables
   - Action space properties

4. **Model Matrices**:
   - A matrices: Observation models P(o|s)
   - B matrices: Transition dynamics P(s'|s,u)
   - C matrices: Preferences/goals
   - D matrices: Prior beliefs over initial states

Here's a step-by-step breakdown of the key concepts and parameters involved in active inference:

1. **State Variables (Hidden States)**:
   - Variable names and dimensions
   - What each state represents conceptually
   - State space structure (discrete/continuous, finite/infinite)

2. **Observation Variables**:
   - Observation modalities and their meanings
   - Sensor/measurement interpretations
   - Noise models or uncertainty characterization

Key parameters:

   - **Precision parameter γ** (default value 0.1): This parameter controls the precision of predictions made by the model. It is set to zero when no action is taken, indicating that there are no actions available for prediction.
   - **Learning rate α**: This parameter determines how quickly the model learns from data and updates its parameters based on new information. It can be adjusted using the `learn_rate` argument in the `model()` function.
   - **Fixed vs. learnable parameters**: These parameters control the learning rate, which is a measure of how well the model converges to the optimal solution. They are set by default and can change based on user input or external data.
   - **Learning rates**: These parameters determine how quickly the model learns from new data points. They are adjusted using the `learn_rate` argument in the `model()` function, which is a parameter that controls the rate at which the model updates its parameters.

3. **Model Matrices**:
   - A matrices: Observation models P(o|s)
   - B matrices: Transition dynamics P(s'|s,u)
   - C matrices: Preferences/goals