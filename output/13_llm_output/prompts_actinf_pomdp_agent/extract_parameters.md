# EXTRACT_PARAMETERS

Based on the provided specifications, here are the key parameters for Active Inference POMDP Agent:

1. **Model Matrices**:
   - A matrices representing the model's structure and interpretation of the input data (e.g., Likelihood Matrix).
   - B matrices representing the action-belief relationships between actions and states.
   - C matrices representing the policy prior, habit distribution, and initial preferences for each mode.
   - D matrices representing the transition probabilities over actions and their corresponding beliefs.

2. **Precision Parameters**:
   - γ (gamma): precision parameters and their roles in determining the agent's performance on the input data.
   - α (alpha): learning rate parameter to adjust the accuracy of predictions based on the model's training data.
   - Other precision/confidence parameters, such as γ or α

3. **Dimensional Parameters**:
   - State space dimensions for each factor:
      - A matrices representing the state-action relationships between actions and states.
      - B matrices representing the action-belief relationships over actions and their corresponding beliefs.
      - C matrices representing the policy prior distribution over actions, with a probability of transitioning from one state to another based on the current state.
      - D matrices representing the transition probabilities over actions, with a probability of transitioning between different states based on the current state.

4. **Temporal Parameters**:
   - Time horizons (T)
   - Temporal dependencies and windows:
      - Time horizon for each mode: 1 step
       - Temporal window size: 30
       - Temporal window length: 256
    - Update frequencies and timescales:
      - Update frequency: 1.0e-4
       - Update time: 1.0e-9

5. **Initial Conditions**:
   - Initial parameters for each mode:
      - Initial belief over initial states: 3x3 matrices representing the initial beliefs of all modes.
      - Initial state and action probabilities: 2x2 matrices representing the initial actions, with a probability distribution over actions based on the current state.
    - Initialization strategies:
      - Random initialization (randomly initialize parameters):
          - Initial belief over initial states: 3x3 matrices representing the initial beliefs of all modes.
          - Initial action probabilities: 2x2 matrices representing the initial actions, with a probability distribution over actions based on the current state.
    - Initialization strategies:
      - Random initialization (randomly initialize parameters):
          - Initial belief over initial states: