# IDENTIFY_COMPONENTS

Here's a systematic breakdown of the GNN specification:

1. **State Variables (Hidden States)**:
   - Variable names and dimensions
   - What each state represents conceptually
   - State space structure (discrete/continuous, finite/infinite)
   - Action variables are denoted as actions or states, but not explicitly stated in the document.

2. **Observation Variables**:
   - Observation modalities and their meanings
   - Sensor/measurement interpretations
   - Noise models or uncertainty characterization
   - Control policies and decision variables
   - Action space properties (e.g., action types)

3. **Action/Control Variables**:
   - Available actions are denoted as actions, but not explicitly stated in the document.
   - Actions can be thought of as a set of possible actions that can influence the state variable or control policy.

4. **Model Matrices**:
   - A matrices: Observation models P(o|s)
   - B matrices: Transition dynamics P(s'|s,u)
   - C matrices: Preferences/goals
   - D matrices: Prior beliefs over initial states
   - D matrix is a state space representation of the action-belief transition.

5. **Parameters and Hyperparameters**:
   - Precision parameters (γ, α, etc.) are denoted as precision weights or prior probabilities.
   - Learning rates and adaptation parameters are denoted as learning rate updates or update parameters.
   - Fixed vs. learnable parameters can be thought of as fixed values that determine the action-belief transition based on a specific choice of actions.

6. **Temporal Structure**:
   - Time horizons and temporal dependencies are not explicitly stated in the document, but they may represent time periods where certain actions or states influence each other.

This is a systematic breakdown:

1. **State Variables (Hidden States)**:
   - Variable names and dimensions
   - What each state represents conceptually
   - State space structure (discrete/continuous, finite/infinite)
   - Action variables are denoted as actions, but not explicitly stated in the document.

2. **Observation Variables**:
   - Observation modalities and their meanings
   - Sensor/measurement interpretations
   - Noise models or uncertainty characterization
   - Control policies and decision variables
   - Action space properties (e.g., action types)

3. **Action/Control Variables**:
   - Available actions are denoted as actions, but not explicitly stated in the document.
   - Actions can be thought of as a set of possible actions that can