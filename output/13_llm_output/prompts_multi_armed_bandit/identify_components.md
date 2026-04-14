# IDENTIFY_COMPONENTS

Here's a systematic breakdown of the GNN specification:

1. **State Variables (Hidden States)**:
   - Variable names and dimensions
   - What each state represents conceptually
   - State space structure (discrete/continuous, finite/infinite)
   - Actions are actions that can be performed in different states (e.g., "pull arm 0", "pull arm 1")

2. **Observation Variables**:
   - Observation modalities and their meanings
   - Sensor/measurement interpretations
   - Noise models or uncertainty characterization

**State Space Structure:**

   - Discrete space: discrete time horizon, finite number of states (3)
   - Continuous space: continuous space with a fixed number of states (1)

3. **Action Variables**:
   - Available actions and their effects
   - Control policies and decision variables
   - Action space properties

**Model Matrices:**

   - A matrix representing the state-action relationship
   
   - B matrices representing the action/observation relationships
   
   - C matrices representing the control policies (policy vectors)

4. **Parameters and Hyperparameters**:

   - Precision parameters (γ, α, etc.)
   - Learning rates and adaptation parameters
   - Fixed vs. learnable parameters

**Temporal Structure:**

   - Time horizons and temporal dependencies
   - Dynamic vs. static components