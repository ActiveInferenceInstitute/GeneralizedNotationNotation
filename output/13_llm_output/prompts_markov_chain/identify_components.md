# IDENTIFY_COMPONENTS

Here's a systematic breakdown of the GNN specification:

1. **State Variables (Hidden States)**:
   - Variable names and dimensions
   - What each state represents conceptually
   - State space structure (discrete/continuous, finite/infinite)
   - The number of states is 3, but it can also be represented as a single state with 2 hidden states.

2. **Observation Variables**:
   - Observation modalities and their meanings
   - Sensor/measurement interpretations
   - Noise models or uncertainty characterization
   - Action/control variables (optional)
   - Action space properties

**State Variable Types:**

   - **Initial State**: A discrete-time state with no actions, no observable states.
   - **Observation**: A continuous-time observation of the same initial state.

3. **Action/Control Variables**:
   - **Actions**: Actions can be either passive (no action) or active (action selection).
   - **Goals** are actions that affect the system's behavior, but not directly observable states.
   - **Prior Beliefs** represent prior beliefs about future states and actions.

4. **Model Matrices**:
   - **Initial State Matrix**: A discrete-time state with no action or observation space.
   - **Observation Matrix**: A continuous-time observation of the same initial state, but with a fixed number of hidden states (1) and an action space that is finite in size.

5. **Parameters and Hyperparameters**:
   - **Learning Rates** are hyperparameter values to control the rate at which the model learns from data.
   - **Adaptive Parameters** can be adjusted based on the amount of training data, allowing for more flexible learning algorithms.
   - **Fixed vs. Learnable Parameters** allow for a fixed number of hidden states and actions without adjusting parameters.

6. **Temporal Structure**:
   - **Time Horizon**: A discrete-time time horizon with a finite number of timesteps (40) to control the rate at which the model learns from data.

This is a concise overview, but keep in mind that this is just one possible representation for GNNs and other Markov Chain models. There are many different representations available depending on the specific implementation choices.