# IDENTIFY_COMPONENTS

Based on the document, here is a systematic breakdown of the key components in the GNN specification:

1. **State Variables (Hidden States)**:
   - Variable names and dimensions
   - What each state represents conceptually
   - State space structure (discrete/continuous, finite/infinite)
   - Observation modalities and their meanings
2. **Observation Variables**:
   - Variable names and dimensions
   - What each observation represents conceptually
   - Observation modality and interpretation

3. **Action/Control Variables**:
   - Available actions and their effects
   - Control policies and decision variables
   - Action space properties

4. **Model Matrices**:
   - A matrices: Observation models P(o|s)
   - B matrices: Transition dynamics P(s'|s,u)
   - C matrices: Preferences/goals
   - D matrices: Prior beliefs over initial states
5. **Parameters and Hyperparameters**:
   - Precision parameters (γ, α, etc.)
   - Learning rates and adaptation parameters

6. **Temporal Structure**:
   - Time horizons and temporal dependencies
   - Dynamic vs. static components