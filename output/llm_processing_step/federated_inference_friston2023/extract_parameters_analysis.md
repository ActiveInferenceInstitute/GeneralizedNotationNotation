# Parameter Extraction and Configuration

**File:** federated_inference_friston2023.md

**Analysis Type:** extract_parameters

**Generated:** 2025-06-23T14:07:14.669175

---

Here is a systematic breakdown of the parameters extracted from the provided GNN specification for the "Federated Inference Multi-Agent Belief Sharing Model v1":

### 1. Model Matrices

#### A Matrices
- **Dimensions**:
  - A_vis_subject_a1, A_vis_subject_a2, A_vis_subject_a3: \(6 \times 9 \times 2 \times 2 \times 3\)
  - A_vis_center_a1, A_vis_center_a2, A_vis_center_a3: \(3 \times 9 \times 2 \times 2 \times 3\)
  - A_vis_left_a1, A_vis_left_a2, A_vis_left_a3: \(3 \times 9 \times 2 \times 2 \times 3\)
  - A_vis_right_a1, A_vis_right_a2, A_vis_right_a3: \(3 \times 9 \times 2 \times 2 \times 3\)
  - A_proprioceptive_a1, A_proprioceptive_a2, A_proprioceptive_a3: \(3 \times 3\)
  - A_comm_location: \(9 \times 9\)
  - A_comm_proximity: \(2 \times 2\)
  - A_comm_pose: \(2 \times 2\)

- **Structure and Interpretation**:
  - A matrices define the likelihood of observations given hidden states. They encode how sensory inputs (visual and auditory) relate to the hidden states of agents, reflecting their perception of the environment.

#### B Matrices
- **Dimensions**:
  - B_location: \(9 \times 9 \times 3\)
  - B_proximity: \(2 \times 2 \times 1\)
  - B_pose: \(2 \times 2 \times 1\)
  - B_gaze_a1, B_gaze_a2, B_gaze_a3: \(3 \times 3 \times 3\)

- **Structure and Interpretation**:
  - B matrices represent the transition dynamics of hidden states over time. They capture how states evolve based on actions taken by agents or environmental influences.

#### C Matrices
- **Dimensions**:
  - C_vis_foveal: \(3\)
  - C_vis_contrast: \(3\)
  - C_comm_location: \(9\)
  - C_comm_proximity: \(2\)
  - C_comm_pose: \(2\)
  - C_proprioceptive: \(3\)

- **Structure and Interpretation**:
  - C matrices are preference vectors that encode the agents' biases or preferences over different observations, influencing their decision-making processes.

#### D Matrices
- **Dimensions**:
  - D_location: \(9\)
  - D_proximity: \(2\)
  - D_pose: \(2\)
  - D_gaze_a1: \(3\)
  - D_gaze_a2: \(3\)
  - D_gaze_a3: \(3\)

- **Structure and Interpretation**:
  - D matrices represent prior distributions over initial states, reflecting the agents' beliefs about their starting conditions before any observations are made.

### 2. Precision Parameters
- **γ (gamma)**:
  - η_precision: \(4.0\) (Precision of belief sharing communication)
  - γ_attention: \(2.0\) (Attentional precision for epistemic foraging)

- **α (alpha)**:
  - Learning rate: \(0.1\) (Rate of adaptation in belief updates)

- **Other Precision/Confidence Parameters**:
  - τ_learning: \(32.0\) (Forgetting timescale, indicating how many experiences are retained)

### 3. Dimensional Parameters
- **State Space Dimensions**:
  - num_locations: \(9\)
  - num_proximity_levels: \(2\)
  - num_pose_states: \(2\)
  - num_gaze_directions: \(3\)

- **Observation Space Dimensions**:
  - num_visual_foveal_outcomes: \(3\)
  - num_visual_contrast_outcomes: \(3\)
  - num_proprioceptive_outcomes: \(3\)
  - num_communication_outcomes: \([9, 2, 2]\)

- **Action Space Dimensions**:
  - num_gaze_actions: \(3\)

### 4. Temporal Parameters
- **Time Horizons (T)**:
  - ModelTimeHorizon: Unbounded (Continuous operation)

- **Temporal Dependencies and Windows**:
  - Temporal dependencies are defined through the transition matrices (B matrices) that dictate how states evolve over time.

- **Update Frequencies and Timescales**:
  - τ_learning: \(32.0\) (Indicates the frequency of learning updates)

### 5. Initial Conditions
- **Prior Beliefs over Initial States**:
  - D_location: Uniform distribution over \(9\) locations.
  - D_proximity: Uniform distribution over \(2\) proximity levels.
  - D_pose: Uniform distribution over \(2\) pose states.
  - D_gaze_a1, D_gaze_a2, D_gaze_a3: Slight preference for center gaze.

- **Initial Parameter Values**:
  - Values for transition matrices (B matrices) and likelihood matrices (A matrices) are defined in the parameterization section.

- **Initialization Strategies**:
  - Uniform distributions for initial states ensure that agents start with no bias towards any specific state.

### 6. Configuration Summary
- **Parameter File Format Recommendations**:
  - Use structured formats (e.g., JSON, YAML) to maintain clarity and ease of access for parameter tuning.

- **Tunable vs. Fixed Parameters**:
  - Tunable: Learning rates, precision parameters, and preference vectors.
  - Fixed: Dimensions of state and observation spaces, structural definitions of matrices.

- **Sensitivity Analysis Priorities**:
  - Focus on the impact of precision parameters (η_precision, γ_attention) and learning rates on model performance and convergence behavior.

This systematic breakdown provides a comprehensive overview of the parameters and their roles within the federated inference model, facilitating a deeper understanding of the model's structure and functionality in the context of Active Inference.

---

*Analysis generated using LLM provider: openai*
