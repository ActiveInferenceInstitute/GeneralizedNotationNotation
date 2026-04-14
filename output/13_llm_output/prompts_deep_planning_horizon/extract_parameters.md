# EXTRACT_PARAMETERS

Based on the document, here is a systematic parameter breakdown for the GNN implementation:

1. **Model Matrices**:
   - A matrices: dimensions, structure, interpretation
   - B matrices: dimensions, structure, interpretation
   - C matrices: dimensions, structure, interpretation
   - D matrices: dimensions, structure, interpretation

**Initial Parameters:**

   - **A (Likelihood Matrix):**
    - Number of hidden states: 4
    - Number of actions: 4
    - Number of timesteps: 30
    - Initial parameter values:
      - γ = 1.5
      - α = 0.25
   - **B (Transition Matrix):**
    - Number of actions: 4
    - Number of observations: 4
    - Number of policies: 64
    - Number of timesteps: 30
    - Initial parameter values:
      - γ = 1.5
      - α = 0.25
   - **C (Policy Prior Vector):**
    - Number of actions: 4
    - Number of observations: 4
    - Number of policies: 64
    - Number of timesteps: 30
    - Initial parameter values:
      - γ = 1.5
      - α = 0.25
   - **D (Prior Over Hidden States):**
    - Number of actions: 4
    - Number of observations: 4
    - Number of policies: 64
    - Number of timesteps: 30
    - Initial parameter values:
      - γ = 1.5
      - α = 0.25
   - **E (Policy Prior Vector):**
    - Number of actions: 4
    - Number of observations: 4
    - Number of policies: 64
    - Number of timesteps: 30
    - Initial parameter values:
      - γ = 1.5
      - α = 0.25
   - **G (Policy Prior Vector):**
    - Number of actions: 4
    - Number of observations: 4
    - Number of policies: 64
    - Number of timesteps: 30
    - Initial parameter values:
      - γ = 1.5
      - α = 0.25
   - **G (Initial Policy Vector):**
    - Number of actions: 4
    - Number of observations: 4
    - Number of policies: 64
    - Number of timesteps: 30
   