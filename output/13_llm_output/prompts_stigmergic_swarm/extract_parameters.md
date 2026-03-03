# EXTRACT_PARAMETERS

Okay, let’s meticulously extract and organize the parameters from this GNN specification for the “Stigmergic Swarm Active Inference” model. This will be a detailed breakdown, aligning with your requested structure.

**1. Model Matrices:**

*   **A Matrices (Likelihood Matrices):**
    *   Dimensions: 4x9x9 for each agent (A1, A2, A3). This represents the likelihood of observing a particular observation (o1, o2, o3) given the agent’s position on the 3x3 grid and the environmental signal.
    *   Structure:  Each A matrix is a 3x3 matrix representing the probability of observing each of the four observation types (empty, signal_low, signal_high, goal) at each of the nine grid cells.
    *   Interpretation: These matrices quantify the agent’s belief about the environment given its location. Higher values indicate a greater probability of observing the corresponding observation.
*   **B Matrices (Transition Matrices):**
    *   Dimensions: 9x9x4 for each agent (B1, B2, B3). This represents the probability of transitioning to a new position on the 3x3 grid given the agent’s current position and the chosen action.
    *   Structure: Each B matrix is a 9x9 matrix, where each element (i, j, k) represents the probability of taking action *k* (N/S/E/W) and moving to cell *j* on the grid.
    *   Interpretation: These matrices define the agent’s movement dynamics.
*   **C Matrices (Preference Vectors):**
    *   Dimensions: 4x4 for each agent (C1, C2, C3). This represents the agent’s preference for each of the four observation types.
    *   Structure: Each C matrix is a 4-dimensional vector, where each element represents the relative preference for the corresponding observation type.
    *   Interpretation: These vectors define the agent’s goals and preferences, guiding its actions.
*   **D Matrices (Position Priors):**
    *   Dimensions: 9x1x1 for each agent (D1, D2, D3). This represents the agent’s prior belief about its position on the 3x3 grid.
    *   Structure: Each D matrix is a 9