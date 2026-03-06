# IDENTIFY_COMPONENTS

Okay, let's meticulously dissect this GNN specification for the Hierarchical Active Inference POMDP. This breakdown will focus on identifying and classifying each component, aligning with your requested emphasis on Active Inference concepts and mathematical relationships.

**1. State Variables (Hidden States)**

*   **Level 1 (Fast Dynamics):**
    *   `s1[4,1,type=float]` - Hidden State: Represents the internal state of the agent at Level 1. Dimension: 4 hidden states.  This is the core hidden state of the fast dynamics, driving the primary inference loop.
    *   `s1_prime[4,1,type=float]` - Next Hidden State:  The predicted next state given the current state and action.  This is a key component of the Active Inference framework, representing the agent's best guess about its future state.
*   **Level 2 (Slow Dynamics):**
    *   `s2[2,1,type=float]` - Contextual Hidden State: Represents the agent's contextual state at Level 2. Dimension: 2 contextual states. This state modulates the likelihood function at Level 1, introducing context-dependent behavior.

**2. Observation Variables**

*   **Level 1:**
    *   `o1[4,1,type=int]` - Observation: The agent's sensory input. Dimension: 4 observations. These observations are generated based on the Level 1 hidden state (`s1`) and the observation model (`A1`).
*   **Level 2:**
    *   `o2[4,1,type=float]` - Higher-Level Observation:  This is a distribution over the Level 1 hidden states, derived from the Level 2 context (`s2`). It effectively represents the agent's belief about the underlying state at Level 1, given the context.

**3. Action/Control Variables**

*   `π1[3,type=float]` - Policy Vector: The agent’s action selection probabilities. Dimension: 3 actions. This defines the policy that the agent uses to select actions.
*   `u1[1,type=int]` - Action: The selected action at Level 1. Dimension: 1 action.
*   `B1[4,4,3,type=float]` - Transition Matrix: This matrix governs the transitions between the Level 1 hidden