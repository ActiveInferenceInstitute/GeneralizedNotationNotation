# IDENTIFY_COMPONENTS

Okay, let's meticulously dissect this GNN specification for the “Multi-Agent Cooperative Active Inference” model. This is a well-structured GNN representation, and we can break it down into its core components, aligning with your requested framework.

**1. State Variables (Hidden States)**

*   **Agent 1 (A1)**:
    *   `s1[4,1,type=float]` – Agent 1 Hidden State: A 4-dimensional hidden state representing the agent's internal belief about its position and potentially other relevant variables. The `[4,1]` indicates a vector of length 4, and the `type=float` specifies the data type.
    *   `o1[4,1,type=int]` – Agent 1 Observation: A 4-dimensional observation vector. The `[4,1]` indicates a vector of length 4, and the `type=int` specifies the data type. This observation includes information about Agent 2’s actions.
*   **Agent 2 (A2)**:
    *   `s2[4,1,type=float]` – Agent 2 Hidden State:  Identical in structure and meaning to `s1`.
    *   `o2[4,1,type=int]` – Agent 2 Observation: Identical in structure and meaning to `o1`.
*   **Shared Environment State (s_joint[16,1,type=float])**:
    *   `s_joint`: Represents the joint state of the environment, defined by the positions of both agents. The `[16,1]` indicates a vector of length 16, and the `type=float` specifies the data type. This is likely a discretized joint state space (e.g., 4x4 = 16 possible configurations).

**State Space Structure:** The state space is discrete, with 16 possible joint configurations (as implied by the 4x4 grid). Each agent’s hidden state `s1` and `s2` are 4-dimensional, representing a belief about its own position within this 16-state space.

**2. Observation Variables**

*   **Agent 1 Observation (o1)**:  This is a crucial element. It’s a 4-dimensional vector, and importantly, it *includes* Agent 2’s last action