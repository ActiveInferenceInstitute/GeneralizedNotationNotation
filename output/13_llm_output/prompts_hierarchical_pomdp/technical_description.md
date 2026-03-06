# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

Okay, this is a remarkably detailed specification of a Hierarchical Active Inference POMDP model implemented using a GNN. Let’s break down this technical description comprehensively, focusing on the key elements and implications for a GNN implementation.

**Overall Architecture & Core Idea**

This model elegantly combines the strengths of Active Inference with a hierarchical structure. It’s designed to represent scenarios where a fast, reactive process (Level 1) interacts with a slower, contextual process (Level 2). The key innovation is the use of a GNN to represent and update these beliefs and actions, allowing for efficient integration of information across the two levels. The top-down prediction constraint is a crucial element for grounding the fast dynamics within a broader, contextualized belief.

**GNN Specification Breakdown**

1. **GNN Version & Flags:** `GNN v1` – Indicates a specific version of the GNN implementation. This is important for reproducibility and tracking changes.

2. **Model Components:**
   * **Two Levels:** The model is explicitly divided into two levels:
      * **Level 1 (Fast):**  This is the primary active inference component. It’s a standard POMDP with 4 hidden states, 4 observations, and 3 actions. The dynamics are represented by matrices `A1`, `B1`, `C1`, and `D1`.
      * **Level 2 (Slow):** This level introduces contextual information. It has 2 contextual states and modulates the likelihood of the Level 1 hidden states. It’s represented by matrices `A2`, `B2`, `C2`, and `D2`.

3. **State Representation:**
   * **`s1` & `s2`:** These are the hidden state distributions for Level 1 and Level 2, respectively. They are represented as vectors of probabilities.
   * **`o1` & `o2`:** These are the observations, representing the outcome of the fast and slow processes, respectively.
   * **`π1` & `s2`:** These are the actions and contextual states, respectively.

4. **GNN Layers & Connections (Crucial for GNN Implementation):**
   * **Node Representation:** Each state (`s1`, `s2`, `o1`, `o2`) is represented as a node in the GNN. The initial values in `A1`, `B1`, `C1`, `