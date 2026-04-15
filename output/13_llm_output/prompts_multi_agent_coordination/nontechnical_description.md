# Nontechnical Description

Prompt:

> Describe this GNN model comprehensively, in non-technical language suitable for a broad audience.

Response:

You can use the following code to generate a GNN model:

1. Create an instance of `MultiAgentCooperativeActiveInference` class with two agents and shared observation space. The agent 1 is initialized with a probability distribution over its own state, while the agent 2 has a probability distribution over its own hidden states.
2. Initialize the initial parameters for the GNN model:
   - `A1`: Probability of Agent 1's last action (probability = 0.85)
   - `B1`: Probability of Agent 1's next hidden state
   - `C1`: Probability of Agent 1's policy vector (probability = 0.05)
   - `D1`: Probability of Agent 1's prior over the last action
3. Initialize the initial parameters for the GNN model:
   - `A2`: Probability of Agent 2's last action (probability = 0.85)
   - `B2`: Probability of Agent 2's next hidden state
   - `C2`: Probability of Agent 2's policy vector (probability = 0.1)
   - `D2`: Probability of Agent 2's prior over the last action
4. Initialize the initial parameters for the GNN model:
   - `A1` and `B1`: Probability of Agent 1's last action (probability = 0.85)
   - `C1` and `D1`: Probability of Agent 1's next hidden state
   - `S_joint` and `G2`: Probability of Agent 2's last action (probability = 0.9)
5. Initialize the initial parameters for the GNN model:
   - `A1`, `B1`, and `C1`: Probability of Agent 1's last action (probability = 0.85)
   - `D1` and `G2`: Probability of Agent 2's last action (probability = 0.9)
6. Initialize the initial parameters for the GNN model:
   - `A2`, `B2`, and `C2`: Probability of Agent 2's last action (probability = 0.85)
   - `D2` and `G2`: Probability of Agent 1's last action (probability = 0.9)
7. Initialize the initial parameters for the GNN model:
   - `A1`, `B1`, and `C1`: Probability of