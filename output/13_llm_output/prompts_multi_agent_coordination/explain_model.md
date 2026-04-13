# EXPLAIN_MODEL

This is a comprehensive explanation of the GNN (Generalized Notation Notation) model and its components:

1. **Model Purpose**: This model represents a cooperative agent-agent system that cooperates to achieve goals based on shared observation space. It captures the idea of "actions" or "observations", which are actions taken by each agent towards their goal, and "prior" beliefs about future actions from other agents.

2. **Core Components**:
   - **Hidden states (s_f0, s_f1, etc.)**: These represent the collective knowledge of all agents in the system. They capture the shared belief or intention of each agent towards their goal.
   - **Observations** (o_m0, o_m1, etc.): These are actions taken by each agent towards their goal based on shared observation space. They represent the collective action set of the system.
   - **Actions/Controls**: These represent the actions or decisions made by individual agents towards their goals. They capture the collective knowledge and beliefs of all agents in the system.

3. **Model Dynamics**: This model implements Active Inference principles, which are based on cooperative decision-making between agents. It allows for the inference of future state transitions from shared observation space to actions taken by each agent towards their goal. The key relationships include:
   - **Initialization**: Each agent starts with a belief in its own goal and then updates it based on shared observation space.
   - **Evolution**: The system evolves over time as new information is gained or lost, allowing for the inference of future state transitions from shared observation space to actions taken by each agent towards their goals.

4. **Active Inference Context**: This model implements Active Inference principles in a way that allows for the inference of P-states (beliefs) and P-actions (action sets). It uses a probabilistic graphical model, which represents the system's beliefs based on shared observation space and actions taken by each agent towards their goals. The key relationships include:
   - **Initialization**: Each agent starts with an initial belief in its own goal and then updates it based on shared observation space to make future state transitions from shared observation space to action sets for each agent.
   - **Evolution**: The system evolves over time as new information is gained or lost, allowing for the inference of P-states (beliefs) and P-actions (action sets).

5. **Practical Implications**: This model can inform decisions by providing a probabilistic graphical