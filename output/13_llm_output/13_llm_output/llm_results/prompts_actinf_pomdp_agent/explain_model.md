# EXPLAIN_MODEL

You're on the right track! Here's a step-by-step guide to understanding the GNN framework:

1. **Initialization**: The GNN consists of two main components:
   - **StatePossession**: Each observation is assigned to one hidden state and that hidden state has an associated probability distribution over actions (`H[y,x]`) or policy (`G[h(s), y=1]`, where `g(s)` denotes the GNN transition operator).

   - **Initialization**: The initial state of each observation is initialized with a random value and then mapped to hidden states. Each observation has an associated probability distribution over actions, which maps to a policy vector (`H[y])`.

2. **Model Representation**: This model represents a general type of active inference agent:
   - **Single-observation**: It assigns beliefs (facts) or decisions based on observed observations ($p(x)) and probabilities $P$ for the chosen action ([π]).

   - **Multiple-observations**: It generates policies, actions, and hidden states in parallel using probability distributions. Each observation has an associated probability distribution over actions/policies with uniform policy prior (`g(s)`.

3. **Constraints**: This model enforces bounds on beliefs based on observed data. The goal is to generate beliefs that are consistent with each action chosen by the agent, i.e., beliefs for all actions:

   - **One-step history**: It generates histories of how the agents (and their policies/actions) change over time due to changes in observations and hidden states. It updates these beliefs based on a sequence of policy transitions (`G[g(s), y=1]`) until it reaches an empty horizon after each step:

   - **One-step History**: The goal is to generate the transition matrix $(A, B)$ for each observation whose action ($y$) has been seen before. This can be done in a greedy way by choosing actions based on observed observations and then moving forward to other possible decisions.

4. **Model Dynamics**: It implements Active Inference principles:
   - **Policies**: Policies are defined based on the available data, i.e., there is no prior knowledge about what action will happen next for all actions. That is, it only knows of 3 options and doesn't know which one to choose.

   - **Actions**: Actions can be defined by using a sequence of policy transitions or other initial guesses (actions) that are given at each time step in the history. There could potentially exist multiple policies with different objectives. This allows for a probabilistic graphical representation, where actions/policies change based on probability distributions and actions are constrained to take one action out of possible options:

    - **One-step History**: This is the most intuitive scenario (every choice is made independently). In this case, there could be no prior knowledge about what action will happen next for all actions.

5. **Active Inference Context**: It generates Actions/Policy histories based on its data collection strategy and initial beliefs ($p(x)$. Each observation can have a unique history, but each policy is defined in terms of previous policies with associated probabilities).

Please keep the following concepts:

  *   **StatePossession** – What are the states (observations)?
  *   **Observable Data** and **Data Collection**: How does it get data?
  *   **Initial History**: What is a history of what actions were given at each time step in the history?
  *   **Constraints**: What policies/actions must be defined to ensure that beliefs are consistent with actions.