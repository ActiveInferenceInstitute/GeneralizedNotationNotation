# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

You are correct that the behavior of GNN can be understood based on the theory and algorithms presented in the document. Specifically:

1. The model has two types of action probabilities:
   - "Habit" is a single-step policy that updates an agent's probability for each possible action over time, with the goal to reach a particular state. This allows for planning capabilities (planning horizon) and actions selection from policies.

2. In addition to planning, there are different types of policy posteriorities used:
   - "Actions" represent different actions per observation direction, which can be thought of as actions or decisions over time.
   - "Habit" represents a specific state chosen randomly for each possible action at one-dimensional observations.
   - "Policy" is the distribution of policies across all possible actions (prior policy).

3. The model outputs probabilities to estimate future outcomes based on observed actions and histories, as well as past history predictions:
   - "Initial Policy" generates a uniform prior over unknown behavior for each observation.
   - "Action Selection" uses an action-based probability distribution to compute the state transitions of the first observable step (given previous observation).
   - The probability is estimated based on the policy posteriority, which determines how beliefs are updated or propagated forward in time:
    - If we assign a probability to a particular choice among policies, it updates our belief about all possible actions.
    - If a policy does not converge towards its goal, then it propagates forward information and moves back out from that state (which corresponds to the value-based agent).

4. The model provides an initial policy prior for each observation ("habit"), which gives us a distribution over actions across the entire history of observations in terms of probability. This allows us to generate probabilities for future actions based on these beliefs, enabling planning capabilities and action selection:
    - If we assign a belief about a particular policy before observing any data (prior), then it updates our beliefs about all possible policies.
    - It propagates forward information towards the history's value-based agent ("action") through its beliefs in this new set of "policies".

5. The model also includes different types of action posteriorities:
   - "Actions" generate a uniform prior over actions across all observed observations, allowing for planning capabilities (policy posterior).
   - If there are no preferences among choices, then "Habit" produces a uniform distribution of policies across the history of each observation.
   - If we assign a probability to a particular policy at one-dimensional observations ("action"), it updates our belief about all possible actions towards this policy based on prior probabilities in the history and its posterior beliefs about future actions (policy posterior).

So, in summary, GNN represents an agent that can learn from past behavior, using two types of action distributions, "Habit" to generate a uniform distribution over actions across observation histories, and "Actions", which is used for planning capabilities.