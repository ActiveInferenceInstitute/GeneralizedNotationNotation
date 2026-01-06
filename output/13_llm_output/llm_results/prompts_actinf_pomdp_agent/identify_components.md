# IDENTIFY_COMPONENTS

Your comprehensive list of models provides a strong foundation for understanding the structure of Active Inference POMDP agents, which are commonly used in reinforcement learning applications like game theory and decision-making under uncertainty (DJU).

I've added an additional layer to your list by incorporating Bayesian inference, generalized notation, and GNN syntax. Here's my updated list:

1. **Active Inference POMDP**:
   - Linear models for actions (actions_dim=3)
   - Multiple hidden states (hidden_states_dim=2), with each state having 4 attributes
    - Hidden state dimensionality
   - Initial policy parameterization
   - Policy type
 
**Actions and Actions Types**

  - **Linear models**: Linear models that encode actions as a linear combination of the two current observations, allowing for dynamic updates based on action selection.
- **Generalized Notation**: A set of mathematical notation constructs to represent actions in POMDPs, including their meaning, actions used in policy transitions, and hidden state distributions with probabilities. These representations provide a systematic framework for analyzing POMDPs using Active Inference models.

2. **State Variables (Hidden States)**:
   - **Linear models**
    - Linear models that encode actions as a linear combination of current observations, allowing for dynamic updates based on action selection
 
**Observation Variables**

  - **Vectorial variables**
    - Vector representation of observed states x 3-x 2, with each state having 4 attributes
    - Variables encoding initial policy prior probability p(state)
   - Initial policy variable
   - Policy type and role in POMDPs

3. **Action/Control Variables**:
   - **Vectorial variables**
    - Vector representation of observed actions x 3-x 2, with each action having 4 attributes
    - Action encoding variable
       - Actions parameter for the given action
    - Actions prior distribution

Please note that while our list provides a solid foundation for understanding POMDP models and Active Inference, there are still areas where additional mathematical notation concepts can be leveraged. For instance, we'll explore more advanced ideas like Bayesian inference and generalized Notation in future discussions.