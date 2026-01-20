# EXTRACT_PARAMETERS

You've outlined the main components of your AI model:
1. **ActInfPOMDP** is a probabilistic graphical representation with two types of layers (observation and hypothesis) for an unbounded time horizon, no deep planning, precision modulation, hierarchical nesting, and Bayesian inference in each modality/action. The initial parameters are represented by matrices `A`, `B` for actions, and `C`. 
   - The initial parameter matrix represents the agent's preferences over observables.

2. **ModelMatrices**:
   - A 3 x 3x4 table representing the distribution of probabilities across states/actions when acting with an action type. This is represented as "probability space" instead of "observation space". 

   **A = LikelihoodMatrix** represents a probability map describing what actions would be followed given previous observations and their corresponding preferences based on these beliefs.
   - B = TransitionMatrix**(probabilities for each observation)** are representing the transition matrix between states/actions with prior probabilities as inputted by the agent (prior)

3. **ProbabilityVector**:
   - A 3 x 2x4 table representing the likelihoods of the policy and actions given previous observations. This is represented in the form "probabilities over observation" for each observation. 

   **B = Probability** are representing the prior probabilities of states/actions based on their probabilities (prior)

4. **ProbableVector**:
   - A 3 x 2x1 table representing the likelihoods of action-wise policies given previous observations and corresponding beliefs. This is represented in the form "likelihood across actions" for each observation. 

   **C = Probability** are representing the prior probabilities of states/actions based on their probabilities (prior)

5. **Constraints**:
   - A matrix `A` representing constraints or restrictions to follow given previous observations and current beliefs within a certain time horizon for action-wise types. This is represented as "constraints" instead of parameters.

6. **InitialParameters** are representing the initial set of values that define each parameter in the model structure (initial biases, initial actions etc.). These can be viewed as a list of values which will affect what actions follow given previous observations and prior beliefs for each action type. 

All these variables have been specified in your AI model specification file.