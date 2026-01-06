# EXPLAIN_MODEL

Here's a concise overview of the GNN (`GNN`) framework:

**Part 1: GNN Definition & Components**

1. **Model Purpose**: This is an Active Inference agent for modeling discrete Bayesian POMDPs with one observation modality and one hidden state factor. The model describes an instance of this type, which represents the policy distribution and initial habit used to generate beliefs from observations.
2. **Core Components**:
   - **Action Policy**: A belief-based inference process that updates prior beliefs based on the observed data (observation probabilities).
   - **Habit**: A probabilistic graphical representation of a state with actions assigned to it, allowing for information sharing and exploration among agents.
   - **Initial Policy**: Initializing initial policies such as "policy" or "habit".
   - **Random History**: Randomly sampling from the policy posterior prior distribution based on observable probability distributions over observations (observable probabilities). This allows for exploring different actions within a state space, without worrying about specific choices.

3. **Model Dynamics**: How does this model evolve over time? What are key relationships and behaviors that can be inferred using it?
4. **Active Inference Context**: What can you learn or predict from this agent? 

**Part 2: GNN Structure & Principles**

1. **GNN Structure**: An Action-Based Bayesian POMDP is represented as a probability distribution over the state space, with initial policy prior and actions initialized based on observable probabilities (observation probabilities). The goal of the agent is to update its beliefs using probability distributions derived from these observed events.
2. **Base Action**: This action sequence represents a single observation modality ("action") in the POMDP universe, without any knowledge about the next state or actions within the next observation. The agent's preferences are encoded as log-probabilities over observations and policies.
3. **Key Components**:
   - **hidden states** (s_f0 through s_f6): The probability distributions representing all possible future observables that can influence action sequences, policy transitions, and belief updates. These probabilities encode the beliefs of other agents in a state space (observation probabilities) within that state.
   - **observable parameters**: All observable probabilities corresponding to actions across states, enabling exploration of different choices with specific policies ("policy" or "habit") as initial policies for actions within each observed observation. These observations are composed by the action sequences themselves and can be used together to generate all possible beliefs for the agent's preferences (choices) over actions in future observations ('actions').
4. **Policy Components**:
   - **fixed policy**: The belief-based inference process that updates prior probabilities based on observations/policy transitions, enabling exploration of policies across states through action sequences, and exploring individual choices with specific policies ("action") by randomly sampling from the observed state space (observation probabilities) within each observable observation ('actions').
5. **Base Actions**: These are actions encoded as a probability distribution over observables in the POMDP universe for all observations (observable probabilities). They encode their behavior based on these prior beliefs and can be used to generate specific beliefs from observable probabilities across observations ("belief" or "action") by randomly sampling from the observed state space.
6. **Transition Policies**: These are actions encoded as a probability distribution over observables in the POMDP universe for all observations (observable probabilities). They encode their behavior based on these prior beliefs and can be used to generate specific preferences across observable paths ("policy" or "habit") by randomly sampling from observed states within each observable observation ('actions').
7. **Action Selection**: These are actions encoded as a probability distribution over observables in the POMDP universe for all observations (observable probabilities). They encode their behavior based on these prior beliefs and can be used to generate specific preferences across observable paths by randomly sampling from observed states within each observable observation ('actions').
8. **Base Belief**: This is the belief represented at the global state space ("policy") when done correctly, with no loss of information due to the agent's actions (observation probabilities) being uncertain and thus not affected if they are influenced by any other choices made in the past or based on prior knowledge/beliefs shared across observations.
9. **Initial Policy**: This is an initial action sequence encoded as a probability distribution over observed states ("policy") that generates beliefs for all subsequent actions within its domain of applicability ("actions"). It encodes the preferences and behavior of previous agents through their policy sequences, which in turn allows exploring and adapting current policies across observation paths (observation probabilities).
10. **Random History**: This is also an action sequence encoded as a probability distribution over observed states ("policy") that generates beliefs for all actions within its domain of applicability ("actions"). It encodes the preferences and behavior of previous agents through their policy sequences, which in turn allows exploring and adapting current policies across observation paths (observation probabilities).
11. **Habit**: A probabilistic graphical representation of a state represented by an action sequence encoded as a probability distribution over observed states ("policy") that generates beliefs for all future actions within its domain of applicability ("actions"). It encodes the preferences and behavior of previous agents through their policy sequences, which in turn allows exploring and adapting current policies across observation paths (observation probabilities).