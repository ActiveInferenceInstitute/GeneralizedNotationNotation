# SUMMARIZE_CONTENT

Here's a concise summary of the GNN specification:

**Overview**
This is an active inference Bayesian network (AIBN) that models multi-armed bandit scenarios as POMDPs. It uses a set of hidden states and actions, with each action having two types of reward distributions (identity rewards vs small rewards). The model learns to balance exploration over arms while exploiting exploitation by choosing the arm with higher reward.

**Key Variables**
The model consists of:

1. **Hidden States**: A list of 3 hidden states representing the "reward context" and "action-observation mapping". Each state has a probability distribution over actions, which can be either identity or small rewards (smaller than big rewards).

2. **Observations**: A list of 3 observations representing reward signals from arm 0 to arm 1. Each observation is initialized with a random value and has an associated probability distribution over arms.

3. **Actions**: A list of actions, which are labeled as "action-observation mapping" (AoM). Each action can be either identity or small rewards (smaller than big rewards), so each action has two types of reward distributions: identity and small rewards.

**Critical Parameters**
The model learns to balance exploration over arms while exploiting exploitation by choosing the arm with higher reward. The key parameters are:

1. **Random Initialization**: Randomly initialize all hidden states, actions, and observations.

2. **Initialization of Hidden States**: Initialize a list of 3 hidden states representing "reward context" (identity rewards vs small rewards). Each state has an associated probability distribution over action-observation mapping.

3. **Initialization of Actions**: Initialize a list of 3 actions labeled as "action-observation mapping". Each action can be either identity or small rewards, so each action has two types of reward distributions: identity and small rewards (smaller than big rewards).

**Notable Features**
The model learns to balance exploration over arms while exploiting exploitation by choosing the arm with higher reward. The key features are:

1. **Random Initialization**: Randomly initialize all hidden states, actions, and observations.

2. **Initialization of Hidden States**: Initialize a list of 3 hidden states representing "reward context" (identity rewards vs small rewards). Each state has an associated probability distribution over action-observation mapping.

3. **Initialization of Actions**: Initialize a list of 3 actions labeled