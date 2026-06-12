# EXPLAIN_MODEL

Here is a detailed explanation of the GNN implementation:

**1. **Model Purpose:**
This document describes how to generate an Active Inference (AI) model on a specific problem domain. The goal is to understand and describe the underlying mechanisms, relationships, and key concepts involved in this task.

The AI model can be described as follows:

1. **Input**: A set of data representing various actions/controls that are available for exploration/learning purposes (e.g., actions, observations). This includes actions like "move towards", "look at", etc.

2. **Output**: The output of the AI model is a list of beliefs or predictions based on the input data and other relevant information. These beliefs can be represented as vectors in a vector space called "inference" matrix (in this case, it represents the belief-based inference).

3. **Model Components**: 
   - **hidden state** (h_m0): A set of hidden states representing the current state/current action/observation and other relevant information.
   - **observations** (o_m1, o_m2, etc.): A list of actions/controls that are available for exploration/learning purposes. These can be represented as vectors in a vector space called "inference" matrix (in this case, it represents the belief-based inference).

4. **actions**: A set of actions representing current state/current action/observation and other relevant information. Actions represent what is currently happening or being observed. Actions are represented by vectors in the vector space called "action_space".

5. **observations** (o_m2): A list of observations representing current state/current observation and other relevant information. Observations represent what is happening at a given time point, which can be represented as vectors in the vector space called "observation" matrix (in this case, it represents the belief-based inference).

6. **actions** (a_m0): A set of actions representing current state/current action/observation and other relevant information. Actions represent what is currently happening or being observed at a given time point. Actions are represented by vectors in the vector space called "action" matrix (in this case, it represents the belief-based inference).

7. **actions** (a_m1): A set of actions representing current state/current observation and other relevant information. Actions represent what is currently happening or being observed at a given time point. Actions are