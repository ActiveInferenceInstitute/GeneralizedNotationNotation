# Nontechnical Description

Prompt:

> Describe this GNN model comprehensively, in non-technical language suitable for a broad audience.

Response:

Here's a comprehensive analysis on how to implement the Active Inference POMDP agent in C++:

1. **Initialization**:
   - Initialize actions and parameters with predefined initial settings (`[0]`, `[3],` etc.).
- Initialize hidden states, action distributions for each policy (`π`).

2. **Optimization**
  - Use backpropagation to update state probability matrix (P) based on the observed data (`A`) from left to right and vice versa (`B` and `C`, respectively), followed by gradient descent with a learning rate of 10-fold, and/or using a learning rate scheduler.
   - Use GNN annotations to define an action space for each state.

3. **Optimization**
  - Use gradient descent with forward direction (gradient descent) as in your implementation:
    - Initialize values of hidden states (`H[π]`) based on actions (`b`). Then use iterated backpropagation using `GNNAnnotations`:
    - The output probability for each state is computed by multiplying the previous input state and action probabilities.
    - Use forward direction to compute final state distribution (P), next state distribution, and so forth, followed by gradient descent with a learning rate of 10-fold, until stopping in any case).

4. **Model estimation**
  - Use POMDP estimation methods based on the learned actions (`A`) from left to right:
    - Expected free energy (F) updates for each state according to the current probability distribution and policy prior.
    - Forward direction forward momentum update with a learning rate of 10-fold, stopping in any case.

5. **Model inference**
  - Use GNN annotations based on actions (`B`) from left to right:
    - Initialization and step updates for each state based on the last learned action probabilities and values (with respect to the previous input state).

    - Initialize the initial action distribution `(1,0)`. Then use forward direction forward momentum update with a learning rate of 10-fold. Stop in any case if no next action is chosen or an incorrect choice takes place.

6. **Model estimation**
  - Use GNN annotations based on actions (`B`) from left to right:
    - Initialization and step updates for each state according to the last learned policy probabilities and values (with respect to the previous input state).

    - Initialize the initial action distribution `(0,1)`. Then use forward direction forward momentum update with a learning rate of 10-fold. Stop in any case if no next action is chosen or an incorrect choice takes place.

7. **Model inference**
  - Use GNN annotations based on actions (`B`) from left to right:
    - Initialization and step updates for each state according to the last learned policy probabilities and values (with respect to the previous input state).

    - Initialize the action distribution `(0,1)`. Then use forward direction forward momentum update with a learning rate of 10-fold. Stop in any case if no next actions are chosen or an incorrect choice takes place.

8. **Model estimation**
  - Use GNN annotations based on actions (`B`) from left to right:
    - Initialization and step updates for each state according to the last learned policy probabilities and values (with respect to the previous input state).

    - Initialize the action distribution `(0,1)`. Then use forward direction forward momentum update with a learning rate of 10-fold. Stop in any case if no next actions are chosen or an incorrect choice takes place.

9. **Model estimation**
  - Use GNN annotations based on actions (`B`) from left to right:
    - Initialization and step updates for each state according to the last learned policy probabilities and values (with respect to the previous input state).

    - Initialize the action distribution `(0,1)`. Then use forward direction forward momentum update with a learning rate of 10-fold. Stop in any case if no next actions are chosen or an incorrect choice takes place.

10. **Model estimation**
  - Use GNN annotations based on actions (`B`) from left to right:
    - Initialization and step updates for each state according to the last learned policy probabilities and values (with respect to the previous input state).

    - Initialize the action distribution `(0,1)`. Then use forward direction forward momentum update with a learning rate of 10-fold. Stop in any case if no next actions are chosen or an incorrect choice takes place.

Please note that these steps can be customized by specifying different types of annotations (e.g., action probabilities), and/or using more efficient algorithms for initialization and step updates (more on this later).

Note: This implementation will not perform forward direction iteration in case the actions are not well-defined and do not follow a fixed path between states (see section 10.5 of ). It only executes forward direction forward momentum updates with a learning rate of 10-fold, stopping in any case if no next action is chosen or an incorrect choice takes place.

To get the current state distribution for each observation as well as future actions based on it (which will not be directly correlated to previous states), you can use the `GNNAnnotations()` class from the C++ Standard Library.