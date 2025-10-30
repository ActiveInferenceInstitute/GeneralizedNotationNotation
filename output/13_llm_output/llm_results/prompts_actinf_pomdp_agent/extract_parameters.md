# EXTRACT_PARAMETERS

Based on the information provided, here's a systematic approach to extract key features and parameters from GNN examples:
1. **Initial Parameters**:
- **State Space Matrices**: A list of matrix entries for each observable parameter in each observation modality or hidden state factor.
* 
**Type(s):** `action` (policy) `actions_dim=3`, `observation` (observable index)
```python
  # Example actions
  Action = Matrix([[[0., 1., 1.], [[2./np.sqrt(self.__attr__("initializer")), 2/np.sqrt(self.__attr__.gamma),
                                                           4/self.__attr__.__init__.number],
                                                  [4^(-n) * (8**((self.__attr__("max_depth")) / self.__attr__) ** n)]]])
  # Example actions
```

2. **Randomized Parameters**:
- **Initialization Strategies**:
   - **Generalized Notation Notations**:
   * 
     - `p` (probability): The initial probability of action A being chosen in each observation modality or hidden state factor.
   **Type(s):**
   - `A_i$`: the probability vector representing all actions, initialized with a uniform distribution over actions on each observable dimension.
   - `v` (value): a list of probability vectors for actions to be sampled from policy values and prior probabilities at the observation index.

3. **Configuration Summary**:
- **Initial Conditions**:
    - **State Space Matrix**: A list of matrix entries representing the state space dimensions of each mode/dimensionality pair, initialized with uniform distributions over actions and biases respectively.

4. **Tunable Parameters**:
   - **Parameter file format recommendations**:
   - Use a consistent parameter file format for GNN examples based on the documentation in the following table:
    - **Initialization parameters (alpha)**: A list of numerical values corresponding to initial alpha parameters, default values and their indices from `self.__attr__("initial_param").** type.

    ```python
  # Example initializers
  Initializer = [Matrix([[[0., 1.], [[4/np.sqrt(2)]]])])

  # Example initialization strategies
```
    - **Initialization parameters (gamma)**: A list of numerical values corresponding to initial gamma parameters, default values and their indices from `self.__attr__("initial_param").** type.

    ```python
  # Example initialization parameters
  Initializer = [Matrix([[[0., 1.], [[4/np.sqrt(2)]]])])

  # Example initialization strategies
    - **Initialization parameters (alpha)**: A list of numerical values corresponding to initial alpha parameters, default values and their indices from `self.__attr__("initial_param").** type.

    ```python
  # Example initialization parameters
```
   - **Type(*args)**: Use a consistent parameter file format for GNN examples based on the documentation in the following table
    - **Initialization parameters (alpha)**: A list of numerical values corresponding to initial alpha parameters, default values and their indices from `self.__attr__("initial_param").** type.

    ```python
  # Example initialization parameters
  Initializer = [Matrix([[[0., 1.], [[4/np.sqrt(2)]])]))

  # Example initialization strategies
    - **Initialization parameters (gamma)**: A list of numerical values corresponding to initial gamma parameters, default values and their indices from `self.__attr__("initial_param").** type.

    ```python
  # Example initialization parameters
```
   - **Type(*args)**: Use a consistent parameter file format for GNN examples based on the documentation in the following table
    - **Initialization parameters (alpha)**: A list of numerical values corresponding to initial alpha parameters, default values and their indices from `self.__attr__("initial_param").** type.

    ```python
  # Example initialization parameters
  Initializer = [Matrix([[[0., 1.], [[4/np.sqrt(2)]])]))

  # Example initialization strategies
    - **Initialization parameters (gamma)**: A list of numerical values corresponding to initial gamma parameters, default values and their indices from `self.__attr__("initial_param").** type.

    ```python
  # Example initialization parameters
```
   - **Type(*args)**: Use a consistent parameter file format for GNN examples based on the documentation in the following table
    - **Initialization parameters (gamma)**: A list of numerical values corresponding to initial gamma parameters, default values and their indices from `self.__attr__("initial_param").** type.

    ```python
  # Example initialization parameters
  Initializer = [Matrix([[[0., 1.], [[4/np.sqrt(2)]])]))

  # Example initialization strategies
    - **Initialization parameters (gamma)**: A list of numerical values corresponding to initial gamma parameters, default values and their indices from `self.__attr__("initial_param").** type.

    ```python
  # Example initialization parameters
```