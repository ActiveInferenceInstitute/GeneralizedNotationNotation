# EXTRACT_PARAMETERS

1. **Model Matrices:**
   - A matrices representation with 3 hidden states, 3 actions per action, and uniform prior over the last state dimension for each action parameter set

2. **Precision Parameters:**
   - γ (gamma): parameter describing how many times the agent chooses a new guess based on its knowledge of previous guesses
   
   - α: learning rate

    - Alpha is used as initialization parameters to give initial idea of what the agent would do in case of certain actions and action types, but this should not be relied upon for analysis.

3. **Dimensional Parameters:**
   - State space dimensions

  - A matrices representation with 3 hidden states, 3 actions per action, and uniform prior over the last state dimension for each action parameter set

  - B matrices:
  - C matrices:
    - D matrices:
      - G matrix represents the belief updating of the agent based on observed patterns. The policy can be seen as a distribution between action selection and beliefs across states

4. **Precision Parameters:**
   - γ (gamma): prediction error parameters with regard to each input unit

    - α = learning rate

**Additional Inputs:**

5. **Configuration Summary:** 

   - Parameter file format recommendations

   - Tuning strategies:

      - Sensitivity analysis priorities