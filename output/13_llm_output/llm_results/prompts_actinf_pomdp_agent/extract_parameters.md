# EXTRACT_PARAMETERS

You've already provided the relevant information to derive a comprehensive API for Active Inference POMDP Agents. However, it's indeed worth mentioning that you have multiple parameters and matrices in your codebase:

1. **Parameter File Format**: You mentioned two types of parameters, which should be listed along with their data types as follows:
   - `A`:
   - `B`:
   - `C`
   - `D`:
   - `γ`
   - `α`:
   - `Other precision/confidence parameters`

2. **Initial Condition**: You mentioned that the initial parameter values and initialization strategies should be described in an API. These are likely based on your current understanding of the behavior and can include dependencies between variables, such as action selection from a policy prior or inference steps within states:
   - `C`:
   - `G` (Generalized Notation Notation)

3. **Transition Matrix**: You mention that it's structured with parameters in one column and matrices in another. However, you don't specify this structure for the initial parameter values themselves but instead refer to a dictionary of initial values:

   ```python
initial_param = {
  "A": 150, # A-T model
  420           # Initial state
  378          # Actions
  69         # Policy prior
  49                # Action sequence
  49                 # Action evaluation

       'F': 4.7    # Initial hypothesis value (initial guess)
}
```