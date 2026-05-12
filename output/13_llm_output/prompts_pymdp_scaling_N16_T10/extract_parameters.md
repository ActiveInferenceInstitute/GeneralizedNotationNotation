# EXTRACT_PARAMETERS

Here are the steps to generate a structured specification for the model:

1. **Model Matrices**:
    - A matrices: dimensions, structure, interpretation
    - B matrices: dimensions, structure, interpretation  
    - C matrices: dimensions, structure, interpretation

    The matrix format recommendations can be used in the following order:

   - **Initialization parameters**
   - **Parameter file formatting recommendations**

2. **Precision Parameters**:
    - γ (gamma): precision parameters and their roles

Here are some examples of how to generate a structured specification for each parameter:

```python
import numpy as np
from pymdp import mdp_ontology, mdp_model

# Define the initial conditions
initial_conditions = [
    # Initialization parameters
    (0.15 * 2 + 0.3) ** 2 / 4 == 0.67895555555555555,
    -0.15 * 2 + 0.3 ** 2 / 4 == 0.67895555555555555
]
```

3. **Parameter file format recommendations**:
   - `Initialization parameters` can be used in the following order:

   - **Initialization parameters**
    - Initializing probabilities and biases

    The parameter file format recommendations can be used in the following order:

   - **Initialization parameters**
    - Initializing probability distributions, bias distributions, and transition matrices

4. **Parameter file formatting recommendations**:
   - `Parameter file formats` can be used in the following order:

   - **Parameters files**
    - Parameter values

    The parameter file format recommendations can be used in the following order:

   - **Parameters files**
    - Parameter value lists (e.g., [0, 1])

5. **Configuration summary**:
   - `Initialization parameters` can be used in the following order:

   - **Initialization parameters**
    - Initializing probabilities and biases

    The parameter file format recommendations can be used in the following order:

   - **Initialization parameters**
    - Initializing probability distributions, bias distributions, and transition matrices

6. **Temporal parameters**:
   - `Parameter values` can be used in the following order:

   - **Parameter value lists (e.g., [0, 1])**

    The parameter file format recommendations can be used in the following order: