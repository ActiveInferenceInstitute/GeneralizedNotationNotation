# EXTRACT_PARAMETERS

Based on the document, here are the key parameters and their corresponding descriptions:

**Model Matrices:**

1. **A matrices**:
   - A matrix representing the ActInfPOMDP model structure (represented by `A`)
   - A matrix representing the GNN inference framework (represented by `B`)
   - A matrix representing the hierarchical POMDP structure (represented by `C`, `D`, and `G2`):
    - A matrix representing the hierarchy of hierarchies (represented by `s1`)
    - A matrix representing the hierarchy of actions (represented by `o1`)
    - A matrix representing the hierarchy of observables (represented by `π1`):
      - A matrix representing the hierarchical information flow (represented by `g2`):
        - A matrix representing the hierarchical information flow structure (represented by `G2`, `h2`):
          - A matrix representing the hierarchical information flow structure (represented by `H`)
            - A matrix representing the hierarchical information flow structure (represented by `A`):
              - A matrix representing the hierarchical information flow structure (represented by `g1`:
                - A matrix representing the hierarchical information flow structure (represented by `h2`):
                  - A matrix representing the hierarchical information flow structure (represented by `H`)
                    - A matrix representing the hierarchical information flow structure (representing
                      "context switches" and "top-down predictions")

**Precision Parameters:**

1. **γ**: The learning rate parameter for each level of hierarchy, which controls the rate at which the model learns from data. It is set to 0.5 in the document.
2. **α**: The learning rate parameter for each modality, which controls the rate at which the model learns from data. It is set to 1/num_hidden_states_l1 in the document.
3. **Other precision/confidence parameters:**
   - γ (gamma): the learning rate parameter for each level of hierarchy, which controls the rate at which the model learns from data. It is set to 0.5 in the document.
   
   - α (alpha): the learning rate parameter for each modality, which controls the rate at which the model learns from data. It is set to 1/num_obs_l1 in the document.
4. **Dimension Parameters:**
   - State space dimensions: `statespace` and `observation`, both are represented by `states`.
   
   - Observation space dimensions: `observations` and