# EXTRACT_PARAMETERS

Based on the provided GNN specification, here are the key parameters that contribute to the model:

1. **Model Matrices**:
   - A matrices representing the representation of the model (e.g., `A`) and its components (`D`).
   - B matrices representing the prior belief over hidden states (`B`) and initial state (`s`, `o`) for each modality (`d`):
   - C matrices representing the prior beliefs over all possible actions (`C`)

2. **Precision Parameters**:
   - γ (gamma): precision parameters, which are used to estimate the likelihood of observing a particular observation.
   - α (alpha): learning rate and adaptation parameters, which control the update frequency for each modality.
   - Other precision/confidence parameters:
   - Sensitivity analysis priorities

To summarize, here's a systematic parameter breakdown based on these key parameters:

1. **Model Matrices**:
   - A matrices representing the representation of the model (e.g., `A`) and its components (`D`).
   - B matrices representing the prior belief over initial states (`B`) and initial state (`s`, `o`) for each modality (`d`):
   - C matrices representing the prior beliefs over all possible actions (`C`)

2. **Precision Parameters**:
   - γ (gamma): precision parameters, which are used to estimate the likelihood of observing a particular observation.
   - α (alpha): learning rate and adaptation parameters, which control the update frequency for each modality.
   - Other precision/confidence parameters:
   - Sensitivity analysis priorities

To illustrate these parameter breakdowns, here's an example of how they can be visualized in a tabular format:

1. **Model Matrices**:
   - A matrices representing the representation of the model (e.g., `A`) and its components (`D`).
   - B matrices representing the prior belief over initial states (`B`) and initial state (`s`, `o`) for each modality (`d`):
   - C matrices representing the prior beliefs over all possible actions (`C`)

2. **Precision Parameters**:
   - γ (gamma): precision parameters, which are used to estimate the likelihood of observing a particular observation.
   - α (alpha): learning rate and adaptation parameters, which control the update frequency for each modality.
   - Other precision/confidence parameters:
   - Sensitivity analysis priorities

This visualization provides a concise representation of the model's parameter structure, allowing you to easily understand how different components interact with each other.