# EXTRACT_PARAMETERS

Your comprehensive list provides a clear overview of the information captured in your GNN model and its components:

This covers the main pieces:

1. **Model Matrices**: `A`, `B`, `C`, `D` are matrices representing the model architecture, decision rules, policy distribution, habit preferences, and prior distributions over observed data and actions.
2. **Pseudo-supervised learning protocols (`GNNVersionAndFlags`)**: `Gnn` is a function that implements Bayesian inference based on Active Inference principles, but it requires parameters to be defined beforehand (in this case, in the signature). The parameter settings depend on the implementation and can be easily adjusted.
3. **Parameter evaluation metrics**: You mentioned `Dynamic`, which measures changes of the model during learning/training processes. It depends on the choice of update rate, change-rate protocol ($\mathcal{M}_{g}) (default to ${\bf G}$), initial state size (`n_states`).
4. **Model evaluation metrics**: You mentioned `Discrete Time Horizon`, which is a feature type that measures actions executed during time in the model. It depends on the choice of update rate, action selection parameters ($\mathcal{M}_{a})$) and other parameter settings based on the implementation (in this case, in the signature).
5. **System Type**: You also mentioned `Dynamic` for inference and `Discrete Time Horizon` for prediction time horizons. These are different feature types but can be useful depending on your needs or data sources (like simulation scenarios).
6. **Integration with other models**: You mentioned `Active Inference POMDP Agent`, which is a meta-model based on the model representations and their corresponding parameters (`A`), `Generative Model`(s) for GNN, etc., respectively. You might consider different interfaces between these models to adapt them towards specific modeling scenarios or use cases (like simulation simulations or inference backends).
6. **Validation**: Your validation metrics are based on evaluation of the model in a realistic scenario (`Scenario`) and comparison with other variants that achieve similar performance for a given parameter set ($\mathcal{M}_{g} \in [0,1]$), while also accounting for data quality issues or any biases (bias) introduced by your implementation.