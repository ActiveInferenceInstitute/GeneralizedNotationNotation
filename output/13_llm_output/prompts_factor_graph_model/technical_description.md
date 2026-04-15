# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

You can use the following code to implement an active inference model on a graph:

1. Create a graph with two observation modalities (visual and proprioceptive) using `FactorGraph`.
2. Use the `GNN` algorithm from the `actinf_gnn` module to generate a GNN representation of your data. The GNN is implemented in `factorgraph`, which can be used for inference on graphs.
3. Use the `gnn_model` function to create an active inference model on your graph and use it as input to `actinf_gnn`.
4. Use the `GNNModelAnnotation` class from `actinf_gnn` to annotate the GNN representation with annotations for each observation modality (visual, proprioceptive, etc.).
5. Use the `gnn_model` function again to generate an inference model on your graph and use it as input to `actinf_gnn`.
6. Use the `GNNModelAnnotation` class from `actinf_gnn` to annotate the GNN representation with annotations for each observation modality (visual, proprioceptive, etc.).
7. Use the `gnn_model` function again to generate an inference model on your graph and use it as input to `actinf_gnn`.