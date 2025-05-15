# GNN Resource Estimation Report

Analyzed 5 files
Average Memory Usage: 6904.27 KB
Average Inference Time: 298.64 units
Average Storage: 8289.34 KB

## gnn_poetic_muse_model.md
Path: src/gnn/examples/gnn_poetic_muse_model.md
Memory Estimate: 1.27 KB
Inference Estimate: 189.36 units
Storage Estimate: 5.29 KB

### Model Info
- variables_count: 22
- edges_count: 0
- time_spec: Static
- equation_count: 7

### Complexity Metrics
- state_space_complexity: 8.3487
- graph_density: 0.0000
- avg_in_degree: 0.0000
- avg_out_degree: 0.0000
- max_in_degree: 0.0000
- max_out_degree: 0.0000
- cyclic_complexity: 0.0000
- temporal_complexity: 0.0000
- equation_complexity: 13.2857
- overall_complexity: 10.0000
- variable_count: 22.0000
- edge_count: 0.0000
- total_state_space_dim: 325.0000
- max_variable_dim: 135.0000

## gnn_POMDP_example.md
Path: src/gnn/examples/gnn_POMDP_example.md
Memory Estimate: 0.57 KB
Inference Estimate: 158.10 units
Storage Estimate: 3.99 KB

### Model Info
- variables_count: 18
- edges_count: 0
- time_spec: Dynamic
- equation_count: 8

### Complexity Metrics
- state_space_complexity: 7.1898
- graph_density: 0.0000
- avg_in_degree: 0.0000
- avg_out_degree: 0.0000
- max_in_degree: 0.0000
- max_out_degree: 0.0000
- cyclic_complexity: 0.0000
- temporal_complexity: 0.0000
- equation_complexity: 9.4375
- overall_complexity: 9.2574
- variable_count: 18.0000
- edge_count: 0.0000
- total_state_space_dim: 145.0000
- max_variable_dim: 54.0000

## gnn_active_inference_language_model.md
Path: src/gnn/examples/gnn_active_inference_language_model.md
Memory Estimate: 34517.93 KB
Inference Estimate: 838.72 units
Storage Estimate: 41429.14 KB

### Model Info
- variables_count: 56
- edges_count: 0
- time_spec: Dynamic
- equation_count: 7

### Complexity Metrics
- state_space_complexity: 23.0751
- graph_density: 0.0000
- avg_in_degree: 0.0000
- avg_out_degree: 0.0000
- max_in_degree: 0.0000
- max_out_degree: 0.0000
- cyclic_complexity: 0.0000
- temporal_complexity: 0.0000
- equation_complexity: 10.3878
- overall_complexity: 10.0000
- variable_count: 56.0000
- edge_count: 0.0000
- total_state_space_dim: 8836591.0000
- max_variable_dim: 4000000.0000

## gnn_airplane_trading_pomdp.md
Path: src/gnn/examples/gnn_airplane_trading_pomdp.md
Memory Estimate: 1.07 KB
Inference Estimate: 152.96 units
Storage Estimate: 4.48 KB

### Model Info
- variables_count: 16
- edges_count: 1
- time_spec: Dynamic
- equation_count: 7

### Complexity Metrics
- state_space_complexity: 8.1085
- graph_density: 0.0042
- avg_in_degree: 1.0000
- avg_out_degree: 1.0000
- max_in_degree: 1.0000
- max_out_degree: 1.0000
- cyclic_complexity: 0.0000
- temporal_complexity: 0.0000
- equation_complexity: 8.1633
- overall_complexity: 8.9543
- variable_count: 16.0000
- edge_count: 1.0000
- total_state_space_dim: 275.0000
- max_variable_dim: 144.0000

## gnn_example_pymdp_agent.md
Path: src/gnn/examples/gnn_example_pymdp_agent.md
Memory Estimate: 0.48 KB
Inference Estimate: 154.07 units
Storage Estimate: 3.83 KB

### Model Info
- variables_count: 21
- edges_count: 2
- time_spec: Dynamic
- equation_count: 5

### Complexity Metrics
- state_space_complexity: 6.9658
- graph_density: 0.0048
- avg_in_degree: 1.0000
- avg_out_degree: 1.0000
- max_in_degree: 1.0000
- max_out_degree: 1.0000
- cyclic_complexity: 0.0000
- temporal_complexity: 0.0000
- equation_complexity: 8.7600
- overall_complexity: 8.7413
- variable_count: 21.0000
- edge_count: 2.0000
- total_state_space_dim: 124.0000
- max_variable_dim: 27.0000

# Metric Definitions

## General Metrics
- **Memory Estimate (KB):** Estimated RAM required to hold the model's variables and data structures in memory. Calculated based on variable dimensions and data types (e.g., float: 4 bytes, int: 4 bytes).
- **Inference Estimate (units):** A relative, abstract measure of computational cost for a single inference pass. It is derived from factors like model type (Static, Dynamic, Hierarchical), the number and type of variables, the complexity of connections (edges), and the operations defined in equations. Higher values indicate a more computationally intensive model. These units are not tied to a specific hardware time (e.g., milliseconds) but allow for comparison between different GNN models.
- **Storage Estimate (KB):** Estimated disk space required to store the model file. This includes the memory footprint of the data plus overhead for the GNN textual representation, metadata, comments, and equations.

## Complexity Metrics (scores are generally relative; higher often means more complex)
- **state_space_complexity:** Logarithmic measure of the total dimensionality of all variables (sum of the product of dimensions for each variable). Represents the model's theoretical information capacity or the size of its state space.
- **graph_density:** Ratio of actual edges to the maximum possible edges in the model graph. A value of 0 indicates no connections, while 1 would mean a fully connected graph. Measures how interconnected the variables are.
- **avg_in_degree:** Average number of incoming connections (edges) per variable.
- **avg_out_degree:** Average number of outgoing connections (edges) per variable.
- **max_in_degree:** Maximum number of incoming connections for any single variable in the model.
- **max_out_degree:** Maximum number of outgoing connections for any single variable in the model.
- **cyclic_complexity:** A score indicating the presence and extent of cyclic patterns or feedback loops in the graph. Approximated based on the ratio of edges to variables; higher values suggest more complex recurrent interactions.
- **temporal_complexity:** Proportion of edges that involve time dependencies (e.g., connecting a variable at time `t` to one at `t+1`). Indicates the degree to which the model's behavior depends on past states or sequences.
- **equation_complexity:** A measure based on the average length, number, and types of mathematical operators (e.g., +, *, log, softmax) used in the model's equations. Higher values suggest more intricate mathematical relationships between variables.
- **overall_complexity:** A weighted composite score (typically scaled, e.g., 0-10) that combines state space size, graph structure (density, cyclicity), temporal aspects, and equation complexity to provide a single, holistic measure of the model's intricacy.
