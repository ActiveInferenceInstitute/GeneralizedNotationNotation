# GNN Resource Estimation Report

Analyzed 4 files
Average Memory Usage: 38.19 KB
Average Inference Time: 433.33 units
Average Storage: 52.24 KB

## pymdp_pomdp_agent.md
Path: src/gnn/examples/pymdp_pomdp_agent.md
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

## self_driving_car_comprehensive.md
Path: src/gnn/examples/self_driving_car_comprehensive.md
Memory Estimate: 151.46 KB
Inference Estimate: 1138.25 units
Storage Estimate: 194.16 KB

### Model Info
- variables_count: 136
- edges_count: 0
- time_spec: Dynamic
- equation_count: 29

### Complexity Metrics
- state_space_complexity: 15.2431
- graph_density: 0.0000
- avg_in_degree: 0.0000
- avg_out_degree: 0.0000
- max_in_degree: 0.0000
- max_out_degree: 0.0000
- cyclic_complexity: 0.0000
- temporal_complexity: 0.0000
- equation_complexity: 3.1879
- overall_complexity: 9.5343
- variable_count: 136.0000
- edge_count: 0.0000
- total_state_space_dim: 38780.0000
- max_variable_dim: 12000.0000

## rxinfer_multiagent_gnn.md
Path: src/gnn/examples/rxinfer_multiagent_gnn.md
Memory Estimate: 0.52 KB
Inference Estimate: 283.16 units
Storage Estimate: 6.76 KB

### Model Info
- variables_count: 60
- edges_count: 1
- time_spec: Dynamic
- equation_count: 15

### Complexity Metrics
- state_space_complexity: 6.8202
- graph_density: 0.0003
- avg_in_degree: 1.0000
- avg_out_degree: 1.0000
- max_in_degree: 1.0000
- max_out_degree: 1.0000
- cyclic_complexity: 0.0000
- temporal_complexity: 0.0000
- equation_complexity: 3.2578
- overall_complexity: 5.3649
- variable_count: 60.0000
- edge_count: 1.0000
- total_state_space_dim: 112.0000
- max_variable_dim: 16.0000

## rxinfer_hidden_markov_model.md
Path: src/gnn/examples/rxinfer_hidden_markov_model.md
Memory Estimate: 0.29 KB
Inference Estimate: 157.83 units
Storage Estimate: 4.22 KB

### Model Info
- variables_count: 17
- edges_count: 7
- time_spec: Dynamic
- equation_count: 12

### Complexity Metrics
- state_space_complexity: 6.3923
- graph_density: 0.0257
- avg_in_degree: 1.7500
- avg_out_degree: 1.1667
- max_in_degree: 3.0000
- max_out_degree: 2.0000
- cyclic_complexity: 0.0000
- temporal_complexity: 0.0000
- equation_complexity: 4.7431
- overall_complexity: 6.0549
- variable_count: 17.0000
- edge_count: 7.0000
- total_state_space_dim: 83.0000
- max_variable_dim: 9.0000

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
