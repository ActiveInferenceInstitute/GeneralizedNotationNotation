# EXTRACT_PARAMETERS

Here is the complete specification for the GNN implementation:
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# List of input files containing data and annotations
input_files = [
    "10_ontology_output/simple_mdp_ontology_report.json",
    "10_ontology_output/multi_armed_bandit_ontology_report.json",
    "10_ontology_output/actinf_pomdp_agent_ontology_report.json",
    "10_ontology_output/time_varying_dynamics_ontology_report.json",
    "10_ontology_output/two_state_bistable_ontology_report.json",
    "10_ontology_output/markov_chain_ontology_report.json",
    "10_ontology_output/tmaze_epistemic_ontology_report.json"
]


def gnn(input_files, num_hidden_states=8):
  """GNN implementation for the input data and annotation files."""

  # List of input files containing data and annotations
  inputs = [
    "10_ontology_output/simple_mdp_ontology_report.json",
    "10_ontology_output/multi_armed_bandit_ontology_report.json",
    "10_ontology_output/actinf_pomdp_agent_ontology_report.json",
    "10_ontology_output/time_varying_dynamics_ontology_report.json",
    "10_ontology_output/two_state_bistable_ontology_report.json",
    "10_ontology_output/markov_chain_ontology_report.json",
    "10_ontology_output/tmaze_epistemic_ontology_report.json"
  ]

  # List of input files containing data and annotations
  inputs = [
    "input_files",
    "input_file_names",
    "annotation_file_names",
    "model_name",
    "statespace_block_number",
    "policy_vector_size",
    "action_vectors_size"
  ]

  # List of input files containing data and annotations
  inputs = [
    "10_ontology_output/simple_mdp_ontology_report.json",
    "input_