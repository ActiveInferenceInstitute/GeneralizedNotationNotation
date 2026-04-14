# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

```python
import json
from collections import Counter

def compute_inference(state):
    # TODO: Implement inference logic for this example

    return {'type': 'int', 'value': state['action']}


def generate_input_and_output_files():
    """Generate input and output files with the following structure.

    Input file:
        input/10_ontology_output/simple_mdp_ontology_report.json

        This is a JSON file containing a list of dictionaries representing
        actions, states, and histories for an action-action pair. Each
        dictionary represents a single action in the input data set. For example,
        `actions = [{'type': 'int', 'value': 1}, {'type': 'int',
            'value': 2}]` represents the actions of the first two actions.
    """

    # Generate input files with the following structure:
    inputs_file = f"input/10_ontology_output/{state['action']}"
    outputs_file = f"outputs/{state['action']}."
    output_files += [f"{inputs_file}|{outputs_file}.json", ".".join(
        ["type"] * len(state["actions"]), sep="\n"])

    return inputs_file, outputs_file


def generate_output_and_input_files():
    """Generate and save output files with the following structure.

    Output file:
        input/10_ontology_output/{state['action']}

        This is a JSON file containing a list of dictionaries representing
        actions, states, and histories for an action-action pair. Each
        dictionary represents a single action in the output data set. For example,
        `actions = [{'type': 'int', 'value': 1}, {'type':
            'int', 'value': 2}]` represents the actions of the first two actions.
    """

    # Generate output files with the following structure:
    outputs_file = f"outputs/{state['action']}."
    inputs_file = f"{inputs_file}|{outputs_file}.json".join(
        ["type"] * len(state["actions"]), sep="\n"])

    return outputs_file, inputs_file


def generate_input_and_output_files():
    """Generate and save input files with the following structure.

    Input file:
        input/10_ontology_output/{state['action']}