# IDENTIFY_COMPONENTS

Here is the list of annotated models:
```python
import numpy as np
from pymdp import mdp_ontology, mdp_model
from pymdp.utils import load_jsonfile


def generate_inference(
    model_name=None,
    input_data=[
        {"id": "10", "type": "int"},
        {"id": "25", "type": "float"}],
        output_path="output/".join(),
        num_hidden_states: int = 128,
        num_actions=128,
    ],
):

    # Load JSON file for input data
    json_file = open(input_data[0])
    json_obj = json.load(json_file)
    mdp_object = mdp_ontology(**json_obj)

    # Generate inference model
    inference_model = mdp_model(*mdp_object, **kwargs)

    # Generate input data and save to JSON file
    with open("input/".join(), "r") as f:
        json.dump(
            [
                {"id": "10", "type": "int"},
                {"id": "25", "type": "float"}],
            )
    mdp_file = open(output_path, "w+")

    # Generate inference model
    inference_model = mdp_model(*mdp_object)

    # Generate input data and save to JSON file
    with open("input/".join(), "r") as f:
        json.dump([
            {"id": "10", "type": "int"},
            {"id": "25", "type": "float"}],
            mode="w+"
        )

    mdp_file = open(output_path, "w+")
    # Generate inference model
    inference_model = mdp_model(*mdp_object)


def generate_inference_on_topology(input_data=[]):
    """Generate Inference on Topology"""
    input_data=input_data[0]

    # Load JSON file for input data
    json_file="output/".join()
    mdp_obj = mdp_ontology(**json_file)

    # Generate inference model
    inference_model = mdp_model(*mdp_object, **kwargs)


def generate_inference(input_data=[]):
    """