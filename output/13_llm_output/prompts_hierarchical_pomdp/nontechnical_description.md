# Nontechnical Description

Prompt:

> Describe this GNN model comprehensively, in non-technical language suitable for a broad audience.

Response:

```python
import json
from typing import List, Dict

class GNNModel:
    def __init__(self):
        self.model = None

    def load(self) -> None:
        """Loads the GNN model from a JSON file."""

        with open("GNN_model.json", "r") as f:
            self.model = json.load(f)
        
        if not isinstance(self.model, dict):
            raise ValueError("Model is not a dictionary.")
            
        for key in self.model["outputs"]:
            try:
                self.model[key] = eval(eval_json(self.model[key]))
            
            except Exception as e:
                print(f"Error loading model from JSON file: {e}")
        
        if isinstance(self.model, dict):
            for key, value in self.model["outputs"].items():
                try:
                    self._load_action(value)
                    
        else:
            raise ValueError("Model is not a dictionary.")

        return

    def _load_action(self, action):
        """Loads the actions from the model."""
        
        if isinstance(self.model[action], dict):
            for key in self.model["outputs"]:
                try:
                    self._load_action(eval(json.loads(str(self.model[key]))))
                    
                except Exception as e:
                    print(f"Error loading action from model: {e}")
                
        else:
            raise ValueError("Model is not a dictionary.")

    def _load_ontology(self, path):
        """Loads the ontology file."""
        
        with open(path + "/node.json", "r") as f:
            self._load_action(eval(json.loads(str(f))))
            
    def _load_ontology(self, path):
        """Loads the ontology file."""
        
        if isinstance(self.model[path], dict):
            for key in self.model["outputs"]:
                try:
                    self._load_node(eval(json.loads(str(self.model[key]))))
                    
                
    def _load_action(self, action):
        """Loads the actions from the model."""
        
        if isinstance(self.model[action], dict):
            for key in self.model["outputs"]:
                try:
                    self._load_node(eval(json.loads(str(self.model[key]))))
                    
                
    def _load_node(self