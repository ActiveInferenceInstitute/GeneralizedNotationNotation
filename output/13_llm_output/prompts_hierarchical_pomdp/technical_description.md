# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

```python
import json
from typing import List


class GNNModel:
    def __init__(self):
        self._model = None

    @staticmethod
    def load_ontology(file_name: str) -> list[str]:
        """Loads a hierarchical ontology from the given file."""

        with open(file_name, 'r') as f:
            data = json.load(f)
        
        if isinstance(data['level'], type):
            self._model = GNNModel()

            for level in data['levels']:
                self._model[level] = {
                    "type": level["type"],
                    "value": level["value"]
                }

        return data

    def load_ontology(self, file: str) -> list[str]:
        """Loads a hierarchical ontology from the given file."""

        with open(file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data['level'], type):
            self._model = GNNModel()

            for level in data['levels']:
                self._model[level] = {
                    "type": level["type"],
                    "value": level["value"]
                }

        return data

    def load_ontology_from_json(self, json_file: str) -> list[str]:
        """Loads a hierarchical ontology from JSON file."""

        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data['level'], type):
            self._model = GNNModel()

            for level in data['levels']:
                self._model[level] = {
                    "type": level["type"],
                    "value": level["value"]
                }

        return data

    def load_ontology_from_pickle(self, file: str) -> list[str]:
        """Loads a hierarchical ontology from pickle file."""

        with open(file, 'rb') as f:
            data = json.load(f)
        
        if isinstance(data['level'], type):
            self._model = GNNModel()

            for level in data['levels']:
                self._model[level] = {
                    "type": level["type"],
                    "value": level["value"]
                }

        return data

    def load_ontology_from_pickle(self, file: str) -> list[str]:
        """Loads a hierarchical ontology from pickle file."""

        with open(file, '