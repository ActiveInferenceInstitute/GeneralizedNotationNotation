# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

```python
import json
from typing import List


class ActInfOntologyAnnotation:
    """Annotated ontology for ActInf agent."""

    def __init__(self):
        self.annotations = {}

        # TODO: Implement this annotation type
        # TODO: Implement this annotation type
        # TODO: Implement this annotation type
        # TODO: Implement this annotation type


class ActInfOntology(ActInfOntologyAnnotation):
    """Annotated ontology for ActInf agent."""

    def __init__(self,
                 entity_type=None,
                 **kwargs):

        self.annotations = {}

        if hasattr(entity_type, "ontology"):
            self.ontology = entity_type["ontology"]
        else:
            self.ontology = None

            # TODO: Implement this annotation type
            # TODO: Implement this annotation type

    def __str__(self):
        """Annotated ontology for ActInf agent."""
        return json.dumps(jsonify(self))


class ActInfOntologyRepresentation(ActInfOntology):
    """Represented by ActInfo object."""

    def __init__(self,
                 entity_type=None,
                 **kwargs):

        self.annotations = {}

        if hasattr(entity_type, "ontology"):
            self.ontology = entity_type["ontology"]
        else:
            self.ontology = None

            # TODO: Implement this annotation type
            # TODO: Implement this annotation type

    def __str__(self):
        """Annotated ontology for ActInf agent."""
        return json.dumps(jsonify(self))


class ActInfoRepresentation(ActInfOntology):
    """Represented by ActInfo object."""

    def __init__(self,
                 entity_type=None,
                 **kwargs):

        self.annotations = {}

        if hasattr(entity_type, "ontology"):
            self.ontology = entity_type["ontology"]
        else:
            self.ontology = None

            # TODO: Implement this annotation type
            # TODO: Implement this annotation type

    def __str__(self):
        """Annotated ontology for ActInf agent."""
        return json.dumps(jsonify(self))


class ActInfoRepresentation(ActInfOntology):
    """Represented by ActInfo object."""

    def __init__(self,
                 entity_type=None,
                 **kwargs):

        self.annotations = {}

        if hasattr(entity_type, "ontology"):
            self.ontology =