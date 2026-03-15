from abc import ABC, abstractmethod
from typing import Dict, Any, List
import json
from .common import GNNInternalRepresentation

class BaseGNNSerializer(ABC):
    """Base class for all GNN serializers with common utility methods."""

    def __init__(self):
        self.format_name = self.__class__.__name__.replace('Serializer', '').lower()

    @abstractmethod
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Serialize the GNN model to string format."""
        pass

    def serialize_to_file(self, model: GNNInternalRepresentation, file_path: str) -> None:
        """Serialize model to file."""
        content = self.serialize(model)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def _serialize_time_spec(self, time_spec) -> Dict[str, Any]:
        """Serialize TimeSpecification object to dict."""
        if not time_spec:
            return {}

        return {
            'time_type': time_spec.time_type,
            'discretization': time_spec.discretization,
            'horizon': time_spec.horizon,
            'step_size': time_spec.step_size
        }

    def _serialize_ontology_mappings(self, mappings) -> List[Dict[str, Any]]:
        """Serialize ontology mappings to list of dicts."""
        if not mappings:
            return []

        result = []
        for mapping in mappings:
            if hasattr(mapping, 'variable_name'):
                result.append({
                    'variable_name': mapping.variable_name,
                    'ontology_term': mapping.ontology_term,
                    'description': mapping.description
                })
            else:
                result.append(str(mapping))
        return result

    def _create_embedded_model_data(self, model: GNNInternalRepresentation) -> Dict[str, Any]:
        """Create complete model data dict for embedding in format-specific comments."""
        return {
            'model_name': model.model_name,
            'version': model.version,
            'annotation': model.annotation,
            'variables': [
                {
                    'name': var.name,
                    'var_type': var.var_type.value if hasattr(var.var_type, 'value') else 'hidden_state',
                    'data_type': var.data_type.value if hasattr(var.data_type, 'value') else 'categorical',
                    'dimensions': var.dimensions,
                    'description': var.description
                }
                for var in model.variables
            ],
            'connections': [
                {
                    'source_variables': conn.source_variables,
                    'target_variables': conn.target_variables,
                    'connection_type': conn.connection_type.value if hasattr(conn.connection_type, 'value') else 'directed',
                    'weight': conn.weight,
                    'description': conn.description
                }
                for conn in model.connections
            ],
            'parameters': [
                {
                    'name': param.name,
                    'value': param.value,
                    'type_hint': param.type_hint,
                    'description': param.description
                }
                for param in model.parameters
            ],
            'equations': [
                {
                    'label': eq.label,
                    'content': eq.content,
                    'format': eq.format,
                    'description': eq.description
                }
                for eq in model.equations
            ],
            'time_specification': self._serialize_time_spec(model.time_specification),
            'ontology_mappings': self._serialize_ontology_mappings(model.ontology_mappings)
        }

    def _get_embedded_comment_prefix(self, format_name: str) -> str:
        """Get the comment prefix for embedding data in different formats."""
        comment_prefixes = {
            'json': '// MODEL_DATA: ',
            'xml': '<!-- MODEL_DATA: ',
            'yaml': '# MODEL_DATA: ',
            'scala': '// MODEL_DATA: ',
            'python': '# MODEL_DATA: ',
            'lean': '-- MODEL_DATA: ',
            'coq': '(* MODEL_DATA: ',
            'alloy': '/* MODEL_DATA: ',
            'asn1': '-- MODEL_DATA: ',
            'protobuf': '// MODEL_DATA: ',
            'haskell': '-- MODEL_DATA: ',
            'isabelle': '(* MODEL_DATA: ',
            'maxima': '/* MODEL_DATA: ',
            'tla': '\\* MODEL_DATA: ',
            'agda': '-- MODEL_DATA: ',
            'z_notation': '%% MODEL_DATA: ',
            'bnf': '; MODEL_DATA: ',
            'ebnf': '(* MODEL_DATA: '
        }
        return comment_prefixes.get(format_name.lower(), '# MODEL_DATA: ')

    def _get_embedded_comment_suffix(self, format_name: str) -> str:
        """Get the comment suffix for embedding data in different formats."""
        comment_suffixes = {
            'xml': ' -->',
            'coq': ' *)',
            'alloy': ' */',
            'isabelle': ' *)',
            'maxima': ' */',
            'ebnf': ' *)'
        }
        return comment_suffixes.get(format_name.lower(), '')

    def _add_embedded_model_data(self, content: str, model: GNNInternalRepresentation) -> str:
        """Add embedded model data to serialized content for round-trip fidelity."""
        model_data = self._create_embedded_model_data(model)
        json_data = json.dumps(model_data, separators=(',', ':'))

        prefix = self._get_embedded_comment_prefix(self.format_name)
        suffix = self._get_embedded_comment_suffix(self.format_name)

        comment_line = f"{prefix}{json_data}{suffix}"

        # Add the comment at the end of the content
        return content + '\n\n' + comment_line + '\n'
