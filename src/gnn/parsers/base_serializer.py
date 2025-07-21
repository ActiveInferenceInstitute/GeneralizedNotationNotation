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
            'time_type': getattr(time_spec, 'time_type', 'Static'),
            'discretization': getattr(time_spec, 'discretization', None),
            'horizon': getattr(time_spec, 'horizon', None),
            'step_size': getattr(time_spec, 'step_size', None)
        }
    
    def _serialize_ontology_mappings(self, mappings) -> List[Dict[str, Any]]:
        """Serialize ontology mappings to list of dicts."""
        if not mappings:
            return []
        
        result = []
        for mapping in mappings:
            if hasattr(mapping, '__dict__'):
                result.append({
                    'variable_name': getattr(mapping, 'variable_name', ''),
                    'ontology_term': getattr(mapping, 'ontology_term', ''),
                    'description': getattr(mapping, 'description', None)
                })
            else:
                result.append(str(mapping))
        return result
    
    def _create_embedded_model_data(self, model: GNNInternalRepresentation) -> Dict[str, Any]:
        """Create complete model data dict for embedding in format-specific comments."""
        return {
            'model_name': model.model_name,
            'version': getattr(model, 'version', '1.0'),
            'annotation': model.annotation,
            'variables': [
                {
                    'name': var.name,
                    'var_type': var.var_type.value if hasattr(var, 'var_type') and hasattr(var.var_type, 'value') else 'hidden_state',
                    'data_type': var.data_type.value if hasattr(var, 'data_type') and hasattr(var.data_type, 'value') else 'categorical',
                    'dimensions': getattr(var, 'dimensions', []),
                    'description': getattr(var, 'description', None)
                }
                for var in model.variables
            ],
            'connections': [
                {
                    'source_variables': getattr(conn, 'source_variables', []),
                    'target_variables': getattr(conn, 'target_variables', []),
                    'connection_type': conn.connection_type.value if hasattr(conn, 'connection_type') and hasattr(conn.connection_type, 'value') else 'directed',
                    'weight': getattr(conn, 'weight', None),
                    'description': getattr(conn, 'description', None)
                }
                for conn in model.connections
            ],
            'parameters': [
                {
                    'name': param.name,
                    'value': getattr(param, 'value', None),
                    'type_hint': getattr(param, 'type_hint', None),
                    'description': getattr(param, 'description', None)
                }
                for param in model.parameters
            ],
            'equations': [
                {
                    'label': getattr(eq, 'label', None),
                    'content': getattr(eq, 'content', str(eq)),
                    'format': getattr(eq, 'format', 'latex'),
                    'description': getattr(eq, 'description', None)
                }
                for eq in getattr(model, 'equations', [])
            ],
            'time_specification': self._serialize_time_spec(getattr(model, 'time_specification', None)),
            'ontology_mappings': self._serialize_ontology_mappings(getattr(model, 'ontology_mappings', []))
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