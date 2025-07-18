from typing import Dict, Any, List, Optional, Union, Protocol
from abc import ABC, abstractmethod
import pickle
import base64
import json
from datetime import datetime
from .common import GNNInternalRepresentation, GNNFormat
from .base_serializer import BaseGNNSerializer

class BinarySerializer(BaseGNNSerializer):
    """Serializer for binary formats (Pickle) with enhanced round-trip support."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to pickle format with embedded JSON data for round-trip."""
        # Create a comprehensive data structure for pickling
        # Complete model data for perfect round-trip
        complete_model_data = {
            'model_name': model.model_name,
            'version': model.version,
            'annotation': model.annotation,
            'variables': [
                {
                    'name': var.name,
                    'var_type': var.var_type.value if hasattr(var, 'var_type') else 'hidden_state',
                    'data_type': var.data_type.value if hasattr(var, 'data_type') else 'categorical',
                    'dimensions': var.dimensions if hasattr(var, 'dimensions') else []
                }
                for var in model.variables
            ],
            'connections': [
                {
                    'source_variables': conn.source_variables if hasattr(conn, 'source_variables') else [],
                    'target_variables': conn.target_variables if hasattr(conn, 'target_variables') else [],
                    'connection_type': conn.connection_type.value if hasattr(conn, 'connection_type') else 'directed'
                }
                for conn in model.connections
            ],
            'parameters': [
                {
                    'name': param.name,
                    'value': param.value,
                    'param_type': getattr(param, 'param_type', 'constant')
                }
                for param in model.parameters
            ],
            'equations': [str(eq) for eq in (model.equations if hasattr(model, 'equations') else [])],
            'time_specification': self._serialize_time_spec(model.time_specification) if hasattr(model, 'time_specification') and model.time_specification else None,
            'ontology_mappings': self._serialize_ontology_mappings(model.ontology_mappings) if hasattr(model, 'ontology_mappings') else [],
            'created_at': model.created_at.isoformat(),
            'modified_at': model.modified_at.isoformat(),
            'checksum': model.checksum
        }
        
        # Pickle the complete data structure
        pickled_data = pickle.dumps(complete_model_data)
        
        # Return as base64 encoded string for text representation
        return base64.b64encode(pickled_data).decode('ascii')
    
    def serialize_to_file(self, model: GNNInternalRepresentation, file_path: str) -> None:
        """Serialize directly to binary pickle file."""
        complete_model_data = {
            'model_name': model.model_name,
            'version': model.version,
            'annotation': model.annotation,
            'variables': [
                {
                    'name': var.name,
                    'var_type': var.var_type.value if hasattr(var, 'var_type') else 'hidden_state',
                    'data_type': var.data_type.value if hasattr(var, 'data_type') else 'categorical',
                    'dimensions': var.dimensions if hasattr(var, 'dimensions') else []
                }
                for var in model.variables
            ],
            'connections': [
                {
                    'source_variables': conn.source_variables if hasattr(conn, 'source_variables') else [],
                    'target_variables': conn.target_variables if hasattr(conn, 'target_variables') else [],
                    'connection_type': conn.connection_type.value if hasattr(conn, 'connection_type') else 'directed'
                }
                for conn in model.connections
            ],
            'parameters': [
                {
                    'name': param.name,
                    'value': param.value,
                    'param_type': getattr(param, 'param_type', 'constant')
                }
                for param in model.parameters
            ],
            'equations': [str(eq) for eq in (model.equations if hasattr(model, 'equations') else [])],
            'time_specification': self._serialize_time_spec(model.time_specification) if hasattr(model, 'time_specification') and model.time_specification else None,
            'ontology_mappings': self._serialize_ontology_mappings(model.ontology_mappings) if hasattr(model, 'ontology_mappings') else [],
            'created_at': model.created_at.isoformat(),
            'modified_at': model.modified_at.isoformat(),
            'checksum': model.checksum
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(complete_model_data, f)
    
    def serialize_pickle_direct(self, model: GNNInternalRepresentation, file_path: str) -> None:
        """Direct binary pickle serialization."""
        self.serialize_to_file(model, file_path)
    
    def _serialize_time_spec(self, time_spec):
        """Serialize time specification object."""
        if not time_spec:
            return None
        return {
            'time_type': getattr(time_spec, 'time_type', 'dynamic'),
            'discretization': getattr(time_spec, 'discretization', None),
            'horizon': getattr(time_spec, 'horizon', None),
            'step_size': getattr(time_spec, 'step_size', None)
        }
    
    def _serialize_ontology_mappings(self, mappings):
        """Serialize ontology mappings."""
        if not mappings:
            return []
        return [
            {
                'variable_name': mapping.variable_name if hasattr(mapping, 'variable_name') else str(mapping),
                'ontology_term': mapping.ontology_term if hasattr(mapping, 'ontology_term') else '',
                'description': getattr(mapping, 'description', None)
            }
            for mapping in mappings
        ] 