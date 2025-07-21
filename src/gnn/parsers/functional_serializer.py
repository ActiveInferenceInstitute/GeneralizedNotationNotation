from typing import Dict, Any, List, Optional, Union, Protocol
from abc import ABC, abstractmethod
import json
from datetime import datetime
from .common import GNNInternalRepresentation, GNNFormat
from .base_serializer import BaseGNNSerializer

class FunctionalSerializer(BaseGNNSerializer):
    """Serializer for functional programming languages."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to Haskell format."""
        lines = []
        
        # Module declaration
        model_name_clean = model.model_name.replace(" ", "").replace("-", "")
        lines.append(f"module {model_name_clean} where")
        lines.append("")
        
        # Imports
        lines.append("import Data.List (sort)")
        lines.append("import Numeric.LinearAlgebra ()")
        lines.append("")
        
        # Data types for variables
        if model.variables:
            lines.append("-- Variable Types")
            for var in sorted(model.variables, key=lambda v: v.name):
                haskell_type = self._map_to_haskell_type(var.data_type.value)
                lines.append(f"data {var.name} = {var.name} {haskell_type}")
            lines.append("")
        
        # Connections as functions
        if model.connections:
            lines.append("-- Connections as Functions")
            for conn in model.connections:
                if hasattr(conn, 'source_variables') and hasattr(conn, 'target_variables'):
                    sources = conn.source_variables
                    targets = conn.target_variables
                    for source in sources:
                        for target in targets:
                            lines.append(f"{source}To{target} :: {source} -> {target}")
                            lines.append(f"{source}To{target} x = undefined  -- TODO: implement connection")
            lines.append("")
        
        # Embed complete model data as Haskell comment for round-trip fidelity
        model_data = {
            'model_name': model.model_name,
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
            'ontology_mappings': self._serialize_ontology_mappings(model.ontology_mappings) if hasattr(model, 'ontology_mappings') else []
        }
        
        # Add embedded JSON data as Haskell comment
        lines.append("-- MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')))
        lines.append("")
        
        return '\n'.join(lines)
    
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
    
    def _map_to_haskell_type(self, data_type: str) -> str:
        """Map GNN data types to Haskell types."""
        mapping = {
            "categorical": "[Int]",
            "continuous": "Double",
            "binary": "Bool",
            "integer": "Int",
            "float": "Double",
            "complex": "String"
        }
        return mapping.get(data_type, "String") 