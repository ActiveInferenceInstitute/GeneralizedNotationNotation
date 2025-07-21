from typing import Dict, Any, List, Optional, Union, Protocol
from abc import ABC, abstractmethod
import json
from datetime import datetime
from .common import GNNInternalRepresentation, GNNFormat
from .base_serializer import BaseGNNSerializer

class GrammarSerializer(BaseGNNSerializer):
    """Serializer for BNF/EBNF grammar format with embedded data support."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to BNF format with embedded data."""
        lines = []
        
        # Header
        lines.append(f"# Grammar for GNN Model: {model.model_name}")
        lines.append(f"# {model.annotation}")
        lines.append("")
        
        # Root rule
        lines.append("<gnn_model> ::= <variables> <connections> <parameters>")
        lines.append("")
        
        # Variables
        if model.variables:
            lines.append("<variables> ::= <variable> | <variable> <variables>")
            lines.append("<variable> ::= <variable_name> <variable_type>")
            var_names = []
            for var in sorted(model.variables, key=lambda v: v.name):
                var_names.append(f'"{var.name}"')
            lines.append("<variable_name> ::= " + " | ".join(var_names))
            lines.append("")
        
        # Connections
        if model.connections:
            lines.append("<connections> ::= <connection> | <connection> <connections>")
            lines.append("<connection> ::= <source_var> <connection_op> <target_var>")
            lines.append('<connection_op> ::= ">" | "-" | "->"')
            lines.append("")
        
        # Parameters  
        if model.parameters:
            lines.append("<parameters> ::= <parameter> | <parameter> <parameters>")
            lines.append("<parameter> ::= <param_name> = <param_value>")
            param_names = []
            for param in sorted(model.parameters, key=lambda p: p.name):
                param_names.append(f'"{param.name}"')
            lines.append("<param_name> ::= " + " | ".join(param_names))
            lines.append("")
        
        # Embed complete model data as BNF comment for round-trip fidelity
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
        
        # Add embedded JSON data as BNF comment
        lines.append("# MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')))
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