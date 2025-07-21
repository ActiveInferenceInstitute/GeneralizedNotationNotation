from typing import Dict, Any, List, Optional, Union, Protocol
from abc import ABC, abstractmethod
import json
from datetime import datetime
from .common import GNNInternalRepresentation, GNNFormat
from .base_serializer import BaseGNNSerializer

class ZNotationSerializer(BaseGNNSerializer):
    """Serializer for Z notation formal specification language."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to Z notation format with embedded data."""
        lines = []
        
        # Z notation header
        model_name_clean = model.model_name.replace(" ", "").replace("-", "")
        lines.append(f"% Z Notation Specification for {model.model_name}")
        if model.annotation:
            lines.append(f"% {model.annotation}")
        lines.append("")
        
        # Schema definitions for variables
        if model.variables:
            lines.append("┌─ " + model_name_clean + " ─┐")
            for var in sorted(model.variables, key=lambda v: v.name):
                z_type = self._map_to_z_type(var.data_type.value)
                lines.append(f"│ {var.name}: {z_type}")
            lines.append("└─────────────────┘")
            lines.append("")
        
        # State schema
        lines.append("┌─ " + model_name_clean + "State ─┐")
        if model.variables:
            for var in sorted(model.variables, key=lambda v: v.name):
                lines.append(f"│ {var.name}: ℕ")
            lines.append("│")
            lines.append("│ // State constraints")
            for var in model.variables:
                lines.append(f"│ {var.name} ∈ ℕ")
        lines.append("└─────────────────┘")
        lines.append("")
        
        # Operations schema
        lines.append("┌─ " + model_name_clean + "Operation ─┐")
        lines.append("│ Δ" + model_name_clean + "State")
        lines.append("│")
        lines.append("│ // Operations preserve model invariants")
        lines.append("└─────────────────┘")
        lines.append("")
        
        # Embed complete model data as Z notation comment for round-trip fidelity
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
        
        # Add embedded JSON data as Z notation comment
        lines.append("% MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')))
        lines.append("")
        
        return '\n'.join(lines)
    
    def _map_to_z_type(self, data_type: str) -> str:
        """Map GNN data types to Z notation types."""
        mapping = {
            "categorical": "ℕ",
            "continuous": "ℝ",
            "binary": "𝔹",
            "integer": "ℤ",
            "float": "ℝ",
            "complex": "ℂ"
        }
        return mapping.get(data_type, "ℕ")
    
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