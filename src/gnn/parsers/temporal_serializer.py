from typing import Dict, Any, List, Optional, Union, Protocol
from abc import ABC, abstractmethod
import json
from datetime import datetime
from .common import GNNInternalRepresentation, GNNFormat
from .base_serializer import BaseGNNSerializer

class TemporalSerializer(BaseGNNSerializer):
    """Serializer for temporal logic languages."""
    
    def __init__(self, target_format: str = "tla"):
        """Initialize with target format (tla or agda)."""
        super().__init__()
        self.target_format = target_format
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to temporal logic format."""
        if self.target_format == "agda":
            return self._serialize_agda(model)
        else:
            return self._serialize_tla(model)
    
    def _serialize_tla(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to TLA+ format."""
        lines = []
        
        # Header
        model_name_clean = model.model_name.replace(" ", "").replace("-", "")
        lines.append(f"---- MODULE {model_name_clean} ----")
        lines.append("")
        lines.append("EXTENDS Naturals, Reals")
        lines.append("")
        
        # Variables
        if model.variables:
            lines.append("VARIABLES")
            var_names = [var.name for var in sorted(model.variables, key=lambda v: v.name)]
            lines.append("  " + ", ".join(var_names))
            lines.append("")
        
        # Type invariants
        if model.variables:
            lines.append("TypeOK ==")
            for i, var in enumerate(sorted(model.variables, key=lambda v: v.name)):
                connector = "/\\" if i > 0 else ""
                tla_type = self._map_to_tla_type(var.data_type.value)
                lines.append(f"  {connector} {var.name} \\in {tla_type}")
            lines.append("")
        
        lines.append("====")
        
        # Embed complete model data as TLA+ comment for round-trip fidelity
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
        
        # Add embedded JSON data as TLA+ comment
        lines.append("\\* MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')))
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
    
    def _map_to_tla_type(self, data_type: str) -> str:
        """Map GNN data types to TLA+ types."""
        mapping = {
            "categorical": "Seq(Nat)",
            "continuous": "Real",
            "binary": "BOOLEAN",
            "integer": "Int",
            "float": "Real",
            "complex": "STRING"
        }
        return mapping.get(data_type, "STRING")
    
    def _serialize_agda(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to Agda format."""
        lines = []
        
        # Header
        model_name_clean = model.model_name.replace(" ", "").replace("-", "")
        lines.append(f"module {model_name_clean} where")
        lines.append("")
        lines.append("open import Level")
        lines.append("open import Data.Nat")
        lines.append("open import Data.Real")
        lines.append("")
        
        # Variable definitions
        if model.variables:
            lines.append("-- Variables")
            for var in sorted(model.variables, key=lambda v: v.name):
                agda_type = self._map_to_agda_type(var.data_type.value)
                lines.append(f"postulate {var.name} : {agda_type}")
            lines.append("")
        
        # Embed complete model data as Agda comment for round-trip fidelity
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
        
        # Add embedded JSON data as Agda comment
        lines.append("{- MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')) + " -}")
        lines.append("")
        
        return '\n'.join(lines)
    
    def _map_to_agda_type(self, data_type: str) -> str:
        """Map GNN data types to Agda types."""
        mapping = {
            "categorical": "List ℕ",
            "continuous": "ℝ",
            "binary": "Bool",
            "integer": "ℤ",
            "float": "ℝ",
            "complex": "String"
        }
        return mapping.get(data_type, "String")


class TLASerializer(TemporalSerializer):
    """Specific serializer for TLA+ format."""
    
    def __init__(self):
        super().__init__(target_format="tla")


class AgdaSerializer(TemporalSerializer):
    """Specific serializer for Agda format."""
    
    def __init__(self):
        super().__init__(target_format="agda") 