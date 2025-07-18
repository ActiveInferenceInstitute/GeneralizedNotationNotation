from typing import Dict, Any, List, Optional, Union, Protocol
from abc import ABC, abstractmethod
import json
from datetime import datetime
from .common import GNNInternalRepresentation, GNNFormat
from .base_serializer import BaseGNNSerializer

class XSDSerializer(BaseGNNSerializer):
    """Serializer for XML Schema Definition format."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to XSD format."""
        lines = []
        
        # XML Schema header
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append('<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"')
        lines.append(f'           targetNamespace="https://gnn/models/{model.model_name.lower().replace(" ", "_")}"')
        lines.append('           elementFormDefault="qualified">')
        lines.append("")
        
        # Model documentation
        lines.append("  <xs:annotation>")
        lines.append("    <xs:documentation>")
        lines.append(f"      Generated GNN Schema for: {model.model_name}")
        lines.append(f"      {model.annotation}")
        lines.append("    </xs:documentation>")
        lines.append("  </xs:annotation>")
        lines.append("")
        
        # Root element
        lines.append("  <xs:element name=\"gnn_model\">")
        lines.append("    <xs:complexType>")
        lines.append("      <xs:sequence>")
        
        # Variables
        if model.variables:
            lines.append("        <xs:element name=\"variables\">")
            lines.append("          <xs:complexType>")
            lines.append("            <xs:sequence>")
            for var in sorted(model.variables, key=lambda v: v.name):
                xsd_type = self._map_to_xsd_type(var.data_type.value)
                lines.append(f'              <xs:element name="{var.name}" type="{xsd_type}"/>')
            lines.append("            </xs:sequence>")
            lines.append("          </xs:complexType>")
            lines.append("        </xs:element>")
        
        lines.append("      </xs:sequence>")
        lines.append("    </xs:complexType>")
        lines.append("  </xs:element>")
        lines.append("")
        
        # Embed complete model data as XML comment for round-trip fidelity
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
        
        # Add embedded JSON data as XML comment
        lines.append("<!-- MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')) + " -->")
        lines.append("")
        lines.append("</xs:schema>")
        
        return '\n'.join(lines)
    
    def _serialize_time_spec(self, time_spec) -> Dict[str, Any]:
        """Serialize TimeSpecification object to dict."""
        if not time_spec or not hasattr(time_spec, '__dict__'):
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
    
    def _map_to_xsd_type(self, data_type: str) -> str:
        """Map GNN data types to XSD types."""
        mapping = {
            "categorical": "xs:string",
            "continuous": "xs:double",
            "binary": "xs:boolean",
            "integer": "xs:int",
            "float": "xs:double",
            "complex": "xs:string"
        }
        return mapping.get(data_type, "xs:string") 