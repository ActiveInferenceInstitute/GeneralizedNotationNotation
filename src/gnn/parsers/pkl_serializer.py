from typing import Dict, Any, List, Optional, Union, Protocol
from abc import ABC, abstractmethod
from datetime import datetime
import json
from .common import GNNInternalRepresentation, GNNFormat
from .base_serializer import BaseGNNSerializer

class PKLSerializer(BaseGNNSerializer):
    """Enhanced serializer for Apple PKL configuration format with complete model preservation."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to PKL format with embedded model data."""
        lines = []
        
        # Module header with comprehensive documentation
        lines.append("///")
        lines.append(f"/// GNN Model: {model.model_name}")
        if model.annotation:
            lines.append(f"/// Annotation: {model.annotation}")
        lines.append(f"/// Generated: {datetime.now().isoformat()}")
        lines.append("/// Enhanced by GNN PKL Serializer")
        lines.append("///")
        lines.append("")
        lines.append('@ModuleInfo { minPklVersion = "0.25.0" }')
        lines.append("")
        
        # Model class with complete structure
        lines.append("class GNNModel {{")
        lines.append(f'  name: String = "{model.model_name}"')
        lines.append(f'  annotation: String = "{model.annotation}"')
        lines.append("")
        
        # Variables with complete type information
        if model.variables:
            lines.append("  variables: Mapping<String, Variable> = new Mapping {{")
            for var in sorted(model.variables, key=lambda v: v.name):
                dims_str = f"List({', '.join(map(str, var.dimensions))})" if hasattr(var, 'dimensions') and var.dimensions else "List()"
                var_type = var.var_type.value if hasattr(var, 'var_type') else 'hidden_state'
                data_type = var.data_type.value if hasattr(var, 'data_type') else 'categorical'
                lines.append(f'    ["{var.name}"] = new Variable {{')
                lines.append(f'      name = "{var.name}"')
                lines.append(f'      varType = "{var_type}"')
                lines.append(f'      dataType = "{data_type}"')
                lines.append(f'      dimensions = {dims_str}')
                lines.append("    }}")
            lines.append("  }}")
            lines.append("")
        
        # Connections with complete relationship data
        if model.connections:
            lines.append("  connections: Mapping<String, Connection> = new Mapping {{")
            for i, conn in enumerate(model.connections):
                sources = conn.source_variables if hasattr(conn, 'source_variables') else []
                targets = conn.target_variables if hasattr(conn, 'target_variables') else []
                conn_type = conn.connection_type.value if hasattr(conn, 'connection_type') else 'directed'
                
                quoted_sources = [f'"{s}"' for s in sources]
                quoted_targets = [f'"{t}"' for t in targets]
                sources_str = f"List({', '.join(quoted_sources)})" if sources else "List()"
                targets_str = f"List({', '.join(quoted_targets)})" if targets else "List()"
                
                lines.append(f'    ["connection_{i}"] = new Connection {{')
                lines.append(f'      sourceVariables = {sources_str}')
                lines.append(f'      targetVariables = {targets_str}')
                lines.append(f'      connectionType = "{conn_type}"')
                lines.append("    }}")
            lines.append("  }}")
            lines.append("")
        
        # Parameters with values and types
        if model.parameters:
            lines.append("  parameters: Mapping<String, Parameter> = new Mapping {{")
            for param in sorted(model.parameters, key=lambda p: p.name):
                value_str = self._format_pkl_value(param.value)
                param_type = getattr(param, 'param_type', 'constant')
                lines.append(f'    ["{param.name}"] = new Parameter {{')
                lines.append(f'      name = "{param.name}"')
                lines.append(f'      value = {value_str}')
                lines.append(f'      paramType = "{param_type}"')
                lines.append("    }}")
            lines.append("  }}")
            lines.append("")
        
        # Equations if present
        if hasattr(model, 'equations') and model.equations:
            lines.append("  equations: List<String> = new List {{")
            for eq in model.equations:
                if isinstance(eq, dict) and 'equation' in eq:
                    lines.append(f'    "{eq["equation"]}"')
                else:
                    lines.append(f'    "{str(eq)}"')
            lines.append("  }}")
            lines.append("")
        
        # Time specification if present
        if hasattr(model, 'time_specification') and model.time_specification:
            time_spec = model.time_specification
            if isinstance(time_spec, dict):
                lines.append("  timeSpecification: TimeSpec = new TimeSpec {{")
                lines.append(f'    timeType = "{time_spec.get("time_type", "discrete")}"')
                lines.append(f'    steps = {time_spec.get("steps", 1)}')
                if 'description' in time_spec:
                    lines.append(f'    description = "{time_spec["description"]}"')
                lines.append("  }}")
                lines.append("")
        
        # Ontology mappings if present
        if hasattr(model, 'ontology_mappings') and model.ontology_mappings:
            lines.append("  ontologyMappings: List<OntologyMapping> = new List {{")
            for mapping in model.ontology_mappings:
                if isinstance(mapping, dict):
                    var_name = mapping.get('variable_name', 'unknown')
                    ontology_term = mapping.get('ontology_term', 'unknown')
                    lines.append("    new OntologyMapping {{")
                    lines.append(f'      variableName = "{var_name}"')
                    lines.append(f'      ontologyTerm = "{ontology_term}"')
                    lines.append("    }}")
            lines.append("  }}")
            lines.append("")
        
        lines.append("}}")
        lines.append("")
        
        # Class definitions
        if model.variables:
            lines.append("class Variable {{")
            lines.append("  name: String")
            lines.append("  varType: String")
            lines.append("  dataType: String")
            lines.append("  dimensions: List<Int>")
            lines.append("}}")
            lines.append("")
        
        if model.connections:
            lines.append("class Connection {{")
            lines.append("  sourceVariables: List<String>")
            lines.append("  targetVariables: List<String>")
            lines.append("  connectionType: String")
            lines.append("}}")
            lines.append("")
        
        if model.parameters:
            lines.append("class Parameter {{")
            lines.append("  name: String")
            lines.append("  value: Any")
            lines.append("  paramType: String")
            lines.append("}}")
            lines.append("")
        
        if hasattr(model, 'time_specification') and model.time_specification:
            lines.append("class TimeSpec {{")
            lines.append("  timeType: String")
            lines.append("  steps: Int")
            lines.append("  description: String?")
            lines.append("}}")
            lines.append("")
        
        if hasattr(model, 'ontology_mappings') and model.ontology_mappings:
            lines.append("class OntologyMapping {{")
            lines.append("  variableName: String")
            lines.append("  ontologyTerm: String")
            lines.append("}}")
            lines.append("")
        
        # Embed complete model data as JSON comment for round-trip fidelity
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
        
        # Add embedded JSON data as block comment
        lines.append("/* MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')) + " */")
        lines.append("")
        
        # Add parsing hints as comments
        lines.append("// Variables:")
        for var in model.variables:
            lines.append(f"// Variable: {var.name} ({var.var_type.value if hasattr(var, 'var_type') else 'unknown'})")
        
        lines.append("// Connections:")
        for conn in model.connections:
            sources = ','.join(conn.source_variables) if hasattr(conn, 'source_variables') else 'unknown'
            targets = ','.join(conn.target_variables) if hasattr(conn, 'target_variables') else 'unknown'
            conn_type = conn.connection_type.value if hasattr(conn, 'connection_type') else 'directed'
            lines.append(f"// Connection: {sources} --{conn_type}--> {targets}")
        
        lines.append("// Parameters:")
        for param in model.parameters:
            lines.append(f"// Parameter: {param.name} = {param.value}")
        
        return '\n'.join(lines)
    
    def _format_pkl_value(self, value) -> str:
        """Format a value for PKL syntax."""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            formatted_items = []
            for item in value:
                if isinstance(item, str):
                    formatted_items.append(f'"{item}"')
                else:
                    formatted_items.append(str(item))
            return f"List({', '.join(formatted_items)})"
        else:
            return f'"{str(value)}"' 