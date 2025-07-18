"""
GNN Serializers - Multi-Format Output Generation

This module provides serialization capabilities for converting GNN internal
representations back to all supported formats.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

from typing import Dict, Any, List, Optional, Union, Protocol
from abc import ABC, abstractmethod
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pickle
import hashlib
from datetime import datetime
import base64

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .common import GNNInternalRepresentation, GNNFormat, ASTNode

# Base Serializer Protocol
class GNNSerializer(Protocol):
    """Protocol for all GNN serializers."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Serialize model to string format."""
        ...
    
    def serialize_to_file(self, model: GNNInternalRepresentation, file_path: str) -> None:
        """Serialize model directly to file."""
        ...

# Base Serializer Class
class BaseGNNSerializer(ABC):
    """Base class for all GNN serializers."""
    
    def __init__(self):
        """Initialize the base serializer."""
        pass
    
    @abstractmethod
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Serialize model to string format."""
        pass
    
    def serialize_to_file(self, model: GNNInternalRepresentation, file_path: str) -> None:
        """Serialize model directly to file."""
        content = self.serialize(model)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

# Markdown Serializer
class MarkdownSerializer(BaseGNNSerializer):
    """Serializer for GNN Markdown format."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model back to Markdown format."""
        sections = []
        
        # GNNVersionAndFlags
        sections.append("## GNNVersionAndFlags")
        sections.append("Version: 1.0")
        sections.append("")
        
        # ModelName
        sections.append("## ModelName")
        sections.append(model.model_name)
        sections.append("")
        
        # ModelAnnotation
        if model.annotation:
            sections.append("## ModelAnnotation")
            sections.append(model.annotation)
            sections.append("")
        
        # StateSpaceBlock
        if model.variables:
            sections.append("## StateSpaceBlock")
            for var in model.variables:
                dims_str = f"[{','.join(map(str, var.dimensions))}]" if var.dimensions else ""
                sections.append(f"{var.name}{dims_str},{var.data_type.value}")
            sections.append("")
        
        # Connections
        if model.connections:
            sections.append("## Connections")
            for conn in model.connections:
                if conn.connection_type.value == "directed":
                    op = ">"
                elif conn.connection_type.value == "undirected":
                    op = "-"
                else:
                    op = "->"
                
                for src in conn.source_variables:
                    for tgt in conn.target_variables:
                        sections.append(f"{src}{op}{tgt}")
            sections.append("")
        
        # InitialParameterization
        if model.parameters:
            sections.append("## InitialParameterization")
            for param in model.parameters:
                sections.append(f"{param.name} = {param.value}")
            sections.append("")
        
        # Equations
        if model.equations:
            sections.append("## Equations")
            for eq in model.equations:
                if eq.label:
                    sections.append(f"**{eq.label}:**")
                sections.append(f"$${eq.content}$$")
                sections.append("")
        
        # Time
        if model.time_specification:
            sections.append("## Time")
            sections.append(model.time_specification.time_type)
            if model.time_specification.discretization:
                sections.append(model.time_specification.discretization)
            if model.time_specification.horizon:
                sections.append(f"ModelTimeHorizon = {model.time_specification.horizon}")
            sections.append("")
        
        # ActInfOntologyAnnotation
        if model.ontology_mappings:
            sections.append("## ActInfOntologyAnnotation")
            for mapping in model.ontology_mappings:
                sections.append(f"{mapping.variable_name} = {mapping.ontology_term}")
            sections.append("")
        
        # Footer
        sections.append("## Footer")
        sections.append(f"Generated: {datetime.now().isoformat()}")
        sections.append("")
        
        # Signature
        sections.append("## Signature")
        if model.checksum:
            sections.append(f"Checksum: {model.checksum}")
        sections.append("")
        
        return "\n".join(sections)

# JSON Serializer
class JSONSerializer(BaseGNNSerializer):
    """Serializer for JSON data interchange format."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to JSON format."""
        return json.dumps(model.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)

# XML Serializer  
class XMLSerializer(BaseGNNSerializer):
    """Serializer for XML format with deterministic output."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to XML format with consistent ordering."""
        root = ET.Element("gnn_model")
        
        # Set attributes in alphabetical order for consistency
        root.set("name", model.model_name)
        root.set("version", model.version)
        
        # Metadata - always first
        metadata = ET.SubElement(root, "metadata")
        ET.SubElement(metadata, "annotation").text = model.annotation or ""
        ET.SubElement(metadata, "created_at").text = model.created_at.isoformat()
        ET.SubElement(metadata, "modified_at").text = model.modified_at.isoformat()
        
        # Variables - sorted by name for consistency
        if model.variables:
            variables = ET.SubElement(root, "variables")
            sorted_vars = sorted(model.variables, key=lambda v: v.name)
            for var in sorted_vars:
                var_elem = ET.SubElement(variables, "variable")
                # Set attributes in alphabetical order
                var_elem.set("data_type", var.data_type.value)
                if var.dimensions:
                    var_elem.set("dimensions", ",".join(map(str, var.dimensions)))
                var_elem.set("name", var.name)
                var_elem.set("type", var.var_type.value)
                if var.description:
                    var_elem.set("description", var.description)
        
        # Connections - sorted by source then target for consistency
        if model.connections:
            connections = ET.SubElement(root, "connections")
            sorted_conns = sorted(model.connections, key=lambda c: (
                ",".join(sorted(c.source_variables)), 
                ",".join(sorted(c.target_variables))
            ))
            for conn in sorted_conns:
                conn_elem = ET.SubElement(connections, "connection")
                conn_elem.set("type", conn.connection_type.value)
                if conn.weight is not None:
                    conn_elem.set("weight", str(conn.weight))
                
                # Create child elements in consistent order
                sources_elem = ET.SubElement(conn_elem, "sources")
                sources_elem.text = ",".join(sorted(conn.source_variables))
                
                targets_elem = ET.SubElement(conn_elem, "targets")
                targets_elem.text = ",".join(sorted(conn.target_variables))
                
                if conn.description:
                    desc_elem = ET.SubElement(conn_elem, "description")
                    desc_elem.text = conn.description
        
        # Parameters - sorted by name
        if model.parameters:
            parameters = ET.SubElement(root, "parameters")
            sorted_params = sorted(model.parameters, key=lambda p: p.name)
            for param in sorted_params:
                param_elem = ET.SubElement(parameters, "parameter")
                param_elem.set("name", param.name)
                if param.type_hint:
                    param_elem.set("type", param.type_hint)
                param_elem.text = str(param.value)
                if param.description:
                    param_elem.set("description", param.description)
        
        # Equations - sorted by label
        if model.equations:
            equations = ET.SubElement(root, "equations")
            sorted_eqs = sorted(model.equations, key=lambda e: e.label or "")
            for eq in sorted_eqs:
                eq_elem = ET.SubElement(equations, "equation")
                if eq.format:
                    eq_elem.set("format", eq.format)
                if eq.label:
                    eq_elem.set("label", eq.label)
                eq_elem.text = eq.content
                if eq.description:
                    eq_elem.set("description", eq.description)
        
        # Time Specification
        if model.time_specification:
            time_elem = ET.SubElement(root, "time_specification")
            if model.time_specification.discretization:
                time_elem.set("discretization", model.time_specification.discretization)
            if model.time_specification.horizon:
                time_elem.set("horizon", str(model.time_specification.horizon))
            if model.time_specification.step_size:
                time_elem.set("step_size", str(model.time_specification.step_size))
            time_elem.set("type", model.time_specification.time_type)
        
        # Ontology Mappings - sorted by variable name
        if model.ontology_mappings:
            ontology = ET.SubElement(root, "ontology_mappings")
            sorted_mappings = sorted(model.ontology_mappings, key=lambda m: m.variable_name)
            for mapping in sorted_mappings:
                mapping_elem = ET.SubElement(ontology, "mapping")
                mapping_elem.set("term", mapping.ontology_term)
                mapping_elem.set("variable", mapping.variable_name)
                if mapping.description:
                    mapping_elem.set("description", mapping.description)
        
        # Convert to string with consistent formatting
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        
        # Get pretty XML and normalize whitespace
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # Remove empty lines and normalize
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        
        return '\n'.join(lines)

# YAML Serializer
class YAMLSerializer(BaseGNNSerializer):
    """Serializer for YAML configuration format."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to YAML format."""
        # Create a cleaner structure for YAML with consistent ordering
        data = {
            'model_name': model.model_name,
            'version': model.version,
            'annotation': model.annotation,
            'variables': [
                {
                    'name': var.name,
                    'type': var.var_type.value,
                    'data_type': var.data_type.value,
                    'dimensions': var.dimensions,
                    'description': var.description
                }
                for var in sorted(model.variables, key=lambda v: v.name)
            ],
            'connections': [
                {
                    'source_variables': sorted(conn.source_variables),
                    'target_variables': sorted(conn.target_variables),
                    'connection_type': conn.connection_type.value,
                    'weight': conn.weight,
                    'description': conn.description
                }
                for conn in sorted(model.connections, key=lambda c: (
                    ",".join(sorted(c.source_variables)), 
                    ",".join(sorted(c.target_variables))
                ))
            ],
            'parameters': [
                {
                    'name': param.name,
                    'value': param.value,
                    'type_hint': param.type_hint,
                    'description': param.description
                }
                for param in sorted(model.parameters, key=lambda p: p.name)
            ],
            'equations': [
                {
                    'content': eq.content,
                    'label': eq.label,
                    'format': eq.format,
                    'description': eq.description
                }
                for eq in sorted(model.equations, key=lambda e: e.label or "")
            ],
            'time_specification': {
                'time_type': model.time_specification.time_type,
                'discretization': model.time_specification.discretization,
                'horizon': model.time_specification.horizon,
                'step_size': model.time_specification.step_size
            } if model.time_specification else None,
            'ontology_mappings': [
                {
                    'variable_name': mapping.variable_name,
                    'ontology_term': mapping.ontology_term,
                    'description': mapping.description
                }
                for mapping in sorted(model.ontology_mappings, key=lambda m: m.variable_name)
            ],
            'created_at': model.created_at.isoformat(),
            'modified_at': model.modified_at.isoformat()
        }
        
        if not HAS_YAML:
            return self._dict_to_yaml_like(data)
        
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=True)
    
    def _dict_to_yaml_like(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Convert dict to YAML-like format when PyYAML is not available."""
        lines = []
        spaces = "  " * indent
        
        for key, value in sorted(data.items()) if isinstance(data, dict) else data.items():
            if isinstance(value, dict):
                lines.append(f"{spaces}{key}:")
                lines.append(self._dict_to_yaml_like(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{spaces}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{spaces}- ")
                        lines.append(self._dict_to_yaml_like(item, indent + 1))
                    else:
                        lines.append(f"{spaces}- {item}")
            else:
                lines.append(f"{spaces}{key}: {value}")
        
        return "\n".join(lines)

# Scala Serializer
class ScalaSerializer(BaseGNNSerializer):
    """Serializer for Scala categorical specifications."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to Scala categorical format."""
        lines = []
        
        # Package and imports
        lines.append("package gnn.categorical")
        lines.append("")
        lines.append("import cats._")
        lines.append("import cats.implicits._")
        lines.append("import cats.arrow.Category")
        lines.append("")
        
        # Model object
        model_name_clean = model.model_name.replace(" ", "").replace("-", "")
        lines.append(f"object {model_name_clean}Model {{")
        lines.append("")
        
        # State space definition
        if model.variables:
            lines.append("  // State Space")
            for var in sorted(model.variables, key=lambda v: v.name):
                var_type = self._map_variable_type(var)
                lines.append(f"  type {var.name} = {var_type}")
            lines.append("")
        
        # Morphisms (connections)
        if model.connections:
            lines.append("  // Morphisms")
            sorted_conns = sorted(model.connections, key=lambda c: (
                ",".join(sorted(c.source_variables)), 
                ",".join(sorted(c.target_variables))
            ))
            for conn in sorted_conns:
                for src in sorted(conn.source_variables):
                    for tgt in sorted(conn.target_variables):
                        lines.append(f"  val {src}To{tgt}: {src} => {tgt} = identity")
            lines.append("")
        
        lines.append("}")
        
        # Embed complete model data as Scala comment for round-trip fidelity
        import json
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
        
        # Add embedded JSON data as Scala comment
        lines.append("// MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')))
        lines.append("")
        
        return "\n".join(lines)
    
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
    
    def _map_variable_type(self, var) -> str:
        """Map GNN variable types to Scala types."""
        if var.data_type.value == "categorical":
            return "List[Double]"
        elif var.data_type.value == "continuous":
            return "Double"
        elif var.data_type.value == "binary":
            return "Boolean"
        else:
            return "Any"

# Protocol Buffers Serializer - Enhanced with Complete Model Preservation
class ProtobufSerializer(BaseGNNSerializer):
    """Enhanced serializer for Protocol Buffers format with complete model preservation."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to Protocol Buffers format with embedded model data."""
        lines = []
        
        # Proto3 syntax
        lines.append('syntax = "proto3";')
        lines.append("")
        
        # Package declaration
        package_name = model.model_name.lower().replace(" ", "_").replace("-", "_")
        lines.append(f"package {package_name};")
        lines.append("")
        
        # Add comprehensive model documentation as comments
        lines.append("// GNN Model: " + model.model_name)
        if model.annotation:
            lines.append("// Annotation: " + model.annotation)
        lines.append("// Generated by GNN Protobuf Serializer")
        lines.append("")
        
        # Variable message
        lines.append("message Variable {")
        lines.append("  string name = 1;")
        lines.append("  string var_type = 2;")
        lines.append("  string data_type = 3;")
        lines.append("  repeated int32 dimensions = 4;")
        lines.append("  string description = 5;")
        lines.append("}")
        lines.append("")
        
        # Connection message
        lines.append("message Connection {")
        lines.append("  repeated string source_variables = 1;")
        lines.append("  repeated string target_variables = 2;")
        lines.append("  string connection_type = 3;")
        lines.append("  string description = 4;")
        lines.append("}")
        lines.append("")
        
        # Parameter message
        lines.append("message Parameter {")
        lines.append("  string name = 1;")
        lines.append("  string value = 2;")
        lines.append("  string param_type = 3;")
        lines.append("}")
        lines.append("")
        
        # Equation message
        lines.append("message Equation {")
        lines.append("  string equation = 1;")
        lines.append("  string type = 2;")
        lines.append("}")
        lines.append("")
        
        # Time specification message
        lines.append("message TimeSpecification {")
        lines.append("  string time_type = 1;")
        lines.append("  int32 steps = 2;")
        lines.append("  string description = 3;")
        lines.append("}")
        lines.append("")
        
        # Ontology mapping message
        lines.append("message OntologyMapping {")
        lines.append("  string variable_name = 1;")
        lines.append("  string ontology_term = 2;")
        lines.append("}")
        lines.append("")
        
        # Main model message
        lines.append("message GNNModel {")
        lines.append("  string name = 1;")
        lines.append("  string annotation = 2;")
        lines.append("  repeated Variable variables = 3;")
        lines.append("  repeated Connection connections = 4;")
        lines.append("  repeated Parameter parameters = 5;")
        lines.append("  repeated Equation equations = 6;")
        lines.append("  TimeSpecification time_specification = 7;")
        lines.append("  repeated OntologyMapping ontology_mappings = 8;")
        lines.append("}")
        lines.append("")
        
        # Embed complete model data as JSON in comments for round-trip fidelity
        import json
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
        
        # Add embedded JSON data for complete round-trip
        lines.append("/* MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')) + " */")
        lines.append("")
        
        # Add individual variable, connection, and parameter comments for parsing
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
        
        return "\n".join(lines)
    
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

# PKL (Apple's configuration language) Serializer - Enhanced with Complete Model Preservation
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
        lines.append("class GNNModel {")
        lines.append(f'  name: String = "{model.model_name}"')
        lines.append(f'  annotation: String = "{model.annotation}"')
        lines.append("")
        
        # Variables with complete type information
        if model.variables:
            lines.append("  variables: Mapping<String, Variable> = new Mapping {")
            for var in sorted(model.variables, key=lambda v: v.name):
                dims_str = f"List({', '.join(map(str, var.dimensions))})" if hasattr(var, 'dimensions') and var.dimensions else "List()"
                var_type = var.var_type.value if hasattr(var, 'var_type') else 'hidden_state'
                data_type = var.data_type.value if hasattr(var, 'data_type') else 'categorical'
                lines.append(f'    ["{var.name}"] = new Variable {{')
                lines.append(f'      name = "{var.name}"')
                lines.append(f'      varType = "{var_type}"')
                lines.append(f'      dataType = "{data_type}"')
                lines.append(f'      dimensions = {dims_str}')
                lines.append("    }")
            lines.append("  }")
            lines.append("")
        
        # Connections with complete relationship data
        if model.connections:
            lines.append("  connections: Mapping<String, Connection> = new Mapping {")
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
                lines.append("    }")
            lines.append("  }")
            lines.append("")
        
        # Parameters with values and types
        if model.parameters:
            lines.append("  parameters: Mapping<String, Parameter> = new Mapping {")
            for param in sorted(model.parameters, key=lambda p: p.name):
                value_str = self._format_pkl_value(param.value)
                param_type = getattr(param, 'param_type', 'constant')
                lines.append(f'    ["{param.name}"] = new Parameter {{')
                lines.append(f'      name = "{param.name}"')
                lines.append(f'      value = {value_str}')
                lines.append(f'      paramType = "{param_type}"')
                lines.append("    }")
            lines.append("  }")
            lines.append("")
        
        # Equations if present
        if hasattr(model, 'equations') and model.equations:
            lines.append("  equations: List<String> = new List {")
            for eq in model.equations:
                if isinstance(eq, dict) and 'equation' in eq:
                    lines.append(f'    "{eq["equation"]}"')
                else:
                    lines.append(f'    "{str(eq)}"')
            lines.append("  }")
            lines.append("")
        
        # Time specification if present
        if hasattr(model, 'time_specification') and model.time_specification:
            time_spec = model.time_specification
            if isinstance(time_spec, dict):
                lines.append("  timeSpecification: TimeSpec = new TimeSpec {")
                lines.append(f'    timeType = "{time_spec.get("time_type", "discrete")}"')
                lines.append(f'    steps = {time_spec.get("steps", 1)}')
                if 'description' in time_spec:
                    lines.append(f'    description = "{time_spec["description"]}"')
                lines.append("  }")
                lines.append("")
        
        # Ontology mappings if present
        if hasattr(model, 'ontology_mappings') and model.ontology_mappings:
            lines.append("  ontologyMappings: List<OntologyMapping> = new List {")
            for mapping in model.ontology_mappings:
                if isinstance(mapping, dict):
                    var_name = mapping.get('variable_name', 'unknown')
                    ontology_term = mapping.get('ontology_term', 'unknown')
                    lines.append("    new OntologyMapping {")
                    lines.append(f'      variableName = "{var_name}"')
                    lines.append(f'      ontologyTerm = "{ontology_term}"')
                    lines.append("    }")
            lines.append("  }")
            lines.append("")
        
        lines.append("}")
        lines.append("")
        
        # Class definitions
        if model.variables:
            lines.append("class Variable {")
            lines.append("  name: String")
            lines.append("  varType: String")
            lines.append("  dataType: String")
            lines.append("  dimensions: List<Int>")
            lines.append("}")
            lines.append("")
        
        if model.connections:
            lines.append("class Connection {")
            lines.append("  sourceVariables: List<String>")
            lines.append("  targetVariables: List<String>")
            lines.append("  connectionType: String")
            lines.append("}")
            lines.append("")
        
        if model.parameters:
            lines.append("class Parameter {")
            lines.append("  name: String")
            lines.append("  value: Any")
            lines.append("  paramType: String")
            lines.append("}")
            lines.append("")
        
        if hasattr(model, 'time_specification') and model.time_specification:
            lines.append("class TimeSpec {")
            lines.append("  timeType: String")
            lines.append("  steps: Int")
            lines.append("  description: String?")
            lines.append("}")
            lines.append("")
        
        if hasattr(model, 'ontology_mappings') and model.ontology_mappings:
            lines.append("class OntologyMapping {")
            lines.append("  variableName: String")
            lines.append("  ontologyTerm: String")
            lines.append("}")
            lines.append("")
        
        # Embed complete model data as JSON comment for round-trip fidelity
        import json
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
        
        return "\n".join(lines)
    
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

# XSD (XML Schema) Serializer
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
        import json
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
        
        return "\n".join(lines)
    
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

# ASN.1 Serializer
class ASN1Serializer(BaseGNNSerializer):
    """Serializer for ASN.1 format."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to ASN.1 format."""
        lines = []
        
        # Module header
        model_name_clean = model.model_name.replace(" ", "").replace("-", "")
        lines.append(f"{model_name_clean}-Schema DEFINITIONS AUTOMATIC TAGS ::= BEGIN")
        lines.append("")
        
        # Main model structure
        lines.append("GNNModel ::= SEQUENCE {")
        lines.append("    modelName       UTF8String,")
        lines.append("    version         UTF8String,")
        lines.append("    annotation      UTF8String OPTIONAL,")
        
        if model.variables:
            lines.append("    variables       SEQUENCE OF Variable OPTIONAL,")
        if model.connections:
            lines.append("    connections     SEQUENCE OF Connection OPTIONAL,")
        if model.parameters:
            lines.append("    parameters      SEQUENCE OF Parameter OPTIONAL")
        
        lines.append("}")
        lines.append("")
        
        # Variable definition
        if model.variables:
            lines.append("Variable ::= SEQUENCE {")
            lines.append("    name            UTF8String,")
            lines.append("    varType         UTF8String,")
            lines.append("    dataType        UTF8String,")
            lines.append("    dimensions      SEQUENCE OF INTEGER OPTIONAL,")
            lines.append("    description     UTF8String OPTIONAL")
            lines.append("}")
            lines.append("")
        
        # Connection definition
        if model.connections:
            lines.append("Connection ::= SEQUENCE {")
            lines.append("    sourceVariables SEQUENCE OF UTF8String,")
            lines.append("    targetVariables SEQUENCE OF UTF8String,")
            lines.append("    connectionType  UTF8String,")
            lines.append("    weight          REAL OPTIONAL,")
            lines.append("    description     UTF8String OPTIONAL")
            lines.append("}")
            lines.append("")
        
        # Parameter definition
        if model.parameters:
            lines.append("Parameter ::= SEQUENCE {")
            lines.append("    name            UTF8String,")
            lines.append("    value           UTF8String,")
            lines.append("    typeHint        UTF8String OPTIONAL,")
            lines.append("    description     UTF8String OPTIONAL")
            lines.append("}")
            lines.append("")
        
        lines.append(f"END -- {model_name_clean}-Schema")
        lines.append("")
        
        # Embed complete model data as ASN.1 comment for round-trip fidelity
        import json
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
        
        # Add embedded JSON data as ASN.1 comment
        lines.append("-- MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')))
        
        return "\n".join(lines)
    
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

# Lean Serializer
class LeanSerializer(BaseGNNSerializer):
    """Serializer for Lean theorem prover format."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to Lean format."""
        lines = []
        
        # Header
        lines.append("-- GNN Model in Lean 4")
        lines.append(f"-- Model: {model.model_name}")
        lines.append(f"-- {model.annotation}")
        lines.append("")
        lines.append("import Mathlib.Data.Matrix.Basic")
        lines.append("import Mathlib.Data.Real.Basic")
        lines.append("")
        
        # Model namespace
        model_name_clean = model.model_name.replace(" ", "").replace("-", "")
        lines.append(f"namespace {model_name_clean}")
        lines.append("")
        
        # Variable definitions
        if model.variables:
            lines.append("-- Variables")
            for var in sorted(model.variables, key=lambda v: v.name):
                lean_type = self._map_to_lean_type(var.data_type.value)
                lines.append(f"variable ({var.name} : {lean_type})")
            lines.append("")
        
        # Structure definition
        lines.append(f"structure {model_name_clean}Model where")
        if model.variables:
            for var in sorted(model.variables, key=lambda v: v.name):
                lean_type = self._map_to_lean_type(var.data_type.value)
                lines.append(f"  {var.name} : {lean_type}")
        lines.append("")
        
        lines.append(f"end {model_name_clean}")
        
        # Embed complete model data as Lean comment for round-trip fidelity
        import json
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
        
        # Add embedded JSON data as Lean comment
        lines.append("-- MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')))
        lines.append("")
        
        return "\n".join(lines)
    
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
    
    def _map_to_lean_type(self, data_type: str) -> str:
        """Map GNN data types to Lean types."""
        mapping = {
            "categorical": "List ℕ",
            "continuous": "ℝ",
            "binary": "Bool",
            "integer": "ℤ",
            "float": "ℝ",
            "complex": "ℂ"
        }
        return mapping.get(data_type, "String")

# Coq Serializer
class CoqSerializer(BaseGNNSerializer):
    """Serializer for Coq format."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to Coq format."""
        lines = []
        
        # Header
        lines.append(f"(* GNN Model: {model.model_name} *)")
        lines.append(f"(* {model.annotation} *)")
        lines.append("")
        lines.append("Require Import Reals.")
        lines.append("Require Import List.")
        lines.append("")
        
        # Module
        model_name_clean = model.model_name.replace(" ", "").replace("-", "")
        lines.append(f"Module {model_name_clean}.")
        lines.append("")
        
        # Variable declarations
        if model.variables:
            lines.append("(* Variables *)")
            for var in sorted(model.variables, key=lambda v: v.name):
                coq_type = self._map_to_coq_type(var.data_type.value)
                lines.append(f"Parameter {var.name} : {coq_type}.")
            lines.append("")
        
        lines.append(f"End {model_name_clean}.")
        
        # Embed complete model data as Coq comment for round-trip fidelity
        import json
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
        
        # Add embedded JSON data as Coq comment
        lines.append("(* MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')) + " *)")
        lines.append("")
        
        return "\n".join(lines)
    
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
    
    def _map_to_coq_type(self, data_type: str) -> str:
        """Map GNN data types to Coq types."""
        mapping = {
            "categorical": "list nat",
            "continuous": "R",
            "binary": "bool",
            "integer": "Z",
            "float": "R",
            "complex": "string"
        }
        return mapping.get(data_type, "string")

# Python Serializer
class PythonSerializer(BaseGNNSerializer):
    """Serializer for Python format."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to Python format."""
        lines = []
        
        # Header
        lines.append(f'"""')
        lines.append(f'GNN Model: {model.model_name}')
        lines.append(f'{model.annotation}')
        lines.append(f'Generated: {datetime.now().isoformat()}')
        lines.append(f'"""')
        lines.append("")
        lines.append("import numpy as np")
        lines.append("from typing import Dict, List, Any")
        lines.append("")
        
        # Model class
        model_name_clean = model.model_name.replace(" ", "").replace("-", "")
        lines.append(f"class {model_name_clean}Model:")
        lines.append(f'    """GNN Model: {model.model_name}"""')
        lines.append("")
        lines.append("    def __init__(self):")
        lines.append(f'        self.model_name = "{model.model_name}"')
        lines.append(f'        self.version = "{model.version}"')
        lines.append(f'        self.annotation = "{model.annotation}"')
        lines.append("")
        
        # Variables
        if model.variables:
            lines.append("        # Variables")
            lines.append("        self.variables = {")
            for var in sorted(model.variables, key=lambda v: v.name):
                lines.append(f'            "{var.name}": {{')
                lines.append(f'                "type": "{var.var_type.value}",')
                lines.append(f'                "data_type": "{var.data_type.value}",')
                lines.append(f'                "dimensions": {var.dimensions},')
                if var.description:
                    lines.append(f'                "description": "{var.description}",')
                lines.append("            },")
            lines.append("        }")
            lines.append("")
        
        # Parameters
        if model.parameters:
            lines.append("        # Parameters")
            lines.append("        self.parameters = {")
            for param in sorted(model.parameters, key=lambda p: p.name):
                value_repr = repr(param.value) if isinstance(param.value, str) else str(param.value)
                lines.append(f'            "{param.name}": {value_repr},')
            lines.append("        }")
            lines.append("")
        
        # Embed complete model data as Python comment for round-trip fidelity
        import json
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
        
        # Add embedded JSON data as Python comment
        lines.append("# MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')))
        lines.append("")
        
        return "\n".join(lines)
    
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

# Grammar Serializer (BNF/EBNF)
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
            lines.append("<connection_op> ::= \">\" | \"-\" | \"->\"")
            lines.append("")
        
        # Parameters  
        if model.parameters:
            lines.append("<parameters> ::= <parameter> | <parameter> <parameters>")
            lines.append("<parameter> ::= <param_name> \"=\" <param_value>")
            param_names = []
            for param in sorted(model.parameters, key=lambda p: p.name):
                param_names.append(f'"{param.name}"')
            lines.append("<param_name> ::= " + " | ".join(param_names))
            lines.append("")
        
        # Embed complete model data as BNF comment for round-trip fidelity
        import json
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
        
        return "\n".join(lines)
    
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

# Isabelle Serializer
class IsabelleSerializer(BaseGNNSerializer):
    """Serializer for Isabelle/HOL format."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to Isabelle/HOL format."""
        lines = []
        
        # Theory header
        model_name_clean = model.model_name.replace(" ", "").replace("-", "")
        lines.append(f"theory {model_name_clean}")
        lines.append("imports Main")
        lines.append("begin")
        lines.append("")
        
        # Documentation
        lines.append(f'text \\<open>{model.model_name}\\<close>')
        lines.append(f'text \\<open>{model.annotation}\\<close>')
        lines.append("")
        
        # Variable types
        if model.variables:
            for var in sorted(model.variables, key=lambda v: v.name):
                isabelle_type = self._map_to_isabelle_type(var.data_type.value)
                lines.append(f'type_synonym {var.name} = "{isabelle_type}"')
            lines.append("")
        
        lines.append("end")
        
        # Embed complete model data as Isabelle comment for round-trip fidelity
        import json
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
        
        # Add embedded JSON data as Isabelle comment
        lines.append("(* MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')) + " *)")
        lines.append("")
        
        return "\n".join(lines)
    
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
    
    def _map_to_isabelle_type(self, data_type: str) -> str:
        """Map GNN data types to Isabelle types."""
        mapping = {
            "categorical": "nat list",
            "continuous": "real",
            "binary": "bool",
            "integer": "int",
            "float": "real",
            "complex": "string"
        }
        return mapping.get(data_type, "string")

# Maxima Serializer
class MaximaSerializer(BaseGNNSerializer):
    """Serializer for Maxima symbolic computation format with embedded data support."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to Maxima format with embedded data."""
        lines = []
        
        # Header
        lines.append(f"/* GNN Model: {model.model_name} */")
        lines.append(f"/* {model.annotation} */")
        lines.append("")
        
        # Variables
        if model.variables:
            lines.append("/* Variables */")
            for var in sorted(model.variables, key=lambda v: v.name):
                if var.dimensions:
                    dims_str = f"[{','.join(map(str, var.dimensions))}]"
                    lines.append(f"{var.name}: matrix{dims_str};")
                else:
                    lines.append(f"{var.name}: 0;")
            lines.append("")
        
        # Parameters
        if model.parameters:
            lines.append("/* Parameters */")
            for param in sorted(model.parameters, key=lambda p: p.name):
                lines.append(f"{param.name}: {param.value};")
            lines.append("")
        
        # Connections as function dependencies
        if model.connections:
            lines.append("/* Connections */")
            for i, conn in enumerate(model.connections):
                if hasattr(conn, 'source_variables') and hasattr(conn, 'target_variables'):
                    sources = conn.source_variables
                    targets = conn.target_variables
                    for target in targets:
                        for source in sources:
                            lines.append(f"/* {target} depends on {source} */")
            lines.append("")
        
        # Equations
        if model.equations:
            lines.append("/* Equations */")
            for eq in model.equations:
                if hasattr(eq, 'content'):
                    lines.append(f"/* {eq.content} */")
            lines.append("")
        
        # Embed complete model data as Maxima comment for round-trip fidelity
        import json
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
        
        # Add embedded JSON data as Maxima comment
        lines.append("/* MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')) + " */")
        lines.append("")
        
        return "\n".join(lines)
    
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

# Alloy Serializer  
class AlloySerializer(BaseGNNSerializer):
    """Serializer for Alloy model checking language."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to Alloy format with embedded data."""
        lines = []
        
        # Module declaration
        model_name_clean = model.model_name.replace(" ", "").replace("-", "")
        lines.append(f"module {model_name_clean}")
        lines.append("")
        
        # Model documentation
        lines.append("// GNN Model: " + model.model_name)
        if model.annotation:
            lines.append("// " + model.annotation)
        lines.append("")
        
        # Signatures for variables
        if model.variables:
            for var in sorted(model.variables, key=lambda v: v.name):
                alloy_type = self._map_to_alloy_type(var.data_type.value)
                lines.append(f"sig {var.name} {{")
                lines.append(f"  value: {alloy_type}")
                lines.append("}")
                lines.append("")
        
        # Facts for connections
        if model.connections:
            lines.append("fact Connections {")
            for conn in model.connections:
                if hasattr(conn, 'source_variables') and hasattr(conn, 'target_variables'):
                    sources = conn.source_variables
                    targets = conn.target_variables
                    for source in sources:
                        for target in targets:
                            lines.append(f"  {source} -> {target}")
                lines.append("")
            lines.append("}")
            lines.append("")
        
        # Predicates for model constraints
        lines.append(f"pred {model_name_clean}Valid {{")
        lines.append("  // Model constraints")
        if model.variables:
            for var in model.variables:
                lines.append(f"  some {var.name}")
        lines.append("}")
        lines.append("")
        
        # Embed complete model data as Alloy comment for round-trip fidelity
        import json
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
        
        # Add embedded JSON data as Alloy comment
        lines.append("/* MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')) + " */")
        lines.append("")
        
        return "\n".join(lines)
    
    def _map_to_alloy_type(self, data_type: str) -> str:
        """Map GNN data types to Alloy types."""
        mapping = {
            "categorical": "Int",
            "continuous": "Int", # Alloy uses Int for numeric values
            "binary": "Int",
            "integer": "Int",
            "float": "Int",
            "complex": "univ"
        }
        return mapping.get(data_type, "univ")
    
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

# Z-notation Serializer
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
        import json
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
        
        return "\n".join(lines)
    
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

# Schema Serializer (for XSD, ASN.1)
class SchemaSerializer(BaseGNNSerializer):
    """Serializer for formal schema languages."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to generic schema format."""
        # Default to XSD format
        return XSDSerializer().serialize(model)

# Temporal Serializer (TLA+, Agda)
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
        import json
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
        
        return "\n".join(lines)
    
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
        
        # Module header
        model_name_clean = model.model_name.replace(" ", "").replace("-", "")
        lines.append(f"module {model_name_clean} where")
        lines.append("")
        lines.append("open import Data.Nat")
        lines.append("open import Data.List")
        lines.append("open import Data.Bool")
        lines.append("")
        
        # Data types for variables
        if model.variables:
            lines.append("-- Variables as data types")
            for var in sorted(model.variables, key=lambda v: v.name):
                agda_type = self._map_to_agda_type(var.data_type.value)
                lines.append(f"{var.name} : {agda_type}")
            lines.append("")
        
        # Parameters as definitions
        if model.parameters:
            lines.append("-- Parameters")
            for param in sorted(model.parameters, key=lambda p: p.name):
                if isinstance(param.value, (int, float)):
                    lines.append(f"{param.name} : ℕ")
                    lines.append(f"{param.name} = {param.value}")
                else:
                    lines.append(f"{param.name} : Set")
            lines.append("")
        
        # Model structure
        lines.append(f"data {model_name_clean}Model : Set where")
        if model.variables:
            lines.append("  model : ")
            var_types = [f"{var.name}" for var in sorted(model.variables, key=lambda v: v.name)]
            if var_types:
                lines.append(f"    {' → '.join(var_types)} → {model_name_clean}Model")
        lines.append("")
        
        # Embed complete model data as Agda comment for round-trip fidelity
        import json
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
        lines.append("-- MODEL_DATA: " + json.dumps(model_data, separators=(',', ':')))
        lines.append("")
        
        return "\n".join(lines)
    
    def _map_to_agda_type(self, data_type: str) -> str:
        """Map GNN data types to Agda types."""
        mapping = {
            "categorical": "List ℕ",
            "continuous": "ℝ",
            "binary": "Bool",
            "integer": "ℕ",
            "float": "ℝ",
            "complex": "String"
        }
        return mapping.get(data_type, "Set")

# Agda Serializer (separate from TLA+)
class AgdaSerializer(TemporalSerializer):
    """Serializer specifically for Agda format."""
    
    def __init__(self):
        super().__init__(target_format="agda")

# Functional Serializer (Haskell)
class FunctionalSerializer(BaseGNNSerializer):
    """Serializer for functional programming languages."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to Haskell format."""
        lines = []
        
        # Module header
        model_name_clean = model.model_name.replace(" ", "").replace("-", "")
        lines.append(f"module {model_name_clean} where")
        lines.append("")
        
        # Imports
        lines.append("import Data.Matrix")
        lines.append("import Data.List")
        lines.append("")
        
        # Data types
        if model.variables:
            for var in sorted(model.variables, key=lambda v: v.name):
                haskell_type = self._map_to_haskell_type(var.data_type.value)
                lines.append(f"type {var.name} = {haskell_type}")
            lines.append("")
        
        # Model data structure
        lines.append(f"data {model_name_clean}Model = {model_name_clean}Model")
        if model.variables:
            lines.append("  {")
            for i, var in enumerate(sorted(model.variables, key=lambda v: v.name)):
                prefix = "  ," if i > 0 else "  "
                lines.append(f"{prefix} {var.name.lower()}Field :: {var.name}")
            lines.append("  }")
        lines.append("")
        
        # Embed complete model data as Haskell comment for round-trip fidelity
        import json
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
        
        return "\n".join(lines)
    
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

# Binary Serializer
class BinarySerializer(BaseGNNSerializer):
    """Serializer for binary formats (Pickle) with enhanced round-trip support."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to pickle format with embedded JSON data for round-trip."""
        # Create a comprehensive data structure for pickling
        import json
        
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
        
        # Pickle the complete data
        pickled_data = pickle.dumps(complete_model_data)
        
        # Return as base64 for text-based storage/transmission
        return base64.b64encode(pickled_data).decode('ascii')
    
    def serialize_to_file(self, model: GNNInternalRepresentation, file_path: str) -> None:
        """Serialize model to binary file."""
        # Create complete model data
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
        """Serialize directly to pickle file without base64 encoding."""
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