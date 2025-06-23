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
    """Serializer for JSON data interchange."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to JSON format."""
        return json.dumps(model.to_dict(), indent=2, ensure_ascii=False)

# XML Serializer  
class XMLSerializer(BaseGNNSerializer):
    """Serializer for XML format."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to XML format."""
        root = ET.Element("gnn_model")
        root.set("name", model.model_name)
        root.set("version", model.version)
        
        # Metadata
        metadata = ET.SubElement(root, "metadata")
        ET.SubElement(metadata, "annotation").text = model.annotation or ""
        ET.SubElement(metadata, "created_at").text = model.created_at.isoformat()
        
        # Variables
        if model.variables:
            variables = ET.SubElement(root, "variables")
            for var in model.variables:
                var_elem = ET.SubElement(variables, "variable")
                var_elem.set("name", var.name)
                var_elem.set("type", var.var_type.value)
                var_elem.set("data_type", var.data_type.value)
                if var.dimensions:
                    var_elem.set("dimensions", ",".join(map(str, var.dimensions)))
        
        # Connections
        if model.connections:
            connections = ET.SubElement(root, "connections")
            for conn in model.connections:
                conn_elem = ET.SubElement(connections, "connection")
                conn_elem.set("type", conn.connection_type.value)
                ET.SubElement(conn_elem, "sources").text = ",".join(conn.source_variables)
                ET.SubElement(conn_elem, "targets").text = ",".join(conn.target_variables)
        
        # Format XML
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

# YAML Serializer
class YAMLSerializer(BaseGNNSerializer):
    """Serializer for YAML configuration format."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to YAML format."""
        if not HAS_YAML:
            # Fallback to JSON-like format
            data = model.to_dict()
            return self._dict_to_yaml_like(data)
        
        data = model.to_dict()
        return yaml.dump(data, default_flow_style=False, allow_unicode=True)
    
    def _dict_to_yaml_like(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Convert dict to YAML-like format when PyYAML is not available."""
        lines = []
        spaces = "  " * indent
        
        for key, value in data.items():
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
            for var in model.variables:
                var_type = self._map_variable_type(var)
                lines.append(f"  type {var.name} = {var_type}")
            lines.append("")
        
        # Morphisms (connections)
        if model.connections:
            lines.append("  // Morphisms")
            for conn in model.connections:
                for src in conn.source_variables:
                    for tgt in conn.target_variables:
                        lines.append(f"  val {src}To{tgt}: {src} => {tgt} = identity")
            lines.append("")
        
        lines.append("}")
        
        return "\n".join(lines)
    
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

# Additional serializers with minimal implementations
class LeanSerializer(BaseGNNSerializer):
    def serialize(self, model: GNNInternalRepresentation) -> str:
        return f"-- Lean implementation for {model.model_name}\n-- TODO: Implement Lean serialization"

class CoqSerializer(BaseGNNSerializer):
    def serialize(self, model: GNNInternalRepresentation) -> str:
        return f"(* Coq implementation for {model.model_name} *)\n(* TODO: Implement Coq serialization *)"

class PythonSerializer(BaseGNNSerializer):
    def serialize(self, model: GNNInternalRepresentation) -> str:
        return f'"""{model.model_name} - Python implementation"""\n# TODO: Implement Python serialization'

class GrammarSerializer(BaseGNNSerializer):
    def serialize(self, model: GNNInternalRepresentation) -> str:
        return f"# Grammar for {model.model_name}\n# TODO: Implement grammar serialization"

class IsabelleSerializer(BaseGNNSerializer):
    def serialize(self, model: GNNInternalRepresentation) -> str:
        return f"theory {model.model_name.replace(' ', '')}\nimports Main\nbegin\n(* TODO: Implement *)\nend"

class MaximaSerializer(BaseGNNSerializer):
    def serialize(self, model: GNNInternalRepresentation) -> str:
        return f"/* {model.model_name} - Maxima implementation */\n/* TODO: Implement Maxima serialization */"

class ProtobufSerializer(BaseGNNSerializer):
    def serialize(self, model: GNNInternalRepresentation) -> str:
        return f'syntax = "proto3";\n// {model.model_name}\n// TODO: Implement protobuf serialization'

class SchemaSerializer(BaseGNNSerializer):
    def serialize(self, model: GNNInternalRepresentation) -> str:
        return f"<!-- Schema for {model.model_name} -->\n<!-- TODO: Implement schema serialization -->"

class TemporalSerializer(BaseGNNSerializer):
    def serialize(self, model: GNNInternalRepresentation) -> str:
        return f"---- MODULE {model.model_name.replace(' ', '')} ----\n---- TODO: Implement temporal logic ----"

class FunctionalSerializer(BaseGNNSerializer):
    def serialize(self, model: GNNInternalRepresentation) -> str:
        return f"-- {model.model_name}\n-- TODO: Implement functional serialization"

class BinarySerializer(BaseGNNSerializer):
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to binary format (base64 encoded)."""
        import base64
        pickled_data = pickle.dumps(model.to_dict())
        return base64.b64encode(pickled_data).decode('ascii')
    
    def serialize_to_file(self, model: GNNInternalRepresentation, file_path: str) -> None:
        """Serialize model to binary file."""
        with open(file_path, 'wb') as f:
            pickle.dump(model.to_dict(), f) 