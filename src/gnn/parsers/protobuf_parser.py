"""
Protocol Buffers Parser for GNN

This module provides parsing capabilities for Protocol Buffers serialization
format specifications of GNN models.

Author: @docxology
Date: 2025-01-11
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from .common import (
    BaseGNNParser, ParseResult, GNNInternalRepresentation,
    Variable, Connection, Parameter, Equation, TimeSpecification,
    VariableType, DataType, ConnectionType, ParseError
)

class ProtobufGNNParser(BaseGNNParser):
    """Parser for Protocol Buffers specifications."""
    
    def __init__(self):
        super().__init__()
        self.message_pattern = re.compile(r'message\s+(\w+)\s*\{([^}]+)\}', re.MULTILINE | re.DOTALL)
        self.field_pattern = re.compile(r'(?:optional|required|repeated)?\s*(\w+)\s+(\w+)\s*=\s*(\d+)(?:\s*\[([^\]]+)\])?;', re.MULTILINE)
        self.enum_pattern = re.compile(r'enum\s+(\w+)\s*\{([^}]+)\}', re.MULTILINE | re.DOTALL)
        self.package_pattern = re.compile(r'package\s+([\w.]+);')
        
    def get_supported_extensions(self) -> List[str]:
        """Get file extensions supported by this parser."""
        return ['.proto']
    
    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a Protocol Buffers file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to read file: {e}")
            return result
    
    def parse_string(self, content: str) -> ParseResult:
        """Parse Protocol Buffers content string."""
        result = ParseResult(model=self.create_empty_model())
        
        try:
            # Extract package name as model name
            package_match = self.package_pattern.search(content)
            if package_match:
                result.model.model_name = package_match.group(1).replace('.', '_')
            else:
                result.model.model_name = "ProtobufModel"
            
            # Parse enums (discrete variable types)
            for match in self.enum_pattern.finditer(content):
                enum_name = match.group(1)
                enum_body = match.group(2)
                
                variable = Variable(
                    name=enum_name,
                    var_type=self._infer_variable_type(enum_name),
                    dimensions=[self._count_enum_values(enum_body)],
                    data_type=DataType.CATEGORICAL,
                    description=f"Protobuf enum: {enum_body.strip()[:50]}..."
                )
                result.model.variables.append(variable)
            
            # Parse messages (structured variables)
            for match in self.message_pattern.finditer(content):
                msg_name = match.group(1)
                msg_body = match.group(2)
                
                # Parse fields within the message
                fields = self._parse_message_fields(msg_body)
                
                # Create variable for the message
                variable = Variable(
                    name=msg_name,
                    var_type=self._infer_variable_type(msg_name),
                    dimensions=[len(fields)] if fields else [1],
                    data_type=DataType.CATEGORICAL,
                    description=f"Protobuf message with {len(fields)} fields"
                )
                result.model.variables.append(variable)
                
                # Create parameters for field default values
                for field in fields:
                    if 'default' in field:
                        parameter = Parameter(
                            name=f"{msg_name}_{field['name']}_default",
                            value=field['default'],
                            type_hint=field['type'],
                            description=f"Default value for {msg_name}.{field['name']}"
                        )
                        result.model.parameters.append(parameter)
                
                # Create connections between message fields
                connections = self._extract_field_connections(msg_name, fields)
                result.model.connections.extend(connections)
            
            result.model.annotation = "Parsed from Protocol Buffers definition"
            
        except Exception as e:
            result.add_error(f"Parsing error: {e}")
        
        return result
    
    def _count_enum_values(self, enum_body: str) -> int:
        """Count the number of values in an enum."""
        values = re.findall(r'\w+\s*=\s*\d+', enum_body)
        return len(values) if values else 1
    
    def _parse_message_fields(self, msg_body: str) -> List[Dict[str, Any]]:
        """Parse fields from a message body."""
        fields = []
        
        for match in self.field_pattern.finditer(msg_body):
            field_type = match.group(1)
            field_name = match.group(2)
            field_number = int(match.group(3))
            field_options = match.group(4) if match.group(4) else ""
            
            field_info = {
                'name': field_name,
                'type': field_type,
                'number': field_number,
                'options': field_options
            }
            
            # Parse default values from options
            if 'default' in field_options:
                default_match = re.search(r'default\s*=\s*([^,\]]+)', field_options)
                if default_match:
                    field_info['default'] = default_match.group(1).strip('"\'')
            
            fields.append(field_info)
        
        return fields
    
    def _infer_variable_type(self, name: str) -> VariableType:
        """Infer variable type from name."""
        name_lower = name.lower()
        if 'state' in name_lower or 's_' in name_lower:
            return VariableType.HIDDEN_STATE
        elif 'observation' in name_lower or 'obs' in name_lower or 'o_' in name_lower:
            return VariableType.OBSERVATION
        elif 'action' in name_lower or 'u_' in name_lower:
            return VariableType.ACTION
        elif 'policy' in name_lower or 'pi_' in name_lower:
            return VariableType.POLICY
        elif 'matrix' in name_lower:
            if 'likelihood' in name_lower or 'a_' in name_lower:
                return VariableType.LIKELIHOOD_MATRIX
            elif 'transition' in name_lower or 'b_' in name_lower:
                return VariableType.TRANSITION_MATRIX
            else:
                return VariableType.LIKELIHOOD_MATRIX
        elif 'preference' in name_lower or 'c_' in name_lower:
            return VariableType.PREFERENCE_VECTOR
        elif 'prior' in name_lower or 'd_' in name_lower:
            return VariableType.PRIOR_VECTOR
        else:
            return VariableType.HIDDEN_STATE
    
    def _extract_field_connections(self, msg_name: str, fields: List[Dict[str, Any]]) -> List[Connection]:
        """Extract connections between message fields."""
        connections = []
        
        # Create connections between fields that reference other message types
        for field in fields:
            field_type = field['type']
            field_name = field['name']
            
            # If field type is another custom message (starts with uppercase)
            if field_type[0].isupper() and field_type != msg_name:
                connection = Connection(
                    source_variables=[field_type],
                    target_variables=[f"{msg_name}_{field_name}"],
                    connection_type=ConnectionType.DIRECTED,
                    description=f"Protobuf field reference: {field_type} -> {msg_name}.{field_name}"
                )
                connections.append(connection)
        
        return connections


# Compatibility alias
ProtobufParser = ProtobufGNNParser 