"""
Schema Parsers for GNN

This module provides parsing capabilities for various schema languages:
- XSD (XML Schema Definition)
- ASN.1 (Abstract Syntax Notation One)
- Alloy (Model checking language)
- Z notation (Formal specification language)

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from pathlib import Path

from .common import (
    BaseGNNParser, ParseResult, GNNInternalRepresentation,
    Variable, Connection, Parameter, Equation, TimeSpecification,
    VariableType, DataType, ConnectionType, ParseError
)

class XSDParser(BaseGNNParser):
    """Parser for XML Schema Definition (XSD) files."""
    
    def __init__(self):
        super().__init__()
        
    def get_supported_extensions(self) -> List[str]:
        return ['.xsd']
    
    def parse_file(self, file_path: str) -> ParseResult:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            return self.parse_xml_element(root)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to parse XSD file: {e}")
            return result
    
    def parse_string(self, content: str) -> ParseResult:
        try:
            root = ET.fromstring(content)
            return self.parse_xml_element(root)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to parse XSD content: {e}")
            return result
    
    def parse_xml_element(self, root: ET.Element) -> ParseResult:
        result = ParseResult(model=self.create_empty_model())
        result.model.model_name = root.get('targetNamespace', 'XSDModel').split('/')[-1]
        
        # Parse element definitions
        for elem in root.findall('.//{http://www.w3.org/2001/XMLSchema}element'):
            name = elem.get('name')
            type_attr = elem.get('type', 'string')
            
            if name:
                variable = Variable(
                    name=name,
                    var_type=self._infer_variable_type(name),
                    dimensions=[1],
                    data_type=self._map_xsd_type(type_attr),
                    description=f"XSD element of type {type_attr}"
                )
                result.model.variables.append(variable)
        
        return result
    
    def _map_xsd_type(self, xsd_type: str) -> DataType:
        type_mapping = {
            'string': DataType.CATEGORICAL,
            'int': DataType.INTEGER,
            'integer': DataType.INTEGER,
            'float': DataType.FLOAT,
            'double': DataType.FLOAT,
            'boolean': DataType.BINARY,
            'decimal': DataType.FLOAT
        }
        return type_mapping.get(xsd_type.split(':')[-1], DataType.CATEGORICAL)
    
    def _infer_variable_type(self, name: str) -> VariableType:
        name_lower = name.lower()
        if 'state' in name_lower:
            return VariableType.HIDDEN_STATE
        elif 'observation' in name_lower:
            return VariableType.OBSERVATION
        else:
            return VariableType.HIDDEN_STATE


class ASN1Parser(BaseGNNParser):
    """Parser for ASN.1 (Abstract Syntax Notation One) specifications."""
    
    def __init__(self):
        super().__init__()
        self.module_pattern = re.compile(r'(\w+)\s+DEFINITIONS\s+::=\s+BEGIN(.*?)END', re.DOTALL | re.IGNORECASE)
        self.type_pattern = re.compile(r'(\w+)\s+::=\s+(.+?)(?=\n\w+\s+::=|\nEND|$)', re.DOTALL)
        
    def get_supported_extensions(self) -> List[str]:
        return ['.asn1', '.asn']
    
    def parse_file(self, file_path: str) -> ParseResult:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to read ASN.1 file: {e}")
            return result
    
    def parse_string(self, content: str) -> ParseResult:
        result = ParseResult(model=self.create_empty_model())
        
        # Parse module
        module_match = self.module_pattern.search(content)
        if module_match:
            result.model.model_name = module_match.group(1)
            module_content = module_match.group(2)
            
            # Parse type definitions
            for match in self.type_pattern.finditer(module_content):
                type_name = match.group(1)
                type_def = match.group(2).strip()
                
                variable = Variable(
                    name=type_name,
                    var_type=self._infer_variable_type(type_name),
                    dimensions=self._parse_asn1_dimensions(type_def),
                    data_type=self._map_asn1_type(type_def),
                    description=f"ASN.1 type: {type_def[:50]}..."
                )
                result.model.variables.append(variable)
        
        return result
    
    def _parse_asn1_dimensions(self, type_def: str) -> List[int]:
        if 'SEQUENCE OF' in type_def:
            return [1]  # Assume 1D sequence
        elif 'SEQUENCE' in type_def:
            # Count fields in sequence
            fields = re.findall(r'\w+\s+\w+', type_def)
            return [len(fields)] if fields else [1]
        return [1]
    
    def _map_asn1_type(self, type_def: str) -> DataType:
        if 'INTEGER' in type_def:
            return DataType.INTEGER
        elif 'REAL' in type_def:
            return DataType.FLOAT
        elif 'BOOLEAN' in type_def:
            return DataType.BINARY
        elif 'ENUMERATED' in type_def:
            return DataType.CATEGORICAL
        else:
            return DataType.CATEGORICAL
    
    def _infer_variable_type(self, name: str) -> VariableType:
        name_lower = name.lower()
        if 'state' in name_lower:
            return VariableType.HIDDEN_STATE
        elif 'observation' in name_lower:
            return VariableType.OBSERVATION
        else:
            return VariableType.HIDDEN_STATE


class AlloyParser(BaseGNNParser):
    """Parser for Alloy model checking language specifications."""
    
    def __init__(self):
        super().__init__()
        self.sig_pattern = re.compile(r'sig\s+(\w+)\s*(?:extends\s+(\w+))?\s*\{([^}]*)\}', re.MULTILINE | re.DOTALL)
        self.fact_pattern = re.compile(r'fact\s+(\w+)?\s*\{([^}]*)\}', re.MULTILINE | re.DOTALL)
        self.pred_pattern = re.compile(r'pred\s+(\w+)\s*\[([^\]]*)\]\s*\{([^}]*)\}', re.MULTILINE | re.DOTALL)
        
    def get_supported_extensions(self) -> List[str]:
        return ['.als']
    
    def parse_file(self, file_path: str) -> ParseResult:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to read Alloy file: {e}")
            return result
    
    def parse_string(self, content: str) -> ParseResult:
        result = ParseResult(model=self.create_empty_model())
        result.model.model_name = "AlloyModel"
        
        # Parse signatures (variables)
        for match in self.sig_pattern.finditer(content):
            sig_name = match.group(1)
            parent_sig = match.group(2)
            sig_body = match.group(3)
            
            variable = Variable(
                name=sig_name,
                var_type=self._infer_variable_type(sig_name),
                dimensions=[1],
                data_type=DataType.CATEGORICAL,
                description=f"Alloy signature{' extending ' + parent_sig if parent_sig else ''}"
            )
            result.model.variables.append(variable)
            
            # Create connection to parent if exists
            if parent_sig:
                connection = Connection(
                    source_variables=[parent_sig],
                    target_variables=[sig_name],
                    connection_type=ConnectionType.DIRECTED,
                    description=f"Alloy inheritance: {parent_sig} -> {sig_name}"
                )
                result.model.connections.append(connection)
        
        # Parse facts (constraints/equations)
        for match in self.fact_pattern.finditer(content):
            fact_name = match.group(1) or f"fact_{len(result.model.equations)}"
            fact_body = match.group(2)
            
            equation = Equation(
                label=fact_name,
                content=fact_body.strip(),
                format="alloy",
                description="Alloy fact (constraint)"
            )
            result.model.equations.append(equation)
        
        return result
    
    def _infer_variable_type(self, name: str) -> VariableType:
        name_lower = name.lower()
        if 'state' in name_lower:
            return VariableType.HIDDEN_STATE
        elif 'observation' in name_lower:
            return VariableType.OBSERVATION
        else:
            return VariableType.HIDDEN_STATE


class ZNotationParser(BaseGNNParser):
    """Parser for Z notation formal specifications."""
    
    def __init__(self):
        super().__init__()
        self.schema_pattern = re.compile(r'(\w+)\s*=\s*\[\s*([^]]*)\s*\|\s*([^]]*)\s*\]', re.MULTILINE | re.DOTALL)
        self.type_pattern = re.compile(r'(\w+)\s*:\s*([^\n;]+)', re.MULTILINE)
        
    def get_supported_extensions(self) -> List[str]:
        return ['.zed', '.z']
    
    def parse_file(self, file_path: str) -> ParseResult:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to read Z notation file: {e}")
            return result
    
    def parse_string(self, content: str) -> ParseResult:
        result = ParseResult(model=self.create_empty_model())
        result.model.model_name = "ZNotationModel"
        
        # Parse schemas
        for match in self.schema_pattern.finditer(content):
            schema_name = match.group(1)
            declarations = match.group(2)
            predicates = match.group(3)
            
            # Parse variable declarations
            for type_match in self.type_pattern.finditer(declarations):
                var_name = type_match.group(1)
                var_type = type_match.group(2).strip()
                
                variable = Variable(
                    name=var_name,
                    var_type=self._infer_variable_type(var_name),
                    dimensions=[1],
                    data_type=self._map_z_type(var_type),
                    description=f"Z notation variable of type {var_type}"
                )
                result.model.variables.append(variable)
            
            # Add predicates as equations
            if predicates.strip():
                equation = Equation(
                    label=f"{schema_name}_predicate",
                    content=predicates.strip(),
                    format="z_notation",
                    description=f"Z notation predicate for schema {schema_name}"
                )
                result.model.equations.append(equation)
        
        return result
    
    def _map_z_type(self, z_type: str) -> DataType:
        if 'ℕ' in z_type or 'nat' in z_type.lower():
            return DataType.INTEGER
        elif 'ℝ' in z_type or 'real' in z_type.lower():
            return DataType.FLOAT
        elif 'BOOL' in z_type.upper():
            return DataType.BINARY
        else:
            return DataType.CATEGORICAL
    
    def _infer_variable_type(self, name: str) -> VariableType:
        name_lower = name.lower()
        if 'state' in name_lower:
            return VariableType.HIDDEN_STATE
        elif 'observation' in name_lower:
            return VariableType.OBSERVATION
        else:
            return VariableType.HIDDEN_STATE


# Compatibility aliases
XSDGNNParser = XSDParser
ASN1GNNParser = ASN1Parser
AlloyGNNParser = AlloyParser
ZNotationGNNParser = ZNotationParser 